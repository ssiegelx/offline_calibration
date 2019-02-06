import os
import sys
import inspect
import datetime
import time
import pickle
import gc

import numpy as np
import h5py

import scipy.constants

from sklearn import gaussian_process
from sklearn.gaussian_process.kernels import Matern, ConstantKernel

import log

from pychfpga import NameSpace, load_yaml_config
from calibration import utils

from ch_util import andata, tools, ephemeris, timing
from ch_util.fluxcat import FluxCatalog

###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':point_source.analysis'))

LOG_FILE = os.environ.get('CALIBRATION_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'offline_cal.log'))

DEFAULT_LOGGING = {
    'formatters': {
         'std': {
             'format': "%(asctime)s %(levelname)s %(name)s: %(message)s",
             'datefmt': "%m/%d %H:%M:%S"},
          },
    'handlers': {
        'stderr': {'class': 'logging.StreamHandler', 'formatter': 'std', 'level': 'DEBUG'}
        },
    'loggers': {
        '': {'handlers': ['stderr'], 'level': 'INFO'}  # root logger

        }
    }

###################################################
# main routine
###################################################

def offline_point_source_calibration(file_list, source, inputmap=None, start=None, stop=None,
                                                        physical_freq=None, tcorr=None,
                                                        logging_params=DEFAULT_LOGGING,
                                                        **kwargs):
    # Load config
    config = DEFAULTS.deepcopy()
    config.merge(NameSpace(kwargs))

    # Setup logging
    log.setup_logging(logging_params)
    mlog = log.get_logger(__name__)

    mlog.info("ephemeris file: %s" % ephemeris.__file__)

    # Set the model to use
    fitter_function = utils.fit_point_source_transit
    model_function = utils.model_point_source_transit

    farg = inspect.getargspec(fitter_function)
    defaults = {key:val for key, val in zip(farg.args[-len(farg.defaults):],farg.defaults)}
    poly_deg_amp = kwargs.get('poly_deg_amp', defaults['poly_deg_amp'])
    poly_deg_phi = kwargs.get('poly_deg_phi', defaults['poly_deg_phi'])
    poly_type = kwargs.get('poly_type', defaults['poly_type'])

    param_name = (['%s_poly_amp_coeff%d' % (poly_type, cc) for cc in range(poly_deg_amp+1)] +
                  ['%s_poly_phi_coeff%d' % (poly_type, cc) for cc in range(poly_deg_phi+1)])

    model_kwargs = [('poly_deg_amp', poly_deg_amp), ('poly_deg_phi', poly_deg_phi), ('poly_type', poly_type)]
    model_name = '.'.join([getattr(model_function, key) for key in ['__module__', '__name__']])

    tval = {}

    # Set where to evaluate gain
    ha_eval_str = ['raw_transit']

    if config.multi_sample:
        ha_eval_str += ['transit', 'peak']
        ha_eval = [0.0, None]
        fitslc = slice(1, 3)

    ind_eval = ha_eval_str.index(config.evaluate_gain_at)

    # Determine dimensions
    direction = ['amp', 'phi']
    nparam = len(param_name)
    ngain = len(ha_eval_str)
    ndir = len(direction)

    # Determine frequencies
    data = andata.CorrData.from_acq_h5(file_list, datasets=(), start=start, stop=stop)
    freq = data.freq

    if physical_freq is not None:
        index_freq = np.array([np.argmin(np.abs(ff - freq)) for ff in physical_freq])
        freq_sel = utils.convert_to_slice(index_freq)
        freq = freq[index_freq]
    else:
        index_freq = np.arange(freq.size)
        freq_sel = None

    nfreq = freq.size

    # Compute flux of source
    inv_rt_flux_density = tools.invert_no_zero(np.sqrt(FluxCatalog[source].predict_flux(freq)))

    # Read in the eigenvaluess for all frequencies
    data = andata.CorrData.from_acq_h5(file_list, datasets=['erms', 'eval'], freq_sel=freq_sel, start=start, stop=stop)

    # Determine source coordinates
    this_csd = np.floor(ephemeris.unix_to_csd(np.median(data.time)))
    timestamp0 = ephemeris.transit_times(FluxCatalog[source].skyfield, ephemeris.csd_to_unix(this_csd))[0]
    src_ra, src_dec = ephemeris.object_coords(FluxCatalog[source].skyfield, date=timestamp0, deg=True)

    ra = ephemeris.lsa(data.time)
    ha = ra - src_ra
    ha = ha - (ha > 180.0) * 360.0 + (ha < -180.0) * 360.0
    ha = np.radians(ha)

    itrans = np.argmin(np.abs(ha))

    window = 0.75 * np.max(np.abs(ha))

    off_source = np.abs(ha) > window

    mlog.info("CSD %d" % this_csd)
    mlog.info("Hour angle at transit (%d of %d):  %0.2f deg   " % (itrans, len(ha), np.degrees(ha[itrans])))
    mlog.info("Hour angle off source: %0.2f deg" % np.median(np.abs(np.degrees(ha[off_source]))))

    src_dec = np.radians(src_dec)
    lat = np.radians(ephemeris.CHIMELATITUDE)

    # Determine division of frequencies
    ninput = data.ninput
    ntime = data.ntime
    nblock_freq = int(np.ceil(nfreq / float(config.nfreq_per_block)))

    # Determine bad inputs
    eps = 10.0 * np.finfo(data['erms'].dtype).eps
    good_freq = np.flatnonzero(np.all(data['erms'][:] > eps, axis=-1))
    ind_sub_freq = good_freq[slice(0, good_freq.size, max(int(good_freq.size / 10), 1))]

    tmp_data = andata.CorrData.from_acq_h5(file_list, datasets=['evec'], freq_sel=ind_sub_freq, start=start, stop=stop)
    eps = 10.0 * np.finfo(tmp_data['evec'].dtype).eps
    bad_input = np.flatnonzero(np.all(np.abs(tmp_data['evec'][:, 0]) < eps,  axis=(0, 2)))

    input_axis = tmp_data.input.copy()

    del tmp_data

    # Query layout database for correlator inputs
    if inputmap is None:
        inputmap = tools.get_correlator_inputs(datetime.datetime.utcfromtimestamp(data.time[itrans]), correlator='chime')

    inputmap = tools.reorder_correlator_inputs(input_axis, inputmap)

    tools.change_chime_location(rotation=config.telescope_rotation)

    # Determine x and y pol index
    xfeeds = np.array([idf for idf, inp in enumerate(inputmap) if (idf not in bad_input) and tools.is_array_x(inp)])
    yfeeds = np.array([idf for idf, inp in enumerate(inputmap) if (idf not in bad_input) and tools.is_array_y(inp)])

    nfeed = xfeeds.size + yfeeds.size

    pol = [yfeeds,  xfeeds]
    polstr = ['Y', 'X']
    npol = len(pol)

    neigen = min(max(npol, config.neigen), data['eval'].shape[1])

    phase_ref = config.phase_reference_index
    phase_ref_by_pol = [pol[pp].tolist().index(phase_ref[pp]) for pp in range(npol)]

    # Calculate dynamic range
    eval0_off_source = np.median(data['eval'][:, 0, off_source], axis=-1)

    dyn = data['eval'][:, 1, :] * tools.invert_no_zero(eval0_off_source[:, np.newaxis])

    # Determine frequencies to mask
    not_rfi = np.ones((nfreq, 1), dtype=np.bool)
    if config.mask_rfi is not None:
        for frng in config.mask_rfi:
            not_rfi[:, 0] &= ((freq < frng[0]) | (freq > frng[1]))

    mlog.info("%0.1f percent of frequencies available after masking RFI." %
             (100.0 * np.sum(not_rfi, dtype=np.float32) / float(nfreq),))

    #dyn_flg = utils.contiguous_flag(dyn > config.dyn_rng_threshold, centre=itrans)
    if source in config.dyn_rng_threshold:
        dyn_rng_threshold = config.dyn_rng_threshold[source]
    else:
        dyn_rng_threshold = config.dyn_rng_threshold.default

    mlog.info("Dynamic range threshold set to %0.1f." % dyn_rng_threshold)

    dyn_flg = dyn > dyn_rng_threshold

    # Calculate fit flag
    fit_flag = np.zeros((nfreq, npol, ntime), dtype=np.bool)
    for pp in range(npol):

        mlog.info("Dynamic Range Nsample, Pol %d:  %s" %
                  (pp, ','.join(["%d" % xx for xx in np.percentile(np.sum(dyn_flg, axis=-1), [25, 50, 75, 100])])))

        if config.nsigma1 is None:
            fit_flag[:, pp, :] = dyn_flg & not_rfi

        else:

            fit_window = config.nsigma1 * np.radians(utils.get_window(freq, pol=polstr[pp], dec=src_dec, deg=True))

            win_flg = np.abs(ha)[np.newaxis, :] <= fit_window[:, np.newaxis]

            fit_flag[:, pp, :] = (dyn_flg & win_flg & not_rfi)

    # Calculate base error
    base_err =  data['erms'][:, np.newaxis, :]

    # Check for sign flips
    ref_resp = andata.CorrData.from_acq_h5(file_list, datasets=['evec'],
                                           input_sel=config.eigen_reference, freq_sel=freq_sel,
                                           start=start, stop=stop)['evec'][:, 0:neigen, 0, :]

    sign0 = 1.0 - 2.0 * (ref_resp.real < 0.0)

    # Check that we have the correct reference feed
    if np.any(np.abs(ref_resp.imag) > 0.0):
        ValueError("Reference feed %d is incorrect." % config.eigen_reference)

    del ref_resp

    # Save index_map
    results = {}
    results['model'] = model_name
    results['param'] = param_name
    results['freq'] = data.index_map['freq'][:]
    results['input'] = input_axis
    results['eval'] = ha_eval_str
    results['dir'] = direction

    for key, val in model_kwargs:
        results[key] = val

    # Initialize numpy arrays to hold results
    if config.return_response:

        results['response'] = np.zeros((nfreq, ninput, ntime), dtype=np.complex64)
        results['response_err'] = np.zeros((nfreq, ninput, ntime), dtype=np.float32)
        results['fit_flag'] = fit_flag
        results['ha_axis'] = ha
        results['ra'] = ra

    else:

        results['gain_eval'] = np.zeros((nfreq, ninput, ngain), dtype=np.complex64)
        results['weight_eval'] = np.zeros((nfreq, ninput, ngain), dtype=np.float32)
        results['frac_gain_err'] = np.zeros((nfreq, ninput, ngain, ndir), dtype=np.float32)

        results['parameter'] = np.zeros((nfreq, ninput, nparam), dtype=np.float32)
        results['parameter_err'] = np.zeros((nfreq, ninput, nparam), dtype=np.float32)

        results['index_eval'] = np.full((nfreq, ninput), -1, dtype=np.int8)
        results['gain'] = np.zeros((nfreq, ninput), dtype=np.complex64)
        results['weight'] = np.zeros((nfreq, ninput), dtype=np.float32)

        results['ndof'] = np.zeros((nfreq, ninput, ndir), dtype=np.float32)
        results['chisq'] = np.zeros((nfreq, ninput, ndir), dtype=np.float32)

        results['timing'] = np.zeros((nfreq, ninput), dtype=np.complex64)

    # Initialize metric like variables
    results['runtime'] = np.zeros((nblock_freq, 2), dtype=np.float64)

    # Compute distances
    dist = tools.get_feed_positions(inputmap)
    for pp, feeds in enumerate(pol):
        dist[feeds, :] -= dist[phase_ref[pp], np.newaxis, :]

    # Loop over frequency blocks
    for gg in range(nblock_freq):

        mlog.info("Frequency block %d of %d." % (gg, nblock_freq))

        fstart = gg*config.nfreq_per_block
        fstop = min((gg+1)*config.nfreq_per_block, nfreq)
        findex = np.arange(fstart, fstop)
        ngroup = findex.size

        freq_sel = utils.convert_to_slice(index_freq[findex])

        timeit_start_gg = time.time()

        #
        if config.return_response:
            gstart = start
            gstop = stop

            tslc = slice(0, ntime)

        else:
            good_times = np.flatnonzero(np.any(fit_flag[findex], axis=(0, 1)))

            if good_times.size == 0:
                continue

            gstart = int(np.min(good_times))
            gstop  = int(np.max(good_times)) + 1

            tslc = slice(gstart, gstop)

            gstart += start
            gstop  += start

        hag = ha[tslc]
        itrans = np.argmin(np.abs(hag))

        # Load eigenvectors.
        nudata = andata.CorrData.from_acq_h5(file_list, datasets=['evec', 'vis', 'flags/vis_weight'], apply_gain=False,
                                             freq_sel=freq_sel, start=gstart, stop=gstop)

        # Save time to load data
        results['runtime'][gg, 0] = time.time() - timeit_start_gg
        timeit_start_gg = time.time()

        mlog.info("Time to load (per frequency):  %0.3f sec" % (results['runtime'][gg, 0] / ngroup,))

        # Loop over polarizations
        for pp, feeds in enumerate(pol):

            # Get timing correction
            if tcorr is not None:
                tgain = tcorr.get_gain(nudata.freq, nudata.input[feeds], nudata.time)
                tgain *= tgain[:, phase_ref_by_pol[pp], np.newaxis, :].conj()

                tgain_transit = tgain[:, :, itrans].copy()
                tgain *= tgain_transit[:, :, np.newaxis].conj()

            # Create the polarization masking vector
            P = np.zeros((1, ninput, 1), dtype=np.float64)
            P[:, feeds, :] = 1.0

            # Loop over frequencies
            for gff, ff in enumerate(findex):

                flg = fit_flag[ff, pp, tslc]

                if (2 * int(np.sum(flg))) < (nparam + 1) and not config.return_response:
                    continue

                # Normalize by eigenvalue and correct for pi phase flips in process.
                resp = (nudata['evec'][gff, 0:neigen, :, :] * np.sqrt(data['eval'][ff, 0:neigen, np.newaxis, tslc]) *
                                                                             sign0[ff, :, np.newaxis, tslc])

                # Rotate to single-pol response
                # Move time to first axis for the matrix multiplication
                invL = tools.invert_no_zero(np.rollaxis(data['eval'][ff, 0:neigen, np.newaxis, tslc], -1, 0))

                UT = np.rollaxis(resp, -1, 0)
                U = np.swapaxes(UT, -1, -2)

                mu, vp = np.linalg.eigh( np.matmul(UT.conj(), P * U) )

                rsign0 = (1.0 - 2.0 * (vp[:, 0, np.newaxis, :].real < 0.0))

                resp = mu[:, np.newaxis, :] * np.matmul(U, rsign0 * vp * invL)

                # Extract feeds of this pol
                # Transpose so that time is back to last axis
                resp = resp[:, feeds, -1].T

                # Compute error on response
                dataflg = ((nudata.weight[gff, feeds, :] > 0.0) & np.isfinite(nudata.weight[gff, feeds, :])).astype(np.float32)

                resp_err =  dataflg * base_err[ff, :, tslc] * np.sqrt(nudata.vis[gff, feeds, :].real) * tools.invert_no_zero(np.sqrt(mu[np.newaxis, :, -1]))

                # Reference to specific input
                resp *= np.exp(-1.0J * np.angle(resp[phase_ref_by_pol[pp], np.newaxis, :]))

                # Apply timing correction
                if tcorr is not None:
                    resp *= tgain[gff]

                    results['timing'][ff, feeds] = tgain_transit[gff]

                # Fringestop
                lmbda = scipy.constants.c * 1e-6 / nudata.freq[gff]

                resp *= tools.fringestop_phase(hag[np.newaxis, :], lat, src_dec,
                                               dist[feeds, 0, np.newaxis] / lmbda,
                                               dist[feeds, 1, np.newaxis] / lmbda)

                # Normalize by source flux
                resp *= inv_rt_flux_density[ff]
                resp_err *= inv_rt_flux_density[ff]

                # If requested, reference phase to the median value
                if config.med_phase_ref:
                    phi0 = np.angle(resp[:, itrans, np.newaxis])
                    resp *= np.exp(-1.0J * phi0)
                    resp *= np.exp(-1.0J * np.median(np.angle(resp), axis=0, keepdims=True))
                    resp *= np.exp(1.0J * phi0)

                # Check if return_response flag was set by user
                if not config.return_response:

                    if config.multi_sample:
                        moving_window = config.nsigma2 and config.nsigma2 * np.radians(utils.get_window(nudata.freq[gff], pol=polstr[pp],
                                                                                     dec=src_dec, deg=True))

                    # Loop over inputs
                    for pii, ii in enumerate(feeds):

                        is_good = flg & (np.abs(resp[pii, :]) > 0.0) & (resp_err[pii, :] > 0.0)

                        # Set the intial gains based on raw response at transit
                        if is_good[itrans]:
                            results['gain_eval'][ff, ii, 0] = tools.invert_no_zero(resp[pii, itrans])
                            results['frac_gain_err'][ff, ii, 0, :] = (resp_err[pii, itrans] *
                                                                      tools.invert_no_zero(np.abs(resp[pii, itrans])))
                            results['weight_eval'][ff, ii, 0] = 0.5 * (np.abs(resp[pii, itrans])**2 *
                                                                tools.invert_no_zero(resp_err[pii, itrans]))**2

                            results['index_eval'][ff, ii] = 0
                            results['gain'][ff, ii] = results['gain_eval'][ff, ii, 0]
                            results['weight'][ff, ii] = results['weight_eval'][ff, ii, 0]


                        # Exit if not performing multi time sample fit
                        if not config.multi_sample:
                            continue

                        if (2 * int(np.sum(is_good))) < (nparam + 1):
                            continue

                        try:
                            param, param_err, gain, gain_err, ndof, chisq, tval = fitter_function(hag[is_good],
                                                                              resp[pii, is_good], resp_err[pii, is_good], ha_eval,
                                                                              window=moving_window, tval=tval, **config.fit)
                        except Exception as rex:
                            if config.verbose:
                                mlog.info("Frequency %0.2f, Feed %d failed with error: %s" % (nudata.freq[gff], ii, rex))
                            continue

                        # Check for nan
                        wfit = (np.abs(gain) * tools.invert_no_zero(np.abs(gain_err)))**2
                        if np.any(~np.isfinite(np.abs(gain))) or np.any(~np.isfinite(wfit)):
                            continue

                        # Save to results using the convention that you should *multiply* the visibilites by the gains
                        results['gain_eval'][ff, ii, fitslc] = tools.invert_no_zero(gain)
                        results['frac_gain_err'][ff, ii, fitslc, 0] = gain_err.real
                        results['frac_gain_err'][ff, ii, fitslc, 1] = gain_err.imag
                        results['weight_eval'][ff, ii, fitslc] = wfit

                        results['parameter'][ff, ii, :] = param
                        results['parameter_err'][ff, ii, :] = param_err

                        results['ndof'][ff, ii, :] = ndof
                        results['chisq'][ff, ii, :] = chisq

                        # Check if the fit was succesful and update the gain evaluation index appropriately
                        if np.all((chisq / ndof.astype(np.float32)) <= config.chisq_per_dof_threshold):
                            results['index_eval'][ff, ii] = ind_eval
                            results['gain'][ff, ii] = results['gain_eval'][ff, ii, ind_eval]
                            results['weight'][ff, ii] = results['weight_eval'][ff, ii, ind_eval]

                else:

                    # Return response only (do not fit model)
                    results['response'][ff, feeds, :] = resp
                    results['response_err'][ff, feeds, :] = resp_err


        # Save time to fit data
        results['runtime'][gg, 1] = time.time() - timeit_start_gg

        mlog.info("Time to fit (per frequency):  %0.3f sec" % (results['runtime'][gg, 1] / ngroup, ))

        # Clean up
        del nudata
        gc.collect()

    # Print total run time
    mlog.info("TOTAL TIME TO LOAD: %0.3f min" % (np.sum(results['runtime'][:, 0]) / 60.0, ))
    mlog.info("TOTAL TIME TO FIT:  %0.3f min" % (np.sum(results['runtime'][:, 1]) / 60.0, ))

    # Set the best estimate of the gain
    if not config.return_response:

        flag = results['index_eval'] >= 0
        gain = results['gain']

        # Compute amplitude
        amp = np.abs(gain)

        # Hard cutoffs on the amplitude
        med_amp = np.median(amp[flag])
        min_amp = med_amp * config.min_amp_scale_factor
        max_amp = med_amp * config.max_amp_scale_factor

        flag &= ((amp >= min_amp) & (amp <= max_amp))

        # Flag outliers in amplitude for each frequency
        for pp, feeds in enumerate(pol):

            med_amp_by_pol = np.zeros(nfreq, dtype=np.float32)
            sig_amp_by_pol = np.zeros(nfreq, dtype=np.float32)

            for ff in range(nfreq):

                this_flag = flag[ff, feeds]

                if np.any(this_flag):

                    med, slow, shigh = utils.estimate_directional_scale(amp[ff, feeds[this_flag]])
                    lower = med - config.nsigma_outlier * slow
                    upper = med + config.nsigma_outlier * shigh

                    flag[ff, feeds] &= ((amp[ff, feeds] >= lower) & (amp[ff, feeds] <= upper))

                    med_amp_by_pol[ff] = med
                    sig_amp_by_pol[ff] = 0.5 * (shigh - slow) / np.sqrt(np.sum(this_flag, dtype=np.float32))


            if config.nsigma_med_outlier:

                med_flag = med_amp_by_pol > 0.0

                not_outlier = flag_outliers(med_amp_by_pol, med_flag, window=config.window_med_outlier,
                                                                      nsigma=config.nsigma_med_outlier)
                flag[:, feeds] &= not_outlier[:, np.newaxis]

                mlog.info("Pol %s:  %d frequencies are outliers." % (polstr[pp], np.sum(~not_outlier & med_flag, dtype=np.int)))

        # Determine bad frequencies
        flag_freq = (np.sum(flag, axis=1, dtype=np.float32) / float(ninput)) > config.threshold_good_freq
        good_freq = np.flatnonzero(flag_freq)

        # Determine bad inputs
        fraction_good = np.sum(flag[good_freq, :], axis=0, dtype=np.float32) / float(good_freq.size)
        flag_input = fraction_good > config.threshold_good_input

        # Finalize flag
        flag &= (flag_freq[:, np.newaxis] & flag_input[np.newaxis, :])

        # Interpolate gains
        interp_gain, interp_weight = interpolate_gain(freq, gain, results['weight'], flag=flag,
                                                      length_scale=config.interpolation_length_scale,
                                                      mlog=mlog)
        # Save gains to object
        results['flag'] = flag
        results['gain'] = interp_gain
        results['weight'] = interp_weight

    # Return results
    return results


def interpolate_gain(freq, gain, weight, flag=None, length_scale=30.0, mlog=None):

    if flag is None:
        flag = weight > 0.0

    nfreq, ninput = gain.shape

    interp_gain = gain.copy()
    interp_weight = weight.copy()

    alpha = tools.invert_no_zero(weight)

    x = freq.reshape(-1, 1)

    if mlog is not None:
        t0 = time.time()
        mlog.info("Interpolating gains over frequency (start time %s)." %
                   datetime.datetime.utcfromtimestamp(t0).strftime('%Y-%m-%d %H:%M:%S'))

    for ii in range(ninput):

        train = np.flatnonzero(flag[:, ii])
        test = np.flatnonzero(~flag[:, ii])

        if train.size > 0.0:

            xtest = x[test, :]

            xtrain = x[train, :]
            ytrain = np.hstack((gain[train, ii, np.newaxis].real,
                                gain[train, ii, np.newaxis].imag))
            # Mean subtract
            ytrain_mu = np.mean(ytrain, axis=0, keepdims=True)
            ytrain = ytrain - ytrain_mu

            # Get initial estimate of variance
            var = 0.5 * np.sum((1.4826 * np.median(np.abs(ytrain -
                                         np.median(ytrain, axis=0, keepdims=True)), axis=0))**2)
            # Define kernel
            kernel = ConstantKernel(var) * Matern(length_scale=length_scale, length_scale_bounds="fixed", nu=1.5)

            # Regress against non-flagged data
            gp = gaussian_process.GaussianProcessRegressor(kernel=kernel, alpha=alpha[train, ii])
            gp.fit(xtrain, ytrain)

            # Predict error
            ypred, err_ypred = gp.predict(xtest, return_std=True)

            interp_gain[test, ii] = (ypred[:, 0] + ytrain_mu[:, 0]) + 1.0J * (ypred[:, 1] + ytrain_mu[:, 1])
            interp_weight[test, ii] = tools.invert_no_zero(err_ypred**2)

        else:
            # No valid data
            interp_gain[:, ii] = 0.0 + 0.0J
            interp_weight[:, ii] = 0.0

    if mlog is not None:
        mlog.info("Done.  Interpolation took %0.1f minutes." % ((time.time() - t0) / 60.0, ))

    return interp_gain, interp_weight


def sliding_window(arr, window):

    # Advanced numpy tricks
    shape = arr.shape[:-1] + (arr.shape[-1]-window+1, window)
    strides = arr.strides + (arr.strides[-1],)
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)


def flag_outliers(raw, flag, window=25, nsigma=5.0):

    # Make sure we have an even window size
    if window % 2:
        window += 1

    hwidth = window // 2 - 1

    nraw = raw.size
    dtype = raw.dtype

    # Replace flagged samples with nan
    good = np.flatnonzero(flag)

    data = np.full((nraw,), np.nan, dtype=dtype)
    data[good] = raw[good]

    # Expand the edges
    expanded_data = np.concatenate((np.full((hwidth,), np.nan, dtype=dtype),
                                    data,
                                    np.full((hwidth+1,), np.nan, dtype=dtype)))

    # Apply median filter
    smooth = np.nanmedian(sliding_window(expanded_data, window), axis=-1)

    # Calculate RMS of residual
    resid = np.abs(data - smooth)

    rwidth = 9 * window
    hrwidth = rwidth // 2 - 1

    expanded_resid = np.concatenate((np.full((hrwidth,), np.nan, dtype=dtype),
                                    resid,
                                    np.full((hrwidth+1,), np.nan, dtype=dtype)))

    sig = 1.4826 * np.nanmedian(sliding_window(expanded_resid, rwidth), axis=-1)

    not_outlier = resid < (nsigma * sig)

    return not_outlier

###################################################
# command line interface
###################################################

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('file_list', help='HDF5 files containing transit to be processed.', type=str, nargs='+')
    parser.add_argument('--source', help='Name of astronomical point source.', type=str, default='CYG_A')
    parser.add_argument('--out', help='Name of output file.', type=str, default=None)
    parser.add_argument('--start', help='Index to start slice of time axis.', type=int, default=None)
    parser.add_argument('--stop',  help='Index to stop slice of time axis.', type=int, default=None)
    parser.add_argument('--freq', help='Frequencies in MHz to process.', type=float, nargs='+')
    parser.add_argument('--apply_timing', help='Load and apply timing correction.', action='store_true')
    parser.add_argument('--config', help='Name of configuration file.', type=str, default=None)
    parser.add_argument('--log', help='Name of log file.', type=str, default=LOG_FILE)

    args = parser.parse_args()

    # Parse output file name
    if os.path.splitext(args.out)[1] != '.pickle':
        ValueError("Command line interface only saves to pickle files.")

    # If calling from the command line, then send logging to log file instead of screen
    containers.mkdir(os.path.dirname(args.log))
    logging_params = DEFAULT_LOGGING
    logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                            'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Load configuration file
    kwargs = load_yaml_config(args.config) if args.config is not None else {}

    # Load timing correction
    tcorr = None
    if args.apply_timing:
        try:
            tcorr = timing.load_timing_correction(args.file_list, start=args.start, stop=args.stop)

        except Exception as e:
            print 'timing.load_timing_correction failed with error: %s' % e

    # Call point source calibration routine
    results = offline_point_source_calibration(args.file_list, args.source, start=args.start, stop=args.stop,
                                               physical_freq=args.freq, tcorr=tcorr,
                                               logging_params=logging_params,
                                               **kwargs)

    # Save results to output file
    with open(out, 'w') as handler:
        pickle.dump(results, handler)
