import os
import sys
import glob
import argparse
import pickle

import numpy as np
import h5py

from pychfpga import NameSpace, load_yaml_config
from calibration.utils import get_window

from ch_util.fluxcat import FluxCatalog
from ch_util import ephemeris, tools, timing

import log

from version import __version__
import containers
import offline_cal

###################################################
# default variables
###################################################

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':point_source'))

LOG_FILE = os.environ.get('CALIBRATION_LOG_FILE',
           os.path.join(os.path.dirname(os.path.realpath(__file__)), 'offline_wrapper.log'))

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

def main(config_file=None, logging_params=DEFAULT_LOGGING):

    # Setup logging
    log.setup_logging(logging_params)
    mlog = log.get_logger(__name__)

    # Set config
    config = DEFAULTS.deepcopy()
    if config_file is not None:
        config.merge(NameSpace(load_yaml_config(config_file)))

    # Create transit tracker
    source_list = FluxCatalog.sort() if not config.source_list else config.source_list

    cal_list = [name for name, obj in FluxCatalog.iteritems()
                if (obj.dec >= config.min_dec) and
                (obj.predict_flux(config.freq_nominal) >= config.min_flux) and
                (name in source_list)]

    if not cal_list:
        raise RuntimeError("No calibrators found.")

    # Sort list by flux at nominal frequency
    cal_list.sort(key=lambda name: FluxCatalog[name].predict_flux(config.freq_nominal))

    # Add to transit tracker
    transit_tracker = containers.TransitTrackerOffline(nsigma=config.nsigma_source, extend_night=config.extend_night)
    for name in cal_list:
        transit_tracker[name] = FluxCatalog[name].skyfield

    mlog.info("Initializing offline point source processing.")

    search_time = config.start_time or 0

    # Find all calibration files
    all_files = sorted(glob.glob(os.path.join(config.acq_dir,'*' + config.correlator + config.acq_suffix, '*.h5')))
    if not all_files:
        return

    # Remove files whose last modified time is before the time of the most recent update
    all_files = [ff for ff in all_files if (os.path.getmtime(ff) > search_time)]
    if not all_files:
        return

    # Remove files that are currently locked
    all_files = [ff for ff in all_files if not os.path.isfile(os.path.splitext(ff)[0] + '.lock')]
    if not all_files:
        return

    # Add files to transit tracker
    for ff in all_files:
        transit_tracker.add_file(ff)

    # Extract point source transits ready for analysis
    all_transits = transit_tracker.get_transits()

    # Create dictionary to hold results
    h5_psrc_fit = {}
    inputmap = None

    # Loop over transits
    for transit in all_transits:

        src, csd, is_day, files, start, stop = transit

        # Discard any point sources with unusual csd value
        if (csd < config.min_csd) or (csd > config.max_csd):
            continue

        # Discard any point sources transiting during the day
        if is_day > config.process_daytime:
            continue

        mlog.info('Processing %s transit on CSD %d (%d files, %d time samples)' %
                         (src, csd, len(files), stop - start + 1))

        # Load inputmap
        if inputmap is None:
            if config.inputmap is None:
                inputmap = tools.get_correlator_inputs(ephemeris.unix_to_datetime(ephemeris.csd_to_unix(csd)),
                                                       correlator=config.correlator)
            else:
                with open(config.inputmap, 'r') as handler:
                    inputmap = pickle.load(handler)


        # Grab the timing correction for this transit
        tcorr = None
        if config.apply_timing:

            if config.timing_glob is not None:

                mlog.info("Loading timing correction from extended timing solutions.")

                timing_files = sorted(glob.glob(config.timing_glob))

                if timing_files:

                    try:
                        tcorr = search_extended_timing_solutions(timing_files, ephemeris.csd_to_unix(csd))

                    except Exception as e:
                        mlog.error('search_extended_timing_solutions failed with error: %s' % e)

                    else:
                        mlog.info(str(tcorr))

            if tcorr is None:

                mlog.info("Loading timing correction from chimetiming acquisitions.")

                try:
                    tcorr = timing.load_timing_correction(files, start=start, stop=stop,
                                                          window=config.timing_window,
                                                          instrument=config.correlator)
                except Exception as e:
                    mlog.error('timing.load_timing_correction failed with error: %s' % e)
                    mlog.warning('No timing correction applied to %s transit on CSD %d.' % (src, csd))
                else:
                    mlog.info(str(tcorr))


        # Call the main routine to process data
        try:
            outdct = offline_cal.offline_point_source_calibration(files, src, start=start, stop=stop, inputmap=inputmap,
                                                              tcorr=tcorr, logging_params=logging_params,
                                                              **config.analysis.as_dict())

        except Exception as e:
            msg = 'offline_cal.offline_point_source_calibration failed with error:  %s' % e
            mlog.error(msg)
            continue
            #raise RuntimeError(msg)

        # Find existing gain files for this particular point source
        if src not in h5_psrc_fit:

            output_files = find_files(config, psrc=src)
            if output_files is not None:
                output_files = output_files[-1]
                mlog.info('Writing %s transit on CSD %d to existing file %s.' % (src, csd, output_files))

            h5_psrc_fit[src] = containers.PointSourceWriter(src, output_file=output_files,
                                                                 output_dir=config.output_dir,
                                                                 output_suffix=point_source_name_to_file_suffix(src),
                                                                 instrument=config.correlator,
                                                                 max_file_size=config.max_file_size,
                                                                 max_num=config.max_num_time,
                                                                 memory_size=0)


        # Associate this gain calibration to the transit time
        this_time = ephemeris.transit_times(FluxCatalog[src].skyfield, ephemeris.csd_to_unix(csd))[0]

        outdct['csd'] = csd
        outdct['is_daytime'] = is_day
        outdct['acquisition'] = os.path.basename(os.path.dirname(files[0]))

        # Write to output file
        mlog.info('Writing to disk results from %s transit on CSD %d.' % (src, csd))
        h5_psrc_fit[src].write(this_time, **outdct)

        # Dump an individual file for this point source transit
        mlog.info('Dumping to disk single file for %s transit on CSD %d.' % (src, csd))
        dump_dir = os.path.join(config.output_dir, 'point_source_gains')
        containers.mkdir(dump_dir)

        dump_file = os.path.join(dump_dir, '%s_csd_%d.h5' % (src.lower(), csd))
        h5_psrc_fit[src].dump(dump_file, datasets=['csd', 'acquisition', 'is_daytime', 'gain', 'weight', 'timing', 'model'])

        mlog.info('Finished analysis of %s transit on CSD %d.' % (src, csd))


###################################################
# ancillary functions
###################################################

def point_source_name_to_suffix(src):
    return src.lower().replace('_', '')

def point_source_name_to_file_suffix(src):
    return point_source_name_to_suffix(src) + 'fit'

def find_files(config, psrc=None):

    output_dir = config.output_dir
    output_suffix = config.output_suffix if psrc is None else point_source_name_to_file_suffix(psrc)

    candidate_files = sorted(glob.glob(os.path.join(output_dir, '*' + output_suffix, '*.h5')))

    output_files = []
    for cf in candidate_files:

        with h5py.File(cf, 'r') as hf:

            file_version = hf.attrs['version']
            instrument = hf.attrs['instrument_name']

            valid = (file_version == __version__) and (instrument == config.correlator)

            if valid:
                 output_files.append(cf)

    return output_files or None


def search_extended_timing_solutions(timing_files, timestamp):

    # Load the timing correction
    nfiles = len(timing_files)
    tstart = np.zeros(nfiles, dtype=np.float32)
    tstop  = np.zeros(nfiles, dtype=np.float32)
    all_tcorr = []

    for ff, filename in enumerate(timing_files):

        kwargs = {}
        with h5py.File(filename, 'r') as handler:

            for key in ['tau', 'avg_phase', 'noise_source', 'time']:

                kwargs[key] = handler[key][:]

        tcorr = timing.TimingCorrection(**kwargs)

        all_tcorr.append( tcorr )
        tstart[ff] = tcorr.time[0]
        tstop[ff]  = tcorr.time[-1]

    # Map timestamp to a timing correction object
    imatch = np.flatnonzero((timestamp >= tstart) & (timestamp <= tstop))

    if imatch.size > 1:
        ValueError("Timing corrections overlap!")
    elif imatch.size < 1:
        ValueError("No timing correction for transit on %s (CSD %d)" %
                  (ephemeris.unix_to_datetime(timestamp).strftime("%Y-%m-%d"),  ephemeris.unix_to_csd(timestamp)))

    return all_tcorr[imatch[0]]


###################################################
# command line interface
###################################################

if __name__ == '__main__':
    """ Command-line interface to launch calibration of archive.
    """

    # Parse arguments
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--config', help='Name of configuration file.', type=str, default=None)
    parser.add_argument('--log', help='Name of log file.', type=str, default=LOG_FILE)
    args = parser.parse_args()

    # If calling from the command line, then send logging to log file instead of screen
    containers.mkdir(os.path.dirname(args.log))
    logging_params = DEFAULT_LOGGING
    logging_params['handlers'] = {'stderr': {'class': 'logging.handlers.WatchedFileHandler',
                                            'filename': args.log, 'formatter': 'std', 'level': 'INFO'}}

    # Call main routine
    main(config_file=args.config, logging_params=logging_params)

