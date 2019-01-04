import os
import datetime
import time

import numpy as np
import h5py

from pychfpga import NameSpace, load_yaml_config
from pychfpga import Hdf5Writer

from ch_util import ephemeris
from calibration.utils import get_window

from version import __version__

DEFAULTS = NameSpace(load_yaml_config(os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                                   'defaults.yaml') + ':point_source'))

def mkdir(directory):
    """ Make a directory if it does not already exist.
    """
    try:
        os.makedirs(directory)
    except OSError:
        if not os.path.isdir(directory):
            raise


class PointSourceWriter(Hdf5Writer):
    """ Interface to an Hdf5Writer containing fits to point source transit.
    """

    _uniq_id = 'csd'
    _grow_ax = 'time'

    _axes = {
        'time': {'dtype': np.float64},
        'freq': {'dtype': np.float32},
        'input': {'dtype': str},
        'param': {'dtype': str},
        'eval': {'dtype': str},
        'dir': {'dtype': str}
    }

    _dataset_spec = {
        'csd': {
            'axes': ['time', ],
            'dtype': h5py.special_dtype(vlen=bytes),
            'metric': False,
        },
        'acquisition': {
            'axes': ['time', ],
            'dtype': h5py.special_dtype(vlen=bytes),
            'metric': False,
        },
        'is_daytime': {
            'axes': ['time', ],
            'dtype': np.uint8,
            'metric': False,
        },
        'gain_eval': {
            'axes': ['time', 'freq', 'input', 'eval'],
            'dtype': np.complex64,
            'metric': False,
        },
        'weight_eval': {
            'axes': ['time', 'freq', 'input', 'eval'],
            'dtype': np.float32,
            'metric': False,
        },
        'index_eval': {
            'axes': ['time', 'freq', 'input'],
            'dtype': np.int8,
            'metric': False,
        },
        'flag': {
            'axes': ['time', 'freq', 'input'],
            'dtype': np.bool,
            'metric': False,
        },
        'gain': {
            'axes': ['time', 'freq', 'input'],
            'dtype': np.complex64,
            'metric': False,
        },
        'weight': {
            'axes': ['time', 'freq', 'input'],
            'dtype': np.float32,
            'metric': False,
        },
        'model': {
            'axes': ['time', ],
            'dtype': h5py.special_dtype(vlen=bytes),
            'metric': False,
        },
        'parameter': {
            'axes': ['time', 'freq', 'input', 'param'],
            'dtype': np.float32,
            'metric': False,
        },
        'parameter_err': {
            'axes': ['time', 'freq', 'input', 'param'],
            'dtype': np.float32,
            'metric': False,
        },
        'frac_gain_err': {
            'axes': ['time', 'freq', 'input', 'eval', 'dir'],
            'dtype': np.float32,
            'metric': False,
        },
        'ndof': {
            'axes': ['time', 'freq', 'input', 'dir'],
            'dtype': np.int,
            'metric': False,
        },
        'chisq': {
            'axes': ['time', 'freq', 'input', 'dir'],
            'dtype': np.float32,
            'metric': False,
        }
    }

    def __init__(self, source, output_dir=DEFAULTS.output_dir, output_suffix=DEFAULTS.output_suffix,
                       instrument=DEFAULTS.correlator, *args, **kwargs):
        """ Instantiates a PointSourceWriter object.

        Parameters
        ----------
        output_dir:  str
            Directory where the hdf5 archive files will be saved.

        output_suffix: str
            Suffix appended to the hdf5 archive filenames.

        instrument:  str
            Name of the instrument/correlator.  Included in hdf5 archive filenames,
            and also saved to file attributes.

        source:  str
            Name of the point source, saved to file attributes.
        """

        # Call superclass
        super(PointSourceWriter, self).__init__(*args, **kwargs)

        # Set parameters that specify output file format
        self.output_dir = output_dir
        self.output_suffix = output_suffix

        # Set attributes
        attrs = {'instrument_name': instrument, 'version': __version__, 'source': source}
        self.set_attrs(**attrs)

        # Set metric name
        self._metric_name = 'point_source'


    def get_output_file(self, smp, **kwargs):
        """ Defines the filenaming conventions for the archive files:

            {output_dir}/{YYYYMMDD}T{HHMMSS}Z_{instrument}_{output_suffix}/{SSSSSSS}.h5

        Parameters
        ----------
        smp: unix time
            Time at which the datasets in kwargs were collected.

        acquisition: str
            Datetime string {YYYYMMDD}T{HHMMSS}Z for the acquisition
            that was used to derive the datasets.
        """

        # Determine directory
        acq = kwargs.get('acquisition', None)
        if acq is not None:
            output_dir = os.path.join(self.output_dir, '_'.join([acq[0:16],
                                      self.attrs['instrument_name'], self.output_suffix]))

            start_time = ephemeris.datetime_to_unix(ephemeris.timestr_to_datetime(acq))

        else:
            this_datetime = kwargs.get('datetime', datetime.datetime.utcfromtimestamp(smp).strftime("%Y%m%dT%H%M%SZ"))

            output_dir = os.path.join(self.output_dir, '_'.join([this_datetime,
                                      self.attrs['instrument_name'], self.output_suffix]))

            start_time = ephemeris.datetime_to_unix(datetime.datetime.strptime(this_datetime, "%Y%m%dT%H%M%SZ"))

        mkdir(output_dir)

        # Determine filename
        seconds_elapsed = smp - start_time

        output_file = os.path.join(output_dir, "%08d.h5" % seconds_elapsed)

        return output_file


class TransitTrackerOffline(object):

    def __init__(self, nsigma=3.0):

        self._entries = {}
        self._nsigma = nsigma

    def add_file(self, filename):

        # Make sure this file is not currently in the transit tracker
        if self.contains_file(filename):
            return

        # Read file time range
        with h5py.File(filename, 'r') as handler:
            timestamp = handler['index_map']['time']['ctime'][:]

        timestamp0 = np.median(timestamp)

        # Convert to right ascension
        ra = ephemeris.lsa(timestamp)
        csd = ephemeris.csd(timestamp)

        # Loop over available sources
        for name, src in self.iteritems():

            src_ra, src_dec = ephemeris.object_coords(src.body, date=timestamp0, deg=True)

            # Determine if any times in this file fall
            # in a window around transit of this source
            hour_angle = ra - src_ra
            hour_angle = hour_angle + 360.0 * (hour_angle < -180.0) - 360.0 * (hour_angle > 180.0)

            good_time = np.flatnonzero(np.abs(hour_angle) < src.window)

            if good_time.size > 0:

                # Determine the csd for the transit contained in this file
                icsd = np.unique(np.floor(csd[good_time] - (hour_angle[good_time] / 360.0)))

                if icsd.size > 1:
                    RuntimeError("Error estimating CSD.")

                key = int(icsd[0])

                min_ha, max_ha = np.percentile(hour_angle, [0, 100])

                # Add to list of files to analyze for this source
                if key in src.files:
                    src.files[key].append((filename, hour_angle))
                    src.file_span[key][0] = min(min_ha, src.file_span[key][0])
                    src.file_span[key][1] = max(max_ha, src.file_span[key][1])

                else:
                    src.files[key] = [(filename, hour_angle)]
                    src.file_span[key] = [min_ha, max_ha]

    def get_transits(self):

        out = []
        for name, src in self.iteritems():

            for csd in sorted(src.file_span.keys()):

                span = src.file_span[csd]

                #if (span[0] <= -src.window) and (span[1] >= src.window):

                files = src.files.pop(csd)

                isort = np.argsort([np.min(ha) for ff, ha in files])

                hour_angle = np.concatenate(tuple([files[ii][1] for ii in isort]))

                if np.all(np.diff(hour_angle) > 0.0):

                    below = np.flatnonzero(hour_angle <= -src.window)
                    aa = int(np.max(below)) if below.size > 0 else 0

                    above = np.flatnonzero(hour_angle >=  src.window)
                    bb = int(np.min(above)) if above.size > 0 else hour_angle.size

                    is_day = self.is_daytime(src, csd)

                    out.append((name, csd, is_day, [files[ii][0] for ii in isort],  aa, bb))

                del src.file_span[csd]

        return out

    def is_daytime(self, key, csd):

        src = self[key] if isinstance(key, basestring) else key

        is_daytime = 0

        src_ra, src_dec = ephemeris.object_coords(src.body, date=ephemeris.csd_to_unix(csd), deg=True)

        transit_start = ephemeris.csd_to_unix(csd + (src_ra - src.window) / 360.0)
        transit_end = ephemeris.csd_to_unix(csd + (src_ra + src.window) / 360.0)

        solar_rise = ephemeris.solar_rising(transit_start - 24.0*3600.0, end_time=transit_end)

        for rr in solar_rise:

            ss = ephemeris.solar_setting(rr)[0]

            if ((transit_start <= ss) and (rr <= transit_end)):

                is_daytime += 1

                tt = ephemeris.solar_transit(rr)[0]
                if (transit_start <= tt) and (tt <= transit_end):
                    is_daytime += 1

                break

        return is_daytime

    def contains_file(self, filename):

        contains = False
        for name, src in self.iteritems():
            for csd, file_list in src.files.iteritems():

                if filename in [ff[0] for ff in file_list]:

                    contains = True

        return contains

    def __setitem__(self, key, body):

        if key not in self:

            if ephemeris._is_skyfield_obj(body):
                pass
            elif isinstance(body, (tuple, list)) and (len(body) == 2):
                ra, dec = body
                body = ephemeris.skyfield_star_from_ra_dec(ra, dec, bd_name=key)
            else:
                ValueError("Item must be skyfield object or tuple (ra, dec).")

            #window = self._nsigma * cal_utils.guess_fwhm(400.0, pol='X', dec=body.dec.radians, sigma=True)
            window = self._nsigma * get_window(400.0, pol='X', dec=body.dec.radians, deg=True)

            self._entries[key] = NameSpace()
            self._entries[key].body = body
            self._entries[key].window = window
            self._entries[key].files = {}
            self._entries[key].file_span = {}

    def __contains__(self, key):

        return key in self._entries

    def __getitem__(self, key):

        if key not in self:
            raise KeyError

        return self._entries[key]

    def iteritems(self):

        return self._entries.iteritems()

    def clear_all(self):

        self._entries = {}

    def remove(self, key):

        self._entries.pop(key)