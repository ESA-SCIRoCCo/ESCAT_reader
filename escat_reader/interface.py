import os
import numpy as np

from datetime import datetime, timedelta
import matplotlib.dates as mpl_dates

import escat_reader.ASPS as ASPS
from pygeobase.io_base import ImageBase, MultiTemporalImageBase
from pygeobase.object_base import Image


class ESCAT_ASPS20(ImageBase):

    def __init__(self, *args, **kwargs):
        super(ESCAT_ASPS20, self).__init__(*args, **kwargs)

    def read(self, timestamp=None, **kwargs):
        raw_data, nNodes = ASPS.read_ASPS_L2(self.filename)

        DSR = raw_data['dsr']['prod']
        nDSR = DSR.size
        nMeas = DSR['node']['sigmaf'].flatten().size

        # check data set records for trailing zeros
        Zeros = (DSR['node']['sigmaf'].flatten() == 0).astype(np.uint8)
        nZeros = np.sum(Zeros)
        indZeros = np.flatnonzero(Zeros)

        if nZeros > 0:
            if (nZeros % nNodes) != 0:
                raise ValueError("Number of trailing zero-records " +
                                 "not a multiple of nodes per line")
            if (nMeas - nZeros) != indZeros[0] or \
               (nMeas - 1) != indZeros[-1]:
                raise ValueError(
                    "Zeros records found in non-trailing data records")
            nDSR = raw_data['mph']['nPDR'][0] - (nZeros / nNodes)
            DSR = DSR[0:nDSR - 1]
            nMeas = DSR['node']['sigmaf'].flatten().size

        # get julian dates of each node
        obsTime = DSR['dsrhdr']['utct'].flatten()
        obsTime = [datetime.strptime(x, "%d-%b-%Y %H:%M:%S.%f")
                   for x in obsTime]
        obsTime = np.array(obsTime)
        jd = mpl_dates.num2julian(mpl_dates.date2num(obsTime))
        jd = np.repeat(jd, nNodes, axis=0)

        lat = DSR['node']['lat'].flatten() * 10 ** (-3)
        lon = DSR['node']['lon'].flatten() * 10 ** (-3)
        westernLon = (lon > 180)
        lon[westernLon] -= 360

        data = {}
        data['jd'] = jd
        data['swath'] = np.full(nMeas, 1)
        data['line_num'] = np.repeat(DSR['dsrhdr']['num'],
                                     nNodes, axis=1).flatten()
        data['node_num'] = np.tile(np.arange(0, nNodes), nDSR)

        passInd = (DSR['dsrhdr']['azi'].flatten() > 270000).astype(np.uint8)
        data['as_des_pass'] = np.repeat(passInd, nNodes, axis=0)

        gcd = np.unpackbits(np.reshape(DSR['node']['gcd'].flatten(),
                                       (DSR['node']['gcd'].flatten().size, 1)), axis=1)
        data['f_land'] = gcd[:, -1]

        ncd1_1 = np.unpackbits(np.reshape(DSR['node']['ncd1']['1'].flatten(),
                                          (DSR['node']['ncd1']['1'].flatten().size, 1)),
                               axis=1)
        ncd2_1 = np.unpackbits(np.reshape(DSR['node']['ncd2']['1'].flatten(),
                                          (DSR['node']['ncd2']['1'].flatten().size, 1)),
                               axis=1)
        ncd = ncd1_1[:, -2] + ncd2_1[:, -3] + ncd2_1[:, -4] + ncd2_1[:, -5] + \
            ncd2_1[:, -6] + ncd2_1[:, -7] + ncd2_1[:, -8]
        data['f_usable'] = np.uint8((ncd > 0)) * 2

        fields = [('incf', 'incf', 0.1),
                  ('incm', 'incm', 0.1),
                  ('inca', 'inca', 0.1),
                  ('sigf', 'sigmaf', 10 ** (-7)),
                  ('sigm', 'sigmam', 10 ** (-7)),
                  ('siga', 'sigmaa', 10 ** (-7)),
                  ('azif', 'lookf', 0.1),
                  ('azim', 'lookm', 0.1),
                  ('azia', 'looka', 0.1),
                  ('kpf', 'kpf', 10 ** (-3)),
                  ('kpm', 'kpm', 10 ** (-3)),
                  ('kpa', 'kpa', 10 ** (-3))]
        for field in fields:
            data[field[0]] = DSR['node'][field[1]].flatten() * field[2]

        data['abs_orbit_nr'] = np.repeat(raw_data['mph']['orbitnr'][0],
                                         data[field[0]].size)
        data['spacecraft_id'] = np.repeat(raw_data['mph']['sat'][0],
                                          data[field[0]].size)

        # get metdata
        metadata = {}
        metadata['station_id'] = raw_data['mph']['stid'][0]
        metadata['nDSR'] = nDSR
        metadata['nNodes'] = nNodes

        return Image(lon, lat, data, metadata, timestamp)

    def read_masked_data(self, **kwargs):
        orbit = self.read(**kwargs)

        valid = np.ones(orbit.data[orbit.data.keys()[0]].shape, dtype=np.bool)

        valid = (valid & (orbit.data['f_usable'] < 2))
        valid = (valid & (orbit.data['f_land'] > 0.95))

        for key in orbit.data.keys():
            orbit.data[key] = orbit.data[key][valid]

        orbit.lon = orbit.lon[valid]
        orbit.lat = orbit.lat[valid]
        return orbit

    def write(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class ESCAT_MultiTemporalImageBase(MultiTemporalImageBase):

    """
    ESCAT base class for reading multi temporal images --> orbits
    """

    def __init__(self, *args, **kwargs):
        kwargs['exact_templ'] = False

        self.date_format = kwargs.pop('date_format', None)

        super(ESCAT_MultiTemporalImageBase, self).__init__(*args, **kwargs)

    def tstamps_for_daterange(self, startdate, enddate):
            file_list = []
            delta_all = enddate - startdate
            timestamps = []

            for i in xrange(delta_all.days + 1):
                timestamp = startdate + timedelta(days=i)

                files = self._search_files(timestamp,
                                           custom_datetime_format=self.date_format)

                file_list.extend(sorted(files))

            for filename in file_list:
                filename = os.path.basename(filename)
                timestamps.append(self.get_tstamp_from_filename(filename))
            return timestamps


class ESCAT_ASPS20_H(ESCAT_MultiTemporalImageBase):

    def __init__(self, *args, **kwargs):
        if 'path' not in kwargs:
            kwargs['path'] = os.path.join(root.r, 'Datapool_raw', 'ERS',
                                          'ERS_main', 'ERS_ASPS_H', '')

        if 'ioclass' not in kwargs:
            kwargs['ioclass'] = ESCAT_ASPS20

        if 'fname_templ' not in kwargs:
            kwargs['fname_templ'] = 'ASPS20_H_{datetime}*'

        if 'datetime_format' not in kwargs:
            kwargs['datetime_format'] = '%y%m%d%H%M%S'

        if 'date_format' not in kwargs:
            kwargs['date_format'] = '%y%m%d'

        if 'subpath_templ' not in kwargs:
            kwargs['subpath_templ'] = ['%Y%m']

        super(ESCAT_ASPS20_H, self).__init__(*args, **kwargs)


class ESCAT_ASPS20_N(ESCAT_MultiTemporalImageBase):

    def __init__(self, *args, **kwargs):
        if 'path' not in kwargs:
            kwargs['path'] = os.path.join(root.r, 'Datapool_raw', 'ERS',
                                          'ERS_main', 'ERS_ASPS_N', '')

        if 'ioclass' not in kwargs:
            kwargs['ioclass'] = ESCAT_ASPS20

        if 'fname_templ' not in kwargs:
            kwargs['fname_templ'] = 'ASPS20_N_{datetime}*'

        if 'datetime_format' not in kwargs:
            kwargs['datetime_format'] = '%y%m%d%H%M%S'

        if 'date_format' not in kwargs:
            kwargs['date_format'] = '%y%m%d'

        if 'subpath_templ' not in kwargs:
            kwargs['subpath_templ'] = ['%Y%m']

        super(ESCAT_ASPS20_N, self).__init__(*args, **kwargs)
