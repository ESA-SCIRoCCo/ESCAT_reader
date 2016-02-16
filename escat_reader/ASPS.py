import os
import numpy as np


class ASPS_L2_product_template(object):

    """
    ASPS Product template class
    """

    def getDate24(self):
        return np.dtype('S24')

    def get_MPH(self):
        struct = np.dtype([('orig', np.dtype('S1')),
                           ('orbitnr', np.uint32),
                           ('uniqID', np.uint8, 12),
                           ('prod', np.uint8),
                           ('sat', np.uint8),
                           ('utct', self.getDate24()),
                           ('stid', np.uint8),
                           ('pcd', np.uint16),
                           ('utcg', self.getDate24()),
                           ('sizeSPH', np.uint32),
                           ('nPDR', np.uint32),
                           ('sizePDR', np.uint32),
                           ('subsys', np.uint8),
                           ('obrc', np.uint8),
                           ('reftime', self.getDate24()),
                           ('clock', np.uint32, 2),
                           ('soft', np.uint8, 8),
                           ('thrd', np.uint16),
                           ('spar', np.uint16),
                           ('ascn', self.getDate24()),
                           ('stvc', np.uint32, 6)])

        return struct

    def get_SPH(self):
        struct = np.dtype([('pdesc', np.ubyte),
                           ('orbit', np.uint32),
                           ('nn3s', np.uint16),
                           ('nn2s', np.uint16),
                           ('nn1s', np.uint16),
                           ('nnland', np.uint16),
                           ('nnice', np.uint16),
                           ('nnarc', np.uint16),
                           ('nnkp', np.uint16),
                           ('nnfsck', np.uint16),
                           ('nniq', np.uint16),
                           ('nnic', np.uint16),
                           ('nncog', np.uint16),
                           ('nnstd', np.uint16),
                           ('nndop', np.uint16),
                           ('nnyaw', np.uint16),
                           ('nnwind', np.uint16),
                           ('nnlowind', np.uint16),
                           ('nnhiwind', np.uint16),
                           ('nnd', np.uint16),
                           ('nnwsb', np.uint16),
                           ('nnwdb', np.uint16),
                           ('mwsb', np.uint16),
                           ('wsbstd', np.uint16),
                           ('mwdb', np.uint16),
                           ('md', np.uint32, 41),
                           ('wspv', np.uint16),
                           ('wspconf', np.uint16),
                           ('meteoID1', np.uint16),
                           ('meteoID2', np.uint16),
                           ('meteoID3', np.uint16),
                           ('meteoID4', np.uint16),
                           ('meteotype', np.uint32),
                           ('spare', np.uint32, 2)])
        return struct

    def get_DSR_HDR(self):
        struct = np.dtype([('num', np.int32),
                           ('utct', self.getDate24()),
                           ('azi', np.int32)])
        return struct

    def get_Node(self):
        struct = np.dtype([('lat', np.int32),
                           ('lon', np.int32),
                           ('timef', np.uint16),
                           ('timem', np.uint16),
                           ('timea', np.uint16),
                           ('sigmaf', np.int32),
                           ('incf', np.int16),
                           ('lookf', np.int16),
                           ('kpf', np.uint16),
                           ('nsf', np.int16),
                           ('sigmam', np.int32),
                           ('incm', np.int16),
                           ('lookm', np.int16),
                           ('kpm', np.uint16),
                           ('nsm', np.int16),
                           ('sigmaa', np.int32),
                           ('inca', np.int16),
                           ('looka', np.int16),
                           ('kpa', np.uint16),
                           ('nsa', np.int16),
                           ('ws1', np.int16),
                           ('wd1', np.int16),
                           ('d1', np.int32),
                           ('ws2', np.int16),
                           ('wd2', np.int16),
                           ('d2', np.int32),
                           ('ws3', np.int16),
                           ('wd3', np.int16),
                           ('d3', np.int32),
                           ('ws4', np.int16),
                           ('wd4', np.int16),
                           ('d4', np.int32),
                           ('wsb', np.int16),
                           ('siProb', np.int16),
                           ('wdb', np.int16),
                           ('ncd1', np.uint16, {'1': (np.uint8, 0),
                                                '2': (np.uint8, 1)}),
                           ('ncd2', np.uint16, {'1': (np.uint8, 0),
                                                '2': (np.uint8, 1)}),
                           ('gcd', np.ubyte)])
        return struct

    def get_DSR(self, nNodes, nLines):
        header = self.get_DSR_HDR()
        node = self.get_Node()

        line = np.dtype([('dsrhdr', header),
                         ('node', node, nNodes)])
        struct = np.dtype([('prod', line, nLines)])
        return struct


class ASPS_L2_Product(object):

    """
    Class for reading ASPS products.
    """

    def __init__(self, filename):
        self.filename = filename
        self.fid = None
        self.eor = None
        self.mph = None
        self.sph = None
        self.dsr = None
        self.filesize = 0

    def _open_read(self):
        """
        Open file for reading.
        """
        self.fid = open(self.filename, "rb")

    def _read_MPH(self):
        size = ASPS_L2_product_template().get_MPH().itemsize
        self.mph = (np.frombuffer(self.fid.read(size),
                                  dtype=ASPS_L2_product_template().get_MPH(),
                                  count=1)).newbyteorder("B")

    def _read_SPH(self):
        size = ASPS_L2_product_template().get_SPH().itemsize
        self.sph = (np.frombuffer(self.fid.read(size),
                                  dtype=ASPS_L2_product_template().get_SPH(),
                                  count=1)).newbyteorder("B")

    def _getNumberOfNodes(self, ProdDesc):
        nNodes = 0
        bits = np.unpackbits(ProdDesc)
        if bits[-2] == 0:
            nNodes = 19
        if bits[-2] == 1:
            nNodes = 41
        return nNodes

    def _read_DSR(self):
        nNodes = self._getNumberOfNodes(self.sph['pdesc'])
        dsrTemp = ASPS_L2_product_template().get_DSR(nNodes,
                                                     self.mph['nPDR'][0])
        size = dsrTemp.itemsize

        self.dsr = (np.frombuffer(self.fid.read(size),
                                  dtype=dsrTemp,
                                  count=1)).newbyteorder("B")

    def read_product(self):
        """
        Read an ASPS file.
        """
        self._open_read()
        self.filesize = os.path.getsize(self.filename)
        self.eor = self.fid.tell()

        # read main product header
        self._read_MPH()
        self._read_SPH()
        self._read_DSR()

        self.fid.close()


def read_ASPS_L2(filename):
    """
    Reader for ASPS Level 2 files.

    Parameters
    ----------
    filename : str
        EPS filename.

    Returns
    -------
    data : numpy.ndarray
        Data records.
    nNodes: int
        Number of nodes per dataset record.
    """

    ASPSProduct = ASPS_L2_Product(filename)
    ASPSProduct.read_product()

    # get number of nodes
    nNodes = ASPSProduct._getNumberOfNodes(ASPSProduct.sph['pdesc'])

    data = {'mph': ASPSProduct.mph,
            'sph': ASPSProduct.sph,
            'dsr': ASPSProduct.dsr}

    return data, nNodes
