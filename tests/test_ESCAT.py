from datetime import datetime
from escat_reader.interface import ESCAT_ASPS20_H,\
    ESCAT_ASPS20_N, \
    ESCAT_ASPS20
import os
import numpy as np
import matplotlib.dates as mpl_dates


def test_ESCAT_ASPS20():
    t = datetime.strptime('19960324T225407', '%Y%m%dT%H%M%S')

    path = os.path.join('R', 'Datapool_raw', 'ERS', 'test_datasets',
                        'ERS1_ASPS_data_201507')
    filename = \
        'ER01_WSC_EWI_0P_19960324T225407_19960325T002305_FS_24538_EEB2_asps20.dat'
    escat = ESCAT_ASPS20()
    orbit = escat.read(os.path.join(path, filename))
    np.testing.assert_almost_equal(mpl_dates.date2num(t),
                                   mpl_dates.julian2num(np.min(orbit.data[
                                                                   'jd'])),
                                   decimal=2)

def test_ESCAT_ASPS20_H():
    t = datetime(1996, 5, 1, 01, 19, 17)
    escat = ESCAT_ASPS20_H()
    orbit = escat.read(t)
    assert orbit.timestamp == t


def test_ESCAT_ASPS20_H_iteration():
    start = datetime(1997, 5, 29)
    end = datetime(1997, 6, 1)
    escat = ESCAT_ASPS20_H()
    dates = []
    for img in escat.iter_images(start, end):
        dates.append(img.timestamp)
    assert dates[0] == datetime(1997, 5, 29, 0, 27, 26)
    assert dates[-1] == datetime(1997, 6, 1, 22, 20, 58)


def test_ESCAT_ASPS20_N():
    t = datetime(1996, 4, 1, 0, 21, 43)
    escat = ESCAT_ASPS20_N()
    orbit = escat.read(t)
    assert orbit.timestamp == t


def test_ESCAT_ASPS20_N_iteration():
    start = datetime(1996, 4, 5)
    end = datetime(1996, 4, 6)
    escat = ESCAT_ASPS20_N()
    dates = []
    for img in escat.iter_images(start, end):
        dates.append(img.timestamp)

    assert dates[0] == datetime(1996, 4, 5, 1, 36, 27)
    assert dates[-1] == datetime(1996, 4, 6, 22, 52, 38)

def test_ESCAT_ASPS20_read_masked_data():
    start = datetime(1996, 4, 5)
    end = datetime(1996, 4, 6)
    escat = ESCAT_ASPS20_N()
    dates = []
    for img in escat.iter_images(start, end, mask=True):
        dates.append(img.timestamp)

    assert dates[0] == datetime(1996, 4, 5, 1, 36, 27)
    assert dates[-1] == datetime(1996, 4, 6, 22, 52, 38)

if __name__ == '__main__':
    test_ESCAT_ASPS20()
    test_ESCAT_ASPS20_H()
    test_ESCAT_ASPS20_H_iteration()
    test_ESCAT_ASPS20_N()
    test_ESCAT_ASPS20_N_iteration()
    test_ESCAT_ASPS20_read_masked_data()