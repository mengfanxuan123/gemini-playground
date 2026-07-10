## ===========================================================================================
from datetime import datetime
import pandas as pd
import os
import re
import numpy as np
from osgeo import gdal


## ===============================Matching date===============================================
# defined functions
def file_filter(f):
    if f[-4:] in ['.tif']:
        return True
    else:
        return False


def S2(x):
    Y, M = re.split('_|\.', x)[2:4]
    date = str(Y) + str(M).zfill(2)
    return date


def s2002(x):
    if '200212' < x < '202301':
        return True
    else:
        return False


def mach(X, Y):
    a = []
    for y in Y:
        a.append(str(X) in str(y))
    return a


def write_tif(newpath, im_data, im_Geotrans, im_proj, datatype):
    if len(im_data.shape) == 3:
        im_bands, im_height, im_width = im_data.shape
    else:
        im_bands, (im_height, im_width) = 1, im_data.shape
    diver = gdal.GetDriverByName('GTiff')
    new_dataset = diver.Create(newpath, im_width, im_height, im_bands, datatype)
    new_dataset.SetGeoTransform(im_Geotrans)
    new_dataset.SetProjection(im_proj)

    if im_bands == 1:
        new_dataset.GetRasterBand(1).WriteArray(im_data)
    else:
        for i in range(im_bands):
            new_dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
    del new_dataset


# Generate a standard time series
date_stan_pd = pd.date_range(start="2003-01-01",
                             end="2023-01-01",
                             freq="M")
path = r'H:\Grace\Mascons\NW_Clip'
fns_meta = os.listdir(path)

# Got filename
fn_filter_e = list(filter(file_filter, fns_meta))

t = map(S2, fn_filter_e)
T = sorted(list(t))

date_real = list(filter(s2002, T))
r = 60 * 135
l = 240
TWS = np.empty([r, l], dtype=float)

i = 0
for date in date_stan_pd:
    Y = date.year
    M = date.month
    date_str = str(Y) + str(M).zfill(2)
    if date_str in date_real:
        # Open TWS.tif data
        fn = fn_filter_e[mach(str(Y) + '_' + str(M) + '_', fn_filter_e).index(True)]
        print(fn)
        dataset = gdal.Open(path + "\\" + fn)
        lwe = dataset.GetRasterBand(1)
        lwe_v = lwe.ReadAsArray().reshape(r, )
        TWS[:, i] = lwe_v
    else:
        TWS[:, i] = np.nan
    i = i + 1
TWS[TWS < -9999] = np.nan
## ===============================Filling missing data===============================================
# Replace fill value as "nan"
# Here, we fill the gap with original data series through two ways
# First, only interpolate data using Random Forest, the climate and soil factors was input
# Second, we use EMD method to decompose and interpolate the meta GRACE data series
# then use Random forest


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

## Data ready
ET = np.empty([r, l], dtype=float)
Tmp = np.empty([r, l], dtype=float)
Swc001 = np.empty([r, l], dtype=float)
Swc0104 = np.empty([r, l], dtype=float)
Swc041 = np.empty([r, l], dtype=float)
Swc12 = np.empty([r, l], dtype=float)
Swe = np.empty([r, l], dtype=float)
Pre = np.empty((r, l), dtype=float)

paths = [r'H:\GLDAS\Water_evaporation_flux\anomalies', r'H:\GLDAS\Temperature\NW_clip',
         r'H:\GLDAS\Soil_moisture_0_01m\anomalies_clip', r'H:\GLDAS\Soil_moisture_01_04m\anomalies_clip',
         r'H:\GLDAS\Soil_moisture_04_1m\anomalies_clip', r'H:\GLDAS\Soil_moisture_1_2m\anomalies_clip',
         r'H:\GLDAS\Snow_water_equivalent\anomalies_clip', r'H:\GLDAS\rainfall_flux\Anomalies']
# Got filename
for path in paths:

    all_fn = os.listdir(path)
    fn_filter_e = np.array(list(filter(file_filter, all_fn)))
    i = 0
    for fn in fn_filter_e:
        if 'Water_evaporation' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            ET[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'Temperature' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Tmp[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'Soil_moisture_0_01m' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Swc001[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'Soil_moisture_01_04m' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Swc0104[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'Soil_moisture_04_1m' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Swc041[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'Soil_moisture_1_2m' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Swc12[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'rainfall_flux' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Pre[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
        elif 'Snow_water_equivalent' in path:
            ds = gdal.Open(path + '\\' + fn)
            b1 = ds.GetRasterBand(1)
            Swe[:, i] = b1.ReadAsArray().reshape(r, )
            i = i + 1
    del ds, b1, i

all_arr = np.stack((ET, Tmp, Swc001,
                    Swc0104, Swc041, Swc12,
                    Pre, Swe, TWS), axis=0)
del ET, Tmp, Swc001, Swc0104, Swc041, Swc12, fn_filter_e, Pre, TWS

all_arr[:, np.isnan(all_arr[0, :, :])] = np.nan

# --------------------------------Random forest interpolate-------------------------------------------------
'''
from pix_wise_gapfill import pix_wise_fit_RF
from scipy.io import savemat

out_path = r'H:\Grace\Mascons\RF_intropolated'
# Here, we have looped tested different number of trees in Random Forest(range from 40 to 100 with step 10),
# Now we deside set the trees = 80

trees = 80

image_tr = dataset.GetGeoTransform()
image_prj = dataset.GetProjection()
data_type = gdal.GDT_Float64

filled_data, RMSE, Bias, gap_idx = pix_wise_fit_RF(trees, all_arr)
gap_date = date_stan_pd[gap_idx]
gap_data = filled_data[:, gap_idx]
print(np.nanmean(RMSE))
for j in range(0, len(gap_date)):
    time = str(gap_date[j].year) + '_' + str(gap_date[j].month) + '_' + str(gap_date[j].day) + '.tif'
    write_tif(out_path + r'\Trees_' + str(trees) + r'\CSR_GRACE_' + time,
              gap_data[:, j].reshape(60, 135),
              image_tr, image_prj, data_type)
write_tif(out_path + '\\' + 'Trees_' + str(trees) + '_RMSE.tif',
          RMSE.reshape(60, 135),
          image_tr, image_prj, data_type)
write_tif(out_path + '\\' + 'Trees_' + str(trees) + '_Bias.tif',
          Bias.reshape(60, 135),
          image_tr, image_prj, data_type)
'''
# --------------------------------EMD & random forest interpolate-------------------------------------------------
from pix_wise_gapfill import pix_wise_fit_EMD_RF

path = r'H:\Grace\Mascons\RF_intropolated\Trees_80'
fns_meta = os.listdir(path)
fn_filter_e = np.array(list(filter(file_filter, fns_meta)))
all_arr[8, :, :] = 0
gap_idx = np.load(r"H:\Grace\Mascons\RF_intropolated\gap_idx.npy")
trees = 80

i = 0
# load data
for date in date_stan_pd:
    Y = date.year
    M = date.month
    fn = fn_filter_e[mach(str(Y) + '_' + str(M) + '_', fn_filter_e).index(True)]
    print(fn)
    ds = gdal.Open(path + '\\' + fn)
    band = ds.GetRasterBand(1)
    value = band.ReadAsArray()
    value[value < -1000] = np.nan
    all_arr[8, :, i] = value.reshape(r, )
    i = i + 1
    del value

# mat = all_arr
# Trees_number = 1

RMSE_imp, Bias_imp, data = pix_wise_fit_EMD_RF(trees, all_arr, ~gap_idx)
print(np.nanmean(Bias_imp[~gap_idx, 0]))

d = data[:, ~gap_idx]

real = all_arr[8, :, ~gap_idx].T

res = np.zeros(8100,)
for i in range(0, 8100):
    if np.any(np.isnan(d[i, :])):
        res[i] = np.nan
    else:
        res[i] = np.sqrt(mean_squared_error(d[i, :], real[i, :]))

# np.nanmean(res)
#
# trans = ds.GetGeoTransform()
# prj = ds.GetProjection()
#
# write_tif(r'H:\Grace\Mascons\RF_intropolated' + '\\' + 'Trees_80_imp_RMSE.tif',
#           res.reshape(60, 135), trans, prj, gdal.GDT_Float64)
#
#
# ds = gdal.Open(r"H:\另一个新建文件夹\地表水Sen值分布.tif")
# band = ds.GetRasterBand(1)
# value = band.ReadAsArray()
# value[value == 255] = np.nan
# np.sum(value<1)

# --------------------------------Spatial mean value of TWA-------------------------------------------------

trans = dataset.GetGeoTransform()
prj = dataset.GetProjection()
TWS_mean = np.nanmean(TWS, axis=1)
write_tif(r'H:\Grace\Mascons' + '\\' + 'Mean_TWS.tif',
           TWS_mean.reshape(60, 135), trans, prj, gdal.GDT_Float64)


