def pix_wise_fit_RF(Trees_number, mat):
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    # got the size of mat
    Size = mat.shape
    ls = Size[0]
    rs = Size[1]
    cs = Size[2]

    # judge whether the series contain 'nan' value
    L = np.isnan(mat)
    # accuracy metrics
    RMSE = []
    Bias = []
    for i in range(0, rs):
        l = np.all(L[:, i, :])
        if l:
            RMSE.append(np.nan)
            Bias.append(np.nan)
        else:
            Slice = mat[:, i, :]
            nan_idx = np.any(np.isnan(Slice), axis=0)
            X = Slice[0: ls - 1, ~nan_idx].T
            Y = Slice[ls - 1, ~nan_idx].T
            # fitting model
            RF_pix = RandomForestRegressor(n_estimators=Trees_number)
            RF_pix.fit(X, Y)
            # calculate the accuracy
            X_pre = Slice[0:ls - 1, :].T
            Y_pre = RF_pix.predict(X_pre)
            rmse = np.sqrt(mean_squared_error(Y, Y_pre[~nan_idx]))
            bias = np.mean(Y_pre[~nan_idx] - Y)
            RMSE.append(rmse)
            Bias.append(bias)
            # fill gap value
            mat[ls - 1, i, nan_idx] = Y_pre[nan_idx]
    filled_data = mat[ls - 1, :, :]
    del mat
    return filled_data, np.array(RMSE), np.array(Bias), nan_idx

def pix_wise_FFT_periods(mat, Fs):
    import numpy as np
    per = []
    ls, r, c = mat.shape
    mat_T_num = np.zeros([r, c])
    for i in range(0, r):
        for j in range(0, c):
            if ~np.any(np.isnan(mat[:, i, j])):
                arr_fft = np.fft.fft(mat[:, i, j])
                amp = np.abs(arr_fft / ls)
                fre = np.fft.fftfreq(ls, 1 / Fs)
                amp_r = amp[0:int(ls / 2)] * 2
                a = np.sort(amp_r, kind='quicksort')[::-1]
                lev = a[10]
                t = 1 / fre[0:ls/2]
                T_idx = amp_r > lev
                T = t[T_idx]
                if np.any(np.isinf(T)):
                    mat_T_num[i, j] = len(T) - 1
                    TT = T[~np.isinf(T)]
                    per.append(TT)
                else:
                    mat_T_num[i, j] = len(T)
                    per.append(T)
            else:
                mat_T_num[i, j] = np.nan
    return mat_T_num, per



def pix_wise_fit_EMD_RF(Trees_number, mat, meta_idx):
    from PyEMD import EMD
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import mean_squared_error
    # got the size of mat
    Size = mat.shape
    ls = Size[0]
    rs = Size[1]
    cs = Size[2]
    # judge whether the series contain 'nan' value
    L = np.isnan(mat)
    # accuracy metrics
    RMSE = Bias = np.zeros([rs, 1], dtype=float)
    X_Trend = np.zeros([ls - 1, rs, cs], dtype=float)
    Y_Trend = Y_pre = np.zeros([rs, cs], dtype=float)
    emd = EMD()

    back_value_idx = np.all(L[0, :, :], axis=1)

    RMSE[back_value_idx] = Bias[back_value_idx] = np.nan
    X_Trend[:, back_value_idx, :] = Y_Trend[back_value_idx, :] = Y_pre[back_value_idx, :] = np.nan
    idx_arr = np.array(np.argwhere(~back_value_idx == True))


    for i in idx_arr:
        # Get the slice of data cubic
        Slice = mat[:, i, :]
        X = Slice[0: ls - 1, :]
        Y = Slice[ls - 1, :]
        imf_Y = emd.emd(Y.reshape(cs, ))
        imfs_X = np.zeros([240, 2], dtype=float)
        Y_Trend[i, :] = imf_Y[-1,:]
        RF_pix = RandomForestRegressor(n_estimators=Trees_number)
        # Gathering of input data for emd

        for j in range(0, ls - 1):
            imf_X = emd.emd(X[j, :].reshape(cs, ))
            X_Trend[j, i, :] = imf_X[-1, :]
            imfs_X = np.concatenate([imfs_X ,imf_X[0:-1, :].T], axis=1)
        X_in = np.delete(imfs_X, [0,1], axis=1)
        Y_out = imf_Y[0: -1,:].T

        # Loop for Random forest
        Y_pre_sum = np.zeros([cs, ])
        for l in range(0, Y_out.shape[1]):
            RF_pix.fit(X_in, Y_out[:, l])
            Y_out_pre = RF_pix.predict(X_in)
            Y_pre_sum = Y_pre_sum + Y_out_pre
        Y_pre[i, :] = Y_Trend[i, :] + Y_pre_sum
        RMSE[i] = np.sqrt(mean_squared_error(Y_pre[i, meta_idx], Y[:, meta_idx].reshape(np.sum(meta_idx), )))
        Bias[i] = np.mean(Y_pre[i, meta_idx] - Y[:, meta_idx].reshape(np.sum(meta_idx), ))
    return RMSE, Bias, Y_pre








def pix_wise_Pearson(x_mat, y_mat):
    from scipy.stats import pearsonr
    import numpy as np

    # got the size of mat
    Size = y_mat.shape
    ls = Size[0]
    rs = Size[1]
    cs = Size[2]
    cor_mat = np.zeros([rs, ls])
    p_mat = np.zeros([rs, ls])
    nan_idx = np.any(np.any(np.isnan(y_mat), axis=0), axis=1)
    cor_mat[nan_idx, :] = np.nan
    p_mat[nan_idx, :] = np.nan
    x_mat_dna = x_mat[~nan_idx, :]
    y_mat_dna = y_mat[:, ~nan_idx, :]
    for i in range(0, ls):
        cor_tmp = []
        p_tmp = []
        for j in range(0, np.sum(~nan_idx)):
            c, p = pearsonr(x_mat_dna[j, :], y_mat_dna[i, j, :])
            cor_tmp.append(c)
            p_tmp.append(p)
        cor_mat[~nan_idx, i] = np.array(cor_tmp)
        p_mat[~nan_idx, i] = np.array(p_tmp)
        del cor_tmp, p_tmp
    return cor_mat, p_mat
