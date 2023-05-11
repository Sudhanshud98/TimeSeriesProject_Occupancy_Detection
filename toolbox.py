import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
from sklearn.model_selection import train_test_split
from statsmodels.tsa.seasonal import STL
import statsmodels.api as sm
import statsmodels.tsa.holtwinters as ets
from statsmodels.graphics.tsaplots import plot_acf , plot_pacf
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from numpy import linalg as LA
from scipy import signal
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import chi2

#####################################################
# Importing Data
#####################################################

with open('datatest2.csv') as x:
    ncols = len(x.readline().split(','))

# Importing the dataset
df = pd.read_csv("datatest2.csv", usecols=range(1, ncols))

date = pd.date_range(start = '2015-02-11 14:48',
                    end = '2015-02-18 09:19',
                    freq='min')

#####################################################
# ACF/PACF Plot
#####################################################

def ACF_PACF_Plot(y, lags):
    acf = sm.tsa.stattools.acf(y, nlags=lags)
    pacf = sm.tsa.stattools.pacf(y, nlags=lags)
    fig = plt.figure()
    plt.subplot(211)
    plt.title('ACF/PACF of the raw data')
    plot_acf(y, ax=plt.gca(), lags=lags)
    plt.subplot(212)
    plot_pacf(y, ax=plt.gca(), lags=lags)
    fig.tight_layout(pad=3)
    plt.show()

#####################################################
# Rolling Mean and Variance
#####################################################
def cal_rolling_mean_var(arg, title):
    p_mean = []
    p_var = []
    for i in range(1, len(arg)):
        p_mean.append(np.mean(arg[:i]))
        p_var.append(np.var(arg[:i]))
    fig, axs = plt.subplots(2)
    axs[0].plot(p_mean)
    axs[0].set_xlabel('Samples')
    axs[0].set_ylabel('Magnitude')
    axs[0].set_title('Rolling Mean - '+title)
    axs[1].plot(p_var)
    axs[1].set_xlabel('Samples')
    axs[1].set_ylabel('Magnitude')
    axs[1].set_title('Rolling Variance - '+title)
    plt.tight_layout()
    plt.show()

#####################################################
# ADF Test
#####################################################

def ADF_Cal(x):
    result = adfuller(x)
    print("ADF Statistic: %f" %result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))

#####################################################
# KPSS Test
#####################################################

def kpss_test(timeseries):
    print ('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic','p-value','Lags Used'])
    for key,value in kpsstest[3].items():
        kpss_output['Critical Value (%s)'%key] = value
    print(kpss_output)

#####################################################
# Non-seasonal Differencing
#####################################################

df1 = df.copy()
def nonseasonal_diff(df1, arg, order):
    p_dif = []
    for i in range(0, order):
        p_dif.append(arg[i] == np.nan)
    for i in range(order, len(arg)):
        p_dif.append(arg[i] - arg[i-1])
    df1['P_Diff'] = np.array(p_dif)

#####################################################
# Decomposition
#####################################################

def stl_decomp(col, col_name, period):
    temp = pd.Series(col.values, index = date, name = col_name)
    STL_raw = STL(temp, period=period)
    res = STL_raw.fit()
    fig = res.plot()
    plt.show()

    T = res.trend
    S = res.seasonal
    R = res.resid
    O = res.observed

    # find the strength of the seasonality and/or trend

    Ft = max(0, 1-(np.var(R) / np.var(T + R)))
    print(f"\nThe strength of trend for {col_name} set is ", round(Ft, 3))

    Fs = max(0, 1-(np.var(R) / np.var(S + R)))
    print(f"\nThe strength of seasonality for {col_name} set is ", round(Fs, 3))

#####################################################
# Holt Winter Method
#####################################################

def holt_winter(train, test):
    fitted_model = ExponentialSmoothing(train['P_Diff'], trend = 'add' ,seasonal = 'add', seasonal_periods = 7).fit()
    train_prediction = fitted_model.predict(start = 0, end = len(train) - 1)
    test_predictions = fitted_model.forecast(len(test))
    return train_prediction, test_predictions

def holt_winter_plot(train, test, test_predictions):
    train['P_Diff'].plot(legend = True, label = 'TRAIN')
    test['P_Diff'].plot(legend = True, label = 'TEST')
    test_predictions.plot(legend = True, label = 'PREDICTION')
    plt.title('Train, Test and Predicted Test using Holt Winters')
    plt.tight_layout()
    plt.show()

#####################################################
# Forecast Plot
#####################################################

def forecast_plot(df, n, prediction, label):
    plt.figure()
    plt.plot(list(df[:n].index.values + 1), df['P_Diff'][:n], label='train')
    plt.plot(list(df[n:].index.values + 1), df['P_Diff'][n:], label='test')
    plt.plot(list(df[n:].index.values + 1), prediction[n:], label='h-step forecast')
    plt.xticks(rotation = 45)
    plt.title(f'{label}  Forecasting')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.tight_layout()
    plt.show()

#####################################################
# Forecasting Methods
#####################################################

def Ave_Forecast(frame, n):
    avg_pred = list(0 for i in range(0, len(frame)))
    for i in range(1, n):
        avg_pred[i] = np.mean(frame[:i])
    for i in range(n, len(frame)):
        avg_pred[i] = np.mean(frame[:n])
    return avg_pred

def naive_forecast(frame, n):
    naive_pred = list(0 for i in range(0, len(frame)))
    for i in range(1, n):
        naive_pred[i] = frame[i - 1]
    for i in range(n, len(frame)):
        naive_pred[i] = frame[n - 1]
    return naive_pred

def drift_forecast(frame, n):
    drift_pred = list(0 for i in range(0, len(frame)))
    for i in range(2, n):
        drift_pred[i] = frame[i-1] + ((frame[i-1]-frame[0]))/(i-1)
    for i in range(n, len(frame)):
        drift_pred[i] = frame[n-1] + (i+1-n)*(frame[n-1]-frame[0])/(n-1)
    return drift_pred

def ses_forecast(frame, n):
    alpha = 0.5
    ses_pred = list(0 for i in range(0, len(frame)))
    l0 = frame[0]
    ses_pred[1] = alpha * l0 + (1 - alpha) * l0
    for i in range(2, n):
        ses_pred[i] = alpha * frame[i-1] + (1-alpha) * ses_pred[i-1]
    for i in range(n, len(frame)):
        ses_pred[i] = alpha * frame[n-1] + (1-alpha) * ses_pred[n-1]
    return ses_pred

#####################################################
# GPAC Table
#####################################################

def cal_gpac(lst_acf, kval=7, jval=7):
	phi = np.zeros((kval, jval))
	for k in range(1, kval):
		for j in range(0, jval):
			num = np.zeros((k, k))
			den = np.zeros((k, k))
			n1 = np.zeros((k, 1))
			if k == 1:
				for h in range(0, k):
					num[0][h] = lst_acf[j+h+1]
					den[0][h] = lst_acf[j-k+h+1]
			else:
				for x in range(0, k):
					for y in range(0, k):
						den[x][y] = lst_acf[abs(j-y+x)]
					num = den[:, :-1]
				for h in range(0, k):
					n1[h][0] = lst_acf[j+h+1]
				num = np.append(num, n1, 1)
			num_d = np.linalg.det(num)
			den_d = np.linalg.det(den)
			a = round((num_d/den_d), 2)
			phi[j][k] = a
	phi1 = phi[:, 1:]
	print(phi1)
	print("\n")
	axs = sns.heatmap(phi1, annot = True, xticklabels = np.arange(1, kval), vmin=-1, vmax=1)
	plt.title('Generalized partial auto correlation function (GPAC)')
	plt.show()

#####################################################
# ACF Plot
#####################################################

def Cal_autocorr(y, lag):
    mean = np.mean(y)
    numerator = 0
    denominator = 0
    for t in range(0, len(y)):
        denominator += (y[t] - mean) ** 2
    for t in range(lag, len(y)):
        numerator += (y[t] - mean)*(y[t-lag] - mean)
    return numerator/denominator


def Cal_autocorr_plot(y, lags, title, plot_show='Yes'):
    ryy = []
    ryy_final = []
    lags_final = []
    for lag in range(0, lags+1):
        ryy.append(Cal_autocorr(y, lag))
    ryy_final.extend(ryy[:0:-1])
    ryy_final.extend(ryy)
    lags = list(range(0, lags+1, 1))
    lags_final.extend(lags[:0:-1])
    lags_final = [value*(-1) for value in lags_final]
    lags_final.extend(lags)
    plt.figure(figsize=(12, 8))
    markers, stemlines, baseline = plt.stem(lags_final, ryy_final)
    plt.setp(markers, color='red', marker='o')
    plt.axhspan((-1.96 / np.sqrt(len(y))), (1.96 / np.sqrt(len(y))), alpha=0.2, color='blue')
    plt.xlabel('Lags')
    plt.ylabel('Magnitude')
    plt.title(title)
    plt.tight_layout()
    if plot_show == 'Yes':
        plt.show()


#####################################################
# ARIMA Model
#####################################################

def ARIMA_model(df, n, na, nb, diff_order, title):
    y_train = df['P_Diff'][:n]
    y_test = df['P_Diff'][n:]
    model = sm.tsa.ARIMA(y_train, order= (na, diff_order, nb))
    model_fit = model.fit()

    model_fit.plot_diagnostics(figsize=(14, 10))
    plt.suptitle(f'ARIMA({na},{diff_order},{nb}) Diagnostic Analysis')
    plt.grid()
    plt.show()

    print(model_fit.summary())
    y_predict = model_fit.predict()
    y_forecast = model_fit.forecast(steps=len(y_test))

    forecast_error = y_test - y_forecast
    residual_error = y_train - y_predict

    params = model._params
    print("Coefficients are: ", params)

    print("#" * 100)
    Q = sm.stats.acorr_ljungbox(residual_error, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
    print(f"Q-Value for training set {title}) : ", np.round(Q, 2))

    model_fit = (np.var(residual_error) / np.var(forecast_error))

    print(f'\nMean of residual error for {title} Method is {np.round(np.mean(residual_error), 2)}')
    print(f'\nMSE of residual error for {title} Method is {np.round(np.mean(residual_error ** 2), 2)}')
    print(f'\nMSE of Forecast error for {title} Method is {np.round(mean_squared_error(y_test, forecast_error), 2)}')
    print(f'\nVariance of residual error for {title} Method is {np.round(np.var(residual_error), 2)}')
    print(f'\nVariance of forecast error for {title} Method is {np.round(np.var(forecast_error), 2)}')
    print(f'\nvariance of the residual errors versus the variance of the forecast errors {title}) : ',
          np.round(model_fit, 2))

    plt.plot(y_train, label='Train')
    plt.plot(y_predict, label='Predicted Data')
    plt.legend()
    plt.title(f'Train versus the Predicted data for {title}')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.xticks(rotation = 45)
    plt.tight_layout()
    plt.show()

    plt.plot(y_test, label='Test')
    plt.plot(y_forecast, label='Forecasted Data')
    plt.legend()
    plt.title(f'Test versus Forecasted data for {title}')
    plt.xlabel('Date')
    plt.ylabel('Temperature')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()




