# Importing packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import toolbox
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
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from scipy.stats import chi2
import warnings
warnings.filterwarnings('ignore')
lags = 50
############################################################
# Importing Dataset
############################################################

with open('datatest2.csv') as x:
    ncols = len(x.readline().split(','))

# Importing the dataset
df = pd.read_csv("datatest2.csv", usecols=range(1, ncols))

date = pd.date_range(start = '2015-02-11 14:48',
                    end = '2015-02-18 09:19',
                    freq='min')
print(df)

############################################################
# Checking for the NA, Null values or missing values
############################################################

print(df.isnull().sum())
#There are no null or missing value in the dataset.

############################################################
# Plotting target variable against time
############################################################

col = df.columns.values
df.index = date
plt.figure(figsize=(6, 4))
df['Temperature'].plot()
plt.xlabel("Date")
plt.ylabel("Temperature")
plt.title("Dependent Variable against time")
plt.grid()
plt.legend(["Temperature"])
plt.tight_layout()
plt.show()

# Here we can see that the data is not stationary. we can see a decreasing trend overall and a slight increasing trend towards the end.
# We can also see strong seasonality in the data. So overall, we can see the data has both trend and seasonality.

############################################################
# ACF plot of Dependent Variable
############################################################

toolbox.ACF_PACF_Plot(df['Temperature'], 20)

############################################################
# Checking the correlation between the variables
############################################################

cor = df.corr()
sns.heatmap(data = cor, annot = True, vmin=-1, vmax=1)
plt.tight_layout()
plt.show()

############################################################
# Checking stationarity of the dependent variable
############################################################

toolbox.cal_rolling_mean_var(df['Temperature'], "Temperature")

############################################################
# ADF and KPSS Test
############################################################

toolbox.ADF_Cal(df['Temperature'])
print()

toolbox.kpss_test(df['Temperature'])
print()

############################################################
# Non Seasonal Differencing
############################################################

df1 = df.copy()

# 1st Order
toolbox.nonseasonal_diff(df1, df1['Temperature'], 1)
toolbox.cal_rolling_mean_var(df1['P_Diff'][1:], "Temperature - 1st Ordering")

toolbox.ACF_PACF_Plot(df1['P_Diff'], 20)

toolbox.ADF_Cal(df1['P_Diff'])
print()

toolbox.kpss_test(df1['P_Diff'])
print()

############################################################
# Time series Decomposition
############################################################

# Decomposition of Raw data
toolbox.stl_decomp(df['Temperature'], 'Temperature', 1440)

#Decomposition of Differenced data
toolbox.stl_decomp(df1['P_Diff'], 'Differenced Temeprature', 1440)

############################################################
#Feature Selection
############################################################

x = df.drop(['Temperature','date'], axis=1)
y = df['Temperature']
x_train1, x_test1,y_train1, y_test1 = train_test_split(x,y, shuffle=False, test_size=0.20)

sc = StandardScaler()

X_train = sc.fit_transform(x_train1)
X_test = sc.transform(x_test1)

pca = PCA(n_components = 'mle')

X_train = pca.fit_transform(X_train)
X_test = pca.fit_transform(X_test)

s,d,v = np.linalg.svd (X_train, full_matrices = True)

print(f'singular values of x are {d}')
print(f'The condition number for x is {LA.cond(X_train)}')

X_train = sm.add_constant(X_train, prepend=False)
model = sm.OLS(y_train1, X_train).fit()
print(model.summary())

print("t-test p-values for all features: \n", model.pvalues)

print("#" * 100)

print("F-test for final model: \n", model.f_pvalue)

prediction = model.predict(X_train)

############################################################
# Holt-winter method
############################################################

df.index.freq = 'T'
train, test = train_test_split(df1, test_size = 0.2, shuffle = False)

train_pred, test_pred = toolbox.holt_winter(train, test)
residual_err_holt = train['P_Diff'] - train_pred
forecast_err_holt = test['P_Diff'] - test_pred

print("#"*100)
Q_holt = sm.stats.acorr_ljungbox(residual_err_holt, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for training set (Holt-Winter Method) : ", np.round(Q_holt, 2))

model_fit = (np.var(residual_err_holt) / np.var(forecast_err_holt))

print(f'\nMean of residual error for Holt-Winter Method is {np.round(np.mean(residual_err_holt), 2)}')
print(f'\nMSE of residual error for Holt-Winter Method is {np.round(np.mean(residual_err_holt ** 2), 2)}')
print(f'\nMSE of Forecast error for Holt-Winter Method is {np.round(mean_squared_error(test["P_Diff"], forecast_err_holt), 2)}')
print(f'\nVariance of residual error for Holt-Winter Method is {np.round(np.var(residual_err_holt), 2)}')
print(f'\nVariance of forecast error for Holt-Winter Method is {np.round(np.var(forecast_err_holt), 2)}')
print('\nvariance of the residual errors versus the variance of the forecast errors (Holt-Winter Method) : ', np.round(model_fit, 2))


toolbox.holt_winter_plot(train, test, test_pred)

############################################################
# Forecasting Techniques
############################################################

n = len(x_train1)

# Average Method

avg_prediction = toolbox.Ave_Forecast(df1['P_Diff'], n)
toolbox.forecast_plot(df1, n, avg_prediction, 'Average')
residual_err_avg = df1['P_Diff'][:n] - avg_prediction[:n]
forecast_err_avg = df1['P_Diff'][n:] - avg_prediction[n:]

print("#"*100)
Q_avg = sm.stats.acorr_ljungbox(residual_err_avg, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for training set (Average Method) : ", np.round(Q_avg, 2))

model_fit_avg = (np.var(residual_err_avg) / np.var(forecast_err_avg))

print(f'\nMean of residual error for Average Method is {np.round(np.mean(residual_err_avg), 2)}')
print(f'\nMSE of residual error for Average Method is {np.round(np.mean(residual_err_avg ** 2), 2)}')
print(f'\nMSE of Forecast error for Average Method is {np.round(mean_squared_error(df1["P_Diff"][n:], forecast_err_avg), 2)}')
print(f'\nVariance of residual error for Average Method is {np.round(np.var(residual_err_avg), 2)}')
print(f'\nVariance of forecast error for Average Method is {np.round(np.var(forecast_err_avg), 2)}')
print('\nvariance of the residual errors versus the variance of the forecast errors (Average Method) : ', np.round(model_fit_avg, 2))


#Naive Method

naive_prediction = toolbox.naive_forecast(df1['P_Diff'], n)
toolbox.forecast_plot(df1, n, naive_prediction, 'Naive')
residual_err_naive = df1['P_Diff'][:n] - naive_prediction[:n]
forecast_err_naive = df1['P_Diff'][n:] - naive_prediction[n:]

print("#"*100)
Q_naive = sm.stats.acorr_ljungbox(residual_err_naive, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for training set (Naive Method) : ", np.round(Q_naive, 2))

model_fit_naive = (np.var(residual_err_naive) / np.var(forecast_err_naive))

print(f'\nMean of residual error for Naive Method is {np.round(np.mean(residual_err_naive), 2)}')
print(f'\nMSE of residual error for Naive Method is {np.round(np.mean(residual_err_naive ** 2), 2)}')
print(f'\nMSE of Forecast error for Naive Method is {np.round(mean_squared_error(df1["P_Diff"][n:], forecast_err_naive), 2)}')
print(f'\nVariance of residual error for Naive Method is {np.round(np.var(residual_err_naive), 2)}')
print(f'\nVariance of forecast error for Naive Method is {np.round(np.var(forecast_err_naive), 2)}')
print('\nvariance of the residual errors versus the variance of the forecast errors (Naive Method) : ', np.round(model_fit_naive, 2))


# Drift Method

drift_prediction = toolbox.drift_forecast(df1['P_Diff'], n)
toolbox.forecast_plot(df1, n, drift_prediction, 'Drift')
residual_err_drift = df1['P_Diff'][:n] - drift_prediction[:n]
forecast_err_drift = df1['P_Diff'][n:] - drift_prediction[n:]

print("#"*100)
Q_drift = sm.stats.acorr_ljungbox(residual_err_drift, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for training set (Drift Method) : ", np.round(Q_drift, 2))

model_fit_drift = (np.var(residual_err_drift) / np.var(forecast_err_drift))

print(f'\nMean of residual error for Drift Method is {np.round(np.mean(residual_err_drift), 2)}')
print(f'\nMSE of residual error for Drift Method is {np.round(np.mean(residual_err_drift ** 2), 2)}')
print(f'\nMSE of Forecast error for Drift Method is {np.round(mean_squared_error(df1["P_Diff"][n:], forecast_err_drift), 2)}')
print(f'\nVariance of residual error for Drift Method is {np.round(np.var(residual_err_drift), 2)}')
print(f'\nVariance of forecast error for Drift Method is {np.round(np.var(forecast_err_drift), 2)}')
print('\nvariance of the residual errors versus the variance of the forecast errors (Drift Method) : ', np.round(model_fit_drift, 2))


# SES Method

ses_prediction = toolbox.ses_forecast(df1['P_Diff'], n)
toolbox.forecast_plot(df1, n, ses_prediction, 'SES')
residual_err_ses = df1['P_Diff'][:n] - ses_prediction[:n]
forecast_err_ses = df1['P_Diff'][n:] - ses_prediction[n:]

print("#"*100)
Q_ses = sm.stats.acorr_ljungbox(residual_err_ses, lags=[50], boxpierce=True, return_df=True)['bp_stat'].values[0]
print("Q-Value for training set (SES Method) : ", np.round(Q_ses, 2))

model_fit_ses = (np.var(residual_err_ses) / np.var(forecast_err_ses))

print(f'\nMean of residual error for SES Method is {np.round(np.mean(residual_err_ses), 2)}')
print(f'\nMSE of residual error for SES Method is {np.round(np.mean(residual_err_ses ** 2), 2)}')
print(f'\nMSE of Forecast error for SES Method is {np.round(mean_squared_error(df1["P_Diff"][n:], forecast_err_ses), 2)}')
print(f'\nVariance of residual error for SES Method is {np.round(np.var(residual_err_ses), 2)}')
print(f'\nVariance of forecast error for SES Method is {np.round(np.var(forecast_err_ses), 2)}')
print('\nvariance of the residual errors versus the variance of the forecast errors (SES Method) : ', np.round(model_fit_ses, 2))


############################################################
# GPAC table
############################################################

acf_lst = []
for i in range(0, lags + 1):
    acf_lst.append(toolbox.Cal_autocorr(df1['P_Diff'].dropna(), i))
toolbox.cal_gpac(acf_lst)

############################################################
# ARIMA Model
############################################################

# ARIMA(4,0,1)

toolbox.ARIMA_model(df1, n, 4, 1, 0, 'ARIMA(4,0,1)')

# ARIMA(5,0,1)

toolbox.ARIMA_model(df1, n, 5, 1, 0, 'ARIMA(5,0,1)')


