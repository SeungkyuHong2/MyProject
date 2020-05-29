# -*- coding: utf-8 -*-

# Do not change this part
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy.stats import f as fdist
from scipy.stats import t as tdist
from scipy.stats import chi2 

# Data load
data=pd.read_csv('https://drive.google.com/uc?export=download&id=1O74eCM8zlPxCFEuEpshmFXA3HpKT1qbC')
data_column = ['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']
#TODO: explanatory analysis
#TODO: B, C, D - Darw pairwise scatter plots
plt.scatter(data['Mean_temperature'],data['Max_temperature'])
plt.scatter(data['Mean_temperature'],data['Min_temperature'])
plt.scatter(data['Mean_temperature'],data['Dewpoint'])
plt.scatter(data['Mean_temperature'],data['Precipitation'])
plt.scatter(data['Mean_temperature'],data['Sea_level_pressure'])
plt.scatter(data['Mean_temperature'],data['Standard_pressure'])
plt.scatter(data['Mean_temperature'],data['Visibility'])
plt.scatter(data['Mean_temperature'],data['Wind_speed'])
plt.scatter(data['Mean_temperature'],data['Max_wind_speed'])
legend = plt.legend(loc='upper left',shadow=False,fontsize='x-small')


#TODO: E, F - Calculate correlation matrix
corr = pd.DataFrame.corr(data)
#TODO: linear regression
#TODO: B - Calculate VIF
model = LinearRegression()

vif = []
model_fit = model.fit(data[['Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Max_temperature'])
model_score1 = model.score(data[['Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Max_temperature'])
vif.append(1/(1-model_score1))
model_fit = model.fit(data[['Max_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Min_temperature'])
model_score2 = model.score(data[['Max_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Min_temperature'])
vif.append(1/(1-model_score2))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Dewpoint'])
model_score3 = model.score(data[['Max_temperature','Min_temperature','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Dewpoint'])
vif.append(1/(1-model_score3))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Dewpoint','Sea_level_pressure',
               'Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Precipitation'])
model_score4 = model.score(data[['Max_temperature','Min_temperature','Dewpoint','Sea_level_pressure',
               'Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Precipitation'])
vif.append(1/(1-model_score4))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Sea_level_pressure'])
model_score5 = model.score(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Standard_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Sea_level_pressure'])
vif.append(1/(1-model_score5))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Standard_pressure'])
model_score6 = model.score(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Visibility','Wind_speed','Max_wind_speed']],data['Standard_pressure'])
vif.append(1/(1-model_score6))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Wind_speed','Max_wind_speed']],data['Visibility'])
model_score7 = model.score(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Wind_speed','Max_wind_speed']],data['Visibility'])
vif.append(1/(1-model_score7))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Max_wind_speed']],data['Wind_speed'])
model_score8 = model.score(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Max_wind_speed']],data['Wind_speed'])
vif.append(1/(1-model_score8))
model_fit = model.fit(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed']],data['Max_wind_speed'])
model_score9 = model.score(data[['Max_temperature','Min_temperature','Dewpoint','Precipitation',
               'Sea_level_pressure','Standard_pressure','Visibility','Wind_speed']],data['Max_wind_speed'])
vif.append(1/(1-model_score9))



#find the expexted values
def prediction(x,b):
    
    predict = []
    n,p = x.shape
    x = np.c_[np.ones(n),x]
    predict = np.matmul(x,b)
    predict = predict.flatten()
    return predict

# Model 1
#TODO: C - Train a lineare regression model
#          Calculate t-test statistics 

Model1 = LinearRegression()
Model1.fit(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']],data['Mean_temperature'])

b = []
b.append(Model1.intercept_)
b.append(Model1.coef_[0])
b.append(Model1.coef_[1])
b.append(Model1.coef_[2])
b.append(Model1.coef_[3])
beta = np.insert(Model1.coef_,0,Model1.intercept_)
#find the ss
y_predict = prediction(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']],b)
SSE = np.sum((data['Mean_temperature']-y_predict)**2)
SSR = np.sum((y_predict-np.mean(data['Mean_temperature']))**2)
SST = np.sum((data['Mean_temperature']-np.mean(data['Mean_temperature']))**2)


#find the t-statistic
MSE = SSE/(data.shape[0]-4-1)
data2 = data.copy()
data2['intercept'] = np.ones(len(data2))
data2 = data2[['intercept','Precipitation','Visibility','Wind_speed','Max_wind_speed']]
XtX = np.matmul(data2.T,data2)
Xinv = np.linalg.inv(XtX)
se_mat = MSE*Xinv

t_value = []
for i in range(len(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']].columns)):
    tv = b[i+1]/np.sqrt(se_mat[i+1,i+1])
    t_value.append(tv)

m = np.sum((data['Visibility']-np.mean(data['Visibility']))**2)
# find the P-value    
p_value = []
for i in t_value:
    p = 1-tdist.cdf(np.abs(i),data.shape[0]-4-1)
    p_value.append(p)

#t-test function
def t_test(p,a):
    boolean = []
    for i in range(len(p)):
        if p[i] < a/2:
            boolean.append(True)
        else:
            boolean.append(False)
    return boolean
ttest = t_test(p_value,0.05)

#TODO: D - Calculate adjusted R^2
adj_R = 1-(SSE/data.shape[0]-4-1)/(SST/data.shape[0]-1)

#TODO: E - Apply F-test
def f_test(x,b,a):
    f = 0
    p_value = 0
    boolean = None
    
    n,p = x.shape
    MSR = SSR/p
    MSE = SSE/n-p-1
    f = MSR/MSE
    
    p_value = 1-fdist.cdf(f,p,n-p-1)
    
    if p_value < a:
        boolean = True
    else:
        boolean = False
    return (f,p_value,boolean)

ftest = f_test(data[['Precipitation','Visibility','Wind_speed','Max_wind_speed']],b,0.05)

# Model 2
#TODO: F - Check the appropriateness of the given set of variables
model2 = LinearRegression()

#correlation matrix
corr2 = pd.DataFrame.corr(data[['Dewpoint','Precipitation','Visibility']])

#vif
vif2 = []
model2.fit(data[['Precipitation','Visibility']],data['Dewpoint'])
model2_score1=model2.score(data[['Precipitation','Visibility']],data['Dewpoint'])
vif2.append(1/(1-model2_score1))
model2.fit(data[['Dewpoint','Visibility']],data['Precipitation'])
model2_score2=model2.score(data[['Dewpoint','Visibility']],data['Precipitation'])
vif2.append(1/(1-model2_score2))
model2.fit(data[['Dewpoint','Precipitation']],data['Visibility'])
model2_score3=model2.score(data[['Dewpoint','Precipitation']],data['Visibility'])
vif2.append(1/(1-model2_score3))



#TODO: G - Train a lineare regression model
#          Calculate t-test statistics 
model2.fit(data[['Dewpoint','Precipitation','Visibility']],data['Mean_temperature'])

b2 = []
b2.append(model2.intercept_)
b2.append(model2.coef_[0])
b2.append(model2.coef_[1])
b2.append(model2.coef_[2])
y2_predict = prediction(data[['Dewpoint','Precipitation','Visibility']],b2)

SST2 = np.sum((data['Mean_temperature']-np.mean(data['Mean_temperature']))**2)
SSR2 = np.sum((y2_predict-np.mean(data['Mean_temperature']))**2)
SSE2 = np.sum((data['Mean_temperature']-y2_predict)**2)

MSE2 = SSE2/(data.shape[0]-3-1)
data3 = data.copy()
data3['intercept'] = np.ones(len(data3))
data3 = data3[['intercept','Dewpoint','Precipitation','Visibility']]
XtX2 = np.matmul(data3.T,data3)
Xinv2 = np.linalg.inv(XtX2)

se_mat2 = MSE2*Xinv2

t_value2 = []
for i in range(len(data[['Dewpoint','Precipitation','Visibility']].columns)):
    tv = b2[i+1]/np.sqrt(se_mat2[i+1,i+1])
    t_value2.append(tv)
    
p_value2 = []
for i in t_value2:
    p = 1-tdist.cdf(np.abs(i),data.shape[0]-3-1)
    p_value2.append(p)
    
ttest2 = t_test(p_value2,0.05)



#TODO: H - Calculate adjusted R^2
adj_R2 = 1-(SSE2/data.shape[0]-3-1)/(SST2/data.shape[0]-1)



#TODO: I - Test normality of residuals

er = data['Mean_temperature'].values-y2_predict
er_mean = np.sum(er)/data.shape[0]
S = 1/data.shape[0]*(np.sum((er-er_mean)**3))/((1/data.shape[0])*np.sum((er-er_mean)**2)**3/2)
C = 1/data.shape[0]*(np.sum((er-er_mean)**4))/((1/data.shape[0])*np.sum((er-er_mean)**2)**2)   

JB = ((data.shape[0]-data.shape[1])/6)*(S**2+1/4*((C-3)**2))
JB_value = 1-chi2.cdf(JB,2)
def JB_test(jb,a):
    boolean = []
    if jb < a:
        boolean = True
    else:
        boolean = False
    return(boolean)

jbtest = JB_test(JB_value,0.05)
    
 
#TODO: J - Test homoskedasticty of residuals

er_2 = er**2
model3 = LinearRegression()
model3.fit(data[['Dewpoint','Precipitation','Visibility']],er_2)
er2_predict = model3.predict(data[['Dewpoint','Precipitation','Visibility']])
SSE3 = np.sum((er_2-er2_predict)**2)
SSR3 = np.sum((er2_predict-np.mean(er_2))**2)

LM = data.shape[0]*model3.score(data[['Dewpoint','Precipitation','Visibility']],er_2)

bp_value=1-chi2.cdf(LM,3)
def Breusch_pagan(bp,a):
    boolean = []
    if bp < a:
        boolean = True
    else:
        boolean = False
    return(boolean)
bptest = Breusch_pagan(bp_value,0.05)



