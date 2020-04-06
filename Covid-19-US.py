# -*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from scipy.optimize  import leastsq
import datetime
import matplotlib as mat
from pandas.core.frame import DataFrame

def err_f(params, t, y):
    return logistic_inc_function(params, t) - y


def logistic_inc_function(params, t):
    K, a, b = params
    exp_value = np.exp(-a * (t - b))
    return K / (1 + exp_value)

def getDailyInc(input_list):
    output_list = []
    for i in range(len(input_list)-1):
        output_list.append(input_list[i+1]-input_list[i])
    return output_list
 
def fitCurve(input_data, title, block_days, fill_dates, start_days):
    logistic_p0 = [500000, 0.3, 28]  # Initialization
     
    t = np.array([i + 1 for i in range(len(input_data))])
#     print(input_data)
#     print(input_data.head())
    target_y = input_data[0].values
    
    logistic_params = leastsq(err_f, logistic_p0, args=(t, target_y))
       
    logistic_p = logistic_params[0]
#     print("logistic_p=",logistic_p) 

    predict_y = logistic_inc_function(logistic_p, t)
    
#     print(predict_y)

    # error
    pred_e = target_y - predict_y
    
    start_time = datetime.datetime(2020,1,22) + datetime.timedelta(days=start_days)
    last_date = start_time + datetime.timedelta(days = (len(target_y) - fill_dates - 1) )
    last_date_str = last_date.strftime('%Y-%m-%d')
#     print(start_time)
#     print(fill_dates)
#     print(len(input_data))
#     print(block_days)
    
    end_time = datetime.datetime(2020, 6, 30)
    inc = datetime.timedelta(days=1)
    dates = mat.dates.drange(start_time, end_time, inc)  

    t2 = np.array([i + 1 for i in range(len(dates))])
    predict_data_long = logistic_inc_function(logistic_p, t2)

#     print(dates)
#     print(target_y)
#     print(len(input_data))
#     print(len(target_y))
    
    plt.plot_date(dates, predict_data_long, label='Predictions')
    if fill_dates >0:
        plt.plot_date(dates[:len(input_data)-fill_dates], target_y[:-fill_dates], label="Actual Cases")
        plt.plot_date(dates[len(input_data)-fill_dates:len(input_data)], target_y[len(input_data)-fill_dates:], label="Pseudo Cases")
    else:
        plt.plot_date(dates[:len(input_data)], target_y[:], label="Actual Cases")
    
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title(title+"_cumulative_cases_"+last_date_str)
    plt.legend(loc='best') 
    plt.rcParams['figure.figsize'] = (12,8)
    plt.savefig(title+"_cumulative_cases_"+last_date_str+".png", dpi=200)
    plt.show()

    plt.plot_date(dates[1:], getDailyInc(predict_data_long), label='Predictions')
    if fill_dates >0:
        plt.plot_date(dates[1:len(input_data)-fill_dates], getDailyInc(target_y)[:-fill_dates], label="Actual Cases")
        plt.plot_date(dates[len(input_data)-fill_dates:len(input_data)], getDailyInc(target_y)[len(input_data)-fill_dates-1:], label="Pseudo Cases")
    else:
        plt.plot_date(dates[1:len(input_data)], getDailyInc(target_y), label="Actual Cases")
        
#     plt.plot_date(dates[1:len(input_data)], getDailyInc(target_y), label="Actual Cases")
    
    plt.xlabel('Date')
    plt.ylabel('Number of Cases')
    plt.title(title+"_daily_cases_"+last_date_str)
    plt.legend(loc='best') 
    plt.rcParams['figure.figsize'] = (12,8)
    plt.savefig(title+"_daily_cases_"+last_date_str+".png", dpi=200)
    plt.show()

def plotConfirmedCases(input_data, start_days, last_date_str):  
    
    start_time = datetime.datetime(2020, 1, 22)
    end_time = start_time + datetime.timedelta(days=len(input_data))
    inc = datetime.timedelta(days=1)
    dates = mat.dates.drange(start_time, end_time, inc)  
    
    plt.style.use("ggplot")
#     plt.style.use("seaborn")
#     print(len(dates))
    plt.plot_date(dates[start_days:], input_data[start_days:], label='Confirmed Cases')
    plt.legend(loc="best")
    plt.xlabel("Date")
    plt.title("US COVID-19 Confirmed Cases_"+ last_date_str)
    plt.rcParams['figure.figsize'] = (12,8)
    plt.savefig("US COVID-19 Confirmed Cases_"+ last_date_str + ".png", dpi=200)
    plt.show()

def fillData(input_data, fill_dates, ratio):
    ori_len = len(input_data)
    num = ori_len
    for i in range(fill_dates):
        input_data.loc[num]= input_data.loc[num-1]*ratio
#         print(input_data.loc[num-1]*(ratio-1))
        num += 1
    return input_data

def calIncRatio(input_data):
    l = []
    x = input_data[0].values.flatten().tolist()
    for i in range(len(x)-1):
        r = float(x[i+1]-x[i])/(x[i])
        l.append(r)
    return l
#         print("---")
#         print(i)
#         print(r)

def go(start_days, fill_dates, ratio, US_data, title):   
    
    block_days = 0
    #     
    #     plt.style.use("seaborn")
    #     plt.plot_date(dates, US_data, label='Confirmed Cases')
    #     plt.show()
    #     
    # print(len(US_data))
    #     exit()
       
         
    US_data = US_data.astype("float")
    
    if block_days<0: 
        fitCurve(US_data[start_days:block_days], title, block_days, fill_dates, start_days)
    else: 
        fitCurve(US_data[start_days:], title, 0, fill_dates, start_days)
        
    pass

confirmed_cases_since_Jan_22 = [1, 1, 2, 2, 5, 5, 5, 5, 5, 7, 8, 8, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 13, 13, 
           13, 13, 13, 13, 13, 13, 15, 15, 15, 51, 51, 57, 58, 60, 68, 74, 98, 118, 149, 217, 
           262, 402, 518, 583, 959, 1281, 1663, 2179, 2727, 3499, 4632, 6421, 7783, 13677, 
           19100, 25489, 33276, 43847, 53740, 65778, 83836, 101657, 121478, 140886, 161807, 188172, 
           213372, 243453, 275586, 308850, 337072] #Until April 5th
US_data=DataFrame({0:confirmed_cases_since_Jan_22})
ori_len = len(US_data)

start_days = 0 
last_date = datetime.datetime(2020,1,22) + datetime.timedelta(days = (len(US_data) - 1) ) 
last_date_str = last_date.strftime('%Y-%m-%d')
  
plotConfirmedCases(US_data, start_days, last_date_str)

fill_dates = 0
ratio = 1.0
US_data = fillData(US_data, fill_dates, ratio)
title = "US_best_scenario_pred"
go(start_days, fill_dates, ratio, US_data, title)

fill_dates = 5
r_list = calIncRatio(US_data)
print(r_list)

ratio_arr = r_list[len(r_list) - 5:]
ratio = 1 + np.mean(ratio_arr) + np.std(ratio_arr,ddof=0)
# print(ratio)
# print(np.mean(ratio_arr))
# print(np.std(ratio_arr,ddof=0))
# print(np.std(ratio_arr,ddof=1))
# exit()
US_data = fillData(US_data, fill_dates, ratio)
title = "US_worst_scenario_pred"
go(start_days, fill_dates, ratio, US_data, title)
# print(US_data)
# calIncRatio(US_data)
#     exit()
