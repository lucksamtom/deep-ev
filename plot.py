#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import datetime as dt
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import dateutil, random  
import pylab as pylab
from pylab import *
from datetime import datetime,timedelta  
from pandas import Series
from preprocessing import extractHourlyPower 
import pickle as pickle
import numpy
import sklearn.metrics as smet


def plot(prediction_dir):

    # get real data
    USING_CACHE = True
    series_cache_path = './data/series_cache.pkl'
    real_data = extractHourlyPower(series_cache_path, USING_CACHE)

    # get prediction data
    fi = open(prediction_dir,'rb')
    prediction = pickle.load(fi)
    fi.close()
    print('prediction amount: '+str(len(prediction)))
    print('prediction sahpe: '+str(numpy.shape(prediction[0])))

    # get prediction data
    #fi = open('./result/pre_history_average.pkl','rb')
    fi = open('./result/pre_ecd-dcd_batch1_lstm1_patience5.pkl','rb')
    history_average = pickle.load(fi)
    fi.close()

    # plot real data
    date = [str(real_data[i:i+1].keys().values[0]) for i in range(len(real_data))]
    real_power = [i for i in real_data]
    real_x = [dt.datetime.strptime(d,'%Y%m%d%H') for d in date]
    pylab.plot_date(pylab.date2num(real_x), real_power, linestyle='-', color='black')
    
    # plot prediction data
    power_pre = prediction
    pre_point_num = len(power_pre)
    data_pre = date[-pre_point_num:]
    #print(data_pre)
    #print(power_pre)
    x_pre = [dt.datetime.strptime(d,'%Y%m%d%H') for d in data_pre]
    pylab.plot_date(pylab.date2num(x_pre), power_pre, linestyle='-', color='green')

    # plot history average
    pylab.plot_date(pylab.date2num(x_pre), history_average, linestyle='-', color='red')
    
    # calculate rmse
    y_true = real_power[-pre_point_num:]
    print('history average rmse: ', sqrt(smet.mean_squared_error(y_true, history_average)))
    print('predection rmse: ', sqrt(smet.mean_squared_error(y_true, power_pre)))
    #print(sqrt(smet.mean_squared_error(y_true, power_val_pre)))
        
    xlabel(u"date & hour")
    ylabel(u"power_used (every hour)")

    grid(True)

    show()
    

if __name__ == '__main__':
    plot('./result/pre_ecd-dcd_batch1_lstm2_patience5.pkl')
    