#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import pickle as pickle
import datetime as dt
import matplotlib.dates as mdates
#import matplotlib.pyplot as plt
import dateutil, random  
#import pylab as pylab
#from pylab import *
from datetime import datetime,timedelta  
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
import sklearn.metrics as smet
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model
from keras import backend as K
from keras.engine.topology import Layer
from keras.layers import Input, add
from keras.layers.core import Dense
from keras.layers.core import Reshape, Activation, Flatten
from keras.layers.core import Dropout
from keras.layers.normalization import BatchNormalization
from keras.layers import ConvLSTM2D, Convolution2D
from keras.layers.recurrent import GRU, LSTM, SimpleRNN
from keras.models import Model
from keras.callbacks import EarlyStopping
from math import sqrt
#import metrics
from keras.optimizers import Adam
import numpy
from preprocessing import extractHourlyPower, save_cache
from utils import MinMaxNormalization, scale
from order import Order, Record
from iLayer import iLayer
import copy
numpy.random.seed(1337)  # for reproducibility


def geo_dict_paser(geo_dir):
	
	IDtoID = {}
	loc_dict = {}

	geo_f = open('GEO/BJ_geoID.csv', "r")
	print (geo_f.name)
	lines = geo_f.readlines()
	geo_f.close()


	for line in lines:
		line = line.strip()
		longID = line.split(',')[0]
		shortID = line.split(',')[1]
		if longID not in IDtoID:
			IDtoID[longID] = shortID
		
		loc = (line.split(',')[2],line.split(',')[3])
		if shortID not in loc_dict:
			loc_dict[shortID] = loc

	return IDtoID, loc_dict

def SeriesToXy (series, window = 13):
    idx = window
    Xy_set = numpy.array([])
    while idx <= series.size:
        Xy_set = numpy.append(Xy_set,series[idx-window:idx], axis = 0)
        idx = idx + 1
    Xy_set = Xy_set.reshape((-1,window))
    X_set, y_set = Xy_set[:,0:window-1], Xy_set[:,-1]
    return X_set.reshape(-1,window-1,1), y_set.reshape(-1)

def SeriesToXy_ed (train_series, test_series, window = 25):
    X_set_list = []
    y_set_list = []
    
    for series in [train_series, test_series]:
	    idx = window
	    Xy_set = numpy.array([])
	    while idx <= series.size:
	        Xy_set = numpy.append(Xy_set,series[idx-window:idx], axis = 0)
	        idx = idx + 1
	    Xy_set = Xy_set.reshape((-1,window))
	    X_set, y_set = Xy_set[:,0:window-1], Xy_set[:,-12:]
	    X_set_list.append(X_set.reshape(-1,window-1,1))
	    y_set_list.append(y_set.reshape(-1,12,1))

    X_train = [X_set_list[0], X_set_list[0][:,12:,:]]
    y_train = y_set_list[0]
    X_test  = [X_set_list[1], X_set_list[1][:,12:,:]]
    y_test  = y_set_list[1]

    return X_train, y_train, X_test, y_test

def SeriesToXy_period (train_series, test_series, window = 73):# 3 days
	X_set_list = []
	y_set_list = []

	for series in [train_series, test_series]:
		idx = window
		Xy_set = numpy.array([])
		while idx <= series.size:
			Xy_set = numpy.append(Xy_set,series[idx-window:idx], axis = 0)
			idx = idx + 1
		Xy_set = Xy_set.reshape((-1,window))
		X_clossness, X_period, y_set = Xy_set[:,-4:-1], Xy_set[:,(0,24,48)], Xy_set[:,-1]
		X_set_list.append([X_clossness.reshape(-1,3,1), X_period.reshape(-1,3,1)])
		y_set_list.append(y_set.reshape(-1,1,1))

	X_train = X_set_list[0]
	y_train = y_set_list[0]
	X_test  = X_set_list[1]
	y_test  = y_set_list[1]
	print('X_train_clossness, X_train_period, y_train shape:', numpy.shape(X_train[0]), numpy.shape(X_train[1]), numpy.shape(y_train))
	print('X_test_clossness, X_test_period, y_test shape:', numpy.shape(X_test[0]), numpy.shape(X_test[1]), numpy.shape(y_test))

	return X_train, y_train, X_test, y_test

def fit_period_lstm(c_conf=(3,1), p_conf=(3,1)):

	main_inputs = []
	outputs = []
	main_outputs = []
	for conf in [c_conf, p_conf]:
	    if conf is not None:
	        len_seq, vec_len = conf
	        input = Input(shape=(len_seq, vec_len))
	        main_inputs.append(input)
	        lstm_1 = LSTM(units=vec_len, activation='tanh', return_sequences=True)(input)
	        lstm_2 = LSTM(units=vec_len, activation='tanh', return_sequences=True)(lstm_1)
	        lstm_3 = LSTM(units=vec_len, activation='tanh', return_sequences=True)(lstm_2)
	        outputs.append(lstm_3)
	if len(outputs) == 1:
	    main_output = outputs[0]
	else:
	    new_outputs = []
	    for output in outputs:

	        new_outputs.append(iLayer()(output))
	    main_output = add(new_outputs)

	reshape_1 = Reshape((1, 3))(main_output)
	main_outputs = Dense(units=1, activation='tanh')(reshape_1)

	model = Model(inputs=main_inputs, outputs=main_outputs)

	early_stopping = EarlyStopping(monitor='val_loss', patience=5, mode='min')
	# Run training
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.summary()
	return model

def fit_lstm(X, y, batch_size, nb_epoch, neurons):
	model = Sequential()
	model.add(LSTM(units=vec_len, activation='tanh', return_sequences=True))
	model.add(Dense(1))
	return model

def history_average(data_series):
	history = data_series[-144:].values
	prediction = numpy.array([])
	temp = numpy.array([])
	for i in range(72):
		average = (history[i] + history[i+24] + history[i+48]) / 3
		prediction = numpy.append(prediction, average)
	return prediction

def prediction_ed (test, model):
	window = 25 #input combined output
	X_test, y_test = SeriesToXy_ed(test, window)

	X_test_decoder = X_test[:,12:,:]
	X_test_encoder = X_test

	test_size = numpy.shape(X_test)[0]
	idx = 0

	Y_predict = numpy.array([])
	while idx < test_size:
	    pre_temp = model.predict([X_test_encoder[idx].reshape(1,24,1), X_test_decoder[idx].reshape(1,12,1)])
	    #print(scaler.inverse_transform(pre_temp))
	    pre_temp = scaler.inverse_transform(pre_temp[0][-1][0])
	    #print(pre_temp)
	    Y_predict = numpy.append(Y_predict, pre_temp)
	    idx = idx + 1
	return Y_predict

def fit_encoder_decoder(latent_dim, num_encoder_tokens, num_decoder_tokens):

	# Define an input sequence and process it.
	encoder_inputs = Input(shape=(None, num_encoder_tokens))
	encoder = LSTM(latent_dim, return_state=True)
	encoder_outputs, state_h, state_c = encoder(encoder_inputs)
	# We discard `encoder_outputs` and only keep the states.
	encoder_states = [state_h, state_c]

	# Set up the decoder, using `encoder_states` as initial state.
	decoder_inputs = Input(shape=(None, num_decoder_tokens))
	# We set up our decoder to return full output sequences,
	# and to return internal states as well. We don't use the 
	# return states in the training model, but we will use them in inference.
	decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
	decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
	                                     initial_state=encoder_states)
	decoder_dense = Dense(num_decoder_tokens, activation='tanh')
	decoder_outputs = decoder_dense(decoder_outputs)

	# Define the model that will turn
	# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
	model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
	
	return model

def rmse(y_true,y_pred):
    return mean_squared_error(y_true, y_pred)**0.5

def build_model():
	'''
	reference only
	'''
	main_inputs = []
	main_outputs = []

	input = Input(shape=(12, 1))
	main_inputs.append(input)
	lstm_1 = LSTM(units=1, activation='relu', return_sequences=True)(input)
	batch_1 = BatchNormalization()(lstm_1)
	drop_1 = Dropout(0.2)(batch_1)
	lstm_2 = LSTM(units=1, activation='relu', return_sequences=True)(drop_1)
	batch_2 = BatchNormalization()(lstm_2)
	drop2 = Dropout(0.2)(batch_2)
	lstm_3 = LSTM(units=1, activation='relu', return_sequences=True)(drop2)
	batch_3 = BatchNormalization()(lstm_3)
	drop3 = Dropout(0.2)(batch_3)
	lstm_4 = LSTM(units=1, activation='relu')(drop3)
	batch_4 = BatchNormalization()(lstm_4)
	main_outputs.append(batch_4)

	model = Model(inputs=main_inputs, outputs=main_outputs)

	lr = 0.0002  # learning rate
	adam = Adam(lr)
	model.compile(loss='mse', optimizer=adam,
	              metrics=[metrics.rmse])
	model.summary()

	return model

def main():
	USING_CACHE = True
	series_cache_path = './data/series_cache.pkl'
	order_set = []

	data_series = extractHourlyPower(series_cache_path, USING_CACHE)
	
	training_day = 20 # 18 for period; 20 for encoder-decoder
	total_day = 24
	train_series, test_series = data_series[:training_day*24], data_series[(training_day-total_day)*24:]
	y_true = copy.copy(test_series[-72:].values)
	
	#history_average 
	#prediction = history_average(data_series)
	
	scaler, train_series, test_series = scale(train_series, test_series)

	#fetch encoder_decoder training data and model
	X_train, y_train, X_test, y_test = SeriesToXy_ed(train_series, test_series, window = 25)
	model = fit_encoder_decoder(12, 1, 1)

	#fetch period training data and model
	#X_train, y_train, X_test, y_test = SeriesToXy_period(train_series, test_series, window = 73)
	#model = fit_period_lstm()

	early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')
	# Run training
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.summary()

	model.fit(X_train, y_train,
	          batch_size=1,
	          epochs=300, 
	          validation_split=0.2,
	          callbacks = [early_stopping])
	
	prediction = model.predict(X_test)

	prediction = scaler.inverse_transform(prediction)

	# encoder decoder 
	prediction = prediction[:,-1,:]

	prediction = prediction.reshape(-1)
	rmse = sqrt(smet.mean_squared_error(y_true, prediction))
	
	print('rmse: ', rmse)
	save_cache(prediction, './data/pre_ecd-dcd_batch1_lstm1_patience5.pkl')

	K.clear_session()# tensorflow bug

if __name__ == '__main__':

	main()
	exit(0)
