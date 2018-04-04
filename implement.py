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
from preprocessing import extractHourlyPower 
from minmax import MinMaxNormalization
from order import Order, Record
from iLayer import iLayer
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

def SeriesToXy_ed (series, window = 13):
    idx = window
    Xy_set = numpy.array([])
    while idx <= series.size:
        Xy_set = numpy.append(Xy_set,series[idx-window:idx], axis = 0)
        idx = idx + 1
    Xy_set = Xy_set.reshape((-1,window))
    X_set, y_set = Xy_set[:,0:window-1], Xy_set[:,-12:]
    return X_set.reshape(-1,window-1,1), y_set.reshape(-1,12,1)

def SeriesToXy_period (series, window = 73):# 3 days
	idx = window
	Xy_set = numpy.array([])
	while idx <= series.size:
		Xy_set = numpy.append(Xy_set,series[idx-window:idx], axis = 0)
		idx = idx + 1
	Xy_set = Xy_set.reshape((-1,window))
	X_train_clossness, X_train_period, y_set = Xy_set[:,-4:-1], Xy_set[:,(0,24,48)], Xy_set[:,-1]
	return X_train_clossness.reshape(-1,3,1), X_train_period.reshape(-1,3,1), y_set.reshape(-1,1,1)

# scale train and test data to [-1, 1]
def scale(train, test):
    train_value = train.values
    test_value = test.values
    temp = numpy.append(train_value,test_value)
    # fit scaler
    scaler = MinMaxNormalization()
    #print(temp.reshape(-1))
    scaler.fit(temp.reshape(-1))
    # transform train
    train_value = train_value.reshape(-1)
    train_scaled = scaler.transform(train_value)
    train_scaled = train_scaled.reshape(-1)
    # transform test
    test_value = test_value.reshape(-1)
    test_scaled = scaler.transform(test_value)
    test_scaled = test_scaled.reshape(-1)
    train[0:] = train_scaled
    test[0:] = test_scaled
    return scaler, train, test

def fit_period_lstm(train, test, epochs, c_conf=(3,1), p_conf=(3,1)):
	
	window = 73 #input combined output
	X_train_clossness, X_train_period, y_train = SeriesToXy_period(train, window)
	X_test_clossness, X_test_period, y_test = SeriesToXy_period(test, window)

	print('X_train_clossness, X_train_period, y_train shape:', numpy.shape(X_train_clossness), numpy.shape(X_train_period), numpy.shape(y_train))
	print('X_test_clossness, X_test_period, y_test shape:', numpy.shape(X_test_clossness), numpy.shape(X_test_period), numpy.shape(y_test))

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
	model.fit([X_train_clossness, X_train_period], y_train,
	          batch_size=3,
	          epochs=epochs,
	          validation_split=0.2)
	prediction = model.predict([X_test_clossness, X_test_period])
	prediction = scaler.inverse_transform(prediction.reshape(-1))
	print('prediction shape: ', numpy.shape(prediction))
	return model, prediction


def fit_lstm(X, y, batch_size, nb_epoch, neurons):
	early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min')
	model = Sequential()
	model.add(LSTM(neurons, batch_input_shape=(batch_size, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(1))
	model.compile(loss='mean_squared_error', optimizer='adam')
	model.fit(X, y, epochs=nb_epoch, batch_size=batch_size, verbose=1, shuffle=False, callbacks=[early_stopping])
	return model

def history_average(test):
	history = test[0:-72].values
	prediction = numpy.array([])
	temp = numpy.array([])
	for i in range(24):
		average = (history[i] + history[i+24] + history[i+48]) / 3
		temp = numpy.append(temp, average)
	prediction = numpy.append(prediction,temp)
	prediction = numpy.append(prediction,temp)
	prediction = numpy.append(prediction,temp)
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

def fit_encoder_decoder(train, test, latent_dim, num_encoder_tokens, num_decoder_tokens, epochs):
	
	window = 25 #input combined output
	X_train, y_train = SeriesToXy_ed(train, window)
	X_test, y_test = SeriesToXy_ed(test, window)

	decoder_input_data = X_train[:,12:,:]
	encoder_input_data = X_train
	decoder_target_data = y_train

	X_test_decoder = X_test[:,12:,:]
	X_test_encoder = X_test

	print('X_train_encoder, X_train_encoder, y_train shape:', numpy.shape(encoder_input_data), numpy.shape(decoder_input_data), numpy.shape(decoder_target_data))
	print('X_test_encoder, X_test_encoder, y_test shape:', numpy.shape(X_test_encoder), numpy.shape(X_test_decoder), numpy.shape(y_test))



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
	early_stopping = EarlyStopping(monitor='loss', patience=5, mode='min')
	# Run training
	model.compile(optimizer='adam', loss='mean_squared_error')
	model.summary()
	model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
	          batch_size=3,
	          epochs=epochs,
	          callbacks=[early_stopping])
	prediction = prediction_ed(test, model)

	return model, prediction

def prediction (X_test,model):
	test_size = numpy.shape(X_test)[0]
	idx = 0

	Y_predict = numpy.array([])
	while idx < test_size:
	    pre_temp = model.predict(X_test[idx].reshape(1,12,1))
	    pre_temp = scaler.inverse_transform(pre_temp)
	    Y_predict = numpy.append(Y_predict,pre_temp)
	    idx = idx + 1
	return Y_predict

def rmse(y_true,y_pred):
    return mean_squared_error(y_true, y_pred)**0.5

def build_model():
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

if __name__ == '__main__':
	
	USING_CACHE = True
	series_cache_path = './data/series_cache.pkl'

	order_set = []
	IDtoID = {}
	loc_dict = {}

	#处理地理信息
	#(IDtoID, loc_dict) = geo_dict_paser(geo_dir)
	#print(loc_dict)
	order_set_geo = []
	'''
	#筛选有地理标示的订单
	for order in order_set:
		if order.stubId in IDtoID:
			order.shortID = IDtoID[order.stubId]
			order.geo_info = loc_dict[order.shortID]
			order_set_geo.append(order)
	
	print ('Order_set amount before filter is:' + str(len(order_set_geo)))
	'''

	'''
	date = [str(i[0]) for i in sorted_power_list]
	power = [i[1] for i in sorted_power_list]
	print(date)
	print(power)

	#x = [time.mktime(time.strptime(str(d), "%Y%m%d%H")) for d in date]
	x = [dt.datetime.strptime(d,'%Y%m%d%H') for d in date]
	#plt.plot(x, power)
	pylab.plot_date(pylab.date2num(x), power, linestyle='-')
	xlabel(u"date & hour")
	ylabel(u"power_used (every hour)")

	grid(True)

	show()
	
	time_save_end = time.time()
	#print('Program cost time:'+str(time_save_end-time_save_start))
	#for order in order_set:
		#print(order.power_dict)
	'''

	series = extractHourlyPower(series_cache_path, USING_CACHE)
	
	training_day = 18
	total_day = 24
	train, test = series[:training_day*24], series[(training_day-total_day)*24:]
	y_true = test[-72:]
	scaler, train, test = scale(train, test)

	
	#lstm_ed_model, prediction = fit_encoder_decoder(train, test, 12, 1, 1, 3000)
	lstm_period_model, prediction = fit_period_lstm(train, test, 300, c_conf=(3,1), p_conf=(3,1))
	#prediction = history_average(test)

	#mse = mean_squared_error(y_true, prediction[-1].tolist())
	#print('mse: '+str(mse))
	'''

	window = 13 #input combined output
	X_train, y_train = SeriesToXy(train, window)
	X_test, y_test = SeriesToXy(test, window)

	print('X_train, y_train shape:', numpy.shape(X_train), numpy.shape(y_train))
	print('X_test, y_test shape:', numpy.shape(X_test), numpy.shape(y_test))
	batch_size = 1

	lstm_model = fit_lstm(X_train, y_train, batch_size, 3000, 4)

	
	#lstm_model = build_model()
	#lstm_model.fit(X_train, y_train, epochs=6000, batch_size=3, verbose=1, shuffle=False)
	
	lstm_model.save('lstm_model_B.h5')  # creates a HDF5 file 'my_model.h5'
	
	#lstm_model = load_model('lstm_model.h5')
	score = lstm_model.evaluate(X_test, y_test, batch_size=1)
	print(X_test)
	prediction = prediction(X_test, lstm_model)
	print(prediction)
	K.clear_session()# tensorflow bug
	'''

	'''
	#predict encoder_decoder
	
	prediction = prediction_ed (test, lstm_ed_model)
	#print(prediction)
	print(numpy.shape(prediction))
	'''

	#open and save predicted data
	
	with open('Y_3lstm_p5_batch3_epoch300.pkl','wb') as f:
		pickle.dump(prediction, f, pickle.HIGHEST_PROTOCOL)
	
	fi = open('Y_3lstm_p5_batch3_epoch300.pkl','rb')
	prediction = pickle.load(fi)
	fi.close()
	print(prediction)
	print('rmse: ', sqrt(smet.mean_squared_error(y_true, prediction)))
	'''
	#predicted_series = series[(training_day-total_day)*24:][-84:]
	#predicted_series[:]=prediction
	#print(predicted_series)

	'''
	'''
	#tide up predicted data
	date = [str(i[0]) for i in sorted_power_list]
	power = [i[1] for i in sorted_power_list]

	x = [dt.datetime.strptime(d,'%Y%m%d%H') for d in date]
	pylab.plot_date(pylab.date2num(x), power, linestyle='-', color='black')


	data_pre = date[-72:]
	power_pre = prediction[-2]
	power_val_pre = val_prediction[-72:]
	#print(data_pre)
	#print(power_pre)
	x_pre = [dt.datetime.strptime(d,'%Y%m%d%H') for d in data_pre]
	pylab.plot_date(pylab.date2num(x_pre), power_pre, linestyle='-', color='red')
	pylab.plot_date(pylab.date2num(x_pre), power_val_pre, linestyle='-', color='green')
	print(sqrt(smet.mean_squared_error(y_true, power_pre)))
	print(sqrt(smet.mean_squared_error(y_true, power_val_pre)))

	
	xlabel(u"date & hour")
	ylabel(u"power_used (every hour)")

	grid(True)

	show()
	'''
