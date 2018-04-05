#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy

class MinMaxNormalization(object):
    """
    MinMax Normalization-->[-1,1]
      x=(x-min)/(max-min)
      x=x*2-1
    """
    def __int__(self):
        pass
    def fit(self, X):
        self._min = X.min()
        self._max = X.max()
        print('min', self._min, 'max', self._max)
    def transform(self,X):
        X = 1.*(X-self._min)/(self._max-self._min)
        X = X*2.-1
        return X
    def fit_transform( self, X):
        self.fit(X)
        return self.transform(X)
    def inverse_transform(self, X):
        X = (X+1.)/2.
        X = 1.*X*(self._max-self._min)+self._min
        return X

def scale(train, test):

    '''
    scale train and test data to [-1, 1]

    '''
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
