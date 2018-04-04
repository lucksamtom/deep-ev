#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import time
import re
import pickle as pickle
import os
import datetime as dt
import dateutil, random 
from datetime import datetime,timedelta  
from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime

class Record:
    voltage = 0.0
    current = 0.0
    soc = 0.0
    power_used = 0.0
    def __init__(self,time,voltage,current,soc,power_used):
        self.time = time
        self.voltage = voltage
        self.current = current
        self.soc = soc
        self.power_used = power_used

class Order:
    
    #订单中详细记录
    record_list = []
    power_dict = {}
    shortID = 0
    geo_info = None
    
    def __init__(self,in_dict):
        
        self.vin = in_dict['vin']
        self.chargeType = in_dict['chargeType']
        self.carTypeNo = in_dict['carTypeNo']
        self.stubId = in_dict['stubId']
        self.stubModelNo = in_dict['stubModelNo']
        self.startType = in_dict['startType']
        self.stubFirmwareVersion = in_dict['stubFirmwareVersion']
        self.stubFirmwareType = in_dict['stubFirmwareType']
                
        self.id = in_dict['id']
        self.userId = in_dict['userId']
        self.endCode = in_dict['endCode']
        
        #时间信息，格式化时间保存
        self.cts = in_dict['cts']
        self.ctl = in_dict['ctl']
        self.timeStart = in_dict['timeStart']
        self.timeEnd = in_dict['timeEnd']

        
        self.socStart = (in_dict['socStart'])
        self.soc = (in_dict['soc'])
        #用电信息，浮点保存
        self.power = float(in_dict['power'])


def get_file_name(file_dir):  
	'''
	return all file name in one dir
	param: string of raw_data_dir

	return: a list of all file names
	''' 
	L=[]   
	for root, dirs, files in os.walk(file_dir):  
		for file in files:  
			L.append(os.path.join(file))
	return L

def save_cache(order_set, saving_dir):
	# write data_cache with pkl
	time_save_start = time.time()
	with open(saving_dir,'wb') as f:
		pickle.dump(order_set, f, pickle.HIGHEST_PROTOCOL)
	time_save_end = time.time()
	print('pkl save cost:'+str(time_save_end-time_save_start))

def read_cache(cache_dir):
	#读取Cache
	fi = open(cache_dir,'rb')
	cache = pickle.load(fi)
	fi.close()
	print("Cache read successfully")
	return cache

def orderPaser (order_line):
	'''
	param: one string of every single order

	return: one Order object containing record and all info
	'''
	use_line = order_line.split(' ',4) #取订单,切分

	#print (use_line[4])    #去除无用字符，仅保留4

	p1  = r"(?<=is\":\[).+?(?=\],\"inf)"
	 
	searchObj = re.compile(p1).search(use_line[4])

	if searchObj:
		record_set = re.compile(p1).search(use_line[4]).group(0)#提取一个订单中的所有记录
	else:
		return None

	p_order = r"(?<=\"info\":\{).+?(?=\}\},\"type\":)"
	order = re.compile(p_order).search(use_line[4]).group(0) #提取一个订单中的信息

	#print (order)

	#提取并转换订单中的时间
	p_time = re.compile(r"20[0-9][0-9]\..+?:[0-9][0-9]:[0-9][0-9]")
	time_str = p_time.findall(order)
	order = p_time.sub(' ',order) #删除字符串中的时间，方便后续对字符串处理
	time_set = []
	for time_s in time_str:
	    time_set.append(time.strptime(time_s,"%Y.%m.%d %H:%M:%S"))
	
	#提取并转换订单中的 carTypeNo
	p_carTypeNo = re.compile(r"(?<=carTypeNo\":).+?(?=,\")")
	carTypeNo = p_carTypeNo.findall(order)[0].replace('\"','')
	#print(carTypeNo)
	carTypeNo = carTypeNo
	order = p_carTypeNo.sub('" "',order)
	#print(order)	

	#将订单信息转换为字典
	order_peice = order.replace('\"','').split(',')
	order_dict = {}
	#print(order_peice)
	
	for order_info in order_peice:
	    order_info = order_info.split(':')
	    #print(order_info)
	    order_dict[order_info[0]] = order_info[1]
	
	#格式化时间
	if time_set[0] < time_set[1]:
	    order_dict['timeStart'] = time_set[0]
	    order_dict['timeEnd'] = time_set[1]
	else:
	    order_dict['timeStart'] = time_set[1]
	    order_dict['timeEnd'] = time_set[0]
	order_dict['cts'] = order_dict['cts']
	order_dict['ctl'] = order_dict['ctl']

	#插入carTypeNo
	order_dict['carTypeNo'] = carTypeNo
	#print(order_dict)
	#生成order对象
	order_temp = Order(order_dict)


	'''
	下面为记录的处理

	'''

	p2 = r"\[.+?\]"
	matcher2 = re.compile(p2).findall(record_set) #将记录集转为List
	record_idx = 0  #取第一条记录
	#print (matcher2[record_idx])
	pattern3 = re.compile(r"(?<=\[).+?(?=\])")
	record_list = []
	#将一条记录分片
	for record in matcher2:
		peice = pattern3.search(record).group(0).replace('\"','').split(',')
		#print (peice)
		#将一条记录保存为Record对象
		r_temp = Record(float(peice[0])*0.001,float(peice[1]),float(peice[2]),\
	      	          float(peice[3]),float(peice[7]))
		#print(r_temp.soc)
		record_list.append(r_temp)
	#print(record_list)

	order_temp.record_list = record_list

	return order_temp

def fileAnalysis (lines):
	'''
	param: all lines in one raw data file

	return: a list of all order objects within one file
	'''
	start_line = 7

	order_set = []
	
	while (start_line <= len(lines)):
		order_line = lines[start_line-1]

		order_temp = orderPaser(order_line)
		if order_temp == None:
			start_line += 1
			continue

		#print('stubId: ' + order_temp.stubId)
		#print('order_record_ammount: ' + str(len(order_temp.record_list)))

		order_set.append(order_temp)
		start_line += 1

	return order_set

def powerCalculator(order):
	'''
	calculate power used in one order
	param: one order object

	no return

	'''


	#order内的所有record按时间排序，从早到晚
	record_list = sorted(order.record_list, key = lambda x:x.time, reverse=False)

	#通过record计算每小时用电量
	#初始化
	start_record = record_list[0]
	start_timeStamp = record_list[0].time
	start_time = int(time.strftime("%Y%m%d%H"\
	                               ,time.localtime(record_list[0].time)))
	pre_record = record_list[0]
	power_rest = 0

	#记录小时用电
	power_dict = {}
	#print(start_time)

	#循环所有记录
	for record in record_list:
	    current_timeStamp = record.time
	    current_time = int(time.strftime("%Y%m%d%H"\
	                               ,time.localtime(record.time)))
	    if pre_record.time == current_timeStamp:   #第一个记录忽略
	        continue
	    if current_timeStamp == record_list[-1].time: #最后一个记录
	        power_hourly = record.power_used - start_record.power_used + power_rest
	        if start_time in power_dict:
	            power_dict[start_time] += power_hourly
	        else:
	            power_dict[start_time] = power_hourly
	    if current_time > start_time : #仅小时对比,充电跨过小时零点
	        cross_timeStamp = int(time.mktime(time.strptime(str(current_time), "%Y%m%d%H")))
	        factor = (cross_timeStamp - pre_record.time) / (current_timeStamp - pre_record.time)
	        power_hourly = pre_record.power_used - start_record.power_used \
	                    + factor * (record.power_used - pre_record.power_used)\
	                    + power_rest
	        power_rest = (1-factor) * (record.power_used - pre_record.power_used)
	        
	        if start_time in power_dict:
	            power_dict[start_time] += power_hourly
	        else:
	            power_dict[start_time] = power_hourly
	            
	        start_record = record
	        start_time = current_time #小时
	        pre_record = record
	    else:                   #一个小时以内，继续
	        pre_record = record
	        continue
	order.power_dict = power_dict

def loadOrderFromRawData(raw_data_dir = './data/raw_data/', USING_RAW_DATA_CACHE = True):
	
	#delete '#' below if reload the raw data
	#USING_RAW_DATA_CACHE = False
	order_set = []

	#获取数据文件夹所有文件名
	file_name = get_file_name(raw_data_dir)
	
	if USING_RAW_DATA_CACHE:
		#读取Cache
		raw_data_cache_path = './data/raw_data_cache.pkl'
		order_set = read_cache(raw_data_cache_path)
		print("Raw data cache read successfully")
	else:
		#读取每个文件，并转换为Record&Order对象
		for each_name in file_name:
			fo = open(raw_data_dir+each_name, "r")
			print ('READING: ', fo.name)
			lines = fo.readlines()
			fo.close()
			print ('FINISHED reading: ', fo.name)
			each_file_set = fileAnalysis(lines)	
			
			#添加到总订单集合
			order_set.extend(each_file_set)
			print ('Orders containing: ', len(each_file_set))

	for order in order_set:
		powerCalculator(order)

	return order_set

def addAllPower(order_set):
	#合并所有order，生成order_set的每小时用电量
	all_power_dict = {}
	for order in order_set:
		for key in order.power_dict:
			if key in all_power_dict:
				all_power_dict[key] += order.power_dict[key]
			else:
				all_power_dict[key] = order.power_dict[key]
	return all_power_dict

def dict2list(dic):
    ''' 将字典转化为列表 '''
    keys = dic.keys()
    vals = dic.values()
    lst = [(key, val) for key, val in zip(keys, vals)]
    return lst

def extractHourlyPower(series_cache_path = './data/series_cache.pkl', USING_CACHE = True):
	'''
	return pandas series eg: 2017040110 499.8
	'''
	if USING_CACHE:
		series_cache = read_cache(series_cache_path)
		print("Series cache read successfully")
		return series_cache
	order_set = loadOrderFromRawData()

	#合并所有order，生成order_set的每小时用电量
	all_power_dict = {}
	for order in order_set:
		for key in order.power_dict:
			if key in all_power_dict:
				all_power_dict[key] += order.power_dict[key]
			else:
				all_power_dict[key] = order.power_dict[key]

	sorted_power_list = sorted(dict2list(all_power_dict), key=lambda x:x[0], reverse=False)
	sorted_power_list = [i for i in sorted_power_list if i[0]<2017042500]
	series = Series(*zip(*((b,a) for a,b in sorted_power_list)))

	return series


if __name__ == '__main__':
	
	MAKE_SERIES_CACHE = False
	MAKE_RAW_DATA_CACHE = False

	if MAKE_RAW_DATA_CACHE:
		order_set = loadOrderFromRawData(raw_data_dir = './data/raw_data/', USING_RAW_DATA_CACHE = False)
		raw_data_cache_path = './data/raw_data_cache.pkl'
		save_cache(order_set, raw_data_cache_path)

	series = Series()

	if MAKE_SERIES_CACHE:
		series = extractHourlyPower(USING_CACHE = False)
		print('saving series cache')
		save_cache(series, './data/series_cache.pkl')
	else:
		series = extractHourlyPower(USING_CACHE = True)

	print(series)
