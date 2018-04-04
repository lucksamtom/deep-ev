#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
        