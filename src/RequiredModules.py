# -*- coding: utf-8 -*-

working_dir_tianhe = '/HOME/zju_qhs_1/BIGDATA/GraphicalSolutionArterialRoad/'
working_dir_local = '/home/qhs/Qhs_Files/Program/Python/GraphicalSolutionArterialRoad/codes/'

#import xml.etree.cElementTree as et
from lxml import etree
from datetime import datetime,timedelta
import json
import networkx as nx

#from anytree import Node, RenderTree
import traceback

#分析过程中常用的类
#from RequiredModules import *
import builtins
import itertools
import pickle

import time

import os
import sys
from imp import reload
reload(sys)
sys.path.append('/BIGDATA/zju_qhs_1/GraphicalSolutionArterialRoad/codes')
sys.path.append('/BIGDATA/zju_qhs_1')
import copy

#gmm
from sklearn import mixture
import sklearn.datasets
from scipy import interpolate
from scipy.stats import multivariate_normal

#support vector regression
from sklearn.svm import SVR

import random
import numpy as np
import math
import pprint
from scipy.interpolate import interp1d
#integral
import scipy.integrate as integrate


import pandas as pd

#gaussian process regression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel


#from graphviz import Digraph

##matplotlib
import matplotlib.pyplot as plt
#from matplotlib.patches import Circle, Wedge
#from matplotlib.patches import Polygon as matplotlibPolygon
#import matplotlib.pyplot as plt

#from matplotlib import cm
#from matplotlib import rcParams
#rcParams['text.usetex']=False
#rcParams['text.latex.unicode']=False

import shapely
from shapely.geometry import MultiLineString
from shapely.geometry import LineString
from shapely.geometry import Polygon
#numpy的保存变量函数numpy.savez(file, *args, **kwds)用到的
#from tempfile import TemporaryFile

#核密度估计方法
#from statsmodels.nonparametric.kde import KDEUnivariate
import statsmodels.api as sm
from scipy import stats

#monte carlo integration.
#from skmonaco import mcquad

#python linked list lib
from llist import dllist, dllistnode
from llist import sllist, sllistnode

import threading,time
import queue
import multiprocessing
#
#import intervaltree
import pyinter#python interval class


#from bisect import bisect_left

#used for interp to get the points
#from bpf4 import bpf 


from Demand import RouteDemand
from RGP import RGP#red-green class
from LaneCapacity import LaneCapacity
from LaneFD import FD

#deal with the discrete constraints. 
from constraint import *




def ReconstructQueueProfileFromN_backup(self, l, spillovers, fd = FD):
	"""
	self is a RouteDemand class.
	re-construct the queue profile from self.raw_N and self.N
	len(self.raw_ts)==len(self.raw_N)
	len(self.ts)==len(self.N)
	-------------------------------------
	Inout: l
		the link length, used to check whether there is a spillover. 
	Input:	fd
		fundamental diagram, the used parameters are fd.kj, fd.vf. 
		They are used to calculate the wave speed.
	Inout: spillovers
		list, spillovers = [(yku1, ykv1), (yku2,ykv2)...]
		each element is (yku,ykv), the onset and ending moment of a spillover. 
	---------------------------------------
	Steps:
		- for each spillover interval yku ykv
			- stepwise compute the stopping wave and starting wave, until they intersect
				- if the stoping wave bigger than l, self spillover happens
				- 
	
	"""
	
	#returned value.
	#	res['xy'][idx_rgp] = 2 dimentional array,
	#		first row is moments and 2nd row is locaiton
	res =  {}
	res['xy'] = {}
	res['startingwave'] = {}
	
	#Set two RouteDemand class to faciliate the following operations.
	#dmd1 is the raw demand, and dmd2 is the spillover influenced spillover.
	dmd1 = RouteDemand(T = 1000)
	dmd1.ts = copy.deepcopy(self.raw_ts)
	dmd1.N = copy.deepcopy(self.raw_N)
	dmd2 = RouteDemand(T = 1000)
	dmd2.ts = copy.deepcopy(self.ts)
	dmd2.N = copy.deepcopy(self.N)
	
	
	#Common moments for raw_N and new N
	#	after this operation, the dmd1.ts == dmd2.ts
	#	and include all ykus and ykvs
	common_moments = set(dmd1.ts + dmd2.ts)
	dmd1.InsertMoment(min(common_moments))
	dmd1.InsertMoment(max(common_moments))
	dmd2.InsertMoment(min(common_moments))
	dmd2.InsertMoment(max(common_moments))
	#	get the common moments, and make the directional moments set the same
	#		note that ykus_ykvs include all the moments.
	ykus_ykvs = [i[0] for i in spillovers]+[i[1] for i in spillovers]
	common_moments = set(dmd1.ts + dmd2.ts + ykus_ykvs)
	diff_moments1 = common_moments.difference(set(dmd1.ts))
	diff_moments2 = common_moments.difference(set(dmd2.ts))
	for t in diff_moments1:
		dmd1.InsertMoment(t)
	for t in diff_moments2:
		dmd2.InsertMoment(t)
	
	#debug
	#return dmd1,dmd2
	
	#for each spillover interval, compute the stopping wave 
	#	and startingwave.
	for idx_spillover, spillover in enumerate(spillovers):
		yku, ykv = spillover
		print(yku,ykv)
		#out of the time horizon, terminate.
		if yku > dmd1.ts[-1]:
			break
		#index of yku amd ykv
		idx_yku1  	= np.where(np.array(dmd1.ts)==yku)[0][0]
		idx_ykv1 	= np.where(np.array(dmd1.ts)==ykv)[0][0]
		idx_yku2  	= np.where(np.array(dmd2.ts)==yku)[0][0]
		idx_ykv2 	= np.where(np.array(dmd2.ts)==ykv)[0][0]
		
		#the initial moment and location of xy and startingwave
		xy_t = [yku];xy_x = [0]
		startingwave_t = [ykv]; startingwave_x = [0]
		
		#moment wise compute the xy and starting wave
		#1, get the potential xy and starting wave
		#2, determine they intersect within l or not, 
		#	if intersect within l,
		#		reformulate the raw_N
		#	if not intersec, then a self spillover
		#		reformulate the raw_N
		#1 get the potential xy
		for i in range(idx_yku1+1, len(dmd1.ts)):
			#if dmd1.N[i] == dmd1.N[i-1]
			if dmd1.N[i] == dmd1.N[i-1]:
				xy_x.append(xy_x[-1])
				xy_t.append(xy_t[-1]+dmd1.ts[i]-dmd1.ts[i-1])
				continue
			
			#stopping wave speed, in m/s, positive
			#	flowrate is in veh/hour, density is in veh/km
			flowrate = ((dmd1.N[i]-dmd1.N[i-1])/(dmd1.ts[i]-dmd1.ts[i-1]))*3600.0
			density = flowrate/fd.vf
			stoppingwavespeed = (-flowrate/(density - fd.kj))/3.6
			
			#queue increment
			#	(dmd1.N[i]-dmd1.N[i-1])/(fd.kj/1000.0) 
			#		is the queue increment
			xy_x.append(xy_x[-1]+ (dmd1.N[i]-dmd1.N[i-1])/(fd.kj/1000.0))
			xy_t.append(xy_t[-1]+(dmd1.N[i]-dmd1.N[i-1])/(fd.kj/1000.0)/(stoppingwavespeed))
			
		#1 get the potantial startingwave
		for i in range(idx_ykv2+1, len(dmd2.ts)):
			#if dmd1.N[i] == dmd1.N[i-1]
			if dmd2.N[i] == dmd2.N[i-1]:
				startingwave_x.append(startingwave_x[-1])
				startingwave_t.append(startingwave_t[-1]+dmd2.ts[i]-dmd2.ts[i-1])
				continue
			
			#startingwave speed, in m/s, positive
			#	flowrate is in veh/hour, density is in veh/km
			flowrate = ((dmd2.N[i]-dmd2.N[i-1])/(dmd2.ts[i]-dmd2.ts[i-1]))*3600.0
			density = flowrate/fd.vf
			startingwave = (-flowrate/(density - fd.kj))/3.6
			
			#startingwave increment
			#	(dmd1.N[i]-dmd1.N[i-1])/(fd.kj/1000.0) 
			#		is the queue increment
			startingwave_x.append(startingwave_x[-1]+ (dmd2.N[i]-dmd2.N[i-1])/(fd.kj/1000.0))
			startingwave_t.append(startingwave_t[-1] + (dmd2.N[i]-dmd2.N[i-1])/(fd.kj/1000.0)/(startingwave))
		
		#determine whether xy and startingwave_x and  intersect or not
		#	(startingwave_t,startingwave_x)
		#	(xy_t, xy_x)
		
		
		new_waves = TwoLinesIntersection_ReconstructQueueProfileFromN(xy_t,xy_x, startingwave_t, startingwave_x, l = l)
		res['xy'][idx_spillover] = new_waves['xy']
		res['startingwave'][idx_spillover] = new_waves['startingwave']
		
	return res





def my_pickle_open(filename, readmode="r"):
        
        return pickle.load( open( filename, readmode ) )




def my_pd_to_rst_table(data_pd, filename = 'temp.txt'):
        """
        print a pandas table to restructuredtext and write it to the file given by arg filename. 
        
        my_pd_to_rst_table(data_pd, filename = 'temp.txt')
        --------------------------------------------------------
        Input: data_pd
        Input: filename
        
        """
        
        f = open(filename,'w')
        from tabulate import tabulate
        f.write(tabulate(data_pd, tablefmt="rst", headers = data_pd.columns))
        f.close()

def my_json_load(filename):
        """
        
        """
        return json.load(open(filename,'r'))




