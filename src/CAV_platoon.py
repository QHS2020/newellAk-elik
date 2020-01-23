# -*- coding: utf-8 -*-

from .RequiredModules import *
import timeit

#基本module
from .Demand import RouteDemand
from .RGP import RGP#red-green class
from .LaneCapacity import LaneCapacity
from .LaneFD import FD
from .ArterialRoad import ARD, Lane
import MCMC_DecisionVariables as MD
import scipy.spatial.distance as scipydistance

#used temporary, used in pso solution, confine the solution feasible. 
#   for instance, green duration should be in [gmin,gmax] 
tmp_signal = MD.SignalParameters()

#minimal stable headway, in sec
#H['NULL] means there is no lead vehicle
#   mode A is HV, mode B is CAV platoon, and C mode is single CAV
H = {'A':{'A':2.0,'B':2.0,'C':2.0},\
    'B':{'A':2.0,'B':1.0,'C':1.0},\
    'C':{'A':2.0,'B':1.0,'C':1.0},\
    'NULL':{'A':2.0,'B':2.0,'C':2.0}}

interval_mean_es = {}
arrival_mode_proba_es = {}
batch_arrival_es = {}
#########################East
#Left 
interval_mean_es['east_left'] = 25.0
arrival_mode_proba_es['east_left'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['east_left'] = {'A':1,'B':1,'C':2}
#Through, east_through
interval_mean_es['east_through'] = 25.0
arrival_mode_proba_es['east_through'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['east_through'] = {'A':1,'B':1,'C':2}
#########################west
#Left, west_left
interval_mean_es['west_left'] = 25.0
arrival_mode_proba_es['west_left'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['west_left'] = {'A':1,'B':1,'C':2}
#Through, west_through
interval_mean_es['west_through'] = 25.0
arrival_mode_proba_es['west_through'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['west_through'] = {'A':1,'B':1,'C':2}
#########################south
#Left, south_left
interval_mean_es['south_left'] = 25.0
arrival_mode_proba_es['south_left'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['south_left'] = {'A':1,'B':1,'C':2}
#Through, south_through
interval_mean_es['south_through'] = 25.0
arrival_mode_proba_es['south_through'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['south_through'] = {'A':1,'B':1,'C':2}
#########################north
#Left, north_left
interval_mean_es['north_left'] = 25.0
arrival_mode_proba_es['north_left'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['north_left'] = {'A':1,'B':1,'C':2}
#Through, north_through
interval_mean_es['north_through'] = 25.0
arrival_mode_proba_es['north_through'] = {'A':.7,'B':.2,'C':.1}
batch_arrival_es['north_through'] = {'A':1,'B':1,'C':2}

#parameters used in the lognormal distribion in headway
sigmas= [0.4128436510190671,
 0.1345446113092352,
 0.10683827147282604,
 0.19924111286803076,
 0.18390767211160008,
 0.0782317644882514,
 0.08813221376574254,
 0.10592931614200946,
 0.03949109303866507,
 0.0746485883543178,
 0.05837536210115604,
 0.0]
mus = [1.1871851824566433,
 1.016862600937631,
 0.933320451838609,
 0.8563486592583159,
 0.9349145477534682,
 0.878104649789996,
 0.8388155628601542,
 0.7717588802776599,
 0.8510828112780652,
 0.8433392600538451,
 0.6231375049891121,
 0.7419373447293773]

# Taguchi Orthogonal Array, 8 columns
TaguchiArray = np.array([[1,1,1,1,1,1,1,1],
[1,1,1,1,2,2,2,2],
[1,1,1,1,3,3,3,3],
[1,2,2,2,1,1,1,2],
[1,2,2,2,2,2,2,3],
[1,2,2,2,3,3,3,1],
[1,3,3,3,1,1,1,3],
[1,3,3,3,2,2,2,1],
[1,3,3,3,3,3,3,2],
[2,1,2,3,1,2,3,1],
[2,1,2,3,2,3,1,2],
[2,1,2,3,3,1,2,3],
[2,2,3,1,1,2,3,2],
[2,2,3,1,2,3,1,3],
[2,2,3,1,3,1,2,1],
[2,3,1,2,1,2,3,3],
[2,3,1,2,2,3,1,1],
[2,3,1,2,3,1,2,2],
[3,1,3,2,1,3,2,1],
[3,1,3,2,2,1,3,2],
[3,1,3,2,3,2,1,3],
[3,2,1,3,1,3,2,2],
[3,2,1,3,2,1,3,3],
[3,2,1,3,3,2,1,1],
[3,3,2,1,1,3,2,3],
[3,3,2,1,2,1,3,1],
[3,3,2,1,3,2,1,2]])

# Taguchi Orthogonal Array, full columns
TaguchiArray_full = 200.0*np.array([[1,1,1,1,1,1,1,1,1,1,1,1,1],
[1,1,1,1,2,2,2,2,2,2,2,2,2],
[1,1,1,1,3,3,3,3,3,3,3,3,3],
[1,2,2,2,1,1,1,2,2,2,3,3,3],
[1,2,2,2,2,2,2,3,3,3,1,1,1],
[1,2,2,2,3,3,3,1,1,1,2,2,2],
[1,3,3,3,1,1,1,3,3,3,2,2,2],
[1,3,3,3,2,2,2,1,1,1,3,3,3],
[1,3,3,3,3,3,3,2,2,2,1,1,1],
[2,1,2,3,1,2,3,1,2,3,1,2,3],
[2,1,2,3,2,3,1,2,3,1,2,3,1],
[2,1,2,3,3,1,2,3,1,2,3,1,2],
[2,2,3,1,1,2,3,2,3,1,3,1,2],
[2,2,3,1,2,3,1,3,1,2,1,2,3],
[2,2,3,1,3,1,2,1,2,3,2,3,1],
[2,3,1,2,1,2,3,3,1,2,2,3,1],
[2,3,1,2,2,3,1,1,2,3,3,1,2],
[2,3,1,2,3,1,2,2,3,1,1,2,3],
[3,1,3,2,1,3,2,1,3,2,1,3,2],
[3,1,3,2,2,1,3,2,1,3,2,1,3],
[3,1,3,2,3,2,1,3,2,1,3,2,1],
[3,2,1,3,1,3,2,2,1,3,3,2,1],
[3,2,1,3,2,1,3,3,2,1,1,3,2],
[3,2,1,3,3,2,1,1,3,2,2,1,3],
[3,3,2,1,1,3,2,3,2,1,2,1,3],
[3,3,2,1,2,1,3,1,3,2,3,2,1],
[3,3,2,1,3,2,1,2,1,3,1,3,2]],dtype=float)

#the 12 movements that corresponding to the TaguchiArray_full array columns
#    note that the 4 inner movements are waiting for the flow input, hence not involved;
#    note that TaguchiArray_full have 13 columns, hence last columns is useless.
TaguchiArray_columns = ['0@west_through','0@west_left','0@south_left','0@north_through','0@south_through','0@north_left','1@east_through','1@south_left','1@east_left','1@north_through','1@south_through','1@north_left']

import scipy.optimize
def intervalmean_finder(batch_arrival = [1,1,1],H=H, q=200,proportation = [.1,.1,.8]):
    """
    find the interval mean (interval between neighboring two modes are assumed to be exponential distirbution). 
    NOTE that batch_arrival should not be changed, due to unknown reasons. 
    
    The flow rate is specified in input q.
    The proportation specify the ratio of each mode. 
    ------------------------------------------------
    output: intervalmean, a scalar. 
    """
    proportation = np.array(proportation, dtype=float)/sum(proportation)
    def flowrate(tao, batch_arrival = batch_arrival, H=H, q=q, proportation = proportation):
        """
        
        """
        q = q/3600.0
        return proportation[0]*2/(0*H['A']['A']+tao) + \
    proportation[1]*batch_arrival[1]/((batch_arrival[1]-1)*H['B']['B']+tao) + \
    proportation[2]/(0*H['C']['C']+tao) - q
    
    return scipy.optimize.brentq(flowrate,a=.0000001,b=100000,args = (batch_arrival, H, q, proportation))


#the probability of the three modes: A, B and C
#   mode A is HV, mode B is CAV platoon, and C mode is single CAV
proportations  = np.array([[.1,.1,.8],[.3,.3,.4],[.8,.1,.1]])
TaguchiArray_columns_0 = ['west_through','west_left','south_left','south_through','north_through','north_left','east_through','east_left']
def IntersectionDemand_from_TaguchiArray_single_intersection(TaguchiArray = 200*TaguchiArray,TaguchiArray_columns=TaguchiArray_columns_0, TaguchiArray_row_id = 0, proportation_id = 0, batch_arrival1 = {'A':1,'B':1,'C':1}):
    """
    
    """
    
    flowrates = {}
    movement_labels = []
    for i in range(len(TaguchiArray_columns)):
        column = TaguchiArray_columns[i]
        movement_labels.append(column)
        flowrates[column] = TaguchiArray[TaguchiArray_row_id,i]
    
    #find the 
    proportation = proportations[proportation_id]
    arrival_mode_proba = {'A':proportation[0],'B':proportation[1],'C':proportation[2]}
    #####################################################
    ##########generate the returnned value
    interval_mean_es = {}
    arrival_mode_proba_es = {}
    batch_arrival_es = {}
    #########################East
    #Left 
    label = 'east_left'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['east_left'] = arrival_mode_proba
    batch_arrival_es['east_left'] = batch_arrival1
    #Through, east_through
    label = 'east_through'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['east_through'] = arrival_mode_proba
    batch_arrival_es['east_through'] = batch_arrival1
    #########################west
    #Left, west_left
    label = 'west_left'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation)
    arrival_mode_proba_es['west_left'] = arrival_mode_proba
    batch_arrival_es['west_left'] = batch_arrival1
    #Through, west_through
    label = 'west_through'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['west_through'] = arrival_mode_proba
    batch_arrival_es['west_through'] = batch_arrival1
    #########################south
    #Left, south_left
    label = 'south_left'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['south_left'] = arrival_mode_proba
    batch_arrival_es['south_left'] = batch_arrival1
    #Through, south_through
    label = 'south_through'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['south_through'] = arrival_mode_proba
    batch_arrival_es['south_through'] = batch_arrival1
    #########################north
    #Left, north_left
    label = 'north_left'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['north_left'] = arrival_mode_proba
    batch_arrival_es['north_left'] = batch_arrival1
    #Through, north_through
    label = 'north_through'
    interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    arrival_mode_proba_es['north_through'] = arrival_mode_proba
    batch_arrival_es['north_through'] = batch_arrival1
    
    return interval_mean_es,arrival_mode_proba_es,batch_arrival_es



#the probability of the three modes: A, B and C
#   mode A is HV, mode B is CAV platoon, and C mode is single CAV
proportations  = np.array([[.1,.1,.8],[.3,.3,.4],[.8,.1,.1]])
def IntersectionDemand_from_TaguchiArray(TaguchiArray = TaguchiArray_full,TaguchiArray_columns=TaguchiArray_columns, intersection_id = 0, TaguchiArray_row_id = 0, proportation_id = 0):
    """
    
    ------------------------------------------------------
    input: TaguchiArray_row_id
        which row of TaguchiArray will be used.
    input: intersection_id
        which intersection will be  generated. 
    input: proportation_id
        which modes share proportations is used. 
        proportation = proportations[proportation_id] is a list. 
    """
    flowrates = {}
    movement_labels = []
    for i in range(len(TaguchiArray_columns)):
        column = TaguchiArray_columns[i]
        if int(column.split('@')[0])==intersection_id:
            movement_labels.append(column.split('@')[1])
            flowrates[column.split('@')[1]] = TaguchiArray[TaguchiArray_row_id,i]
    #find the 
    proportation = proportations[proportation_id]
    arrival_mode_proba = {'A':proportation[0],'B':proportation[1],'C':proportation[2]}
    batch_arrival1 = {'A':1,'B':1,'C':1}
    
    #####################################################
    ##########generate the returnned value
    interval_mean_es = {}
    arrival_mode_proba_es = {}
    batch_arrival_es = {}
    #########################East
    #Left 
    label = 'east_left'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['east_left'] = arrival_mode_proba
    batch_arrival_es['east_left'] = {'A':1,'B':1,'C':1}
    #Through, east_through
    label = 'east_through'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['east_through'] = arrival_mode_proba
    batch_arrival_es['east_through'] = {'A':1,'B':1,'C':1}
    #########################west
    #Left, west_left
    label = 'west_left'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['west_left'] = arrival_mode_proba
    batch_arrival_es['west_left'] = {'A':1,'B':1,'C':1}
    #Through, west_through
    label = 'west_through'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['west_through'] = arrival_mode_proba
    batch_arrival_es['west_through'] = {'A':1,'B':1,'C':1}
    #########################south
    #Left, south_left
    label = 'south_left'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['south_left'] = arrival_mode_proba
    batch_arrival_es['south_left'] = {'A':1,'B':1,'C':1}
    #Through, south_through
    label = 'south_through'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['south_through'] = arrival_mode_proba
    batch_arrival_es['south_through'] = {'A':1,'B':1,'C':1}
    #########################north
    #Left, north_left
    label = 'north_left'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['north_left'] = arrival_mode_proba
    batch_arrival_es['north_left'] = {'A':1,'B':1,'C':1}
    #Through, north_through
    label = 'north_through'
    if label in movement_labels:
        interval_mean_es[label] = intervalmean_finder(q=flowrates[label], proportation=proportation) 
    else:
        interval_mean_es[label] = 25
    arrival_mode_proba_es['north_through'] = arrival_mode_proba
    batch_arrival_es['north_through'] = {'A':1,'B':1,'C':1}
    
    return interval_mean_es,arrival_mode_proba_es,batch_arrival_es
    
    pass




def IntersectionDemand_from_proportation(proportation_id =  1,proportations=proportations, flowrate = 200,batch_arrival = [1,1,1]):
    """
    Considering the proportation of CAVs and CAV platoons. Return the demand of only one intersection. 
    Note that the parameters are the same for all approaches. 
    -------------------------------------
    input: proportation_id
        the index in the proportations
    input: flowrate_vehh
        unit is veh/h
        a float, the average flow rate. which can be formulated as
            q= p_A*2\(E(tao)+2*H['A']['A'])+p_B*2\(E(tao)+n_B*H['B']['B'])+p_C\(E(tao)+H['C']['C']).
        Note that the above specify the mean batch arrival of mode 'A' is 2 (and surely the mean of mode C is 1, because it is single)
    output: 
        interval_mean_es,arrival_mode_proba_es,batch_arrival_es
        
    """
    #find the 
    proportation = proportations[proportation_id]
    interval_mean = intervalmean_finder(q=flowrate, proportation=proportation)
    arrival_mode_proba = {'A':proportation[0],'B':proportation[1],'C':proportation[2]}
    batch_arrival1 = {'A':1,'B':1,'C':1}
    
    #####################################################
    ##########generate the returnned value
    interval_mean_es = {}
    arrival_mode_proba_es = {}
    batch_arrival_es = {}
    #########################East
    #Left 
    interval_mean_es['east_left'] = interval_mean
    arrival_mode_proba_es['east_left'] = arrival_mode_proba
    batch_arrival_es['east_left'] = {'A':1,'B':1,'C':1}
    #Through, east_through
    interval_mean_es['east_through'] = interval_mean
    arrival_mode_proba_es['east_through'] = arrival_mode_proba
    batch_arrival_es['east_through'] = {'A':1,'B':1,'C':1}
    #########################west
    #Left, west_left
    interval_mean_es['west_left'] = interval_mean
    arrival_mode_proba_es['west_left'] = arrival_mode_proba
    batch_arrival_es['west_left'] = {'A':1,'B':1,'C':1}
    #Through, west_through
    interval_mean_es['west_through'] = interval_mean
    arrival_mode_proba_es['west_through'] = arrival_mode_proba
    batch_arrival_es['west_through'] = {'A':1,'B':1,'C':1}
    #########################south
    #Left, south_left
    interval_mean_es['south_left'] = interval_mean
    arrival_mode_proba_es['south_left'] = arrival_mode_proba
    batch_arrival_es['south_left'] = {'A':1,'B':1,'C':1}
    #Through, south_through
    interval_mean_es['south_through'] = interval_mean
    arrival_mode_proba_es['south_through'] = arrival_mode_proba
    batch_arrival_es['south_through'] = {'A':1,'B':1,'C':1}
    #########################north
    #Left, north_left
    interval_mean_es['north_left'] = interval_mean
    arrival_mode_proba_es['north_left'] = arrival_mode_proba
    batch_arrival_es['north_left'] = {'A':1,'B':1,'C':1}
    #Through, north_through
    interval_mean_es['north_through'] = interval_mean
    arrival_mode_proba_es['north_through'] = arrival_mode_proba
    batch_arrival_es['north_through'] = {'A':1,'B':1,'C':1}
    
    return interval_mean_es,arrival_mode_proba_es,batch_arrival_es





def IntersectionDemand():
    """
    return the intersection demand parameters, include:
        - H, dict of dict
        - interval_mean_es, dict, keys are 'east_left'..., length is 8(=4*2);
        - arrival_mode_proba_es, dict, keys are 'east_left'..., length is 8(=4*2);
        - batch_arrival_es, dict, keys are 'east_left'..., length is 8(=4*2);
    """
    
    #minimal stable headway, in sec
    #H['NULL] means there is no lead vehicle
    H = {'A':{'A':2.0,'B':4.0,'C':2.0},\
        'B':{'A':2.0,'B':4.0,'C':2.0},\
        'C':{'A':2.0,'B':4.0,'C':2.0},\
        'NULL':{'A':2.0,'B':2.0,'C':2.0}}
    
    interval_mean_es = {}
    arrival_mode_proba_es = {}
    batch_arrival_es = {}
    #########################East
    #Left 
    interval_mean_es['east_left'] = 25.0
    arrival_mode_proba_es['east_left'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['east_left'] = {'A':1,'B':1,'C':2}
    #Through, east_through
    interval_mean_es['east_through'] = 25.0
    arrival_mode_proba_es['east_through'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['east_through'] = {'A':1,'B':1,'C':2}
    #########################west
    #Left, west_left
    interval_mean_es['west_left'] = 25.0
    arrival_mode_proba_es['west_left'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['west_left'] = {'A':1,'B':1,'C':2}
    #Through, west_through
    interval_mean_es['west_through'] = 25.0
    arrival_mode_proba_es['west_through'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['west_through'] = {'A':1,'B':1,'C':2}
    #########################south
    #Left, south_left
    interval_mean_es['south_left'] = 25.0
    arrival_mode_proba_es['south_left'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['south_left'] = {'A':1,'B':1,'C':2}
    #Through, south_through
    interval_mean_es['south_through'] = 25.0
    arrival_mode_proba_es['south_through'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['south_through'] = {'A':1,'B':1,'C':2}
    #########################north
    #Left, north_left
    interval_mean_es['north_left'] = 25.0
    arrival_mode_proba_es['north_left'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['north_left'] = {'A':1,'B':1,'C':2}
    #Through, north_through
    interval_mean_es['north_through'] = 25.0
    arrival_mode_proba_es['north_through'] = {'A':.7,'B':.2,'C':.1}
    batch_arrival_es['north_through'] = {'A':1,'B':1,'C':2}
    
    return H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es

import scipy
def normpdf(t = 0, loc  = .0, scale = 1.0):
    """
    normal distribution cumulative distribution
    """
    return scipy.stats.norm.pdf(t, loc = loc, scale = scale)
    
    pass

def normcdf(t = 0, loc  = .0, scale = 1.0):
    """
    normal distribution cumulative distribution
    """
    return scipy.stats.norm.cdf(t, loc = loc, scale = scale)
    
def truncatednorm_pdf(x, a = 40, b = 50, loc = .0, scale = 1.0):
    """
    the probability density function of truncated distribution 
    the lower and upper bound are specified in a and b 
    
    """
    if b<=a:raise ValueError('upper bound 'b' should be greater than lower bound')
    if x<a or x>b:return 0
    tmp_x = normcdf(b, loc = loc, scale = scale)-normcdf(a, loc = loc, scale = scale)
    
    if tmp_x==0:raise ValueError('tmp_x ==0, will return NaN')
    
    return normpdf(x,loc=loc,scale=scale)/tmp_x

def truncatednorm_cdf(x, a = 40, b = 50, loc = .0, scale = 1.0):
    """
    the probability density function of truncated distribution 
    the lower and upper bound are specified in a and b 
    
    """
    if b<=a:raise ValueError('upper bound 'b' should be greater than lower bound')
    if x<=a:return 0
    if x>=b:return 1
    tmp = normcdf(a, loc = loc, scale = scale)
    tmp1 = normcdf(x, loc=loc, scale = scale)-tmp
    tmp2 = normcdf(b, loc=loc, scale = scale)-tmp
    if tmp2==0:raise ValueError('tmp2 ==0, will return NaN')
    return tmp1/tmp2


def Get_cumulativeN(t, arrival_ts, N):
    """
    Find the cumualtive N at instance t, given the cumulative moment list arrival_ts and cumulative number N.
    note that len(arrival_ts) = len(N)
    --------------------------------------------------------
    input: arrival_ts N
        both are list. the same length
    """
    #
    if t<=arrival_ts[0]:return N[0]
    if t>=arrival_ts[-1]:return N[-1]
    idx =  np.where(arrival_ts==t)[0]
    if len(idx)>0:return N[idx[0]]
    idx0 =  np.where(arrival_ts>t)[0][0]
    idx1 =  np.where(arrival_ts<t)[0][-1]
    return N[idx0]+ (t-arrival_ts[idx0])*(N[idx1]-N[idx0])/(arrival_ts[idx1]-arrival_ts[idx0])

def VehiclesFrequency_from_N(arrival_ts, N, interval = 30):
    """
    get the arrival moment of the 1st vehice, 2nd vehicle...
    ---------------------------
    input: arrival_ts and N
        both are lists. the same length
    input: interval
        the time interval for counting vehicles, in seconds
    """
    bins = int((arrival_ts[-1]-arrival_ts[0])/interval)
    
    arrival_ts = np.array(arrival_ts, dtype = float)
    N = np.array(N, dtype = float)
    
    ts = np.linspace(arrival_ts[0], arrival_ts[-1], bins)
    frequences = []
    for t0,t1 in zip(ts[:-1], ts[1:]):
        #t0 and t1 are the onset and termination moment of the bin
        
        #find the N at t0
        n0 = Get_cumulativeN(t0,arrival_ts, N)
        n1 = Get_cumulativeN(t1,arrival_ts, N)
        
        frequences.append(n1-n0)
        
    return ts,frequences




def myhist(data, bins = 15,figsize= (8,4), ax = 'asf',alpha=.5, color = np.random.random((1,3))[0]):
    if isinstance(ax,str):
        fig,ax = plt.subplots(figsize= figsize)
    frequency, binsedge  = np.histogram(data,bins=bins)
    frequency= 1.0*frequency/len(data)/((max(data)-min(data))/bins)
    patched = []
    for i in range(len(frequency)):
        if frequency[i]==0:continue
        x = [binsedge[i],binsedge[i+1], binsedge[i+1],binsedge[i]]
        y = [0,0,frequency[i],frequency[i]]
        ax.fill(x,y,alpha=alpha,color = color)
    return ax
    
def myfrquencyplot(binsedge,frequency,color,ax = 'ssdf',figsize= (8,4),alpha=.7):
    #len(binsedge) = len(frequency) + 1
    #both binsedge and frequency are list. 
    if isinstance(ax,str):
        fig,ax = plt.subplots(figsize= figsize)
    for i in range(len(frequency)):
        if frequency[i]==0:continue
        x = [binsedge[i],binsedge[i+1], binsedge[i+1],binsedge[i]]
        y = [0,0,frequency[i],frequency[i]]
        ax.fill(x,y,alpha=alpha,color = color)
    return ax


def platoon_CAV_ratios(somesignal, interval_mean = 16, batch_arrival = {'A':3,'B':3,'C':3},arrival_mode_proba = {'A':.1,'B':.8,'C':.1}, lognormal=True,T_horizon=7200, interval = 30, l=500, scale=40 ,vmin = 5, vmax = 60, vmean = 25, H = H):
    """
    mode A is HV, mode B is CAV platoon, and C mode is single CAV
    -------------------------------------
    output: (bins_original, frequency_original, bins_later, frequency_later)
        All are list. len(bins_original)=len(frequency_original)+1
    input: somesignal
        rgp instance
    input: interval
        the intervals for counting the frequency
    """
    
    
    #the cumulative for overlap
    ForOverlap = MovementTraffic(H = H, interval_mean = interval_mean, arrival_mode_proba=arrival_mode_proba, batch_arrival = batch_arrival, T_horizon = T_horizon)
    #   intiialize to zero arrival. 
    ForOverlap.N = [0];ForOverlap.arrival_t = [0];ForOverlap.N_modes = [];
    
    #initialize and run traffic flow dynamics
    test = MovementTraffic(H = H, interval_mean = interval_mean, arrival_mode_proba=arrival_mode_proba, batch_arrival = batch_arrival, T_horizon = T_horizon)
    test.N_construct()
    test.lognormal = lognormal
    test.D_construct(some_rgp=somesignal)
    D0,departure_t0 = test.reorganize_D()
    #get the departure within the first green signal
    t0 = test.cycle_red_end_t[0]
    t1 = test.cycle_end_t[0]
    idx0 = test.find_idxx_greaterthan(array=departure_t0,threshold=t0)
    idx1 = test.find_idxx_smallerthan(array=departure_t0,threshold=t1)
    #departure_t[0] should be not 0
    departure_t = departure_t0[idx0:idx1+1]
    D = D0[idx0:idx1+1]
    N_modes = test.N_modes[D[0]-1:D[-1]-1]
    
    #couting the departure when without dispersion
    bins_original,frequency_original = VehiclesFrequency_from_N(departure_t,D, interval=interval)
    
    #split the departure
    #   HV_departure_t, HV_D, if the mode is 
    HV_departure_t = [.0]
    HV_modes = []
    CAV_departure_t = [.0]
    CAV_modes = []
    for i in range(len(N_modes)):
        if N_modes[i]=='A':
            HV_departure_t.append(departure_t[i])
            HV_modes.append(N_modes[i])
        else:
            CAV_departure_t.append(departure_t[i])
            CAV_modes.append(N_modes[i])
    
    HV_D = range(len(HV_departure_t))
    CAV_D = range(len(CAV_departure_t))
    
    if len(HV_D)==1:
        bins_later = [i+l/(vmax/3.6) for i in bins_original] 
        frequency_later = frequency_original
        return (bins_original,frequency_original, bins_later, frequency_later)
    
    #CAV overlap
    ForOverlap.overlap_N(N1 = CAV_D, arrival_t1=[i+l/(vmax/3.6) for i in CAV_departure_t], N_modes1=CAV_modes)
    
    #HVs dispersion
    #   len(HV_N1)=len(HV_arrival_t1)
    #   note that HV_N1 may not be integers
    HV_N1,HV_arrival_t1 = test.platoondispersion_truncatednormal1(l=l,N=HV_D,arrival_t= HV_departure_t,scale = scale ,vmin = vmin, vmax = vmax, vmean = vmean,returnedinterval= interval)
    
    ts = sorted(ForOverlap.arrival_t + list(HV_arrival_t1))
    Ns = []
    for t in ts:
        print 'len(HV_arrival_t1)=',len(HV_arrival_t1),'len(HV_departure_t)=',len(HV_departure_t),'len(ForOverlap.arrival_t)',len(ForOverlap.arrival_t)
        tmp_n1 = Get_cumulativeN(t, HV_arrival_t1, HV_N1)
        tmp_n2 = Get_cumulativeN(t, ForOverlap.arrival_t, ForOverlap.N)
        Ns.append(tmp_n1+ tmp_n2)
    bins_later,frequency_later = VehiclesFrequency_from_N(ts,Ns, interval=interval)
    
    return (bins_original,frequency_original, bins_later,frequency_later)


class newell():
    """
    neweell impletation. 
    """
    reactiontime = 1#reaction time of the HVs. 
    def __init__(self):
        """"""
        
        pass
    
    @classmethod
    def saturationheadway_theoretical(self, tao = .1, d= 7.5, vf=50.0):
        """
        calculate the theoretical headway, when the vehicles obey the Newell rules.  
        --------------------------------------------------
        input: tao, unit is sec
            the reaction time. For  human, it should be 1.3 s. For CAV, it should be .1 sec. 
        input: d, unit is meter
            minimal space. 7.5 m
        input: vf, unit is km/h.
            the freeflow speed. unit is km/h
        """
        #convert to the 
        vf1 = 1.0*vf/3.6
        return tao+d/vf1
        
        pass
    
    
    @classmethod
    def saturationheadways(self, somergp, trajectories, influencedbyleading, l = 400, smallbuffer = .5):
        """
        get the saturated headway for each cycle. 
        ----------------------------------
        input: trajectories, influencedbyleading
            trajectories[i] = (ts,xs). ts and xs are both list. 
            influencedbyleading[i]= True or False.
            
            
        input: influencedbyleading
        """
        #the key is the idx of cycle, start from 0.
        headways = {}
        
        #find the moment when depart from the stopline
        #   moments is a list of float
        moments = self.departing_modments(trajectories=trajectories, l = l, smallbuffer=smallbuffer)
        if len(influencedbyleading)!=len(moments):
            raise ValueError('the lenggh shoubd be the same, and len(influencedbyleading),len(moments) = ',len(influencedbyleading),len(moments))
            
        for t,influenced in zip(moments, influencedbyleading):
            idxx = somergp.idx_rgp(t)
            if not headways.has_key(idxx):
                if influenced:
                    headways[idxx] = [t]
            else:
                if influenced:
                    headways[idxx].append(t)
                else:
                    continue
                
        
        
        return headways
    
    @classmethod
    def trajectorys2MultiString(self,trajectories):
        """
        convert the trajectories to multistring.
        """
        formultistring = []
        for traj in trajectories:
            ts,xs = traj
            formultistring.append([(t,x) for t,x in zip(ts,xs)])
            pass
        return MultiLineString(formultistring)
    
    
    @classmethod
    def departing_modments(self, trajectories, l, smallbuffer = .5):
        """
        get the departureing moment of the vehicle.
        input: trajectories
            a list. trajectories[i] = (ts,xs). ts and xs are both list. 
        input: l
            the lane length. unit is meter
        input: smallbuffer 
            because the vehicle stop exactly at the stopline, we need to move it alittle further. 
        """
        tmin = trajectories[0][0][0]
        tmax = trajectories[-1][0][-1]
        ts = [tmin, tmax]
        xs = [l+smallbuffer , l+smallbuffer]
        redss = self.tsxs2shapelyline(ts,xs)
        
        shapelytrajectories =  self.trajectorys2MultiString(trajectories)
        points = list(redss.intersection(shapelytrajectories))
        
        #each p is shapely.point instance
        return [p.x for p in points]
        
    @classmethod
    def plot_trajectories(self, trajectories, N_modes, termini = 100):
        """
        
        """
        if len(trajectories)!=len(N_modes):
            raise ValueError("the length of the two input should be the same.")
        i=0
        fig,ax = plt.subplots()
        for tra,mode in zip(trajectories,N_modes):
            if i>=termini:break
            if mode=='A':
                linewidth = 1.5
                color = 'blue'
            else:
                linewidth = .2
                color = 'red'
            ax.plot(tra[0],tra[1],linewidth=linewidth, color = color)
            i=i+1
            
        return ax
    
    
    @classmethod
    def freeflow_tsxs(self,enter_t=1, vmax = 50, l = 500, delta_t = .1):
        """
        OK.
        NOTE that the buffer is not considered, i.e returned xs[-1]<=l. 
        --------------------------------
        input: enter_t
            the entering moment of the vehicle
        inout: vmax, l
            vmax is the speed limit, l is the lane length. 
            vmax unit is km/h, l unit is meter.
        input: delta_t
            is the time step when constructing the ts and xs
        output: ts,xs
            len(ts)=len(xs) and xs[-1]<=l, ts[0]=enter_t
        """
        t_end = 1.0*l/(vmax/3.6) + enter_t
        ts = np.arange(enter_t, t_end, delta_t)
        xs = [(t-enter_t)*vmax/3.6 for t in ts]
        
        return ts,xs
    
    @classmethod
    def construct(self, ):
        """
        
        """
        
        
        pass
    
    
    @classmethod
    def fusion_twolines(self, ts1, xs1, ts2,xs2):
        """
        combind the two lines when the two lines intersect. 
        If they does not intersect, return ts1,xs1
        ----------------------------------------
        input: ts1,xs1, ts2,xs2
            all are lists. len(ts1)=len(xs1), len(ts2)=len(xs2).
            The trajectoty
        output: combined_ts,combined_xs
            the resulting 
        """
        l1 = self.tsxs2shapelyline(ts1,xs1)
        l2 = self.tsxs2shapelyline(ts2,xs2)
        if l1.intersection(l2).is_empty:
            return ts1,xs1, False
            
            raise ValueError("the two lines shoule be intersect")
        
        #find the intersection point
        #   l1.intersection(l2).bounds=minx, miny, maxx, maxy
        #   minx is the moment, and miny is the location, 
        #       the entrance of road is location=0
        #       the stop line is location=road length. 
        minx,miny,_,_ =  l1.intersection(l2).bounds
        
        first_idxs = np.where(np.array(ts1)<minx)
        second_idxs = np.where(np.array(ts2)>minx)
        combined_ts = list(np.array(ts1)[first_idxs]) + list(np.array(ts2)[second_idxs])
        combined_xs = list(np.array(xs1)[first_idxs]) + list(np.array(xs2)[second_idxs])
        
        return combined_ts,combined_xs,True
    
    def construct_mixedflow(self, arrival_N, ts, N_modes,rgp, l = 400, vmax=40):
        """
        construc the trajectory when the arrival and signal are given. All the vehicles are 
        -----------------------------------------
        input: l
            the lane length. 
        input: vmax
            the maximum speed, unit is km/h.
        input: tao and dn
            the parameters in the newell model.
        input: somergp
            the singals. somergp.reds gives all the red durations using (e,s). 
        input: arrival_N, ts, N_modes
            len(arrival_N)=len(ts)=len(N_modes)+1, because ts[0]=0 and arrival_N[0] = 0.
            ts is a list containing the arrival moments of each vehicle. 
            The mode of each vehicle are given at N_modes. 
        
        output: trajectory
            
        --------------------------------------
        Methods:
            the cases for vehicles trajectory. 
                - first vehicles in the queue.
        Steps:
            first construct the `
        
        """
        
        
        
        pass
    
    @classmethod
    def reds2shapely(self,somergps, l =500):
        """
        OK.
        convert the red bar to shapely instance. 
        Check the signals intersect with some trajectory (it should be a LineString instance), using :
            res = LineString.intersection(MultiLineString).
        If res.is_empty is True, then there is no intersection point. Eitherwise, 
            res.bounds will give (minx, miny, maxx, maxy).
        --------------------------------------------
        input: l
            the lane length. unit is meters
        output MultiLineString
        """
        Alllines = []
        for s,e in zip(somergps.StartMomentsOfReds, somergps.EndMomentsOfReds):
            Alllines.append(((s,l),(e,l)))
            
            
        return MultiLineString(Alllines)
        
    @classmethod
    def tsxs2shapelyline(self,ts,xs):
        """
        convert the trajectory to shapey.LineString class
        """
        return LineString((t,x) for t,x in zip(ts,xs))
    
    @classmethod
    def decelaration_truncated(self, ts, xs, delta_tao, vmax=50.0):
        """
        when the vehicle decelerate, but before it decelerate to speed zero, the green begins. Then the time difference between the green start and the expected arrival moment (free flow speed) is defined as delta_tao
        ---------------------------------------
        input: vmax
            the speed limit, unit is km/h. 
        input: ts an xs
            len(ts)=len(xs), they are the deceleration profiles obtaied from self.HV_deceleration_trajectory().
            
            ts[0]=0, and xs[0]=0.
        output: tss, xss
            part of the ts and xs
        """
        ts = list(ts)
        xs = list(xs)
        
        if ts[-1]<=delta_tao:
            return ts,xs
            
        slope = [abs(x/(t-delta_tao)-vmax/3.6) for t,x in zip(ts[1:],xs[1:]) if t>delta_tao]
        ts_temp =  [t for t,x in zip(ts[1:],xs[1:]) if t>delta_tao]
        t = ts_temp[slope.index(min(slope))]
        idxx = ts.index(t)
        return ts[:idxx+1],xs[:idxx+1]
        
        
    @classmethod
    def HV_deceleration_trajectory(self, m=0.587, delta_t = .1, vmax = 40, a_ma = 2.69):
        """
        the deceleration profile of the vehicles. 
        Input parameters referr to self.HV_acceleration_trajectory()
        
        """
        #the parameter of r in acceleration equation. 
        r  = np.power((1.0+2.0*m),(2+1.0/m))/(4*m*m)
        t_a = vmax/(3.6*r*a_ma*(0.5-2.0/(m+2)+1.0/(2*m+2)))
        print t_a
        #moments for results
        ts = np.linspace(0, t_a, max(int(t_a/delta_t),10))
        ass  = [r*a_ma*t/t_a*(1-np.power(t/t_a, m))**2 for t in ts]
            
        vs = [vmax-3.6*r*a_ma*t_a*np.power(t/t_a,2)*(0.5-2*np.power(t/t_a,m)/(m+2)+np.power(t/t_a,2*m)/(2*m+2)) for t in ts]
        
        xs = [vmax*t/3.6 - r*a_ma*np.power(t_a,2)*np.power(t/t_a,3)*(1.0/6-2*np.power(t/t_a,m)/((m+2.0)*(m+3.0))+np.power(t/t_a, 2.0*m)/((2*m+2)*(2*m+3))) for t in ts]
        
        return ts,ass,vs,xs
    
        
        pass
    
    
    @classmethod
    def HV_acceleration_trajectory_initial_speed(self, m=0.587, delta_t = .1, vmax = 40, a_ma = 2.69, v_init = 0, v_final=40, minsamplesize = 10):
        """
        OK. 
        
        This function allows the acceleration and deceleration with non zero initial speed. 
        
        Logic: the time duration depend of the maximum acceleration and the intiial final speed. 
        
        return the trajectory of the vehicle that start from speed zero. 
        The acceration is polynomal function:
            a(t) = r a_m (theta)(1-\theta^m)^2.
        The a_m is maximam acceleration. Theta=t/t_a, where t_a is accelaration duration. 
        
        This function works for both acceleration and decelration. 
        ---------------------------------
        input: m
            the parameter is the model, which can be found at:
                Akçelik, Rahmi, and Mark Besley. "Acceleration and deceleration models." 23rd Conference of Australian Institutes of Transport Research (CAITR 2001), Monash University, Melbourne, Australia. Vol. 10. 2001.    
        input: a_ma
            the maximum acceleration. unit is m/s^2
        input: t_a
            the time for acceleration. unit is sec
        input: delta_t
            the time increment of the trajectory computation. unit is sec
        input: vmax
            the speed limit of the vehicle. veh/h
        input: v_init, unit is km/h 
            the initial speed of the vehicle. 
        input: v_finial
            the final speed of the vehicle. 
        input: minsamplesize
            when calculating the speed, the 
        output: 
            (ts, as,vs,xs). All are list. The length of the three shoule be the same. 
            ts is the list of moments. Begin from 0.
            ass is the list of accelerations
            vs is the list of velocity
            xs is the list of locations. 
            
            ts[0]=0, ass[0]=0, vs[0]=0, xs[0]=0
        -----------------------------------
        
        """
        if v_init == v_final:
            raise ValueError('Initial speed and final speed should be different ')
        
        v_init = v_init/3.6
        v_final = v_final/3.6
        accele_or_decele = 1 if v_final>v_init else -1
        
        #the parameter of r in acceleration equation. 
        r  = np.power((1.0+2.0*m),(2+1.0/m))/(4*m*m)
        
        #t_a is the time for acceleration or deceleration. 
        t_a = abs((v_final-v_init)/(3.6*r*a_ma*(0.5-2.0/(m+2)+1.0/(2*m+2))))
        
        #moments for results
        ts = np.linspace(0, t_a, max(minsamplesize, abs(int(t_a/delta_t))))
        ass  = [accele_or_decele*r*a_ma*t/t_a*(1-np.power(t/t_a, m))**2 for t in ts]
            
        vs = [v_init + accele_or_decele*3.6*r*a_ma*t_a*np.power(t/t_a,2)*(0.5-2*np.power(t/t_a,m)/(m+2)+np.power(t/t_a,2*m)/(2*m+2)) for t in ts]
        
        xs = [v_init*t +r*a_ma*np.power(t_a,2)*np.power(t/t_a,3)*(1.0/6-2*np.power(t/t_a,m)/((m+2.0)*(m+3.0))+np.power(t/t_a, 2.0*m)/((2*m+2)*(2*m+3))) for t in ts]
        
        return ts,ass,vs,xs
    
    
    @classmethod
    def HV_acceleration_trajectory(self, m=0.587, delta_t = .1, vmax = 40, a_ma = 2.69):
        """
        return the trajectory of the vehicle that start from speed zero. 
        The acceration is polynomal function:
            a(t) = r a_m (theta)(1-\theta^m)^2.
        The a_m is maximam acceleration. Theta=t/t_a, where t_a is accelaration duration. 
        
        This function works for both acceleration and decelration. 
        ---------------------------------
        input: m
            the parameter is the model, which can be found at:
                Akçelik, Rahmi, and Mark Besley. "Acceleration and deceleration models." 23rd Conference of Australian Institutes of Transport Research (CAITR 2001), Monash University, Melbourne, Australia. Vol. 10. 2001.    
        input: a_ma
            the maximum acceleration. unit is m/s^2
        input: t_a
            the time for acceleration. unit is sec
        input: delta_t
            the time increment of the trajectory computation. unit is sec
        input: vmax
            the speed limit of the vehicle. veh/h
        output: 
            (ts, as,vs,xs). All are list. The length of the three shoule be the same. 
            ts is the list of moments. Begin from 0.
            ass is the list of accelerations
            vs is the list of velocity
            xs is the list of locations. 
            
            ts[0]=0, ass[0]=0, vs[0]=0, xs[0]=0
        -----------------------------------
        
        """
        #the parameter of r in acceleration equation. 
        r  = np.power((1.0+2.0*m),(2+1.0/m))/(4*m*m)
        t_a = vmax/(3.6*r*a_ma*(0.5-2.0/(m+2)+1.0/(2*m+2)))
        
        #moments for results
        ts = np.linspace(0, t_a, int(t_a/delta_t))
        ass  = [r*a_ma*t/t_a*(1-np.power(t/t_a, m))**2 for t in ts]
            
        vs = [3.6*r*a_ma*t_a*np.power(t/t_a,2)*(0.5-2*np.power(t/t_a,m)/(m+2)+np.power(t/t_a,2*m)/(2*m+2)) for t in ts]
        
        xs = [r*a_ma*np.power(t_a,2)*np.power(t/t_a,3)*(1.0/6-2*np.power(t/t_a,m)/((m+2.0)*(m+3.0))+np.power(t/t_a, 2.0*m)/((2*m+2)*(2*m+3))) for t in ts]
        
        return ts,ass,vs,xs
    
    @classmethod
    def trajectory_firstvehicleinqueue(self, traversemoment, somergp, ts, xs, reactiontime = 1.0, vmax = 40.0, l = 400, a_md = 2.9, a_ma = 2.9, delta_t = .1, spacebuffer = 400):
        """
        The trajectory of the first vehicle in the queue. 
        ts,xs is the trajectory, and it should intersect with the red horizon. 
        -----------------------------
        input: a_md and a_ma
            the maximum deceleration and acceleration. 
        input: somergp
            somergp.reds gives all the red durations. 
        input: traversemoment
            the moment when the trajectory traverse the red duration. 
        input: spacebuffer
            the vehicle still traverse the stopline, and travels extra length of spacebuffer.
        """
        #convert to m/s.
        vmax1 = vmax/3.6
        
        #get the expected moment when travering the stop line. 
        expected_idx =  somergp.idx_rgp(traversemoment)
        delta_tao = somergp.EndMomentsOfReds[expected_idx] - traversemoment
        #   the moment of the red end
        redend = somergp.EndMomentsOfReds[expected_idx]
        #   the moment of start to accelerate.
        accelerationstart = redend + reactiontime
        
        #the truncated deleleration cueve, which is stored in ts_dece1,xs_dece1
        #   NOTE that xs_dece1[0]=0
        ts_dece,as_dece,vs_dece,xs_dece = self.HV_deceleration_trajectory(vmax  = vmax, a_ma=a_md)
        ts_dece1, xs_dece1 = self.decelaration_truncated(ts= ts_dece,xs=xs_dece,delta_tao = delta_tao)
        
        #combine the trajectory with deceleration profile.
        idxx =  np.argmin(np.abs(np.array(xs)-(l-xs_dece1[-1])))
        firstpart_ts = list( ts[:idxx+1])
        firstpart_xs = list( xs[:idxx+1])
        #   ts_dece1[1:] because ts_dece1[0]=0
        secondpart_ts = [t+firstpart_ts[-1] for t in ts_dece1[1:]]
        secondpart_xs = [min(x+firstpart_xs[-1],l) for x in xs_dece1[1:]]
        
        #third part: stoping
        thirdpart_ts = list(np.arange(secondpart_ts[-1], accelerationstart,delta_t))
        thirdpart_xs = [l for i in thirdpart_ts]
        
        #fourth part: acceleration. 
        #   ts_acce0[0]=0 and xs_acce0[0]=0
        ts_acce,_,_,xs_acce = self.HV_acceleration_trajectory( m=0.587, delta_t = delta_t, vmax = vmax, a_ma = a_ma)
        fourth_ts = [t+thirdpart_ts[-1] for t in ts_acce[1:]]
        fourth_xs = [x+l for x in xs_acce[1:]]
        
        #fifth
        fifth_ts = []
        fifth_xs = []
        if fourth_xs[-1]<l+spacebuffer:
            fifth_ts = [fourth_ts[-1]+delta_t]
            fifth_xs = [fourth_xs[-1]+delta_t*vmax1]
            while fifth_xs[-1]<l+spacebuffer:
                tmp = fifth_ts[-1]+delta_t
                fifth_ts.append(tmp)
                
                tmp = fifth_xs[-1]+delta_t*vmax1
                fifth_xs.append(tmp)
        
        return firstpart_ts+secondpart_ts+thirdpart_ts+fourth_ts+fifth_ts,firstpart_xs+secondpart_xs+thirdpart_xs+fourth_xs+fifth_xs
    
    @classmethod
    def traversemoment_green_or_red(self,ts,xs, somergpshapely):
        """
        check whether the trajectory traverse the stopline in red or green
        If red, return True, if green return False. 
        Note that, at the end of green, it can still traverse
        ------------------------------------------
        input:ts xs
            len(ts)=len(xs)
        input: somergpshapely
            the shapely instance of all the red signals.
        output:
            bool,time
            when trajectory can pass the stopline during green, bool is False
            else return the moment when the trajectory intersect with red
        """
        if self.tsxs2shapelyline(ts,xs).intersection(somergpshapely).is_empty:
            return False,None
        else:
            minx,_,_,_ = self.tsxs2shapelyline(ts,xs).intersection(somergpshapely).bounds
            return True,minx
    
    
    @classmethod
    def shiftcurve(self, ts, xs, reactiontime = 1.3, dn = 7.5, xmin = 0, xmax= 600, vmax = 50, delta_t = .1):
        """
        According to newell theory, the trajectory is translated. But the xs (spatial coordinate) of reuslting trajectory will be within [xmin,xmax].
        
        The lane entrance is x=0, and the stopline is located at x=l, l is road length. 
        
        xmax should be l+buffer. 
        
        For 
        ------------------------------------------
        input: reactiontime = 1.3, dn = 7.5
            the parameters in the newell models. reactiontime unit is sec, and dn unit is meter. 
        input: xmin and xmax
            the minimal and maximal location of the shifted trajectory. 
        input: vmax and delta_t
            the speed limit, unit is km/h.
            delta_t is the time interval, unit it sec
            
            The two are used in the extnsion. 
        output: ts1,xs1
        """
        if reactiontime<=0 or dn<=0:
            print reactiontime,dn
            raise ValueError('Dsdfsadf')
        
        vmax1 = vmax/3.6
        
        #first translate the x and t
        xs1 = []
        ts1 = []
        for t,x in zip(ts,xs):
            if x-dn<xmin:continue
            xs1.append(x-dn)
            ts1.append(t+reactiontime)
        
        #confine the location within [xmin, xmax]
        newxs = [xs1[-1]]
        newts = [ts1[-1]]
        while True:
            if newxs[-1]>xmax:break
            newts.append(newts[-1]+delta_t)
            newxs.append(newxs[-1]+delta_t*vmax1)
        
        return ts1+newts[1:],xs1+newxs[1:]
    
    @classmethod
    def tao_lognormal(self, tao, lognormal = False, mode = 'A',sigma=.01):
        """
        mode = 'A' is HV.
        """
        if lognormal and mode=='A':
            return np.random.lognormal(mean =tao,sigma = sigma) 
        else:
            return np.exp(tao)
    
    @classmethod
    def trajectories(self, arrival_t, N, N_modes, somergp, l = 400, vmax = 50.0, reactiontime = {'B':.3,'A':1.3, 'C':.3}, dn = {'B':7.5,'A':7.5, 'C':7.5}, spacebuffer = 400, delta_t = .1, a_md = 2.9, a_ma = 2.9, lognormal = False,sigma=.1):
        """
        
        
        -------------------------------------------
        input: reactiontime_HV,reactiontime_CAV,dn_HV,dn_CAV
            parameters in the newell model, HV is human vehicle, 
        input: arrival_t, N, N_modes
            all are list. 
            len(arrival_t)=len(N)=len(N_modes)+1, because arrival_t[0]=0 and N[0]=0.
            
            N_modes is a list of modes, each mode is indicated using 'A', 'B' or 'C'. 'A' is for HV, and 'B' 'C' is for CAV.
        input:  l and vmax
            the lane length and the speed limit.
        input: somergp
            the signal. somergp.reds gives all the red durations. 
        input: spacebuffer
            the space that the vehicle still travel after it pass the stopline.
        input: a_md = 2.9, a_ma = 2.9
            the maximum deceleration and acceleration.
        input: lognormal and sigma
            whether the tao in newell theory obey the lognormal.
        ----------------------------------------
        Steps:
            - for each arrival_t
        """
        #store the results. 
        #   trajectories[i] = (ts,xs)
        trajectories = []
        influencedbyleading = []#if this vehicle is influenced by the leading, then true. len(influencedbyleading)=len(trajectories)
        
        #convert the reds to shapely
        shapelyreds = self.reds2shapely(somergps = somergp, l = l)
        
        #first vehicle
        #   get the freeflow ts and xs
        ts0,xs0 = self.freeflow_tsxs(enter_t=arrival_t[1], vmax = vmax, l = l+spacebuffer, delta_t = delta_t)
        #   check wiether freeflow intersect with the red duration. 
        bolll,traversemoment = self.traversemoment_green_or_red(ts0,xs0, shapelyreds)
        if bolll:
            ts, xs = self.trajectory_firstvehicleinqueue(traversemoment=traversemoment, somergp=somergp, ts=ts0, xs=xs0, reactiontime = reactiontime[N_modes[0]], vmax = vmax, l = l, a_md = a_md, a_ma = a_ma, delta_t =delta_t, spacebuffer = spacebuffer)
            trajectories.append((ts,xs))
            influencedbyleading.append(True)
        else:
            trajectories.append((ts0,xs0))
            influencedbyleading.append(True)
        
        #not so many vehicles
        if len(arrival_t)<=2:
            return trajectories
        
        #for the remaining vehicles
        #   steps:  - construc the freeflow trajectory
        #           - merge the shifted trajectory and freeflowtrajectory
        #           - check whether the vehicle can pass the stopline. 
        for enter_t0,mode in zip(arrival_t[2:],N_modes[1:]):
            #get the shifted trajectory of the former vehicle
            #   which is stored in shifted_ts,shifted_xs
            ts0 = trajectories[-1][0];xs0 = trajectories[-1][1]
            #   note that self.tao_lognormal(tao=reactiontime[mode], lognormal = lognormal)
            shifted_ts,shifted_xs = self.shiftcurve(ts=ts0, xs=xs0, reactiontime = self.tao_lognormal(tao = np.log(reactiontime[mode]), lognormal = lognormal, mode=mode,sigma=sigma), dn = dn[mode], xmin = 0, xmax= l+spacebuffer, vmax = vmax, delta_t = delta_t)
            #freeflow trajectory
            enter_t = max(shifted_ts[0], enter_t0)
            ts0,xs0 = self.freeflow_tsxs(enter_t=enter_t, vmax = vmax, l = l+spacebuffer, delta_t = delta_t)
            
            #check the intersection and merge it. 
            ts1, xs1, influenced = self.fusion_twolines(ts1=ts0, xs1=xs0, ts2=shifted_ts, xs2=shifted_xs)
            
            #check wiether intersect with the red duration. 
            bolll,traversemoment = self.traversemoment_green_or_red(ts1,xs1, shapelyreds)
            if bolll:
                ts, xs = self.trajectory_firstvehicleinqueue(traversemoment=traversemoment, somergp=somergp, ts=ts1, xs=xs1, reactiontime = reactiontime[mode], vmax = vmax, l = l, a_md = a_md, a_ma = a_ma, delta_t =delta_t, spacebuffer = spacebuffer)
                trajectories.append((ts,xs))
                influencedbyleading.append(influenced)
            else:
                trajectories.append((ts1,xs1))
                influencedbyleading.append(influenced)
            
        return trajectories,influencedbyleading
    
    
    @classmethod
    def trajectory_HV(self, enter_t, greenstart, reactiontime = 1.0, vmax = 40.0, l = 400):
        """
        the trajectory of single vehicle. This vehicle should be the first vehicle in the queue, because the trajectory of the following vehicles can be obtained from this leading vehicle. 
        
        input: enter_t
            the entering moment, float
        input: greenstart
            the startmoment of the green signal
        input: reactiontime
            the reaction time of the vehicle
        input: vmax
            the speed limit of the vehicle
        input: l
            the road length. 
        ----------------------------------
        
        """
        #convert to m/s.
        vmax1 = vmax/3.6
        if enter_t+l/vmax1>=greenstart:
            raise ValueError('The green start should greater than the ...')
        
        
        pass
    
    def trajectory_HVs(self, arrival_N,  ts, somergp, l = 400, vmax=40, tao = 1, dn = 7.5):
        """
        construc the trajectory when the arrival and signal are given. All the vehicles are 
        --------------------------------------
        input: l
            the lane length. 
        input: vmax
            the maximum speed, unit is km/h.
        input: tao and dn
            the parameters in the newell model.
        input: somergp
            the singals. somergp.reds gives all the red durations using (e,s). 
        input: arrival_N, ts
            ts is a list containing the arrival moments of each vehicle. 
            The 
        
        output: trajectory
            
        --------------------------------------
        Methods:
            the cases for vehicles trajectory. 
                - first vehicles in the queue.
        Steps:
            first construct the `
        
        """
        #convert to m/s
        vmax1  = vmax/3.6
        
        pass
    
    
    pass

class IntersectionTraffic:
    """
    The class of intersection traffic flow, including the signals, the movement of each turning movements.
    """
    #class attributes, keys inlcude 8 types (4 approaches and each approach have two turning directions, i.e. left and through)
    #   'west_left', 'west_through', ....
    #   Movements['west_left'] is a MovementTraffic instance.
    
    
   

    
    def get_pi(self, pi='delay'):
        """
        
        ------------------------------------
        input: pi
            a str. The delay  of the 
        -----------------------------------
        output: 
            if pi=='delay', then return a tuple, i.e. 
                (totaldelay, averagedelay), both unit is minutes.
        
        """
        if pi=='delay':
            totaldelay = .0
            totalN = 0
            for m in self.movements.keys():
                totalN = totalN + self.movements[m].N[-1]
                totaldelay = totaldelay + self.movements[m].get_pi(pi=pi)
            return totaldelay,1.0*totaldelay/totalN
        
        
    def RandomSampleCyclePlan(self, signal = MD.SignalParameters(),plan_id = 0):
        """
        random sample the phases. The output 'phases_greens' is a pd.Series. 
        signal.To_rgps(phases_greens) will conver the phases duratings to RGPs. signal.To_rgps(phases_greens) will generate a movement:RGP dict. 
        
        --------------------------------------
        input: signal
            a MD.SignalParameters instance. 
            signal.phases gives a list of all the phases.
        input: plan_id
            a scalar, range(49). 
        output: phases_greens
            a pd.Series data. The index is 0,1,2...7. Each correspond to a specific phase. For instance if the index is 0, it means the phase signal.phases[0].
            signal.phases[0] is {'west_left', 'west_through'}
            
            the value is scalar, i.e. green duration. 
            
        """
        
        ps,ds,plan_id = signal.mcts_D1_sample_randomly(plan_id=plan_id)
        phases_greens = {}
        for p,d in zip(ps,ds):phases_greens[signal.phases.index(p)]=d
        self.IntersectionSignal.current = pd.Series(phases_greens)
        
        return pd.Series(phases_greens)
    


    def SignalInput(self, GreenPhases,allredmatrix,TempSignal = MD.SignalParameters(),losstimetyle = 'static',intergreentype = 'static',lognormal = False):
        """
        input the signals of the intersection. and get the:
            - self.movements[direction].departure_t
            - self.movements[direction].D
            - self.movements[direction].queue_t
            - self.movements[direction].Q
            - self.movements[direction].delays
        -----------------------------------------
        input: GreenPhases
            pd.Series data. The green duration of each phase. 
        input: allredmatrix
            pandas.DataFrame. The columns and the 
        input: losstimetyle
            a string, the start loss time of the green signal. 
            either "static" or "dynamic"
        input: intergreentype
            a string, the intergreen time. it equals yellow plus the red clearance. 
            either "static" or "dynamic"
        input: lognormal
            
        ----------------------------------------------
        
        """
        #first convert the GreenPhases to rgps
        #   intersection_rgps is a dict. 
        intersection_rgps = TempSignal.To_rgps(GreenPhases)
        for direction in intersection_rgps.keys():
            self.movements[direction].rgp = intersection_rgps[direction]
            if losstimetyle=='static' and intergreentype=='static':
                self.movements[direction].D_construct_static_intergreen(some_rgp = intersection_rgps[direction],yellow= 3.0,redclearance = 1.0, speedlimit  =40,reactiontime = 1.0, startloss = .5,acceleration = 3.0, lognormal = False)
    
    
    def __init__(self, H = H, \
    batch_arrival_es = batch_arrival_es, \
    interval_mean_es = interval_mean_es, \
    arrival_mode_proba_es = arrival_mode_proba_es, \
    intersection_signal = MD.SignalParameters(), \
    T_horizon = 3600.0, \
    modes = ['A','B','C'],yellowtype = 'static',lognormal=False, fixedstartloss=False):
        """
        
        ------------------------------------------
        input: H
            the minimal headway between different modes
        input: batch_arrival_es
            dict. 
            batch_arrival_es['east_left']={'A':1,'B':1,'C':2}
            The mean of the probability.
        input: interval_mean_es
            a dict. 
            interval_mean_es['east_left']= a scalar. The mean of the interval between modes
        input: arrival_mode_proba_es
            a dict. 
            arrival_mode_proba_es['east_left']= {'A':.7,'B':.2,'C':.1}. The relative probability of the arrival modes. 
        input: intersection_signal
            MD.SignalParameters() instance. 
            intersection_signal.plans and intersection_signal.phases. 
        input: T_horizon = 3600.0
            the time horizon of the signal and demands. 
        ---------------------------------------------------
        output:
            no output. but the following are constructed:
                - self.movements
                - self.movements[direction].arrivals_raw
                - self.movements[direction].N
                - self.movements[direction].arrival_t
        """
        self.movements  = {}
        #IntersectionSignal is a MD.SignalParameters instance
        #   IntersectionSignal.current is a pd.Series instance.
        self.IntersectionSignal = MD.SignalParameters()
        for direction in batch_arrival_es.keys():
            #initialize the movement traffic. 
            self.movements[direction]  = MovementTraffic(H= H ,\
            interval_mean = interval_mean_es[direction], \
            arrival_mode_proba = arrival_mode_proba_es[direction], \
            batch_arrival = batch_arrival_es[direction], \
            T_horizon = T_horizon, \
            modes = modes, label =direction , yellowtype=yellowtype,lognormal=lognormal, fixedstartloss=fixedstartloss)
            
            #construct the arrival cumulative number. 
            self.movements[direction].N_construct()
        
class MovementTraffic:
    """
    the class for the movement arrival.
    -   self.N self.arrival_t is the arrival, both are list, the same length
    -   self.N_mode is the list of modes, lengh is len(self.N)-1, because the self.N begins from zero.
    -   self.D self.departure_t is the departure,both are list and the same length
    -   self.Q self.queue_t is the queue length, both are list and the same length
    ---------------------------------------
    """
    
    def platoondispersion_truncatednormal1(self, N, arrival_t, l=100, vmin = 10, vmax = 50, vmean = 20, loc = 20, scale = 1.0, returnedinterval = 3):
        """
        platoon dispersion model for truncted normal speed distribution.
        The pseed obeys the truncated normal distribution with lower and upper specified in vmin and vmax.
        
        The method works as follows:
            - first get the cumulative curve at destination section
            - then linear interploate the moment for each vehicle number n (within [1,N[-1])
        --------------------------------------------------
        input: N, arrival_t
            all are lists. len(N)=len(arrival_t)=len(N_modes)+1
            !!!!N starts from 0, hence len(N_modes)=len(N)-1
        input: l
            the road length for the platoon dispersion.
        input: vmin and vmax vmean
            the minimal speed and maximal speed and mean speed. The unit is km/.
            
        input: loc and scale
            the parameters in the normal distribution. 
            loc is the mean and scale is the std of normal distribution.
            Actually the loc is not used. 
        input: returnedinterval
            the interval for the returned value arrival_t1
        -----------------------------------------------------------
        output: N1, arrival_t1
            the cumulative arrival at downstream point. 
        -----------------------------------------------------------
        Steps:
            - get the time horizon, in t0 and t1
            - for each intersection moment
        """
        #convert the speed to m/s
        vmin1 = vmin/3.6
        vmax1 = vmax/3.6
        vmean1 = vmean/3.6
        #minimal travel time and maximal travel time
        tao_min = l/vmax1
        tao_max = l/vmin1
        tao_mean = l/vmean1
        
        #find the downstream arrival time lower and upper bound
        #   arrival_t[1:] because the N begins from 0, i.e. N[0]=0
        arrival_times_lower = np.array(arrival_t[1:],dtype=float) + tao_min
        arrival_times_upper = np.array(arrival_t[1:],dtype=float) + tao_max
        arrival_times_mean  = np.array(arrival_t[1:],dtype=float) + tao_mean
        
        #initilize the returned value, i.e. N1 and arrival_t1.
        interval = min(returnedinterval, (arrival_times_upper[-1]-arrival_times_lower[0])/5.0)
        arrival_t1 = np.arange(arrival_times_lower[0],arrival_times_upper[-1]+2*interval,interval)
        N1 = 0*arrival_t1
        
        #for each vehicle, the three moment are lower limit, upper limit and the average moment. 
        for at0,atmean,at1 in zip(arrival_times_lower,arrival_times_mean,arrival_times_upper):
            idxx_at1 = np.where(arrival_t1>=at1)[0][0]
            N1[idxx_at1:] = N1[idxx_at1:] + 1
            
            idxx_es = np.where((arrival_t1<at1) & (arrival_t1>at0))[0]
            for idxx in idxx_es:
                t = arrival_t1[idxx]
                #compute the cdf of truncated normal
                cdf  =  truncatednorm_cdf(t, a = at0, b = at1, loc = atmean, scale = scale)
                #assign the cdf
                #   find the idxx that arrival_t1[idxx]>=t
                N1[idxx] = N1[idxx] + cdf
        
        return N1,arrival_t1
    
    def platoondispersion_truncatednormal(self, N, arrival_t, l=100, vmin = 10, vmax = 50, vmean = 20, loc = 20, scale = 1.0):
        """
        platoon dispersion model for truncted normal speed distribution.
        The pseed obeys the truncated normal distribution with lower and upper specified in vmin and vmax.
        
        The method works as follows:
            - first get the cumulative curve at destination section
            - then linear interploate the moment for each vehicle number n (within [1,N[-1])
        --------------------------------------------------
        input: N, arrival_t
            all are lists. len(N)=len(arrival_t)=len(N_modes)+1
            !!!!N starts from 0, hence len(N_modes)=len(N)-1
        input: l
            the road length for the platoon dispersion.
        input: vmin and vmax vmean
            the minimal speed and maximal speed and mean speed. The unit is km/.
            
        input: loc and scale
            the parameters in the normal distribution. 
            loc is the mean and scale is the std of normal distribution.
            Actually the loc is not used. 
        -----------------------------------------------------------
        output: N1, arrival_t1
            the cumulative arrival at downstream point. 
        -----------------------------------------------------------
        Steps:
            - get the time horizon, in t0 and t1
            - for each intersection moment
        """
        #convert the speed to m/s
        vmin1 = vmin/3.6
        vmax1 = vmax/3.6
        vmean1 = vmean/3.6
        #minimal travel time and maximal travel time, and average travel time.
        tao_min = l/vmax1
        tao_max = l/vmin1
        tao_mean = l/vmean1
        
        #find the downstream arrival time lower and upper bound
        #   arrival_t[1:] because the N begins from 0, i.e. N[0]=0
        arrival_times_lower = np.array(arrival_t[1:]) + tao_min
        arrival_times_upper = np.array(arrival_t[1:]) + tao_max
        arrival_times_mean  = np.array(arrival_t[1:]) + tao_mean
        
        #sorted switching moment
        #   moments is a list, and without repeating
        
        moments = sorted(np.unique(np.concatenate([arrival_times_lower,arrival_times_upper])))
        
        arrival_t1 = [.0]
        N1 = [0]
        for t0,t1 in zip(moments[:-1], moments[1:]):
            
            #will store the cumulative N at moment t1 at downstream spot.
            cumulativeN1 = N1[-1]
            
            #find all the vehicles that may arrive during [t0,t1]
            #   the results is stored in arrival_vehicles_indexes, a list
            #   arrival_vehicles_indexes[i]=5, means that vehicle id 5 contriute to the arrival of [t0,t1]
            tmp0 = np.where(arrival_times_lower<=t0)[0]+1
            tmp1 = np.where(arrival_times_upper>=t1)[0]+1
            arrival_vehicles_indexes = np.intersect1d(tmp0, tmp1)
            #print 'len(arrival_vehicles_indexes)=',len(arrival_vehicles_indexes)
            for idd in arrival_vehicles_indexes:
                #compute the cdf(t0)-cdf(t0)
                #   first get the lower, mean and uppter of arrival moment
                lowermoment = arrival_times_lower[idd-1]
                uppermoment = arrival_times_upper[idd-1]
                mean_moment = arrival_times_mean[idd-1]
                #print 'lowermoment=',lowermoment,'  uppermoment=',uppermoment,'  mean_moment=',mean_moment
                
                cumulativeN1  =  cumulativeN1 + truncatednorm_cdf(t1, a = lowermoment, b = uppermoment, loc = mean_moment, scale = scale) - truncatednorm_cdf(t0, a = lowermoment, b = uppermoment, loc = mean_moment, scale = scale)
            
            arrival_t1.append(t1)
            N1.append(cumulativeN1)
        
        return N1,arrival_t1
    
    
    @classmethod
    def GenerateArrival(cls,H ,interval_mean, arrival_mode_proba, batch_arrival, T_horizon = 3600.0, modes = ['A','B','C']):
        """
        generate arival (include arrival_t, N, N_modes), given H,interval_mean, arrival_mode_proba, batch_arrival
        ---------------------------------------
        Input: H
            the minimal headway, or stable head of each mode, It is a dict
            H['A']['B'] is a scalar, which means the headway when the leading and following vehicle are mode A and mode B respectively. 
        Input: interval_mean
            a scalar, unit is second, the mean of the interval between neighboring arrival modes. It is exponential distributed. 
        Input: arrival_mode_proba
            the probability of different arrival modes. It is a dict, the keys are 'A', 'B' and 'C'. arrival_mode_proba['A'] is a scalar. 
        Input: batch_arrival
            the arrival vehicles number of single mode. It is possion distributed. It is a dict. batch_arrival['A'] is the mean vehicles number arriving. 
        Input: the demand horizon of the arrivals; 
        --------------------------------------------------
        output: N, arrival_t, N_modes
            all are lists. 
        """
        T=0.0
        sum_pro = 1.0*sum([arrival_mode_proba[i] for i in ['A','B','C']])
        arrivals_raw = []
        while T<T_horizon:
            #Get three variables: interval, mode, vehicle_n
            #   determin the interval;
            interval = np.random.exponential(interval_mean)
            #   select the mode, mode is a str, i.e. 'A';
            mode = np.random.choice(['A','B','C'],p=[arrival_mode_proba[i]/sum_pro for i in ['A','B','C']])
            #   random determine the vehicles number
            if mode=='C':
                vehicle_n = 1
            else:
                #note the "1+"
                vehicle_n = 1 + np.random.poisson(lam = batch_arrival[mode])
                
            arrival_raw = {'interval':interval,'mode':mode,'batch_n':vehicle_n}
            arrivals_raw.append(arrival_raw)
            
            T = T  + interval + (vehicle_n-1)*H[mode][mode]
    
    
        arrival_t = [.0]#in sec
        N = [0]# it is int
        N_modes = []
        #for each arrival batch, each batch corresponds to 
        #   a specific mode. 
        for idxx in range(len(arrivals_raw)):
            #
            interval  = arrivals_raw[idxx]['interval']
            mode  = arrivals_raw[idxx]['mode']
            batch_n  = arrivals_raw[idxx]['batch_n']
            
            #the head vehicle
            new_t = arrival_t[-1] + interval
            new_n = N[-1] + 1
            arrival_t.append(new_t)
            N.append(new_n)
            N_modes.append(mode)
            
            #the remaining vehicle
            for i in range(batch_n-1):
                new_t = arrival_t[-1] + H[mode][mode]
                new_n = 1 + N[-1]
                arrival_t.append(new_t)
                N.append(new_n)
                N_modes.append(mode)
        return arrival_t, N, N_modes
    
    def printcheck(self, t = []):
        """
        to see whether the vehicles conservation holds.
        The conservation is checked by:
            - self.D(t)+self.Q(t)=self.N(t)
        If not hold then the conservation is not hold.
        ------------------------------------
        input: t
            if it is [], then simply return the 
        """
        if isinstance(t,list):
            print 'self.D[-1]=',self.D[-1],' self.Q[-1]=',self.Q[-1], 'self.N(self.queue_t[-1])=',self.get_N(self.queue_t[-1]), 'self.N(cycle_end_t)=',self.get_N(self.cycle_end_t[self.current_idx])
        else:
            
            idxxq = self.find_idxx_smallerthan(array=self.queue_t, threshold = t, equal = True)
            print 'self.D(t)=',self.get_D(t),' self.Q(t)=',self.Q[idxxq], 'self.N(self.queue_t[-1])=',self.get_N(t)
            pass
    
    @classmethod
    def get_flowrate(cls, H, interval_mean, arrival_mode_proba, batch_arrival):
        """
        compute the average flow rate.
        --------------------------------------------------
        Input: H
            the minimal headway, or stable head of each mode, It is a dict
            H['A']['B'] is a scalar, which means the headway when the leading and following vehicle are mode A and mode B respectively. 
        Input: interval_mean
            a scalar, unit is second, the mean of the interval between neighboring arrival modes. It is exponential distributed. 
        Input: arrival_mode_proba
            the probability of different arrival modes. It is a dict, the keys are 'A', 'B' and 'C'. arrival_mode_proba['A'] is a scalar. 
        Input: batch_arrival
            the arrival vehicles number of single mode. It is possion distributed. It is a dict. batch_arrival['A'] is the mean vehicles number arriving. 
        """
        
        batch_mean = batch_arrival['A']+1
        q_A = 3600.0*batch_mean/(interval_mean+(batch_mean-1.0)*H['A']['A'])
        
        batch_mean = batch_arrival['B']+1
        q_B = 3600.0*batch_mean/(interval_mean+(batch_mean-1.0)*H['B']['B'])
        
        batch_mean = batch_arrival['C']+1
        q_C = 3600.0*batch_mean/(interval_mean+(batch_mean-1.0)*H['C']['C'])
        q_C1 = 3600.0*1/(interval_mean+1*H['C']['C'])
        
        return arrival_mode_proba['A']*q_A+arrival_mode_proba['B']*q_B+arrival_mode_proba['C']*q_C
    
    def get_pi(self,pi='delay'):
        """
        Compute the performances index. 
        ---------------------------------
        input: pi
            a str. Either 'delay', 'capacity'
            if it is delay, then the returned is TOTAL delay
        """
        if pi=='delay':
            delays = self.get_delay_using_queue()
            return sum(delays.values())
            
    def __init__(self, H ,interval_mean, arrival_mode_proba, batch_arrival, T_horizon = 3600.0, modes = ['A','B','C'],label = 'Temp',redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False, yellowtype = 'static'):
        """
        THe initialize of the movement arrival. The arrival is a mixture of the HVs and the CAVs. 
        The method will 
        -----------------------------------
        Input: H
            the minimal headway, or stable head of each mode, It is a dict
            H['A']['B'] is a scalar, which means the headway when the leading and following vehicle are mode A and mode B respectively. 
        Input: interval_mean
            a scalar, unit is second, the mean of the interval between neighboring arrival modes. It is exponential distributed. 
        Input: arrival_mode_proba
            the probability of different arrival modes. It is a dict, the keys are 'A', 'B' and 'C'. arrival_mode_proba['A'] is a scalar. 
        Input: batch_arrival
            the arrival vehicles number of single mode. It is possion distributed. It is a dict. batch_arrival['A'] is the mean vehicles number arriving. 
        Input: the demand horizon of the arrivals; 
        -------------------------------------------
        Output: 
            no output, but self.arrivals_raw is generated
        """
        self.redclearance = 0
        self.speedlimit = 0
        self.reactiontime = 0
        self.acceleration = 0
        self.startloss = 0
        self.sigmas = 0
        self.mus = 0
        self.fixedstartloss = False
        self.lognormal = False
        
        #yellow time that does not change 
        self.staticyellow = 0
        #both for dynamic and static. 
        #   the lengh is the same as self.rgp.reds.
        #   for the last cycle, the yellow time is zero.
        self.yellowtimes = []
        self.yellowtype = 'static'
        #the idx of the current signal
        self.current_idx = 0
        self.movementlabel = ''
        #movement signal. 
        #   rgp.reds will return all the red signals. 
        self.rgp = RGP()
        #arrival of single mode
        #   arrival of single mode
        self.arrival_raw = {'interval':.0,'mode':'A','batch_n':7}
        #   the whole arrival is a list. Each element is the format as 
        #   arrival_raw
        self.arrivals_raw = []
        self.H = {}#H['A']['B'] is the headway when leading is A and following vehicle is B
        #arrival and departure 
        #   self.arrival_t and self.N is a step function and is right-consinuous
        #       arrival N and departure both begins from 0.0
        #       arrival_t and departure_t also begins from 0.0
        self.arrival_t = []#len(arrival_t)==len(N)
        self.N = []#arrival N, increase one by one exactly. 
        #the mode of each vehicle, any one in 'A','B', 'C';
        #   the length is len(slef.N)-1, because first element in slef.N is zero.
        self.N_modes = []
        self.departure_t = []#len(departure_t)==len(D)
        self.D = []#departure cumulative N, increase one by one exactly. 
        #queue length
        self.queue_t = []
        self.Q = []
        #delays[cycle_idx] = [d1,d2,d3....]
        #   each scalar is the delay a specific vehicle within this cycle.
        self.delays = {}
        
        #the start moment and ending moment of each cycle. 
        #   note that, since the yellow moment is not included in the 
        #   self.rgp, hence using self.rgp there is error. 
        self.cycle_start_t = []
        self.cycle_end_t = []
        self.cycle_green_end_t = []#the yellow is not included.
        self.cycle_red_end_t = []#the moment when green of each cycle ends. 
        
        #############################################
        #############################################
        self.yellowtype = yellowtype
        self.redclearance = redclearance
        self.speedlimit = speedlimit
        self.reactiontime = reactiontime
        self.acceleration = acceleration
        self.startloss = startloss
        self.sigmas = sigmas
        self.mus = mus
        self.fixedstartloss = False
        self.lognormal = False
        #convert the speed to m/s instead of km/h.
        speed = 1.0*speedlimit/3.6
        self.staticyellow = reactiontime + speed/(2.0*acceleration)
        
        self.label = label
        
        T = .0
        sum_pro = 1.0*sum([arrival_mode_proba[i] for i in ['A','B','C']])
        self.H = copy.deepcopy(H)
        self.arrivals_raw = []
        while T<T_horizon:
            #Get three variables: interval, mode, vehicle_n
            #   determin the interval;
            interval = np.random.exponential(interval_mean)
            #   select the mode, mode is a str, i.e. 'A';
            mode = np.random.choice(['A','B','C'],p=[arrival_mode_proba[i]/sum_pro for i in ['A','B','C']])
            #   random determine the vehicles number
            if mode=='C':
                vehicle_n = 1
            else:
                #note the "1+"
                vehicle_n = 1 + np.random.poisson(lam = batch_arrival[mode])
                
            arrivals_raw = {'interval':interval,'mode':mode,'batch_n':vehicle_n}
            self.arrivals_raw.append(arrivals_raw)
            
            T = T  + interval + (vehicle_n-1)*H[mode][mode]
    
    
    
    
    def RandomSampleCyclePlan(self, signal = MD.SignalParameters(),plan_id = 0):
        """
        random sample the phases. The output 'phases_greens' is a pd.Series. 
        signal.To_rgps(phases_greens) will conver the phases duratings to RGPs. signal.To_rgps(phases_greens) will generate a movement:RGP dict. 
        
        --------------------------------------
        input: signal
            a MD.SignalParameters instance. 
            signal.phases gives a list of all the phases.
        input: plan_id
            a scalar, range(49). 
        output: phases_greens
            a pd.Series data. The index is 0,1,2...7. Each correspond to a specific phase. For instance if the index is 0, it means the phase signal.phases[0].
            signal.phases[0] is {'west_left', 'west_through'}
            
            the value is scalar, i.e. green duration. 
            
        """
        
        ps,ds,plan_id = signal.mcts_D1_sample_randomly(plan_id=plan_id)
        phases_greens = {}
        for p,d in zip(ps,ds):phases_greens[signal.phases.index(p)]=d
        
        return pd.Series(phases_greens)
    
    
    def DynamicYelow(self, y_end, departure_ts,modes, speedlimit = 50.0, reactiontime = 1.0, deceleration = 3.05):
        """
        Considering the dynamic nature of the yellow time. The last several vehicle may not pass the stop line. 
        
        The yellow time considering the reactiontime is:
            reactiontime + speed/(2.0*acceleration)
        The yellow time don't consider the reaction time is:
            speed/(2.0*acceleration)
        Therefore, the last human driven vehicle within the spatial interval [speed^2/(2.0*acceleration), reactiontime*speed + speed^2/(2.0*acceleration)] determine the dynamic yellow time 
        
        NOTE that if there is dynamic yellow time, the vehicle that corresponding to the exactly ending will pass the stop line either.  
        ----------------------------------------------
        Input: departure_ts
            the expected departure moments. absolute moments
        inpus: modes
            the modes of the vehicles corresponding to each departure t in departure_ts, hence len(departure_ts)==len(modes)
        input: y_end
            the end moment of the yellow signal. Absolute moment. 
            It is calculated based on the green_end + 
        input: speedlimit, km/h; 
        input: reactiontime, second
        input: deceleration, unit is m/s^2
        ---------------------------------------------
        output: new_departure_ts
            a list. The length may equal the departure_ts. The last several elements are discarded due to the fact that they cannot 
        output: delta_yellow
                the original yellow is calculated based on:
                   yellowtime = reactiontime + speed/(2.0*acceleration)
                and the new yellow is based on the vehicle location  within [speed^2/(2.0*acceleration), reactiontime*speed+speed^2/(2.0*acceleration)] that is closest to the stop line. Suppose the location is x \in [speed^2/(2.0*acceleration), reactiontime*speed+speed^2/(2.0*acceleration)]. 
                And the **delta_yellow** is calculated as:
                reactiontime+speed/(2.0*acceleration)-x/speed.
                
                If there is no human driven vehicle that within such space interval, then 
        """
        #if the current cycle is the last signal, it means that ther is no following red. Hence all the vehicles can pass
        if y_end==np.inf:
            return departure_ts,0.0
        
        if y_end<departure_ts[-1]:
            raise ValueError('yellow end is smaller than departure t')
        
        #from the last element, 
        #   range(len(departure_ts)-1,-1,-1)=[len()-1, len()-2,....0]
        for idxx in range(len(departure_ts)-1,-1,-1):
            #mode==A means it is a human driven vehicle.
            if y_end-departure_ts[idxx]<=reactiontime and modes[idxx]=='A':
                return copy.deepcopy(departure_ts[:idxx+1]),y_end-departure_ts[idxx]
            elif y_end-departure_ts[idxx]>reactiontime:
                return copy.deepcopy(departure_ts[:idxx+1]),reactiontime
        return departure_ts,0
    
    def MonteCarlo_departure(self, mus, sigmas, G,iterations = 1000):
        """
        Simulate the depature vehicle number gives the greed duration, G, if the headway obeys a lognormal distribution. 
        ----------------------------------------------
        Input: mus and sigmas (len(mus)==len(sigmas))
            Both are lists. 
            the parameters of the lognormal distribution. For vehicle location n which satisfy n>len(mus), the mu and sigam will adopt mus[-1] and sigmas[-1].
        Input: G
            the green duration. 
        Input: the iterations by the Monte Carlo.s
        -----------------------------------------------
        Output: ns
            a list, constaing the simulation results of expected departures.
        Output: pdf
            a mixture gmm fit of the simulation results. 
        """
        NS = []
        for iter in range(iterations):
            exiting_ts = []
            while True:
                if sum(exiting_ts)>G:break
                mu = mus[len(exiting_ts)] if len(exiting_ts)<len(mus) else mus[-1]
                sigma = sigmas[len(exiting_ts)] if len(exiting_ts)<len(sigmas) else sigmas[-2]
                #lognormal distribution random number
                delta_t = np.random.lognormal(mean = mu,sigma = sigma)
                if len(exiting_ts)>0:
                    exiting_ts.append(exiting_ts[-1]+delta_t)
                else:
                    exiting_ts.append(delta_t)
            NS.append(len(exiting_ts))
        return NS
        
    def TimeLine(self, departure_ts,queues_ts,queues, arrival_ts):
        """
        Construct the time line of the input data. 
        This function is used for debug
        --------------------------------------
        input: departure_ts,queues_ts,queues
            all are list. 
            len(queues_ts)=len(queues), the queue 
            departure_ts is the moments for each depature. 
            arrival_ts is the 
        
        """
        strs_departure_ts = ['departure' for i in departure_ts]
        strs_arrival_ts = ['arrive' for i in arrival_ts]
        strs_queue_ts = ['queue='+str(j) for i,j in zip(queues_ts,queues)]
        
        tmp1 = departure_ts + arrival_ts + queues_ts#moments list
        tmp2 = strs_departure_ts + strs_arrival_ts + strs_queue_ts#str list
        #   get the sorted index
        sorted_idxx = sorted(range(len(tmp1)), key=tmp1.__getitem__)
        #new arrival_t and new modes
        new_tmp1 = sorted(tmp1)
        new_tmp2 = [tmp2[i] for i in sorted_idxx]
        
        res= pd.DataFrame()
        for t in sorted(set(tmp1)):
            i = res.shape[0]
            res.loc[i,0] = t
            idxs = np.where(np.array(new_tmp1)==t)[0]
            tmp = ''
            for idx in idxs:tmp = tmp + '_' + new_tmp2[idx]
            res.loc[i,1] = tmp
        
        return res
        
        
    def get_expected_departures(self, H, arrival_t, N, D0, N_modes, t0, firstexpected_delta_t, Q0, T = 1000.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss=False):
        """
        Get the exptcted departure, including 
            - the expected departure moments
            - the expected queue variation
            - the expected queue variation moments.
            
        NOTE that the vehicle can depart at instance t0+T. T is the time horizon for the vehicles departure. 
        
        
        
        The algirithm need to know that
            - the already departed cumulative number just before t0
            - the initial queue length before moment t0.
            
        -------------------------------------------
        input: H, the minimal headway between modes. 
            a dict. H['A']['B'] is the headway when leading is A and following vehicle is B.
        input: arrival_t, N
            the arrival of vehicles. self.arrival_t and self.N is a step function and is right-consinuous. Both begins from 0. len(arrival_t)==len(N). N is a list that increase one by one exactly. 
        input: D0
            the cumulative departures before instance t0
        input: N_modes
            the mode of each vehicle, any one in 'A','B', 'C';
            the length is len(slef.N)-1, because first element in slef.N is zero.
        input: t0
            a scalar, indicates the moment when the time horizon begins. 
        input: firstexpected_delta_t
            the moment when the first vehicle expect to leave(with respect to t0). The "expected" means that 
            - in the onset green, there is a start loss
            - when the first vehicle is CAV, there is no lost.
        input: Q0
            a scalar. queue length at before instance t0. 
        input: T
            the time horizon length for deparure. It is a relative lenth respect to the t0. 
        input: lognormal
            whether the vehicles headway is stochastic. If it is, then the headway will be stochastically. 
        input: sigmas and mus
            both are list. They are calculated from the NGSIM data. 
            mus[0] and sigmas[0] represent the mean and sigma of the lognormal distribution of the first vehicle.
            random sample using np.random.lognormal(mean = mu,sigma = sigma)
        input: fixedstartloss
            if fixedstartloss==False, it means that the start loss is dynamic:
                - if the first vehicle is cav, there is no start loss, and the first cav can pass the stopline exactly when the green starts. 
        --------------------------------------------
        output: departure_ts, queues, queues_ts
            departure_ts, a list of scalar, each element is the expected departure moment. Note that each moment is absolute momemt. 
            queues, a list of scalar. the queue length at each moment. len(queues)=len(queues_ts)
            queues_ts: each moment corresponding to the queue. it is also absolute moment. 
        --------------------------------------------
        Steps:
            (Keep the arrival temporal index in idxx)
            (Keep the departure moment in delarture_t)
            For each expected departure
                - calculate the expected depart moment in virtualdepart_t
                - 
            (arrival_t[idxx]>delarture_t always hold)
            - from moment t0
                - if there is remaining queue,  
        """
        #all vehicles are cleared
        #if t0>arrival_t[-1]:return False,False.False
        
        
        vehiclesleaved = 0
        
        #the returned value. 
        departure_ts = []
        queues = []
        queues_ts = []
        departure_modes = []#len(departure_modes)==len()
        rollingqueue = Q0
        
        #find the idx in N that arrival_t[idxx]>=t0 and arrival_t[idxx-1]<t0.
        #   if idxx is a list, it means that t0>arrival_t[-1].
        #   note that this arrival have NOT been included in the self.Q yet. 
        idxx = self.find_idxx_greaterthan(arrival_t, t0, equal = True)
        
        #one while iteration deals with only one departure. 
        while True:
            #if all vehicles are cleared, then return
            if D0+len(departure_ts)==N[-1]:
                return departure_ts,queues,queues_ts,departure_modes
            #steps:
            #   - 1 find the expected departure t, as virtualdepart_t. 
            #   - 2 if the expected departure is within horizon
            #       - assign the departutre.
            #       - queue minus 1
            #       - refresh the idxx
            #step1: find the expected departure as virtualdepart_t
            #   the result is stored in virtualmode and virtualdepart_t
            #   len(departure_ts)==0 means it is the first vehicle. 
            if len(departure_ts)==0 and Q0>0:
                #the first vehicle and there is queue. 
                #print('self.D[-1]',self.D[-1],'self.N[-1]',self.N[-1],'Q0',Q0,'len N_mode', len(N_modes))
                virtualmode = N_modes[D0]
                if virtualmode!='A' and (not fixedstartloss):
                    #no start loss
                    #mode is not 'A' means that the vehicle is CAV
                    virtualdepart_t = t0
                else:
                    virtualdepart_t = t0+firstexpected_delta_t
            #the first vehicle and no remaining queue
            elif len(departure_ts)==0 and Q0==0:
                if isinstance(idxx,list):
                    print('t0==',t0,',arrival_t[-1]=',arrival_t[-1],'N[-1]',self.N[-1],'Q[-1]',self.Q[-1],'D[-1]',self.D[-1],'idxx=',idxx)
                    #store the parameters for debug
                    builtins.tmp = {'N':self.N,'arrival_t':self.arrival_t,'N_modes':self.N_modes, 'RGP':self.rgp}
                    
                virtualdepart_t = arrival_t[idxx]
                virtualmode = N_modes[D0]
            elif len(departure_ts)>0 and rollingqueue==0:
                #not the first vehicle and no remaining queue
                #print('self.N[-1]',self.N[-1],'D0+len(departure_ts)',D0+len(departure_ts))
                former_mode = N_modes[D0+len(departure_ts)-1]
                latter_mode = N_modes[D0+len(departure_ts)]
                virtualdepart_t1 = departure_ts[-1]+ H[former_mode][latter_mode]
                if idxx>=len(arrival_t):
                    virtualdepart_t = virtualdepart_t1
                else:
                    virtualdepart_t = max(virtualdepart_t1,arrival_t[idxx])
                virtualmode = latter_mode
            elif len(departure_ts)>0 and rollingqueue>0:
                #not the first vehicle and there is remaining queue
                #print('self.N[-1]',self.N[-1],'self.D[-1]',self.D[-1],rollingqueue)
                former_mode = N_modes[D0+len(departure_ts)-1]
                latter_mode = N_modes[D0+len(departure_ts)]
                if latter_mode=='A' and lognormal:
                    if len(departure_ts)>len(mus):
                        tmp_headway = np.random.lognormal(mean = mus[-1],sigma = sigmas[-1])
                    else:
                        tmp_idx = len(departure_ts)-1
                        tmp_headway = np.random.lognormal(mean = mus[tmp_idx],sigma = sigmas[tmp_idx])
                else:
                    tmp_headway = H[former_mode][latter_mode]
                virtualdepart_t = departure_ts[-1] + tmp_headway
                virtualmode = latter_mode
                
            #step 2, assign the departure. 
            ever_arrival_equal_depart_t = False
            #   if satisfied, it means that the time horizon is not reached. 
            if virtualdepart_t<=t0+T:
                departure_ts.append(virtualdepart_t)
                departure_modes.append(virtualmode)
                #refresh the queue and the idxx, 
                #   iterate over arrivals, each arrival will add to one queue, if there is queue. 
                while True:
                    
                    #it means that arrival horizon is exceeded. 
                    if isinstance(idxx,list):
                        rollingqueue = max(rollingqueue - 1,0)
                        queues.append(rollingqueue)
                        queues_ts.append(virtualdepart_t)
                        break
                    
                    #if satisfied, it means that all arrivals are dealt with. hence break
                    if idxx>=len(arrival_t):
                        #if min(1,rollingqueue)
                        rollingqueue = max(rollingqueue - 1,0)
                        queues.append(rollingqueue)
                        queues_ts.append(virtualdepart_t)
                        #idxx = idxx + 1
                        break
                    
                    #idxx may be [], hence if idxx==[], it means that the t0 is 
                    if arrival_t[idxx]<virtualdepart_t:
                        #if min(1,rollingqueue)
                        rollingqueue = rollingqueue + 1
                        queues.append(rollingqueue)
                        queues_ts.append(arrival_t[idxx])
                        idxx= idxx + 1
                    elif arrival_t[idxx]==virtualdepart_t:
                        #one enter and one leave, so queue does not change.
                        queues.append(rollingqueue)
                        queues_ts.append(arrival_t[idxx])
                        idxx= idxx + 1
                        ever_arrival_equal_depart_t = True
                        break
                    elif arrival_t[idxx]>virtualdepart_t:
                        if not ever_arrival_equal_depart_t:
                            #one departure
                            rollingqueue = min(1,rollingqueue)*(rollingqueue - 1)
                            queues.append(rollingqueue)
                            queues_ts.append(virtualdepart_t)
                        break
                        
            #means virtualdepart_t>t0+T
            else:
                #if arrival before to+T and cannot pass the stopline, then it need to be included in the queue.
                #if idxx<len(arrival_t):
                if not isinstance(idxx,list):
                    while idxx<len(arrival_t) and arrival_t[idxx]<=t0+T:
                        queues.append(queues[-1]+1)
                        queues_ts.append(arrival_t[idxx])
                        idxx = idxx + 1
                
                #it means the time domain is tackled. 
                return departure_ts,queues,queues_ts,departure_modes
            
            vehiclesleaved = vehiclesleaved+1
            #print('Vehicles Departure: ',vehiclesleaved)
    
    def get_expected_departures0(self, H, arrival_t, N, D0, N_modes, t0, firstexpected_delta_t, Q0, T = 1000.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss=False):
        """
        
        """
        
        vehiclesleaved = 0
        
        #the returned value. 
        departure_ts = []
        queues = []
        queues_ts = []
        departure_modes = []#len(departure_modes)==len()
        rollingqueue = Q0
        
        #find the idx in N that arrival_t[idxx]>=t0 and arrival_t[idxx-1]<t0.
        #   if idxx is a list, it means that t0>arrival_t[-1].
        #   note that this arrival have NOT been included in the self.Q yet. 
        idxx = self.find_idxx_greaterthan(arrival_t, t0, equal = True)
        
        
        
        
        
        
        
        pass
        
        
    
    
    def get_expected_departures_backup(self, H, arrival_t, N, D0, N_modes, t0, firstexpected_delta_t, Q0, T = 1000.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss=False):
        """
        Get the exptcted departure, including 
            - the expected departure moments
            - the expected queue variation
            - the expected queue variation moments.
            
        NOTE that the vehicle can depart at instance t0+T. T is the time horizon for the vehicles departure. 
        
        
        
        The algirithm need to know that
            - the already departed cumulative number just before t0
            - the initial queue length before moment t0.
            
        -------------------------------------------
        input: H, the minimal headway between modes. 
            a dict. H['A']['A'] is the headway when leading and following vehicles are both mode A
        input: arrival_t, N
            the arrival of vehicles. self.arrival_t and self.N is a step function and is right-consinuous. Both begins from 0. len(arrival_t)==len(N). N is a list that increase one by one exactly. 
        input: D0
            the cumulative departures before instance t0
        input: N_modes
            the mode of each vehicle, any one in 'A','B', 'C';
            the length is len(slef.N)-1, because first element in slef.N is zero.
        input: t0
            a scalar, indicates the moment when the time horizon begins. 
        input: firstexpected_delta_t
            the moment when the first vehicle expect to leave(with respect to t0). The "expected" means that 
            - in the onset green, there is a start loss
            - when the first vehicle is CAV, there is no lost.
        input: Q0
            a scalar. queue length at before instance t0. 
        input: T
            the time horizon length for deparure. It is a relative lenth respect to the t0. 
        input: lognormal
            whether the vehicles headway is stochastic. If it is, then the headway will be stochastically. 
        input: sigmas and mus
            both are list. They are calculated from the NGSIM data. 
            mus[0] and sigmas[0] represent the mean and sigma of the lognormal distribution of the first vehicle.
            random sample using np.random.lognormal(mean = mu,sigma = sigma)
        input: fixedstartloss
            if fixedstartloss==False, it means that the start loss is dynamic:
                - if the first vehicle is cav, there is no start loss, and the first cav can pass the stopline exactly when the green starts. 
        --------------------------------------------
        output: departure_ts, queues, queues_ts
            departure_ts, a list of scalar, each element is the expected departure moment. Note that each moment is absolute momemt. 
            queues, a list of scalar. the queue length at each moment. len(queues)=len(queues_ts)
            queues_ts: each moment corresponding to the queue. it is also absolute moment. 
        --------------------------------------------
        Steps:
            (Keep the arrival temporal index in idxx)
            (Keep the departure moment in delarture_t)
            For each expected departure
                - calculate the expected depart moment in virtualdepart_t
                - 
            (arrival_t[idxx]>delarture_t always hold)
            - from moment t0
                - if there is remaining queue,  
        """
        #all vehicles are cleared
        #if t0>arrival_t[-1]:return False,False.False
        
        
        vehiclesleaved = 0
        
        #the returned value. 
        departure_ts = []
        queues = []
        queues_ts = []
        departure_modes = []#len(departure_modes)==len()
        rollingqueue = Q0
        
        #find the idx in N that arrival_t[idxx]>=t0 and arrival_t[idxx-1]<t0.
        #   if idxx is a list, it means that t0>arrival_t[-1].
        idxx = self.find_idxx_greaterthan(arrival_t, t0, equal = True)
        
        #one while iteration deals with only one departure. 
        while True:
            
            #if all vehicles are cleared, then return
            if D0+len(departure_ts)==N[-1]:
                return departure_ts,queues,queues_ts,departure_modes
            
            #steps:
            #   - 1 find the expected departure t, as virtualdepart_t. 
            #   - 2 if the expected departure is within horizon
            #       - assign the departutre.
            #       - queue minus 1
            #       - refresh the idxx
            #step1: find the expected departure virtualdepart_t
            #   the result is stored in virtualmode and virtualdepart_t
            #   len(departure_ts)==0 means it is the first vehicle. 
            if len(departure_ts)==0 and Q0>0:
                #the first vehicle and there is queue. 
                virtualmode = N_modes[D0]
                if virtualmode!='A' and (not fixedstartloss):
                    #no start loss
                    #mode is not 'A' means that the vehicle is CAV
                    virtualdepart_t = t0
                else:
                    virtualdepart_t = t0+firstexpected_delta_t
            #the first vehicle and no remaining queue
            elif len(departure_ts)==0 and Q0==0:
                if isinstance(idxx,list):
                    print('t0==',t0,',arrival_t[-1]=',arrival_t[-1],'N[-1]','Q[-1]',self.Q[-1],self.N[-1],'D[-1]',self.D[-1],'idxx=',idxx)
                virtualdepart_t = arrival_t[idxx]
                virtualmode = N_modes[D0]
            elif len(departure_ts)>0 and rollingqueue==0:
                #not the first vehicle and no remaining queue
                former_mode = N_modes[D0+len(departure_ts)-1]
                latter_mode = N_modes[D0+len(departure_ts)]
                virtualdepart_t1 = departure_ts[-1]+ H[former_mode][latter_mode]
                if idxx>=len(arrival_t):
                    virtualdepart_t = virtualdepart_t1
                else:
                    virtualdepart_t2 =  arrival_t[idxx]
                    virtualdepart_t = max(virtualdepart_t1,virtualdepart_t2)
                virtualmode = latter_mode
            elif len(departure_ts)>0 and rollingqueue>0:
                #not the first vehicle and there is remaining queue
                former_mode = N_modes[D0+len(departure_ts)-1]
                latter_mode = N_modes[D0+len(departure_ts)]
                if latter_mode=='A' and lognormal:
                    if len(departure_ts)>len(mus):
                        tmp_headway = np.random.lognormal(mean = mus[-1],sigma = sigmas[-1])
                    else:
                        tmp_idx = len(departure_ts)-1
                        tmp_headway = np.random.lognormal(mean = mus[tmp_idx],sigma = sigmas[tmp_idx])
                else:
                    tmp_headway = H[former_mode][latter_mode]
                virtualdepart_t = departure_ts[-1] + tmp_headway
                virtualmode = latter_mode
                
            #step 2, assign the departure. 
            ever_arrival_equal_depart_t = False
            #   if satisfied, it means that the time horizon is not reached. 
            if virtualdepart_t<=t0+T:
                departure_ts.append(virtualdepart_t)
                departure_modes.append(virtualmode)
                #refresh the queue and the idxx.
                while True:
                    if isinstance(idxx,list):
                        rollingqueue = max(rollingqueue - 1,0)
                        queues.append(rollingqueue)
                        queues_ts.append(virtualdepart_t)
                        break
                    
                    #if satisfied, it means that 
                    if idxx>=len(arrival_t):
                        #if min(1,rollingqueue)
                        rollingqueue = max(rollingqueue - 1,0)
                        queues.append(rollingqueue)
                        queues_ts.append(virtualdepart_t)
                        #idxx = idxx + 1
                        break
                    #idxx may be [], hence if idxx==[], it means that the t0 is 
                    if arrival_t[idxx]<virtualdepart_t:
                        #if min(1,rollingqueue)
                        rollingqueue = min(1,rollingqueue)*(rollingqueue + 1)
                        queues.append(rollingqueue)
                        queues_ts.append(arrival_t[idxx])
                        idxx= idxx + 1
                    elif arrival_t[idxx]==virtualdepart_t:
                        #one enter and one leave, so queue does not change.
                        queues.append(rollingqueue)
                        queues_ts.append(arrival_t[idxx])
                        idxx= idxx + 1
                        ever_arrival_equal_depart_t = True
                    elif arrival_t[idxx]>virtualdepart_t:
                        if not ever_arrival_equal_depart_t:
                            #one departure
                            rollingqueue = min(1,rollingqueue)*(rollingqueue - 1)
                            queues.append(rollingqueue)
                            queues_ts.append(virtualdepart_t)
                        break
            else:
                #if arrival before to+T and cannot pass the stopline, then it need to be included in the queue.
                if arrival_t[idxx]<=t0+T:
                    queues.append(queues[-1]+1)
                    queues_ts.append(arrival_t[idxx])
                
                #it means the time domain is tackled. 
                return departure_ts,queues,queues_ts,departure_modes
            
            vehiclesleaved = vehiclesleaved+1
            #print('Vehicles Departure: ',vehiclesleaved)
            
    
    
    def get_departure_t(self,n):
        """
        Get the departure moment of the n-th vehicle
        """
        if n<0:raise ValueError('n must be positive int, but now n is ', n)
        if n>self.D[-1]:raise ValueError('n must be smaller than vehicle N, but now n is ', n,' and vehicle N is ', self.D[-1])
        
        #find the idx
        idxx = np.where(np.array(self.D)==n)[0]
        #if len(idxx)>1:raise ValueError('N increase one by one, monotonous, get multiple res, idxs are', idxx)
        
        return self.departure_t[idxx[0]]
    
    def get_arrivals_interval(self, array ,t0, t1, start_equal = True, end_equal = True):
        """
        get all the arrival moments between two moments, t0 and t1.
        ----------------------------------------
        input: t0 t1
            t0 and t1 are two moments
        input: start_equal
            bool. If true, then the first arrival_t can equal t0
        input: end_equal
            bool. If true, then the last arrival_t can equal to t1.
        """
        if t1<=t0:
            raise ValueError("Input t1 should be greater than t0.")
            
        #find the idx of the t0, may be null means there is no such idx
        idx0 = self.find_idxx_greaterthan(array = self.arrival_t, threshold  = t0, equal = start_equal)
        #find the idx of the t1, may be null
        idx1 = self.find_idxx_smallerthan(array = self.arrival_t, threshold  =t1, equal = end_equal)
        if isinstance(idx0, list) or isinstance(idx1, list):
            #if isinstance(idx0, list) true, means that t0 out of scope
            return []
        else:
            return array[idx0:idx1+1]
        
    
    def get_arrival_t(self,n):
        """
        Get the arrival moment of the n-th vehicle. 
        """
        if n<0:raise ValueError('n must be positive int, but now n is ', n)
        if n>self.N[-1]:raise ValueError('n must be smaller than vehicle N, but now n is ', n,' and vehicle N is ', self.N[-1])
        
        #find the idx
        idxx = np.where(np.array(self.N)==n)[0]
        if len(idxx)>1:raise ValueError('N increase one by one, monotonous, get multiple res, idxs are', idxx)
        
        return self.arrival_t[idxx[0]]
        
    
    def N_construct(self,):
        """
        Construct the cumulative arrival based on self.arrivals_raw. self.arrivals_raw is a list. Each element is a dict. 
        --------------------------
        Output: no output,  
            Generate self.t and self.N. Both are list. 
            The function is setp wise. 
        """
        self.arrival_t = [.0]#in sec
        self.N = [0]# it is int
        self.N_modes = []
        #for each arrival batch, each batch corresponds to 
        #   a specific mode. 
        for idxx in range(len(self.arrivals_raw)):
            #
            interval  = self.arrivals_raw[idxx]['interval']
            mode  = self.arrivals_raw[idxx]['mode']
            batch_n  = self.arrivals_raw[idxx]['batch_n']
            
            #the head vehicle
            new_t = self.arrival_t[-1] + interval
            new_n = self.N[-1] + 1
            self.arrival_t.append(new_t)
            self.N.append(new_n)
            self.N_modes.append(mode)
            
            #the remaining vehicle
            for i in range(batch_n-1):
                new_t = self.arrival_t[-1] + self.H[mode][mode]
                new_n = 1 + self.N[-1]
                self.arrival_t.append(new_t)
                self.N.append(new_n)
                self.N_modes.append(mode)


    def find_idxx_smallerthan(self, array, threshold, equal = False):
        """
        find the first index idxx so that array[idxx]<threshold and array[idxx+1]>=threshold. 
        -----------------------------------------------
        Input: array
            a list, or 1-D array
        Input: threshold
            a scalar
        Input: equal
            Whether allow array[idxx]==threshold. if True, then the arrar[idxx]==threshold
            else arrar[idxx]<threshold
        -------------------------------------------------
        Output: idxx
            an index in the array
        """
        if equal:
            #tmp is a list
            tmp = np.where(np.array(array) <= threshold)[0]
            if len(tmp)==0:
                return []
            else:
                return tmp[-1]
        else:
            #tmp is a list
            tmp = np.where(np.array(array) < threshold)[0]
            if len(tmp)==0:
                return []
            else:
                return tmp[-1] 

    def find_idxx_greaterthan(self, array, threshold, equal = False):
        """
        find the first index idxx so that array[idxx-1]<threshold and array[idxx]>=threshold. 
        -----------------------------------------------
        Input: array
            a list, or 1-D array
        Input: threshold
            a scalar
        Input: equal
            Whether allow array[idxx]==threshold. if True, then the arrar[idxx]==threshold
            else arrar[idxx]<threshold
        -------------------------------------------------
        Output: idxx
            an index in the array
        """
        if equal:
            #tmp is a list
            tmp = np.where(np.array(array) >= threshold)[0]
            if len(tmp)==0:
                return []
            else:
                return tmp[0]
        else:
            #tmp is a list
            tmp = np.where(np.array(array) > threshold)[0]
            if len(tmp)==0:
                return []
            else:
                return tmp[0]   
    
    def get_Q(self,t):
        """
        get the queue length at instance t. 
        If there is exactly idxx that self.queue_t[idxx]==t, then return self.Q[idxx].
        
        else find the idxx that self.queue_t[idxx]<t<self.queue_t[idxx+1], and return self.Q[idxx].
        """
        if t<self.queue_t[0]:
            return 0
        if t>self.queue_t[-1]:
            return self.Q[-1]
        
        #find the index that self.arrival_t[idxx]<=t and self.arrival_t[idxx+1]>t
        idxxs = self.find_idxx_smallerthan(array=self.queue_t, threshold = t, equal = True)
        return self.Q[idxxs]
        
        
        np.where(np.array(self.departure_t)==t)[0]
        if len(idxxs)==0:
            idxxs = np.where(np.array(self.departure_t)<t)[0]
            return self.Q[idxxs[-1]]
        else:
            return self.Q[idxxs[0]]
    
    def get_D(self,t):
        """get the cumulative departure N at moment t"""
        if t<self.departure_t[0]:
            return self.D[0]
        if t>self.departure_t[-1]:
            return self.D[-1]
        
        #find the index that self.arrival_t[idxx]<=t and self.arrival_t[idxx+1]>t
        idxxs = np.where(np.array(self.departure_t)<=t)[0]
        return self.D[idxxs[-1]]
    
    def get_N_remain(self, t):
        """
        get the vehicles that remain unprocessed as instance t.
        
        """
        
        
        pass
    
    def get_idxxes_in_cycle(self, array,  idxx = 0, equal = True):
        """
        
        """
        #red start and green end
        r_start  = self.cycle_start_t[idxx]
        g_end = self.cycle_end_t[idxx]
        
        #find the start idx and end idx.
        start_idx = self.find_idxx_greaterthan(np.array(array), r_start, equal = equal)
        end_idx = self.find_idxx_smallerthan(np.array(array), g_end, equal = equal)
        
        if isinstance(start_idx,list):
            if isinstance(end_idx,list):
                return []
            else:
                return range(end_idx+1)
        else:
            if isinstance(end_idx,list):
                return range(start_idx,len(array))
            else:
                return range(start_idx,end_idx+1)
        
    
    def get_idxxes_in_red(self, array,  idxx = 0, equal = True):
        """
        find the idxxes  with in red signal in specific cycle idxx.
        """
        #red start and r_end
        r_start  = self.cycle_start_t[idxx]
        r_end = r_start + self.rgp.EndMomentsOfReds[idxx]-self.rgp.StartMomentsOfReds[idxx]
        
        #find the start idx and end idx.
        start_idx = self.find_idxx_greaterthan(np.array(array), r_start, equal = equal)
        end_idx = self.find_idxx_smallerthan(np.array(array), r_end, equal = equal)
        
        if isinstance(start_idx,list):
            if isinstance(end_idx,list):
                return []
            else:
                return range(end_idx+1)
        else:
            if isinstance(end_idx,list):
                return range(start_idx,len(array))
            else:
                return range(start_idx,end_idx+1)
    
    def get_N(self,t):
        """get the cumulative arrival N at moment t"""
        
        if t<self.arrival_t[0]:
            return self.N[0]
        if t>self.arrival_t[-1]:
            return self.N[-1]
        
        #find the index that self.arrival_t[idxx]<=t and self.arrival_t[idxx+1]>t
        idxxs = np.where(np.array(self.arrival_t)<=t)[0]
        return self.N[idxxs[-1]]
        
    
    def get_mode_vehicle(self,n):
        """
        Get the mode of the n-th vehicle, based on self.N_modes. n must positive
        """
        #self.N_modes is a list, length is len(self.N)-1, because the self.N[0]=0
        if n<=0:raise ValueError('n must be positive and int, and now n=',n)
        return self.N_modes[n-1]
    
    def AssignDQ_red_duration(self):
        """
        assign the departure and the queue length during the red signal. 
        
        """
        
        
        pass
        
    def Check_monotonous(self):
        """
        check whether the variables are strict increasing. 
        """
        #arrival check
        if not np.all(np.diff(self.N) > 0):
            raise ValueError('self.N is not strict increasing')
        if not np.all(np.diff(self.arrival_t) > 0):
            raise ValueError('self.arrival_t is not strict increasing')
        print('The self.arrival_t and self.N are correct.')
        
        #departure check
        if not np.all(np.diff(self.D) >= 0):
            raise ValueError('self.D is not strict increasing')
        if not np.all(np.diff(self.departure_t) >= 0):
            raise ValueError('self.departure_t is not strict increasing')
        print('The self.departure_t and self.D are correct.')
        

    
    def D_construct_dynamic_intergreen(self, some_rgp, redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        Given the signal,  the departure cumulative using self.H
        and self.arrival_t and self.N. 
        The role of:
            self.H, it is the minimum headway for different modes
            self.arrival_t and self.N are the arrival moment of the n-th vehicle. The two are identical. 
        
        different from self.D_construct() in that, the static intergreen duration is taken into account. 
        
        The following are constructed:
            self.D
            self.departure_t 
            self.Q
            self.queue_t
            (len(self.departure_t)==len(self.D) and 
            (len(self.queue_t)==len(self.Q)))
        ------------------------------------------
        Input: somergp
            the signal, an RGP instance. somergp.reds return a list of all red duration. 
        Input: yellow
            a scalar, it is the yellow duration
        Input: redclearance
            a scalar. all-red duration that is used to clear the intersection. 
        Input: speedlimit
            a scalar. unit is km/h
        Input: reactiontime, unit is second
            reaction time of the human drivers.
        Input: startloss
            the start loss time due to the acceleration of vehicles. unit is second. 
        Input: acceleration, unit is m/(s^2)
            the deceleration for yellow time. Then the static yellow time is calculated as reactiontime+\frac{v}{2*acceleration}
        input: lognormal
            whether the headway obey the lognormal distirbuton. 
        input: sigmas and mus
            both are list. They are calculated from the NGSIM data. 
            mus[0] and sigmas[0] represent the mean and sigma of the lognormal distribution of the first vehicle.
            random sample using np.random.lognormal(mean = mu,sigma = sigma)
        ------------------------------------------
        Output: no output, but the following are constructed:
            self.departure_t
            self.D
            self.queue_t
            self.Q #the unit is vehicle number
        ---------------------------------------------
        Steps:
            - find the idxx of the first red signal
                - all vehicles arrive before this red start can passs without delay
            - For each red duration, [rs, re]
                - determine the remaining queue at moment rs
                - determine the queue lenggh at moment re
                - For each pair of vehicles, determine the headway, then accumulate the vehicles in self.D and decrease the self.Q
                
        """
        #store the middle results.
        self.tmp = {}
        self.cycle_start_t = []
        self.cycle_end_t = []
        
        #convert the speed to m/s instead of km/h.
        speed = 1.0*speedlimit/3.6
        yellowtime = reactiontime + speed/(2.0*acceleration)
        accumulative_yellowtime = 0.0
        
        self.rgp = copy.deepcopy(some_rgp)
        
        #the begining of the first red signal
        r0 = some_rgp.StartMomentsOfReds[0]
        #   for any moment before r_0, the self.D is the same as self.N
        idxx = self.find_idxx_smallerthan(self.arrival_t, r0, equal = True)
        #len(idxx)==0 means that no arrival is earlier than r0.
        if isinstance(idxx,list):
            self.departure_t  = [-.001]
            self.D  = [.0]
            self.queue_t  = [-0.001]
            self.Q  = [.0]
        else:
            #all arrival before r0 will depart immediately. 
            self.departure_t  = copy.deepcopy(self.arrival_t[:idxx+1])
            self.D  = copy.deepcopy(self.N[:idxx+1])
            self.queue_t  = [r0]
            self.Q  = [0]
        
        #for each red signal
        for rgp_idxx in range(len(some_rgp.reds)):
            #the red start, red end and the green of the current cycle
            r_start = some_rgp.StartMomentsOfReds[rgp_idxx]
            r_end = some_rgp.EndMomentsOfReds[rgp_idxx]
            r_start  = r_start + accumulative_yellowtime
            r_end  = r_end + accumulative_yellowtime
            #demand is out of the signal time horizon.
            #if r_start>=self.arrival_t[-1]:
            #    return
            
            #print 'rgp_idxx=',rgp_idxx,'r end=',r_end
            
            #get the green duraition available as green_duration
            #   it is either np.inf or actual green duration
            if rgp_idxx+1>=len(some_rgp.StartMomentsOfReds):
                next_red_start = np.inf
                green_duration = np.inf
            else:
                next_red_start = some_rgp.StartMomentsOfReds[rgp_idxx+1]+ accumulative_yellowtime + yellowtime
                #the yellow time is included
                green_duration = next_red_start - r_end
            
            self.cycle_start_t.append(r_start)
            
            #assign the self.D, self.Q during red signal.
            self.departure_t.append(r_start)
            self.D.append(self.D[-1])
            self.departure_t.append(r_end)
            self.D.append(self.D[-1])
            #    tmp_idx0 and tmp_idx1 corresponding to the idx of r_start and r_end
            tmp_idx0 = self.find_idxx_greaterthan(self.arrival_t, r_start, equal = False)
            tmp_idx1 = self.find_idxx_smallerthan(self.arrival_t, r_end, equal = False)
            if not isinstance(tmp_idx0,list):
                if isinstance(tmp_idx1,list):tmp_idx1=len(self.arrival_t)
                for i in range(tmp_idx0,tmp_idx1+1):
                    self.Q.append(self.Q[-1]+1)
                    self.queue_t.append(self.arrival_t[i])
            
            #preparing D0, Q0, and get the expected departure, Q
            #   in departure_ts, queues and queues_ts
            #   len(queues_ts)=len(queues)
            D0 = int(self.D[-1])
            Q0 =  self.Q[-1]
            t0 = r_end
            departure_ts,queues,queues_ts,departures_mode= self.get_expected_departures(H=self.H, arrival_t=self.arrival_t, N=self.N, D0=D0, N_modes=self.N_modes,t0=t0, firstexpected_delta_t=startloss, Q0=Q0, T = green_duration,lognormal = lognormal, sigmas = sigmas,mus = mus,fixedstartloss = fixedstartloss)
            
            #dynamic yellow considering
            #   the returned value delta_yellow is 
            new_departure_ts,delta_yellow= self.DynamicYelow(y_end=t0+green_duration, departure_ts=departure_ts, modes=departures_mode, speedlimit = speedlimit, reactiontime =reactiontime, deceleration = acceleration)
            #yellow time and accumulative yellow time.
            tmp_yellow = yellowtime - delta_yellow
            #tmp_yellow = yellowtime - max(delta_yellow, t0+green_duration - new_departure_ts[-1])
            if next_red_start!=np.inf:
                tmp_yellow = max(tmp_yellow, new_departure_ts[-1]-(r_start + some_rgp.StartMomentsOfReds[rgp_idxx+1]- some_rgp.EndMomentsOfReds[rgp_idxx]))
            
            
            accumulative_yellowtime = accumulative_yellowtime + tmp_yellow
            
            #the true next cycle start moment considering the dynamic yellow
            if next_red_start!=np.inf:
                self.cycle_end_t.append(some_rgp.StartMomentsOfReds[rgp_idxx+1]+ accumulative_yellowtime)
            else:
                self.cycle_end_t.append(np.inf)
            
            departure_ts= new_departure_ts
            queues = queues[:len(departure_ts)]
            queues_ts = queues[:len(queues_ts)]
            #print('dynamic yellow time is',tmp_yellow,'delta yellow time is ',delta_yellow,'vehicle deleted is ',len(departure_ts)-len(new_departure_ts))
            
            
            self.tmp[rgp_idxx]={}
            self.tmp[rgp_idxx]['departure_ts']=departure_ts
            self.tmp[rgp_idxx]['queues']=queues
            self.tmp[rgp_idxx]['queues_ts']=queues_ts
            #assign the self.D and self.departure_t during the green duration. 
            #   self.Q and self.queue_t
            
            for tmp in departure_ts:
                if tmp==self.departure_t[-1]:print("asfasfasfasd")
                self.departure_t.append(tmp)
                self.D.append(self.D[-1]+1)
            for tmp_q,tmp_t in zip(queues, queues_ts):
                self.Q.append(tmp_q)
                self.queue_t.append(tmp_t)

    
    
    def D_construct_static_intergreen(self, some_rgp, yellow=3.0,redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        Given the signal,  the departure cumulative using self.H
        and self.arrival_t and self.N. 
        The role of:
            self.H, it is the minimum headway for different modes
            self.arrival_t and self.N are the arrival moment of the n-th vehicle. The two are identical. 
        
        different from self.D_construct() in that, the static intergreen duration is taken into account. 
        
        The following are constructed:
            self.D
            self.departure_t 
            self.Q
            self.queue_t
            (len(self.departure_t)==len(self.D) and 
            (len(self.queue_t)==len(self.Q)))
        ------------------------------------------
        Input: somergp
            the signal, an RGP instance. somergp.reds return a list of all red duration. 
        Input: yellow
            a scalar, it is the yellow duration
        Input: redclearance
            a scalar. all-red duration that is used to clear the intersection. 
        Input: speedlimit
            a scalar. unit is km/h
        Input: reactiontime, unit is second
            reaction time of the human drivers.
        Input: startloss
            the start loss time due to the acceleration of vehicles. unit is second. 
        Input: acceleration, unit is m/(s^2)
            the deceleration for yellow time. Then the static yellow time is calculated as reactiontime+\frac{v}{2*acceleration}
        input: lognormal
            Bool. whether the headway obey the lognormal distirbuton. 
        input: fixedstartloss
            Bool. if fixedstartloss==False, it means that the start loss is dynamic:
                - if the first vehicle is cav, there is no start loss, and the first cav can pass the stopline exactly when the green starts. 
        input: sigmas and mus
            both are list. They are calculated from the NGSIM data. 
            mus[0] and sigmas[0] represent the mean and sigma of the lognormal distribution of the first vehicle.
            random sample using np.random.lognormal(mean = mu,sigma = sigma)
        ------------------------------------------
        Output: no output, but the following are constructed:
            self.departure_t
            self.D
            self.queue_t
            self.Q #the unit is vehicle number
        ---------------------------------------------
        Steps:
            - find the idxx of the first red signal
                - all vehicles arrive before this red start can passs without delay
            - For each red duration, [rs, re]
                - determine the remaining queue at moment rs
                - determine the queue lenggh at moment re
                - For each pair of vehicles, determine the headway, then accumulate the vehicles in self.D and decrease the self.Q
                
        """
        #store the middle results.
        self.tmp = {}
        self.cycle_start_t = []
        self.cycle_end_t = []
        
        #convert the speed to m/s instead of km/h.
        speed = 1.0*speedlimit/3.6
        yellowtime = reactiontime + speed/(2.0*acceleration)
        accumulative_yellowtime = 0.0
        
        self.rgp = copy.deepcopy(some_rgp)
        
        #the begining of the first red signal
        r0 = some_rgp.StartMomentsOfReds[0]
        #   for any moment before r_0, the self.D is the same as self.N
        idxx = self.find_idxx_smallerthan(self.arrival_t, r0, equal = True)
        #len(idxx)==0 means that no arrival is earlier than r0.
        if isinstance(idxx,list):
            self.departure_t  = [-.001]
            self.D  = [0]
            self.queue_t  = [-0.001]
            self.Q  = [0]
        else:
            #all arrival before r0 will depart immediately. 
            self.departure_t  = copy.deepcopy(self.arrival_t[:idxx+1])
            self.D  = copy.deepcopy(self.N[:idxx+1])
            self.queue_t  = [r0]
            self.Q  = [.0]
        
        #for each red signal
        for rgp_idxx in range(len(some_rgp.reds)):
            #the red start, red end and the green of the current cycle
            r_start = some_rgp.StartMomentsOfReds[rgp_idxx]
            r_end = some_rgp.EndMomentsOfReds[rgp_idxx]
            r_start  = r_start + yellowtime*rgp_idxx
            r_end  = r_end + yellowtime*rgp_idxx
            #demand is out of the signal time horizon.
            #if r_start>=self.arrival_t[-1]:
            #    return
            
            #print 'rgp_idxx=',rgp_idxx,'r end=',r_end
            #get the green duraition available as green_duration
            #   it is either np.inf or actual green duration
            if rgp_idxx+1>=len(some_rgp.StartMomentsOfReds):
                next_red_start = np.inf
                green_duration = np.inf
            else:
                next_red_start = some_rgp.StartMomentsOfReds[rgp_idxx+1]+yellowtime*(rgp_idxx+1)
                green_duration = next_red_start - r_end
            
            self.cycle_start_t.append(r_start)
            self.cycle_end_t.append(next_red_start)
            
            #assign the self.D, self.Q during red signal.
            self.departure_t.append(r_start)
            self.D.append(self.D[-1])
            self.departure_t.append(r_end)
            self.D.append(self.D[-1])
            #    tmp_idx0 and tmp_idx1 corresponding to the idx of r_start and r_end
            tmp_idx0 = self.find_idxx_greaterthan(self.arrival_t, r_start, equal = False)
            tmp_idx1 = self.find_idxx_smallerthan(self.arrival_t, r_end, equal = False)
            if not isinstance(tmp_idx0,list):
                if isinstance(tmp_idx1,list):tmp_idx1=len(self.arrival_t)
                for i in range(tmp_idx0,tmp_idx1+1):
                    self.Q.append(self.Q[-1]+1)
                    self.queue_t.append(self.arrival_t[i])
            
            #preparing D0, Q0, and get the expected departure, Q
            #   in departure_ts, queues and queues_ts
            #   len(queues_ts)=len(queues)
            D0 = int(self.D[-1])
            Q0 =  self.Q[-1]
            t0 = r_end
            departure_ts,queues,queues_ts,departures_mode= self.get_expected_departures(H=self.H, arrival_t=self.arrival_t, N=self.N, D0=D0, N_modes=self.N_modes,t0=t0, firstexpected_delta_t=startloss, Q0=Q0, T = green_duration,lognormal = lognormal, sigmas = sigmas,mus = mus,fixedstartloss = fixedstartloss)
            
            self.tmp[rgp_idxx]={}
            self.tmp[rgp_idxx]['departure_ts']=departure_ts
            self.tmp[rgp_idxx]['queues']=queues
            self.tmp[rgp_idxx]['queues_ts']=queues_ts
            #assign the self.D and self.departure_t during the green duration. 
            #   self.Q and self.queue_t
            
            for tmp in departure_ts:
                #if tmp==self.departure_t[-1]:print("asfasfasfasd")
                self.departure_t.append(tmp)
                self.D.append(int(self.D[-1]+1))
            for tmp_q,tmp_t in zip(queues, queues_ts):
                self.Q.append(int(tmp_q))
                self.queue_t.append(tmp_t)
    
    def D_assign_green(self, some_rgp, t0, D0, Q0, greenduration,yellowtype = 'static'):
        """
        
        NOTE:
            the index of the cycle is stored at self.current_idx.
        
        The following are changed:
            self.departure_t
            self.D
            self.Q
            self.queue_t
            self.yellowtimes
            self.cycle_end_t
            self.cycle_start_t (of next cycle!!!!!)
        ---------------------
        input: t0
            the moment when green begins. It is generated by the self.D_assign_red().
        input: D0
            the departure at the instance of red termination. 
        input: Q0
            the queue lengh at the instance of red termination. 
            It is possible that, a vehicle arrive at instance Q0
        input: greenduration
            the green duration that for the vehicles to pass. Note that greenduration include the static yellow time into account.
            
        ----------------------------------------
        Steps:
            - get the expected departure series
            - assign the D and Q
            - set the self.cycle_end_t
        
        
        """
        #
        departure_ts,queues,queues_ts,departures_mode= self.get_expected_departures(D0=D0,Q0=Q0,t0=t0,T = greenduration,H=self.H,arrival_t=self.arrival_t,N=self.N,N_modes=self.N_modes,firstexpected_delta_t=self.startloss,lognormal = self.lognormal,sigmas = self.sigmas,mus = self.mus,fixedstartloss = self.fixedstartloss)
        #print('len(departure_ts)==',departure_ts)
        if yellowtype=='dynamic' and len(departure_ts)>0:
            #dynamic yellow considering
            #   the returned value delta_yellow is
            #builtins.tmp1 = (t0,greenduration,departure_ts,departures_mode,self.speedlimit,self.reactiontime,self.acceleration)
            #builtins.tmp = (self.arrival_t,self.N,self.rgp)
            #print 'resulttttt=',self.DynamicYelow(y_end=t0+greenduration, departure_ts=departure_ts, modes=departures_mode, speedlimit = self.speedlimit, reactiontime =self.reactiontime, deceleration = self.acceleration)
            new_departure_ts,delta_yellow = self.DynamicYelow(y_end=t0+greenduration, departure_ts=departure_ts, modes=departures_mode, speedlimit = self.speedlimit, reactiontime =self.reactiontime, deceleration = self.acceleration)
            #print('len(new_departure_ts)=',len(new_departure_ts), 'len(departure_ts)=',len(departure_ts),'new_departure_ts=',new_departure_ts,'\n departure_ts=',departure_ts   )
            #yellow time and accumulative yellow time.
            tmp_yellow = self.staticyellow - delta_yellow
            departure_ts = copy.deepcopy(new_departure_ts)
            queues_ts = [i for i in queues_ts if i<=t0+greenduration-delta_yellow]
            queues = queues[:len(queues_ts)]
            
        elif yellowtype=='dynamic' and len(departure_ts)==0:
            delta_yellow = self.reactiontime
            tmp_yellow = self.staticyellow - delta_yellow
            
        #assign the D and Q
        self.departure_t.extend(departure_ts)
        tmp = self.D[-1]
        self.D.extend([i+1+tmp for i in range(len(departure_ts))])
        #   self.Q
        self.queue_t.extend(queues_ts)
        self.Q.extend(queues)
        
        #set the yellow time and cycle_end_t
        if yellowtype=='static':
            self.yellowtimes.append(self.staticyellow)
            self.cycle_end_t.append(t0+greenduration)
            self.cycle_start_t.append(t0+greenduration)
        elif yellowtype=='dynamic':
            self.yellowtimes.append(tmp_yellow)
            self.cycle_end_t.append(t0+greenduration-delta_yellow)
            self.cycle_start_t.append(t0+greenduration-delta_yellow)
    
        return departure_ts,queues_ts,queues
        
    def D_assign_red(self, some_rgp):
        """
        assign self.departure_t and self.D during the red signal. The index of the cycle is stored in self.current_idx. ALso self.queue_t and self.Q are changed. 
        
        The required variables:
            - self.current_idx
        
        NOTE:
            (1)the vehicle can depart at instance self.cycle_start_t[self.current_idx] CAN depart. 
            (2) at moment r_end, the self.D and self.Q are not assigned. 
        ----------------------------------
        output:
            the follwoing are changed:
                self.D self.departure_t
                self.Q self.queue_t
            t0, D0, Q0, greenduration
                - t0 is the termination moment of red.
                - D0 is the cumulative departure at termination moment
                - Q0 is the queue length at the termination moment of red.
                - greenduration is the green duration of current cycle.(NOTE that this green duration contains the static yellow time.) 
            
        -------------------------------------
        Steps:
            self.current_idx
            self.yellowtimes
            self.cycle_start_t
            self.cycle_red_end_t
            self.cycle_green_end_t
            
        """
        #the index of the signal
        #rgp_idxx= self.current_idx
        
        #the red start, red end and the green of the current cycle
        r_start = self.cycle_start_t[self.current_idx]
        r_end = r_start + some_rgp.EndMomentsOfReds[self.current_idx]-some_rgp.StartMomentsOfReds[self.current_idx]
        #r_start = some_rgp.StartMomentsOfReds[self.current_idx]+sum(self.yellowtimes[:self.current_idx])
        #r_end = some_rgp.EndMomentsOfReds[self.current_idx]+sum(self.yellowtimes[:self.current_idx])
        self.cycle_red_end_t.append(r_end)
        #demand is out of the signal time horizon.
        #if r_start>=self.arrival_t[-1]:
        #    return
        
        #get the green duraition available as green_duration
        #   it is either np.inf or actual green duration
        #   NOTE that if it is dynamic yellow, the green_duration0 changes. 
        if self.current_idx+1>=len(some_rgp.StartMomentsOfReds):
            next_red_start = np.inf
            green_duration = np.inf
            self.cycle_green_end_t.append(np.inf)
        else:
            next_red_start = some_rgp.StartMomentsOfReds[self.current_idx+1]+sum(self.yellowtimes[:self.current_idx])+self.staticyellow
            green_duration = next_red_start - r_end
            #the green termination moment
            self.cycle_green_end_t.append(some_rgp.StartMomentsOfReds[self.current_idx+1]+sum(self.yellowtimes[:self.current_idx]))
        
        #assign the self.D, self.Q during red signal.
        #    tmp_idx0 and tmp_idx1 corresponding to the idx of r_start and r_end
        tmp_idx0 = self.find_idxx_greaterthan(self.arrival_t, r_start, equal = False)
        tmp_idx1 = self.find_idxx_smallerthan(self.arrival_t, r_end, equal = False)
        if not isinstance(tmp_idx0,list):
            if isinstance(tmp_idx1,list):tmp_idx1=len(self.arrival_t)
            for i in range(tmp_idx0,tmp_idx1+1):
                self.Q.append(self.Q[-1]+1)
                self.queue_t.append(self.arrival_t[i])
        
        #preparing D0, Q0, and get the expected departure, Q
        #   in departure_ts, queues and queues_ts
        #   len(queues_ts)=len(queues)
        D0 = int(self.D[-1])
        Q0 =  self.Q[-1]
        t0 = r_end
        
        return t0,D0,Q0,green_duration
    

    
    
    def D_assign_before_signal_horizon(self, some_rgp, yellowtype = 'static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        assign the following that before the start of the first signal.
        
        Data flow:
            self.D_assign_before_signal_horizon()
                - self.cycle_start_t
            self.D_assign_red()
            self.D_assign_green()
                - self.cycle_end_t
                - self.cycle_start_t
                - self.yellowtimes
        ---------------------------------------------
        output: 
            no output, but the following are changed:
                - self.D
                - self.departure_t
                - self.queue_t
                - self.Q
                - self.cycle_start_t
        """
        #store the middle results.
        self.tmp = {}
        self.cycle_start_t = []
        self.cycle_end_t = []
        
        #convert the speed to m/s instead of km/h.
        speed = 1.0*speedlimit/3.6
        yellowtime = reactiontime + speed/(2.0*acceleration)
        accumulative_yellowtime = 0.0
        
        self.rgp = copy.deepcopy(some_rgp)
        
        #the begining of the first red signal
        r0 = some_rgp.StartMomentsOfReds[0]
        self.cycle_start_t = [r0]
        
        #   for any moment before r_0, the self.D is the same as self.N
        idxx = self.find_idxx_smallerthan(self.arrival_t, r0, equal = True)
        #len(idxx)==0 means that no arrival is earlier than r0.
        if isinstance(idxx,list):
            self.departure_t  = [-.001]
            self.D  = [.0]
            self.queue_t  = [-0.001]
            self.Q  = [.0]
        else:
            #all arrival before r0 will depart immediately. 
            self.departure_t  = copy.deepcopy(self.arrival_t[:idxx+1])
            self.D  = copy.deepcopy(self.N[:idxx+1])
            self.queue_t  = [r0]
            self.Q  = [0]
        
        #self.cycle_start_t = [r0]

        
    def state_by_cycle(self, idxx = 0):
        """
        
        
        """
        
        
        print 'cycle_start_t        =',self.cycle_start_t[idxx]
        print 'cycle_end_t          =',self.cycle_end_t[idxx]
        print 'N at cycle end       =',self.get_N(self.cycle_end_t[idxx])
        print 'self.Q[-1]       =',self.Q[-1],' self.queue_t[-1]=',self.queue_t[-1]
        print 'self.D[-1]       =',self.D[-1],' self.departure_t[-1]=',self.departure_t[-1]
        print 'self.N[self.departure_t[-1]]       =',self.get_N(self.departure_t[-1])
        
        if idxx+1>=len(self.rgp.StartMomentsOfReds):
            next_red_start = np.inf
        else:
            next_red_start = self.rgp.StartMomentsOfReds[idxx+1]+sum(self.yellowtimes[:idxx])+self.staticyellow
        original_cycle_end = next_red_start
        
        dictt = {'cycle_start_t':self.cycle_start_t[idxx],'cycle_end_t':self.cycle_end_t[idxx], 'N at cycle end':self.get_N(self.cycle_end_t[idxx]), 'self.Q[-1]':self.Q[-1], 'self.queue_t[-1]':self.queue_t[-1], 'self.D[-1]':self.D[-1], 'self.departure_t[-1]':self.departure_t[-1], 'self.N[self.departure_t[-1]]':self.get_N(self.departure_t[-1]), 'original cycle end':original_cycle_end,'yellow time':self.yellowtimes[idxx],'static yellow':self.staticyellow}
        
        return pd.Series(dictt)

    def reorganize_D(self):
        """
        reorganize the self.D and self.departure_t.
        -------------------------------------
        input: no input, but use the self.departure_t and self.D
        output: new_D, new_departure_t
            both are lists. 
        """
        new_D = []
        new_departure_t = []
        for i in range(self.D[-1]+1):
            #find the index in list
            idx = self.D.index(i)
            new_D.append(i)
            new_departure_t.append(self.departure_t[idx])
            
        return new_D,new_departure_t
    
    def overlap_N(self, N1, arrival_t1, N_modes1, delta_e = .00001):
        """
        overlap the arrival demand. The following are changed:
            - self.N
            - self.arrival_t
            - self.N_modes
        The input N1 sould begin from 0 and increase 1 by 1. 
        If the original demand need to be set to null, then:
            self.N = [0]
            self.arrival_t = [0]
            self.N_modes = []
            self.overlap_N(N1 = N1, arrival_t1=arrival_t1, N_modes1=N_modes1)
        ------------------------------------------------
        input: N1, arrival_t1, N_modes1
            all are lists. len(N1)=len(arrival_t1)=len(N_modes1)+1, because the N1 begins from 0. And N_modes1 gives the mode of each vehicle. 
        input: delta_e
            a very small scalar, used to increment 
        ----------------------------------------------
        Steps:
            - combind the self.arrival_t and arrival_t1 into arrival_t
            - sort the arrival_t. 
        """
        if arrival_t1[0]!=0:
            raise ValueError('The first element of arrival_t1 should be 0')
        #because arrival_t1[0]=0
        arrival_t = copy.deepcopy(self.arrival_t + arrival_t1[1:])
        #   now len(N_modes)=len(arrival_t)-1
        N_modes = copy.deepcopy(self.N_modes + N_modes1)
        #   the first of sorted_idxx should be 0, because arrival_t[0]=0
        sorted_idxx = sorted(range(len(arrival_t)), key=arrival_t.__getitem__)
        
        #new arrival_t and new modes
        new_arrival_t = sorted(arrival_t)
        new_N_modes = [N_modes[i-1] for i in sorted_idxx[1:]]
        
        #check whether there is identical arrival moments. 
        while True:
            #   duplicated_idxes is a 1d array. 
            try:
                duplicated_idxes = np.where(np.diff(new_arrival_t)<=0)[0]
            except:
                print 'len(new_arrival_t) = ',len(new_arrival_t),new_arrival_t
            if len(duplicated_idxes)==0:break
            for i in range(duplicated_idxes[0], len(new_arrival_t)+1):
                if new_arrival_t[i+1] <= new_arrival_t[i]:
                    new_arrival_t[i+1] = new_arrival_t[i] + delta_e
                else:
                    break
                
        self.arrival_t = copy.deepcopy(new_arrival_t)
        self.N_modes = copy.deepcopy(new_N_modes)
        self.N = range(len(new_arrival_t))
    
    def split_D(self, D, departure_t, N_modes, ratios0 = [.3,.3,.4]):
        """
        split the output, i.e. D into several lists. 
        The self.D should be begin from 0 and increase one by one exactly. 
        If not, the function self.reorganize_D(self) will performe such task. 
            D,departure_t, N_modes = self.reorganize_D()
            res = self.split_D(D=D, departure_t=departure_t)
            a dict. 
            res['idx']['D'] = a list.
            res['idx']['departure_t'] = a list.
            res['idx']['N_modes'] = a list.
            
            NOTE that len(res['idx']['N_modes'])+1=len(res['idx']['D'])
            
            idx is the index in ratios0.
        -------------------------------------
        input: D and departure_t and N_modes
            both are lists. D is the cumulative departure, begins from 0. len(D)=len(departure_t)=len(N_modes)+1, because the D begins from 0. 
            
        input: ratios0
            a list, constaing the share ratios among all movements. 
            if the sum of the ratio is not 1, then the function first will convert the sum to 1
        output: splited_D
            a dict. 
            splited_D['idx']['D'] = a list.
            splited_D['idx']['departure_t'] = a list.
            splited_D['idx']['N_modes'] = a list.
        """
        #returned value initialization.
        splited_D  = {}
        
        #make the sum to 1
        ratios  = np.array(ratios0)/(sum(ratios0))
        
        for i in range(len(D)):
            d  = D[i]
            dt = departure_t[i]
            if i==0:continue#because in the following, splited_D[i]['D'] = [0] is assigned.
            mode = N_modes[i-1]
            
            #random choose the idx
            idx = np.random.choice(range(len(ratios)),p=ratios)
            
            if not splited_D.has_key(idx):
                splited_D[idx] = {}
                splited_D[idx]['D'] = [0]
                splited_D[idx]['departure_t'] = [0]
                splited_D[idx]['N_modes'] = []
            #
            splited_D[idx]['D'].append(splited_D[idx]['D'][-1]+1)
            splited_D[idx]['departure_t'].append(dt)
            splited_D[idx]['N_modes'].append(mode)
            
        return splited_D
    
    def D_itera_cycle(self, some_rgp,yellowtype = 'static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        iter over each rgps
        -------------------------------------------
        output: 
            no output, but the following are changed:
                - self.D
                - self.departure_t
                - self.queue_t
                - self.Q
        """
        
        
        
        
        pass
    
    def debug_0(self, some_rgp):
        """
        
        """
        self.rgp = copy.deepcopy(some_rgp)
        
        self.current_idx = 0
        self.cycle_start_t = []
        self.cycle_end_t = []
        self.cycle_green_end_t = []
        self.D = []
        self.departure_t = []
        self.queue_t = []
        self.Q = []
        self.yellowtimes = []
        
        yellowtype = self.yellowtype
        
        #before hotizon
        self.D_assign_before_signal_horizon(some_rgp = some_rgp, yellowtype=yellowtype, redclearance=self.redclearance, speedlimit=self.speedlimit, reactiontime=self.reactiontime, startloss=self.startloss, acceleration=self.acceleration, lognormal=self.lognormal, sigmas=self.sigmas, mus=self.mus, fixedstartloss=self.fixedstartloss)
        
    def debug_onestep(self,):
        """"""
        yellowtype = self.yellowtype
        cyclesN = len(self.rgp.reds)
        t0, D0, Q0, greenduration = self.D_assign_red(some_rgp=self.rgp)
        #green
        departure_ts,queues_ts,queues = self.D_assign_green(some_rgp=self.rgp, t0=t0, D0=D0, Q0=Q0,greenduration=greenduration,yellowtype = yellowtype)
        self.current_idx = self.current_idx+1
        
    
    def D_construct_until_rgp(self,some_rgp, terminal_idx = 'sfaf', run_next_red = False):
        """
        Run the Dconstruct only part of the rgp cycles. 
        terminal_idx is the idx that the D_construct will still run.
        terminal_idx begins from 0. 
        For instance, if terminal_idx==0, then the D_construct process will only process one cycle.
        ------------------------------------------
        input: terminal_idx
            int, the range is from 0 to len(self.rgp.reds)-1.
            NOTE that the cycle terminal_idx is constructed.
        input: run_next_red
            boolean, if true, the next red is run. 
        """
        if isinstance(terminal_idx,int):
            if terminal_idx>len(some_rgp.reds)-1:
                raise ValueError('The terminal idxx should be smaller than len(rgp)-1')
        else:
            terminal_idx = len(some_rgp.reds)-1
        
        self.rgp = copy.deepcopy(some_rgp)
        
        self.current_idx = 0
        self.cycle_start_t = []
        self.cycle_end_t = []
        self.cycle_green_end_t = []
        self.D = []
        self.departure_t = []
        self.queue_t = []
        self.Q = []
        self.yellowtimes = []
        
        yellowtype = self.yellowtype
        #before hotizon
        self.D_assign_before_signal_horizon(some_rgp = some_rgp, yellowtype=yellowtype, redclearance=self.redclearance, speedlimit=self.speedlimit, reactiontime=self.reactiontime, startloss=self.startloss, acceleration=self.acceleration, lognormal=self.lognormal, sigmas=self.sigmas, mus=self.mus, fixedstartloss=self.fixedstartloss)
        #first cycle green
        for i in range(terminal_idx+1):
            #first cycle red
            t0, D0, Q0, greenduration = self.D_assign_red(some_rgp=self.rgp)
            #green
            departure_ts,queues_ts,queues = self.D_assign_green(some_rgp=self.rgp, t0=t0, D0=D0, Q0=Q0,greenduration=greenduration,yellowtype = yellowtype)
            self.current_idx = self.current_idx+1
        

    
    def D_construct(self, some_rgp, checkconservation = False):
        """
        Given the signal, construct the departure cumulative using self.H
        and self.arrival_t and self.N. 
        The role of:
            self.H, it is the minimum headway for different modes
            self.arrival_t and self.N are the arrival moment of the n-th vehicle. The two are identical. 
            
        
        ----------------------------------------
        Input: some_rgp
            RGP class. some_rgp.reds gives a list of each red signal, [r_start,r_end]
        input: yellowtype
            a str. either 'static' or 'dynamic'. If it is 'static', the following function will be called.
                self.D_construct_static_intergreen()
            if it is 'dynamic', the following function will be called:
                self.D_construct_dynamic_intergreen()
        input: checkconservation
            after each cycle termination, check the conservatio of vehicle numbers. The conservation means:
                self.D[-1]+self.Q[-1]==self.get_N[self.queue_t[-1]]
            
        ------------------------------------------
        Output: no output, but the following are constructed:
            self.departure_t
            self.D
            self.queue_t
            self.Q #the unit is vehicle number
        -----------------------------------------
        Steps:
        
        """
        self.rgp = copy.deepcopy(some_rgp)
        
        self.current_idx = 0
        self.cycle_start_t = []
        self.cycle_end_t = []
        self.cycle_green_end_t = []
        self.D = []
        self.departure_t = []
        self.queue_t = []
        self.Q = []
        self.yellowtimes = []
        
        yellowtype = self.yellowtype
        
        #before hotizon
        self.D_assign_before_signal_horizon(some_rgp = some_rgp, yellowtype=yellowtype, redclearance=self.redclearance, speedlimit=self.speedlimit, reactiontime=self.reactiontime, startloss=self.startloss, acceleration=self.acceleration, lognormal=self.lognormal, sigmas=self.sigmas, mus=self.mus, fixedstartloss=self.fixedstartloss)
        #first cycle green
        cyclesN = len(self.rgp.reds)
        for i in range(cyclesN):
            #first cycle red
            t0, D0, Q0, greenduration = self.D_assign_red(some_rgp=self.rgp)
            #green
            departure_ts,queues_ts,queues = self.D_assign_green(some_rgp=self.rgp, t0=t0, D0=D0, Q0=Q0,greenduration=greenduration,yellowtype = yellowtype)
            
            if checkconservation:
                #check the conservation of vehicles, if not satisfyed, then store the following:
                #   - part of D N Q
                self.conservationcheck()
            self.current_idx = self.current_idx+1
    
    
    
    def D_construct0(self, some_rgp,yellowtype = 'static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        Given the signal, construct the departure cumulative using self.H
        and self.arrival_t and self.N. 
        The role of:
            self.H, it is the minimum headway for different modes
            self.arrival_t and self.N are the arrival moment of the n-th vehicle. The two are identical. 
            
        
        ----------------------------------------
        Input: some_rgp
            RGP class. some_rgp.reds gives a list of each red signal, [r_start,r_end]
        input: yellowtype
            a str. either 'static' or 'dynamic'. If it is 'static', the following function will be called.
                self.D_construct_static_intergreen()
            if it is 'dynamic', the following function will be called:
                self.D_construct_dynamic_intergreen()
        ------------------------------------------
        Output: no output, but the following are constructed:
            self.departure_t
            self.D
            self.queue_t
            self.Q #the unit is vehicle number
        -----------------------------------------
        Steps:
        
        """
        
        if yellowtype=='static':
            self.D_construct_static_intergreen(some_rgp=some_rgp, redclearance = redclearance,  speedlimit  = speedlimit, reactiontime = reactiontime, startloss = startloss, acceleration = acceleration, lognormal = lognormal, sigmas = sigmas,mus = mus, fixedstartloss = fixedstartloss)
        elif yellowtype=='dynamic':
            self.D_construct_dynamic_intergreen(some_rgp=some_rgp, redclearance = redclearance,  speedlimit  = speedlimit, reactiontime = reactiontime, startloss = startloss, acceleration = acceleration, lognormal = lognormal, sigmas = sigmas,mus = mus, fixedstartloss = fixedstartloss)
        else:
            raise ValueError("The 'yellowtype' arg should be either 'static' or 'dynamic'")
        
        
    
    def Q_construct(self,):
        """
        onstruct the que length using 
            - arrivals, i.e. self.N self.arrival_t 
            - departures, i.e. self.D self.departure_t
        ----------------------------------------------
        Output: 
            no output, but the self.queue_t and self.Q are changed.
        ----------------------------------------
        Steps:
            the queue is the difference between arrival (self.N) and departure (self.D).
        """
        
        moments = sorted(set(self.arrival_t + self.departure_t))
        self.Q = [self.get_N(t)-self.get_D(t) for t in moments]
        self.queue_t = copy.deepcopy(moments)
    
    def get_delay_using_queue(self,):
        """
        compute the delay.
        Different from self.get_delay() in that, the self.get_delay() using the arrival moment and departure moment of each vehicle and then get the delay of this vehicle in each cycle. 
        
        this function, self.get_delay_using_queue(), using the self.Q and self.queue_t. 
        --------------------------------------
        input: 
            no input, but the following are required:
                self.Q
                self.queue_t (the two should have the same length).
        Output: delay
            delay[cycle_idx] = d, d is a scalalr. 
        ---------------------------------------------
        Steps:
            - for each cycle
                - find the start_idx and end_idx of the cycle. 
                - using self.Q*delta_t as the delay measurement.
        """
        delays  = {}
        for cycle_idx in range(len(self.rgp.reds)):
            #get the start moment and ending moment of this cycle.
            #   the start momnt is the red start
            #   the end momen is the moment of the next start.
            start_t = self.cycle_start_t[cycle_idx]
            end_t = self.cycle_end_t[cycle_idx]
            
            #get the start index and ending index of in the self.queue_t
            start_idx = self.find_idxx_greaterthan(self.queue_t, start_t, equal = True)
            end_idx  = self.find_idxx_smallerthan(self.queue_t, end_t, equal = True)
            if isinstance(start_idx,list):
                delays[cycle_idx] = 0
                continue
            if isinstance(end_idx,list):
                end_idx  = len(self.queue_t)-1
            
            #delta_ts[i] = self.queue_t[i+1]-self.queue_t[i]
            delta_ts  = np.diff(self.queue_t[start_idx:end_idx+1])
            qs = np.array(self.Q[start_idx:end_idx])
            delays[cycle_idx] = sum(delta_ts*qs)
            
            
        return delays
    
    def get_yellow_time(self,speedlimit  =40, reactiontime = 1.0, acceleration = 3.0):
        """
        
        """
        speed = 1.0*speedlimit/3.6
        return reactiontime + speed/(2.0*acceleration)
        
    
    def conservationcheck(self,equal = True):
        """
        store the part of self.D self.N self.Q in self.tmp.
        It is a dict. 
        --------------------------------------
        Basic ideas:
            - at the end of red signal, the following two quantities should be identical:
                * self.D[-1]+self.Q[-1]
                * self.get_N(self.queue_t[-1])
        """
        #store the variables that for debug. 
        #   res include the following:
        res = {}
        if not self.D[-1]+self.Q[-1]==self.get_N(self.queue_t[-1]):
            #the D 
            #   self.current_idx is the current cycle index in self.rgp.reds
            idxxes = self.get_idxxes_in_cycle(self.departure_t,  idxx = self.current_idx, equal = equal)
            res['D'] = pd.DataFrame([np.array(self.departure_t)[idxxes],np.array(self.D)[idxxes]])
            
            #the N
            idxxes = self.get_idxxes_in_cycle(self.arrival_t,  idxx = self.current_idx, equal = equal)
            res['N'] = pd.DataFrame([np.array(self.arrival_t)[idxxes],np.array(self.N)[idxxes]])
            
            #the Q
            idxxes = self.get_idxxes_in_cycle(self.queue_t,  idxx = self.current_idx, equal = equal)
            res['Q'] = pd.DataFrame([np.array(self.queue_t)[idxxes],np.array(self.Q)[idxxes]])
        
            self.tmp = res
            
            raise ValueError('Conservation is broken.')
            
    
    def get_delay(self):
        """
        Obetain the delay, both the cyclic delay and the overall delay.
        Delay in one cycle definition:
            The delay within one cycle is defined as the area surrounded by 1) arrival curve 2) departure curve; 3) Cycle start and 4) cycle end
        Average delay within one cycle is defined as the delay within this cycle divided by the vehicles number
            the vehicles number is defined as the vehicles have trajectory in this cycle
        Total delay
            is defined as the overall delay
        Average delay is defined as
            
        --------------------------------------
        Input: 
            no input, using the 
        Output: delay
            delay[cycle_idx] = [d1,d2,d3,,,,]
            di is the i-th delayed vehicles in cycle cycle_idx    
        ------------------------------------
        Step:
            - for each vehicle
                - get the enter moment an exitmoment
                    - find the start cycle idx and end cycle index
                    - assign the delay to each cycle. 
        """
        #dict is cycle index. 
        #   cycles_delay[0]=[d1,d2,d3...]
        #   length is the vehicles and sum the the overal delay for this cycle
        cycles_delay = {}
        #for each red duration
        for car_idx in range(self.N[-1]):
            #enxtering_t and exiting_t
            entering_t = self.get_arrival_t(car_idx)
            exiting_t = self.get_departure_t(car_idx)
            exiting_t = max(0,exiting_t)
            #get the entering cycle idx and exiting cycle idx
            entering_idx = self.rgp.idx_rgp(entering_t)
            exiting_idx = self.rgp.idx_rgp(exiting_t)
            for cycle_idx in range(entering_idx,exiting_idx+1):
                if not cycles_delay.has_key(cycle_idx):
                    cycles_delay[cycle_idx] = []
                r0 = self.rgp.StartMomentsOfReds[cycle_idx]
                r2 = self.rgp.NextCycleStartMoment(cycle_idx)
                cycles_delay[cycle_idx].append(min(exiting_t, r2)-max(r0,entering_t))
        
        return cycles_delay


class PSO():
    """
    particle swarm optimization of the intersection signals. 
    
    """
    #w in [0,1.2]
    w = .5
    #c1,c2 in [0,2], r1 and r2 in [0,1]
    c1 = .5
    c2 = .5
    
    
    #IntersectionTraffic instance. 
    #   intersection_flows.movements is a dict. 
    intersection_flows = IntersectionTraffic()
    
    popu_size = 100
    populations = [0]*popu_size
    neighbores = {}#each value is a pd.Index 
    populations_velocity = [0]*popu_size
    populations_distances = []#it should be a pd.DataFrame data. 
    populations_fitnesses = [np.inf]*popu_size
    populations_velocities =  [0]*popu_size
    population_history_best  = [0]*popu_size
    population_history_fit = [np.inf]*popu_size
    #best among the neighbores
    currentbest_idx_populations = [0]*popu_size
    currentbest_fit_populations = [np.inf]*popu_size
    #best of the currrent iteration and globest.
    currentbest_idx = []
    currentbest_fit = np.inf
    globalbest = 0
    globalbest_fit = np.inf
    
    T_horizon=3600.0
    
    results_fits = []
    
    #used in distance calculation
    #   scipydistance.euclidean(u,v)
    
    def Distances(self, typee = 'eculid'):
        """
        compute the distances between individuals and the self.populations_distances is changed. It will be used to find the neighbores of one individual. 
        ----------------------------------------
        input: typee
            a str. the distance type. 'eculid' is most common distance measurement.
       output:
            no output. But the following are changed:
                self.populations_distances
        """
        
        #initialize the self.populations_distances to DataFrame.
        #   self.populations_distances
        self.populations_distances  = pd.DataFrame( np.zeros((self.popu_size,self.popu_size)),index = range(self.popu_size),columns = range(self.popu_size))
        
        for i,j in itertools.combinations(range(self.popu_size),r=2):
            if typee=='eculid':
                d = scipydistance.euclidean(self.populations[i],self.populations[j])
                self.populations_distances.loc[i,j]  = d
                self.populations_distances.loc[j,i]  = d
    
    
    
    def get_neighbores(self, percent =  1):
        """
        get the neighbores of an individual using self.populations_distances. 
        -----------------------------------
        input: percent
            a scalar within (0,1]. percent of the neighbores. If it is 1, then all the individuals are neighbores. 
        output: neighbores
            a dict. keys are range(self.popu_size). 
            neighbores[0] is a pd.Index, containing all the neighbores's index. 
        """
        self.neighbores = {}
        #idxx is the rank that within neighbores. 
        idxx = max(1,percent*self.popu_size)
        for i in range(self.popu_size):
            #find the neighbores. 
            #   first rank the distances. 
            #       rank is a series, the index is the same as self.populations_distances.loc[i,:]
            #       the value is from 1 to n
            rank = self.populations_distances.loc[i,:].rank(ascending=True)
            
            #   get the index of neighbores.
            #       neighbores_idx is a pd.Index instance. 
            neighbores_idx = rank[rank<=idxx].index
            self.neighbores[i] = neighbores_idx
            
    def TaguchiOrthogonalDesign(self, TaguchiArray, FlowLevels, idx):
        """"""
        
        
        pass
        
    def __init__(self,H = H, \
    batch_arrival_es = batch_arrival_es, \
    interval_mean_es = interval_mean_es, \
    arrival_mode_proba_es = arrival_mode_proba_es, \
    intersection_signal = MD.SignalParameters(), \
    T_horizon = 3600.0,\
    modes = ['A','B','C'],yellowtype='static', \
    lognormal=False,fixedstartloss=False):
        """
        NOTE:
            IntersectionTraffic.__init__(self, H = H, \
            batch_arrival_es = batch_arrival_es, \
            interval_mean_es = interval_mean_es, \
            arrival_mode_proba_es = arrival_mode_proba_es, \
            intersection_signal = MD.SignalParameters(), \
            T_horizon = 3600.0, \
            modes = ['A','B','C'],yellowtype = 'static')
        ------------------------------------
        input: T_horizon
            the hotozon of the vehicle arrival. 
        """
        
        self.T_horizon = T_horizon
        self.intersection_flows = IntersectionTraffic(H=H,batch_arrival_es = batch_arrival_es, interval_mean_es = interval_mean_es, arrival_mode_proba_es = arrival_mode_proba_es, T_horizon = T_horizon,intersection_signal=intersection_signal, yellowtype=yellowtype,lognormal=lognormal, fixedstartloss=fixedstartloss)
    
    
    def reformulatesignal(self,solution,signal=MD.SignalParameters()):
        """
        --------------------------------------
        input: signal
            a MD.SignalParameters() instance. signal.gmin and signal.gmax
        input: solution
            one solution of the pso. It should be a pd.Series data. 
            The index is the phase, i.e. signal.phases[i].
            The values are the green durations. 
        """
        
        
        pass
    
    
    def fitness_population(self, pi = 'delay', yellowtype='static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        compute the fitness of the population. 
        self.populations_fitnesses are changed. 
        
        """
        self.populations_fitnesses = []
        for idxx in range(len(self.populations)):
            popu = self.populations[idxx]
            #compute the res, a scalar
            temp_res =  self.fitness_individual(individual=popu, pi = pi, yellowtype = yellowtype, redclearance = redclearance, speedlimit  = speedlimit, reactiontime = reactiontime, startloss = startloss, acceleration = acceleration, lognormal = lognormal, sigmas = sigmas,mus = mus, fixedstartloss = fixedstartloss)
            #store the res
            self.populations_fitnesses.append(temp_res)
        #store the best in the self.results_fits
        if pi=='delay':
            self.results_fits.append(min(self.populations_fitnesses))
            
    
    def fitness_individual(self, individual, pi = 'delay', yellowtype='static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        compute the fitness of each population
        ----------------------------------------------
        
        
        input: individual
            an individual of the pso. It should be a pd.Series
        input: lognormal and fixedstartloss. 
            Parameters used in MovementTraffic instance. 
            Bool. whether the headway obey the lognormal distirbuton. 
        input: fixedstartloss
            Bool. if fixedstartloss==False, it means that the start loss is dynamic:
                - if the first vehicle is cav, there is no start loss, and the first cav can pass the stopline exactly when the green starts. 
        ------------------------------------------
        output: fitness
            a scalar. 
        ------------------------------------------
        steps:
            - first convert the signals (in the format of pd.Series) to the RGP instance. 
            - input the RGP instances to each movement and then get the pi
        """
        if pi=='delay':
            #Step1: convert the individual to RGP
            movements_rgps = MD.SignalParameters().To_rgps(individual, T = self.T_horizon)
            
            #Step2: input the signal to MovementTraffic instance and
            #   construc the cumulative departure and queue
            for m in self.intersection_flows.movements.keys():
                #movement assignment
                movement = self.intersection_flows.movements[m]
                #get the departure and queue in 
                #   movement.departure_t, movement.D
                #   movement.queue_t, movement.Q
                movement.D_construct(some_rgp =movements_rgps[m])
                
            #get the  pi of the movement. 
            totaldelay,averagedelay  = self.intersection_flows.get_pi(pi = pi)
            
                    
            return averagedelay
            
            pass
        
        
        pass
    def r1(self,):
        return np.random.random()
    
    def r2(self,):
        return np.random.random()
    
    
    def velocity_and_move(self,):
        """
        Compute the velocity of one indiviudual. 
        
        """
        #calculate the velocity to self.populations_velocities
        for i in range(self.popu_size):
            #get the distances to the current best 
            d_best = self.populations[self.currentbest_idx_populations[i]]-self.populations[i]
            #get the distance to the individual history best. 
            d_history_best = self.population_history_best[i] -self.populations[i]
            #calculate and assign the velocity
            self.populations_velocities[i]  = self.w*self.populations_velocities[i] + self.c1*self.r1()*d_best + self.c2*self.r2()*d_history_best
        
        #move
        for i in range(self.popu_size):
            self.populations[i] = self.populations[i] + self.populations_velocities[i]
            #confine the solution within gmin and gmax
            #   tmp_signal is a MD.SignalParameters() instance
            self.populations[i][self.populations[i]<tmp_signal.gmin]=tmp_signal.gmin
            self.populations[i][self.populations[i]>tmp_signal.gmax]=tmp_signal.gmax
            
    def termination(self,):
        """
        compare the history best and current best and return the res
        """
        
        
        
        pass
    
    def Best_assign(self,):
        """
        assign the (a)current best and (b)indiviaual history best
        Based on the fllowing:
            self.populations_fitnesses
            self.
        -----------------------------------------------
        output: 
            no output, but the following are changed:
                - self.currentbest_idx_populations
                - self.currentbest_fit_populations
                - self.population_history_best
                - self.population_history_fit
                - self.globalbest
                - self.globalbest_fit
        """
        #current best among neighbores
        #   get the idx of the minimal fit
        for i in range(self.popu_size):
            #all neighbores
            indexes = self.neighbores[i]
            #find the minimal
            minidxx = pd.Series(self.populations_fitnesses)[indexes].idxmin()
            self.currentbest_idx_populations[i] = minidxx
            self.currentbest_fit_populations[i] = self.populations_fitnesses[minidxx]
        
        #indiviaual history best
        for i in range(self.popu_size):
            if self.populations_fitnesses[i]<self.population_history_fit[i]:
                self.population_history_best[i]  = copy.deepcopy(self.populations[i])
                self.population_history_fits = self.populations_fitnesses[i]
        
        #global best and global fit
        #the current best
        minidxx = pd.Series(self.populations_fitnesses).idxmin()
        if self.populations_fitnesses[minidxx]<self.globalbest_fit:
            self.globalbest = self.populations[minidxx]
            self.globalbest_fit = self.populations_fitnesses[minidxx]
        
    def update(self):
        """
        update the solution using :
            self.velocity
            self.populations
        """
        for i in range(self.popu_size):
            self.populations[i] = self.populations[i] + self.populations_velocity[i]
    
    
    def test_onestep(self,itera = 1000, neighborespercent = 1,plan_id = 0, popu_size = 100,pi = 'delay', yellowtype='static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        
        """
        print('fitness')
        #compute the fitness
        self.fitness_population(pi =pi, yellowtype=yellowtype, redclearance = redclearance, speedlimit  =speedlimit, reactiontime = reactiontime, startloss = startloss, acceleration = acceleration, lognormal = lognormal, sigmas = sigmas,mus = mus, fixedstartloss = fixedstartloss)
        
        #distances
        print('distances')
        self.Distances(typee = 'eculid')
        
        print('get_neighbores')
        self.get_neighbores(percent =  neighborespercent)
        
        print('Best_assign')
        self.Best_assign()
        
        #move
        print('velocity_and_move')
        self.velocity_and_move()
        
        pass
    
    def test_init(self,itera = 1000, neighborespercent = 1,plan_id = 0, popu_size = 100,pi = 'delay', yellowtype='static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        
        
        """
        self.popu_size = popu_size
        self.populations = [0]*popu_size
        self.neighbores = {}#each value is a pd.Index 
        self.populations_velocity = [0]*popu_size
        self.populations_distances = []#it should be a pd.DataFrame data. 
        self.populations_fitnesses = [np.inf]*popu_size
        self.populations_velocities =  [0]*popu_size
        self.population_history_best  = [0]*popu_size
        self.population_history_fit = [np.inf]*popu_size
        #best among the neighbores
        self.currentbest_idx_populations = [0]*popu_size
        self.currentbest_fit_populations = [np.inf]*popu_size
        #best of the currrent iteration and globest.
        self.currentbest_idx = []
        self.currentbest_fit = np.inf
        self.globalbest = 0
        self.globalbest_fit = np.inf
        
        
        
        #set the self.popu_size and self.populations
        self.pso_initialization(plan_id = plan_id, popu_size = popu_size)
        
    
    def pso(self,itera = 1000, neighborespercent = 1,plan_id = 0, popu_size = 100,pi = 'delay', yellowtype='static', redclearance = 1.0, speedlimit  =40, reactiontime = 1.0, startloss = .5, acceleration = 3.0, lognormal = False, sigmas = sigmas,mus = mus, fixedstartloss = False):
        """
        
        -------------------------------
        Call steps:
            self.pso_initialization()
            self.fitness_population()
            self.Distances()
            self.DistancesBest()
            self.velocity_populations()
            self.update()
        """
        #set the self.popu_size and self.populations
        self.pso_initialization(plan_id = plan_id, popu_size = popu_size)
        
        #compute the fitness
        self.fitness_population(pi =pi, yellowtype=yellowtype, redclearance = redclearance, speedlimit  =speedlimit, reactiontime = reactiontime, startloss = startloss, acceleration = acceleration, lognormal = lognormal, sigmas = sigmas,mus = mus, fixedstartloss = fixedstartloss)
        
        #distances
        self.Distances(typee = 'eculid')
        self.get_neighbores(percent =  neighborespercent)
        self.Best_assign()
        
        #move
        self.velocity_and_move()
        
        for i in range(itera):
            #compute the fitness
            self.fitness_population(pi =pi, yellowtype=yellowtype, redclearance = redclearance, speedlimit  =speedlimit, reactiontime = reactiontime, startloss = startloss, acceleration = acceleration, lognormal = lognormal, sigmas = sigmas,mus = mus, fixedstartloss = fixedstartloss)
            
            #distances
            self.Distances(typee = 'eculid')
            self.get_neighbores(percent =  neighborespercent)
            self.Best_assign()
            
            #move
            self.velocity_and_move()
            
    
    def pso_initialization(self, plan_id = 0, popu_size = 100):
        """
        assign the populations. 
        --------------------------------------------------
        input: plan_id
            the id of the 
        output:
            no output. But the following are set:
                - self.populations (it is a list. Element is a )
                - self.populations_velocities (it is a list. )
        """
        self.popu_size = popu_size
        #initialize the self.populations
        #   each element is a pd.Series
        self.populations = [self.intersection_flows.RandomSampleCyclePlan(plan_id=plan_id) for i in range(popu_size)]
    
    def optimization(self, objective = 'delay', typee = 'minimize',neighboring = 1.0):
        """
        ----------------------------------------
        input: objective
            a str, the objective of the pso. 
            Can be 'delay', 'capacity',
        input: typee
            a str. Either 'minimize' or 'maximize'. 
        Input: neighboring
            the neighbors of one individual. 1 means 100%. If neighboring=.5, it will select the half of the populations as neighbores. 

        --------------------------------------
        output: 
            population_historys.
                history of each individual.
            population_history_fits.
                history of the fitness of each individual. 
            best_individual.
                the best individual. It should be a pd.Series. 
                The indexes are phases index in signal
                The values are green durations. 
            best_fitness
                the fitness of the best_individual.
            
        """
        
        
        
        pass
    
    pass

class TwoIntersectionsTraffic():
    """
    the class for two adjacent intersections. The approach for the two intersections all have three lanes, each for left-turn, through and right-turn. 
    
    Distance is in self.distance
    offset is in self.offset
    -----------------------------------------
    Importat attributes: intersectiontrafficflow
        self.intersectiontraffics[intersectionlabel] is an IntersectionTrafic instance.
        self.intersectiontraffics[intersectionlabel].movements['east_lest'] is a MovementTraffic instance.
    Important attributes: signal
        - pd.Series format:
            self.intersectionssignals[intersectionlabel] is a pd.Series
        - RGP format:
            self.movementsrgps_dict[intersectionlabel]['east_left'] is a RGP instances. 
        - offset
            self.offset[intersectionlabel] is a scalar.
    
    """
    
    #a list containing all the coordinated movements. 
    #   each element is a list, which means one coordination directions.
    #       coordinatemovements[i]=[(intersection_label1, movement_label1),(intersection_label2, movement_label2),...]
    #       (intersection_label1, movement_label1) is the first movement in the starting intersection
    coordinates = [[(0,'west_through'),(1,'west_through')],[(1,'east_through'),(0,'east_through')]]
    #offset of the second section with respect to the first one.
    #the distance between the two intersections. unit is meter.
    distance = 400#m
    speedlimit = 40#km/h
    traveltime = 0#unit is sec
    
    #each movemelts are '1_east_left', means intersection 1, the east approach, left-turn movement.
    #   number of incoming movements number
    #InputMovements_N = {'1_west_through':2, '1_west_left':2, '1_west_right':2,'0_east_left':2, '0_east_through':2, '0_east_right':2}
    #   number of each incoming movements
    #InputMovements = {'1_west_left':['0_west_through_','0_north_left'],'1_west_through':['0_west_through_','0_north_left'], '0_east_left':['1_']}
    
    #TurningRatio[movement1, movement2] is a scalar, the share of movement1 that will input to movement2. For a movement M, if M is not in the TurningRatios.index, then it means there is no output for M. 
    TurningRatios = pd.DataFrame()
    TurningRatios.loc['0_west_through','1_west_left']=.3
    TurningRatios.loc['0_west_through','1_west_through']=.3
    TurningRatios.loc['0_north_left','1_west_left']=.3
    TurningRatios.loc['0_north_left','1_west_through']=.3
    #   another input
    TurningRatios.loc['1_east_through','0_east_left']=.3
    TurningRatios.loc['1_east_through','0_east_through']=.3
    TurningRatios.loc['1_south_left','0_east_left']=.3
    TurningRatios.loc['1_south_left','0_east_through']=.3
    
    def __init__(self,speedlimit = 40, distance = 40):
        """
        
        --------------------------------
        input: speedlimit
            unit is km/h
        input: distance
            unit is meter
        """
        self.speedlimit = speedlimit
        self.distance =  distance
        self.traveltime = 1.0*distance/(speedlimit/3.6)
        #each intersectiontraffics value is an IntersectionTraffic instance.
        #   the key is the intersection label
        self.intersectiontraffics = {0:IntersectionTraffic(),1:IntersectionTraffic()}
        #   the keys should be consistent with self.intersectiontraffics
        self.intersectionssignals = {0:pd.Series(),1:pd.Series()}
        #self.offset is a dict
        self.offset = {0:0,1:10}
        
        #no waitingmovements and waiting movements
        
        #
    
    def Benchmark_SFR_calculated(self, TurningRatios, Demands_dict, plan_id=3, t_loss = 3, traveltime = 15, offset = False, vf = 50, tao_HV = 1.3, tao_CAV = .1, S = 1800.0):
        """
        signal settings for benchamrk plan.
        Different from self.Benchmark() in that, self.Benchmark() uses the fixed saturation flow rate. And self.Benchmark_SFR_calculated() use the weighted saturated, based on the ratio of HVs, i.e.:
            SFR = w_HV*3600/headway_HV + w_CAV*3600/headway_CAV
        w_HV is the ratio of HVs and w_CAV is the ratio of CAVs. 
        headway_HV is the stable headway of HV and 
        headway_CAV is the stable headway of CAV. 
        
        the benchamrk plan will gives the following results:
            - self.offset, a dict, keys are intersection labels;
            - self.intersectionssignals, a dict, keys are intersection labels., each value is a pd.Series data
        ---------------------------------------------------
        input: TurningRatios
            pd.DataFrame, columns and index are like '0_west_through', 0 is intersection id, west_through is movement. Sum of row is 1.
        input: Demands_dict
            dict, demands. Keys are indeterction ids. Each value is like:
                (H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es)
            Except H, the rest three are all dict, keys are like 'west_through'.
            
        input: t_loss
            loss time for each phase.
        input: plan_id
            the id of the plans in MD.Signalparameters.plans (a tuple.)
            plan_id =0, is the one-phase-one-approach
            plan_id = 3, is the commong four-leg plan
        input: traveltime
            a scalar, which is used in the offset optimization.
        input: vf tao_HV and tao_CAV
            the paramters used to calculate the stable headway for human drivers (HV) and CAVs. 
            vf is the speed limit, unit is km/h
            tao_HV and tao_CAV are reaction time, unit is sec. 
            
            newell.saturationheadway_theoretical(tao = .1, d= 7.5, vf=50.0) will give the headways. 
        output:
            no output, but the following are changed:
                self.intersectionssignals, a dict, {intersection_id:pd.Series}
                self.offset, a dict, intersection_id:offset_value, offset of the first intersection is set to zero. 
        -------------------------------------------------------
        Steps:
            - Find the flow rate for each movement at each intersection in Flowrates_dict
            - compute the SFR based on 
            - Find the Y for each phases, in Y_frame,
        """
        #stable headways used in the weighted SFR
        headway_HV = newell.saturationheadway_theoretical(tao = tao_HV, vf = vf)
        headway_CAV = newell.saturationheadway_theoretical(tao = tao_CAV, vf = vf)
        
        #Get Flowrates_dict and SFR_dict. The struct is the same. 
        #Flowrates_dict.loc[node_id,movement] = float, flow rate, unit is veh/h
        #    Flowrates_dict.loc[0,'east_through']
        #    SFR_dict.loc[0,'east_through'] is also float.
        Flowrates_dict = pd.DataFrame()#
        #   used in the weighted SFR calculating. 
        Flowrates_dict_HV = pd.DataFrame()
        Flowrates_dict_CAV = pd.DataFrame()
        SFR_dict = pd.DataFrame()
        
        #Find the flow rate for each movement at each intersection
        for node_id in Demands_dict.keys():
            #Demands_dict[node_id] = H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es
            for mt_label in Demands_dict[node_id][1].keys():
                
                #if True, means that this movement is waiting for input
                if str(node_id) + '_' + mt_label in self.TurningRatios.columns:
                    Flowrates_dict.loc[node_id,mt_label] = 0
                    Flowrates_dict_HV.loc[node_id,mt_label] = 0
                    Flowrates_dict_CAV.loc[node_id,mt_label] = 0
                    continue
                H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es = Demands_dict[node_id]
                arrival_t, N, N_modes = MovementTraffic.GenerateArrival(H =H,interval_mean=interval_mean_es[mt_label], arrival_mode_proba = arrival_mode_proba_es[mt_label], batch_arrival = batch_arrival_es[mt_label], T_horizon = 3600.0,)
                
                #get the flow rate in Flowrates_dict.
                Flowrates_dict.loc[node_id,mt_label] = 3600.0*N[-1]/(arrival_t[-1]-arrival_t[1])
                
                #   Flowrates_dict_HV and Flowrates_dict_CAV
                Flowrates_dict_HV.loc[node_id,mt_label] = Flowrates_dict.loc[node_id,mt_label]*arrival_mode_proba_es[mt_label]['A']/(sum(arrival_mode_proba_es[mt_label].values()))
                Flowrates_dict_CAV.loc[node_id,mt_label] = Flowrates_dict.loc[node_id,mt_label] - Flowrates_dict_HV.loc[node_id,mt_label]
                
                #get the weighted SFR in Flowrates_dict
                r_HV = arrival_mode_proba_es[mt_label]['A']/sum(arrival_mode_proba_es[mt_label].values())
                r_CAV = 1 - r_HV
                SFR_dict.loc[node_id,mt_label] = r_HV*3600.0/headway_HV + r_CAV*3600.0/headway_CAV
                
        #for movements that wait for flow input
        for node_id in Demands_dict.keys():
            #Demands_dict[node_id] = H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es
            for mt_label in Demands_dict[node_id][1].keys():
                
                #if True, means that this movement is waiting for input
                compound_label = str(node_id)+ '_' + mt_label
                if compound_label not in self.TurningRatios.columns:continue
                
                #incoming movements labels set, in effectivelabels
                effectivelabels = self.TurningRatios.loc[:,compound_label].index[self.TurningRatios.loc[:,compound_label].notna()]
                
                tmp_sum = 1.0*sum(self.TurningRatios.loc[effectivelabels,compound_label])
                for incoming in effectivelabels:
                    #note Flowrates_dict.loc[node_id,mt_label] is initial to zero in the above. 
                    #   incoming[2:] is 'east_left', because income format is like '0_east_left'
                    Flowrates_dict.loc[node_id,mt_label] = Flowrates_dict.loc[node_id,mt_label] + Flowrates_dict.loc[int(incoming[0]),incoming[2:]]*self.TurningRatios.loc[incoming,compound_label]/tmp_sum
                    
                for incoming in effectivelabels:
                    #Flowrates_dict_HV and Flowrates_dict_CAV
                    #   Demands_dict[node_id][2] is arrival_mode_proba_es,
                    #       arrival_mode_proba_es['east_left']
                    r_HV = Demands_dict[int(incoming[0])][2][incoming[2:]]['A']/sum(Demands_dict[int(incoming[0])][2][incoming[2:]].values())
                    r_CAV = 1 - r_HV
                    #   Flowrates_dict_HV
                    Flowrates_dict_HV.loc[node_id,mt_label] = Flowrates_dict_HV.loc[node_id,mt_label] + Flowrates_dict.loc[int(incoming[0]),incoming[2:]]*r_HV*self.TurningRatios.loc[incoming,compound_label]/tmp_sum
                    #   Flowrates_dict_CAV
                    Flowrates_dict_CAV.loc[node_id,mt_label] = Flowrates_dict_CAV.loc[node_id,mt_label] + Flowrates_dict.loc[int(incoming[0]),incoming[2:]]*r_CAV*self.TurningRatios.loc[incoming,compound_label]/tmp_sum
                
                #compute SFR
                #   first compute the cav ratio and HV ratio
                r_CAV = Flowrates_dict_CAV.loc[node_id,mt_label]/Flowrates_dict.loc[node_id,mt_label]
                r_HV = 1 - r_CAV
                #   then computer the SFR
                SFR_dict.loc[node_id,mt_label] = r_HV*3600.0/headway_HV + r_CAV*3600.0/headway_CAV
        
        #intiailize the singals in self.intersectionssignals.
        #   keys are intersection ids, and values are pd.Series. 
        self.RandomSampleAllSignals(plan_id = plan_id)
        #   a dict, 
        Ys = copy.deepcopy(self.intersectionssignals)
        for node_id in Ys.keys():
            for phase_id in Ys[node_id].index:
                #find all the movements belong to the phase
                tmp_movements = MD.DeltaPhasesMovements.loc[phase_id,:][MD.DeltaPhasesMovements.loc[phase_id,:]].index
                #Ys[node_id][phase_id] = max(Flowrates_dict.loc[node_id,tmp_movements])/(1.0*S)
                Ys[node_id][phase_id] = max(Flowrates_dict.loc[node_id,tmp_movements].div(SFR_dict.loc[node_id,tmp_movements]))
        
        #determine C, clcyel length
        C = 0
        for node_id in Ys.keys():
            C_max = (MD.SignalParameters.gmax+t_loss)*len(Ys[node_id])
            C_min = (MD.SignalParameters.gmin+t_loss)*len(Ys[node_id])
            C = max(C_min,min(C_max, len(Ys[node_id])*t_loss/max(1-sum(Ys[node_id]),.0001)))
        
        #determin g in self.intersectionssignals[node_id]
        for node_id in Ys.keys():
            self.intersectionssignals[node_id] = C*(Ys[node_id]/sum(Ys[node_id]))
            #self.intersectionssignals[node_id][self.intersectionssignals[node_id]<MD.SignalParameters.gmin] = MD.SignalParameters.gmin
            #self.intersectionssignals[node_id][self.intersectionssignals[node_id]>MD.SignalParameters.gmax] = MD.SignalParameters.gmax
        
        #determing the offset
        if offset:
            print 'Now try to optimize and set offset in self.offset'
        #offset optimization
            bandwidths,offsets  =self.optimization_offset_2intersections(coordinates = self.coordinates, traveltime=traveltime)
            self.offset[1] = offsets[bandwidths.index(max(bandwidths))]
        
        #Flowrates_dict, Ys, SFR_dict, Flowrates_dict_CAV,Flowrates_dict_HV
        return Flowrates_dict, Ys, SFR_dict, Flowrates_dict_CAV,Flowrates_dict_HV
    
    
    def Benchmark(self, TurningRatios, Demands_dict, plan_id=3, t_loss = 3, S  =1800.0,traveltime = 15, offset = False):
        """
        signal settings for benchamrk plan.
        the benchamrk plan will gives the following results:
            - self.offset, a dict, keys are intersection labels;
            - self.intersectionssignals, a dict, keys are intersection labels., each value is a pd.Series data
        ---------------------------------------------------
        input: TurningRatios
            pd.DataFrame, columns and index are like '0_west_through', 0 is intersection id, west_through is movement. Sum of row is 1.
        input: Demands_dict
            dict, demands. Keys are indeterction ids. Each value is like:
                H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es
            Except H, the rest three are all dict, keys are like 'west_through'.
        input: t_loss
            loss time for each phase.
        input: S
            the saturation flow rate, unit is 1800 veh/h.
        input: plan_id
            the id of the plans in MD.Signalparameters.plans (a tuple.)
            plan_id =0, is the one-phase-one-approach
            plan_id = 3, is the commong four-leg plan
        input: traveltime
            a scalae, which is used in the offset optimization.
        output:
            no output, but the following are changed:
                self.intersectionssignals, a dict, {intersection_id:pd.Series}
                self.offset, a dict, intersection_id:offset_value, offset of the first intersection is set to zero. 
        -------------------------------------------------------
        Steps:
            - Find the flow rate for each movement at each intersection in Flowrates_dict
            - Find the Y for each phases, in Y_frame,
        """
        #Get Flowrates_dict
        #Flowrates_dict.loc[node_id,movement] = float, flow rate, unit is veh/h
        #   Flowrates_dict.loc[0,'east_through']
        Flowrates_dict = pd.DataFrame()
        #Find the flow rate for each movement at each intersection
        for node_id in Demands_dict.keys():
            #Demands_dict[node_id] = H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es
            for mt_label in Demands_dict[node_id][1].keys():
                #if True, means that this movement is waiting for input
                if str(node_id)+ '_' + mt_label in self.TurningRatios.columns:
                    Flowrates_dict.loc[node_id,mt_label] = 0
                    continue
                H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es = Demands_dict[node_id]
                arrival_t, N, N_modes = MovementTraffic.GenerateArrival(H =H,interval_mean=interval_mean_es[mt_label], arrival_mode_proba = arrival_mode_proba_es[mt_label], batch_arrival = batch_arrival_es[mt_label], T_horizon = 3600.0,)
                Flowrates_dict.loc[node_id,mt_label] = 3600.0*N[-1]/(arrival_t[-1]-arrival_t[1])
        #for movements that wait for flow input
        for node_id in Demands_dict.keys():
            #Demands_dict[node_id] = H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es
            for mt_label in Demands_dict[node_id][1].keys():
                #if True, means that this movement is waiting for input
                compound_label = str(node_id)+'_' + mt_label
                if compound_label not in self.TurningRatios.columns:continue
                #in coming movements
                effectivelabels = self.TurningRatios.loc[:,compound_label].index[self.TurningRatios.loc[:,compound_label].notna()]
                tmp_sum = 1.0*sum(self.TurningRatios.loc[effectivelabels,compound_label])
                for incoming in effectivelabels:
                    Flowrates_dict.loc[node_id,mt_label] = Flowrates_dict.loc[node_id,mt_label] + self.TurningRatios.loc[incoming,compound_label]*Flowrates_dict.loc[int(incoming[0]),incoming[2:]]/tmp_sum
        
        #intiailize the singals in self.intersectionssignals.
        #   keys are intersection ids, and values are pd.Series. 
        self.RandomSampleAllSignals(plan_id = plan_id)
        #   a dict
        Ys = copy.deepcopy(self.intersectionssignals)
        for node_id in Ys.keys():
            for phase_id in Ys[node_id].index:
                #find the movements of the phase
                tmp_movements = MD.DeltaPhasesMovements.loc[phase_id,:][MD.DeltaPhasesMovements.loc[phase_id,:]].index
                Ys[node_id][phase_id] = max(Flowrates_dict.loc[node_id,tmp_movements])/(1.0*S)
        
        #determine C, clcyel length
        C = 0
        for node_id in Ys.keys():
            C_max = MD.SignalParameters.gmax*len(Ys[node_id])
            C = min(C_max, len(Ys[node_id])*t_loss/max(1-sum(Ys[node_id]),.0001))
        
        #determin g in self.intersectionssignals[node_id]
        for node_id in Ys.keys():
            self.intersectionssignals[node_id] = C*(Ys[node_id]/sum(Ys[node_id]))
            #self.intersectionssignals[node_id][self.intersectionssignals[node_id]<MD.SignalParameters.gmin] = MD.SignalParameters.gmin
            #self.intersectionssignals[node_id][self.intersectionssignals[node_id]>MD.SignalParameters.gmax] = MD.SignalParameters.gmax
        
        #determing the offset
        if offset:
            print 'Now try to optimize and set offset in self.offset'
        #offset optimization
            bandwidths,offsets  =self.optimization_offset_2intersections(coordinates = self.coordinates, traveltime=traveltime)
            self.offset[1] = offsets[bandwidths.index(max(bandwidths))]
        return Flowrates_dict,Ys
    
    
    def get_pi(self, pi='delay'):
        """
        If input is 'delay', then return the 
            - (totaldelay, averagedelay)
        ------------------------------------
        input: pi
            a str. The delay  of the intersection.
        -----------------------------------
        output: 
            if pi=='delay', then return a tuple, i.e. 
                (totaldelay, averagedelay), both unit is minutes.
        
        """
        if pi=='delay':
            delay = 0
            N = 0
            for intersectionlabel,nodetraffic in self.intersectiontraffics.iteritems():
                overalldelay,averdelay = nodetraffic.get_pi(pi=pi)
                delay = delay + overalldelay
                N = N + 1.0*overalldelay/averdelay
            #return 1.0*delay/N
            return 1.0*delay/N
    
    def reset_demand(self,):
        """
        reset the demand of the movements that need input. 
        The demand shoul be null:
            - movement.arrival_t = [0]
            - movement.N = [0]
            - movement.N_modes = []
        
        """
        #iter over intersections by intersectionlabel
        for intersectionlabel,NodeTraffic in self.intersectiontraffics.iteritems():
            #iter over movements by m
            for m in NodeTraffic.movements.keys():
                output_m = str(intersectionlabel)+'_'+m
                #if output_m in the columns, it means there is some input, hence the movement m need to run here
                if output_m not in self.TurningRatios.columns:continue
                
                #D construct
                self.intersectiontraffics[intersectionlabel].movements[m].arrival_t = [.0]
                self.intersectiontraffics[intersectionlabel].movements[m].N = [.0]
                self.intersectiontraffics[intersectionlabel].movements[m].N_modes = []
    
    
    def demandsetting(self, Demands_dict, T_horizon = 3600.0):
        """
        set the demand of all the intersections.
        --------------------------------------------
        input: Demands_dict
            demands of all intersections. The keys are consistent with self.intersectiontraffics.keys().
            Demands_dict[intersectionlabel] = (H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es).
            interval_mean_es are dict, keys are 'east_left'....
        output: 
            no outout. But the follwing are changed:
                - self.intersectiontraffics[label].movements['east_left'].N, arrival_t, N_modes
        ---------------------------------------------
        steps: 
            - for each movement, generate the N, arrival_t and N_modes
            - input these variables
        """
        #iter over intersections by intersectionlabel
        for intersectionlabel,NodeTraffic in self.intersectiontraffics.iteritems():
            #iter over movements by m
            H,interval_mean_es,arrival_mode_proba_es,batch_arrival_es = Demands_dict[intersectionlabel]
            for m in NodeTraffic.movements.keys():
                target_m = str(intersectionlabel)+'_'+m
                if target_m in self.TurningRatios.columns:
                    NodeTraffic.movements[m].N = [0]
                    NodeTraffic.movements[m].arrival_t = [0]
                    NodeTraffic.movements[m].N_modes = []
                else:
                    arrival_t, N, N_modes = NodeTraffic.movements[m].GenerateArrival(H =H,interval_mean = interval_mean_es[m], arrival_mode_proba = arrival_mode_proba_es[m], batch_arrival = batch_arrival_es[m], T_horizon = T_horizon)
                    NodeTraffic.movements[m].N = N
                    NodeTraffic.movements[m].arrival_t = arrival_t
                    NodeTraffic.movements[m].N_modes = N_modes
                    

    def platoondispersion_truncatednormal1(self, N, arrival_t, l=100, vmin = 10, vmax = 50, vmean = 20, loc = 20, scale = 1.0):
        """
        platoon dispersion model for truncted normal speed distribution.
        The speed obeys the truncated normal distribution with lower and upper specified in vmin and vmax.
        
        The method works as follows:
            - first get the cumulative curve at destination section
            - then linear interploate the moment for each vehicle number n (within [1,N[-1])
        --------------------------------------------------
        input: N, arrival_t
            all are lists. len(N)=len(arrival_t)=len(N_modes)+1
            !!!!N starts from 0, hence len(N_modes)=len(N)-1
        input: l
            the road length for the platoon dispersion.
        input: vmin and vmax vmean
            the minimal speed and maximal speed and mean speed. The unit is km/.
            
        input: loc and scale
            the parameters in the normal distribution. 
            loc is the mean and scale is the std of normal distribution.
            Actually the loc is not used. 
        -----------------------------------------------------------
        output: N1, arrival_t1
            the cumulative arrival at downstream point. 
        -----------------------------------------------------------
        Steps:
            - get the time horizon, in t0 and t1
            - for each intersection moment
        """
        #convert the speed to m/s
        vmin1 = vmin/3.6
        vmax1 = vmax/3.6
        vmean1 = vmean/3.6
        #minimal travel time and maximal travel time
        tao_min = l/vmax1
        tao_max = l/vmin1
        tao_mean = l/vmean1
        
        #find the downstream arrival time lower and upper bound
        #   arrival_t[1:] because the N begins from 0, i.e. N[0]=0
        arrival_times_lower = np.array(arrival_t[1:],dtype=float) + tao_min
        arrival_times_upper = np.array(arrival_t[1:],dtype=float) + tao_max
        arrival_times_mean  = np.array(arrival_t[1:],dtype=float) + tao_mean
        
        arrival_t1 = np.arange(arrival_times_lower[0],arrival_times_upper[-1],max(1, (arrival_times_upper[-1]-arrival_times_lower[0])/5.0))
        N1 = 0*arrival_t1
        #for each vehicle
        for at0,atmean,at1 in zip(arrival_times_lower,arrival_times_mean,arrival_times_upper):
            ts = np.arange(at0,at1, max(1, (at1-at0)/3.0))[1:]
            for t in ts:
                #compute the cdf of truncated normal
                cdf  =  truncatednorm_cdf(t, a = at0, b = at1, loc = atmean, scale = scale)
                
                #assign the cdf
                #   find the idxx that arrival_t1[idxx]>=t
                idxx = np.where(arrival_t1>=t)[0][0]
                N1[idxx:] = N1[idxx:]+cdf
        
        return N1,arrival_t1
        
    def platoondispersion_truncatednormal2_PROBLEM(self, N, arrival_t, l=100, vmin = 10, vmax = 50, vmean = 20, loc = 20, scale = 1.0):
        """
        platoon dispersion model for truncted normal speed distribution.
        The pseed obeys the truncated normal distribution with lower and upper specified in vmin and vmax.
        
        The method works as follows:
            - first get the cumulative curve at destination section
            - then linear interploate the moment for each vehicle number n (within [1,N[-1])
        --------------------------------------------------
        input: N, arrival_t
            all are lists. len(N)=len(arrival_t)=len(N_modes)+1
            !!!!N starts from 0, hence len(N_modes)=len(N)-1
        input: l
            the road length for the platoon dispersion.
        input: vmin and vmax vmean
            the minimal speed and maximal speed and mean speed. The unit is km/.
            
        input: loc and scale
            the parameters in the normal distribution. 
            loc is the mean and scale is the std of normal distribution.
            Actually the loc is not used. 
        -----------------------------------------------------------
        output: N1, arrival_t1
            the cumulative arrival at downstream point. 
        -----------------------------------------------------------
        Steps:
            - get the time horizon, in t0 and t1
            - for each intersection moment
        """
        #convert the speed to m/s
        vmin1 = vmin/3.6
        vmax1 = vmax/3.6
        vmean1 = vmean/3.6
        #minimal travel time and maximal travel time
        tao_min = l/vmax1
        tao_max = l/vmin1
        tao_mean = l/vmean1
        
        #find the downstream arrival time lower and upper bound
        #   arrival_t[1:] because the N begins from 0, i.e. N[0]=0
        arrival_times_lower = np.array(arrival_t[1:]) + tao_min
        arrival_times_upper = np.array(arrival_t[1:]) + tao_max
        arrival_times_mean  = np.array(arrival_t[1:]) + tao_mean
        
        #sorted switching moment
        #   moments is a list, and without repeating
        
        moments = sorted(np.unique(np.concatenate([arrival_times_lower,arrival_times_upper])))
        
        arrival_t1 = [.0]
        N1 = [0]
        for t0,t1 in zip(moments[:-1], moments[1:]):
            #will store the cumulative N at moment t1 at downstream spot.
            cumulativeN1 = N1[-1]
            
            #find all the vehicles that may arrive during [t0,t1]
            #   the results is stored in arrival_vehicles_indexes, a list
            #   arrival_vehicles_indexes[i]=5, means that vehicle id 5 contriute to the arrival of [t0,t1]
            tmp0 = np.where(arrival_times_lower<=t0)[0]+1
            tmp1 = np.where(arrival_times_lower>=t1)[0]+1
            arrival_vehicles_indexes = np.intersect1d(tmp0, tmp1)
            
            for idd in arrival_vehicles_indexes:
                #compute the cdf(t0)-cdf(t0)
                #   first get the lower, mean and uppter of arrival moment
                lowermoment = arrival_times_lower[idd-1]
                uppermoment = arrival_times_upper[idd-1]
                mean_moment = arrival_times_mean[idd-1]
                
                cumulativeN1  =  cumulativeN1 + truncatednorm_cdf(t1, a = lowermoment, b = uppermoment, loc = mean_moment, scale = scale) - truncatednorm_cdf(t0, a = lowermoment, b = uppermoment, loc = mean_moment, scale = scale)
            
            arrival_t1.append(t1)
            N1.append(cumulativeN1)
        
        return N1,arrival_t1
        
    
    def platoondispersion_return_frequ_ts(self, departure_t, D, N_modes, l=500, scale=40 ,vmin = 5, vmax = 60, vmean = 25,returnedinterval=30):
        """
        
        input: D, departure_t
            all are lists. len(D)=len(departure_t)=len(N_modes)+1
            !!!!D starts from 0, hence len(N_modes)=len(D)-1
        input: l
            the road length for the platoon dispersion.
        input: vmin and vmax vmean
            the minimal speed and maximal speed and mean speed. The unit is km/.
            
        input: scale
            the parameters in the normal distribution. 
            loc is the mean and scale is the std of normal distribution.
        """
        
        
        #first split the 
        
        
        pass
    
    
    def run_trafficflow_waitingmovement(self,):
        """
        run the traffic flow dynamics that needs further demand input. 
        The demand input is done by self.run_flowsplits_overlap()
        Note that, the signals are stored in self.movementsrgps_dict:
            movementsrgps_dict[intersectionlabel]['east_left']=RGP instance. 
        ---------------------------------------
        input: 
            no input, but use the following attributes
        output: 
            no output. but the departure and queue of some movements are changed. 
        """
        
        #iter over intersections by intersectionlabel
        for intersectionlabel,NodeTraffic in self.intersectiontraffics.iteritems():
            #iter over movements by m
            for m in NodeTraffic.movements.keys():
                output_m = str(intersectionlabel)+'_'+m
                #if output_m in the columns, it means there is some input, hence the movement m need to run here
                if output_m not in self.TurningRatios.columns:continue
                
                #D construct
                NodeTraffic.movements[m].D_construct(some_rgp=self.movementsrgps_dict[intersectionlabel][m])
    
    def run_flowsplits_overlap(self,):
        """
        after running the self.run_trafficflow_of_nowaitingmovement(), we have the departures of some movements. Now we will split these departures into different movements. 
        -------------------------------------
        input: no input, but will use the following :
            self.intersectiontraffics[somenode].movements['east_left'].D
        output: 
            no output, but changes the arrival_t N N_modes of some movments
        -------------------------------------
        Steps:
            - for each movement, which is the movement waiting for input.
                - find the receiving movement labels
                - find the receiving ratioes, which is determined as self.TurningRatios
                - split the departures
                - for each receving movement
                    - overlap the splited departure for each receiving movements
        """
        #iter over intersections by intersectionlabel
        for intersectionlabel,NodeTraffic in self.intersectiontraffics.iteritems():
            #iter over movements by m
            for m in NodeTraffic.movements.keys():
                output_m = str(intersectionlabel)+'_'+m
                #if output_m in the index, it means there is some output from movement output_m.
                if output_m not in self.TurningRatios.index:continue
                
                #get receiving movements (in receiving_ms) and normalized ratios.
                #   a index.
                receiving_ms = self.TurningRatios.columns[self.TurningRatios.loc[output_m,:].notnull()]
                #   receiving_ratios, a array
                tmp  = sum(self.TurningRatios.loc[output_m,receiving_ms])
                receiving_ratios = (self.TurningRatios.loc[output_m,receiving_ms]/tmp).values
                
                #split the movements.
                #   data prepare
                D,departure_t = NodeTraffic.movements[m].reorganize_D()
                #   get the splited flows
                #   splited_D[i]['D']= list,splited_D[i]['departure_t']= list
                #   splited_D[i]['N_modes']= list, 
                splited_D =  NodeTraffic.movements[m].split_D(D=D, departure_t=departure_t, N_modes = NodeTraffic.movements[m].N_modes, ratios0 = receiving_ratios)
                
                #shift and platoon dispersion, and overlap
                #   vals is a dict, keys include 'D','departure_t','N_modes'
                for idxx,vals in splited_D.iteritems():
                    #   shift  the demand
                    N, arrival_t, N_modes = self.platoondispersion_shift(N = vals['D'], arrival_t=vals['departure_t'], N_modes = vals['N_modes'], traveltime = self.traveltime)
                    
                    #   overlap the demand.
                    receiving_m = receiving_ms[idxx][2:]
                    #print receiving_ms[idxx][:2],receiving_m,len(N),len(arrival_t),len(N_modes)
                    note_temp = self.intersectiontraffics[int(receiving_ms[idxx][:1])]
                    
                    note_temp.movements[receiving_m].overlap_N(N1 = N, arrival_t1 = arrival_t, N_modes1 = N_modes)
                    
                    #print '\t',len(note_temp.movements[receiving_m].N),len(NodeTraffic.movements[receiving_m].arrival_t),len(NodeTraffic.movements[receiving_m].N_modes)
                    
    def run_trafficflow_nowaitingmovement(self,T_horizon = 3600.0):
        """
        run the traffic flow dynamics for each intersection and generate the departure_t and D and queue_t and Q
        NOTE that the runed movements are all movements that does not need any further input. 
        
        
        ------------------------------
        
        """
        #get the signals. 
        #   movementsrgps_dict[intersectionlabel]['east_label']=RGP instance.
        movementsrgps_dict = self.get_signals(T_horizon = T_horizon)
        for intersectionlabel,NodeTraffic in self.intersectiontraffics.iteritems():
            movements_rgps = movementsrgps_dict[intersectionlabel]
            #construct the departure and queue compute the delay
            for m in NodeTraffic.movements.keys():
                #if m is in self.TurningRatios.columns, there is some further input
                target_m = str(intersectionlabel)+'_'+m
                if target_m in self.TurningRatios.columns:continue
                #construct the departure_t D queue_t Q
                try:
                    NodeTraffic.movements[m].D_construct(some_rgp=movements_rgps[m])
                except Exception as e:
                    builtins.tmp = {'N':NodeTraffic.movements[m].N,'arrival_t':NodeTraffic.movements[m].arrival_t,'N_modes':NodeTraffic.movements[m].N_modes, 'RGP':NodeTraffic.movements[m].rgp}
                    raise ValueError(e)
                    
    
    def run_trafficflow_oneshot(self,T_horizon = 3600.0):
        """
        run traffic flow dynamics given demand and signal.
        demand is stored in:
            self.intersectiontraffics[someintersection].intersection_flows.movements. It is a dict.
        signals is stored in:
            self.intersectiontraffics[someintersection].intersection_flows.IntersectionSignal, it is a MD.SignalParameters instance.
            
            Also the self.offset gives the offset between intersection signals. 
        ----------------------------------------
        Steps:
            - first run the intersection-movements that are not in self.coordinates
            - then run all the first movement in the coordinates
            - output the demand and overlap the demand. 
                - run the remaing demand.
        """
        
        #movements that does not need input
        #   following are changed:
        #       self.movementsrgps_dict
        self.run_trafficflow_nowaitingmovement(T_horizon = T_horizon)
        
        #demand split and output and overlap
        #   the arrival_t N N_modes of respective movement are changed
        self.run_flowsplits_overlap()
        
        #traffic flow dynamics changed
        #   the movement that need flow input. 
        self.run_trafficflow_waitingmovement()
    
    
    def input_signals(self,intersectionssignals):
        """
        ---------------------
        input: intersectionsignals
            should be the dict. keys are consistent with self.intersectiontraffics.
        """
        self.intersectionssignals = intersectionssignals
        self.reset_demand()
        
        
        pass
    
    
    def input_signals_as_RGP(self,dictofphasesgreen):
        """
        input the signal (as dict of phases greens) to the self, and generate self.movementsrgps_dict
        -----------------------------------------------
        input: dictofphasesgreen
            the input signal. keys are the intersection labels
        """
        movementsrgps_dict = {}
        for node,phasesgreens in dictofphasesgreeniteritems():
            movementsrgps_dict[node]=MD.SignalParameters().To_rgps(phasesgreens)
            #offset
            if self.offset[node]!=0:
                _ = self.offset_intersection(movementsrgps_dict[node], offset=self.offset[node], substitute = True)
        self.movementsrgps_dict = movementsrgps_dict
    
    def get_signals(self,assign=True, T_horizon=3600.0):
        """
        return the movements RGPS of the two intersections. 
        -------------------------------
        input: 
            no input, using the following :
                - self.intersectionssignals, a dict, keys are the intersection labels, values are pd.Series
                - self.offset, a dict. self.offset[intersectionlabel]=a salar.
        input: assign
            bool. If true, then self.movementsrgps_dict will be assigned.
        output: movementsrgps_dict
            a dict, keys are intersection labels, 
            movementsrgps_dict[intersection_label] = {'east_left':RGP instance,...}
        """
        movementsrgps_dict = {}
        for node,phasesgreens in self.intersectionssignals.iteritems():
            movementsrgps_dict[node]=MD.SignalParameters().To_rgps(phasesgreens, T = T_horizon)
            #offset
            if self.offset[node]!=0:
                _ = self.offset_intersection(movementsrgps_dict[node], offset=self.offset[node], substitute = True)
        if assign==True:
            self.movementsrgps_dict = movementsrgps_dict
        return movementsrgps_dict
        
    def RandomSampleSignal(self, intersection = 0, plan_id = 3):
        """
        random sample the signal for only one intersection,  which is specified in intersection
        ---------------------------------------------
        input: intersection
            the intersectio label that need sampled. It should be the key in self.intersectiontraffics.
        input: plan_id
            an int \in range(49).
        """
        res = self.intersectiontraffics[intersection].RandomSampleCyclePlan(plan_id=plan_id)
        #self.intersectiontraffics[intersection].IntersectionSignal.current = res
        
        self.intersectionssignals[intersection] = res
        
    
    def RandomSampleAllSignals(self,plan_id = 3):
        """
        randomly sample intersections signals. 
        --------------------------------------------------
        #plan id =0, is the one-phase-one-approach
        #plan id = 3, is the commong four-leg plan
        """
        for i,j in self.intersectiontraffics.iteritems():
            #will return a pd.Series data class
            #j.IntersectionSignal.current = j.RandomSampleCyclePlan(plan_id=plan_id)
            
            self.intersectionssignals[i] = j.RandomSampleCyclePlan(plan_id=plan_id)
        
        self.reset_demand()
    
    def RedsNull(self,):
        """
        return a null pyinter.IntervalSet class, that used to calculate the new signal by offset
        """
        return 
        
        pass
    
    
    def offset_intersection(self, movements_rgps, offset=0, substitute=True):
        """
        offset all the signals of one intersection.
        After the operation,
            self.intersectiontraffics[someintersection].intersection_flows.IntersectionSignal.rgps
        ----------------------------------------------
        input: movements_rgps
            a dict. The keys are 'west_through'...., values are RGP instance. 
        input: offset
                a scalae, that take the offset of the intersection signals.
        """
        for i,j in movements_rgps.iteritems():
            _ = self.offset_RGP(j, offset=offset, substitute = substitute)
        
    
    def offset_RGP(self,signal, offset=0, substitute = False):
        """
        OK. NOTE that the signal is RGP instance!!!!!!!!!!!!!.
        take a offset to the signal. 
        after al
        ----------------------------------
        input: signal
            RGP.RGP instance. signal.reds will give all red durations. signal.reds is a pyinterval.IntervalSet instance.
        input: offset
            float. The offset of the signal
        input: substitute
            bool. If true. signal.reds will be changed by calling signal.reset_signal()
        -------------------------------------------
        output: reds
            the shifted red duration of each cycle. 
        """
        #red_starts is np.array
        red_starts = signal.StartMomentsOfReds + offset
        red_ends = signal.EndMomentsOfReds + offset
        reds = [(i,j) for i,j in zip(red_starts,red_ends)]
        if substitute:
            signal.reset_signals(reds)
        return reds
        
    def bandwidth_among_reds(self,redss):
        """
        OK. 
        
        It should be noted that, it would be better that the horizon of the signal is long enough
        obtain the bandwidth of several reds. 
        ------------------------
        input: redss
            a list. Each element is a pyinter.IntervalSet instance.
            redss[i] will give all the red durations of the signal. 
            for r in redss[i]:
                red_duration = r.upper_value - r.lower_value
        output: bandwidth
            a float. The bandwidth of the reds
        --------------------------
        Steps:
            - ovarlap all reds get the union of reds
            - find all the legal green time sum
            - average of all greens are the results. 
        """
        #tmp will store all the union of reds
        #the null interval set. In order to get the union of red, which will be stored in the tmp.
        tmp = pyinter.IntervalSet()
        #take the union of all the reds. The returned
        for reds in redss:
            tmp = tmp.union(reds)
        
        #the tmp1 will store all the legal green. 
        tmp1 = pyinter.IntervalSet([pyinter.open(min(tmp).lower_value,max(tmp).upper_value)])
        tmp1 = tmp1.difference(tmp)
        #a list of green durations.
        Gs = [red.upper_value-red.lower_value for red in tmp1]
        if len(Gs)==0:return 0
        
        return np.mean(Gs)
    
    def platoondispersion_shift(self, N, arrival_t, N_modes, traveltime = 15):
        """
        sift the arrivals, considering the dispersion between two intersections.
        ----------------------------------------
        input: N, arrival_t, N_modes
            all are lists. len(N)=len(arrival_t)=len(N_modes)+1
            N starts from 0, hence len(N_modes)=len(N)-1
        input: traveltime
            a scalar, the travel time between neighboring intersections.
        
        """
        arrival_t1 = [t+traveltime if t>0 else 0 for t in arrival_t]
        
        return N,arrival_t1,N_modes
    
    def plot_ND(self, intersectionid = 0):
        """
        
        """
        fig,ax = plt.subplots()
        for label,mt in self.intersectiontraffics[intersectionid].movements.iteritems():
            ax.plot(mt.arrival_t, mt.N,label = label)
            ax.plot(mt.departure_t, mt.D)
        ax.legend()
        plt.show()
        return ax
        
    import time
    builtins.tmp = []
    def optimization_offset_2intersections(self,coordinates, traveltime = 15, samplesize = 50):
        """
        different from the self.optimization_offset_2intersections0() in that, the signal is stored in 
            - self.intersectionssignals, a dict. 
                - self.intersectionssignals[intersectionlabel] is a pd.Series, keys are phase, and values are durations.
        ----------------------------------------------------
        input: coordinates
            a list. Each element is a list:
                coordinate = coordinates[i]
                coordinate = [(label1:movement), (label2:movement)], a list containing the intersection label and the corresponding movement
        input: traveltime
            the traveltime between the two intersections. It will be convered when optimize the offset. 
        input: samplesize
            a int. Because the function using the enumerate, hence computational burden. 
            It is the number of the evaluation points within [0, cycle]
        ------------------------------------------------------
        output: 
            (bands_offset,offset). The band width and its correspondint offset
        """
        #check the cycle length
        #   find the maximum 
        cycle = max([1.0*sum(j) for i,j in self.intersectionssignals.iteritems()])
        for i,j in self.intersectionssignals.iteritems():
            self.intersectionssignals[i] = j*cycle/sum(j)
        #convert to rgps.
        #   rgps are stored in self.movementsrgps_dict
        #   keys are intersection labels
        _ = self.get_signals(assign = True)
        #store the bandwidth of each offset
        bands_offset = []
        offsets = np.linspace(0,cycle, samplesize)
        # for each offset
        for offset in offsets:
            #store the band width of each coordinate
            #   len(bands_coordinate)=len(coordinates)
            bands_coordinates = []
            #coordinate is a list, 
            #   coordinate=[(intersectionlabel1,movement),()...]
            for coordinate in coordinates:
                #coor_reds store all the reds of the movement
                #   each reds is a pyinter.IntervalSet
                #start = time.time()
                intersection = coordinate[0][0]
                movement = coordinate[0][1]
                #   will input to the self.bandwidth_among_reds(), hence a list. 
                coor_reds = [self.movementsrgps_dict[intersection][movement].reds]
                #tmp reds
                tmp_rgp = RGP()
                
                intersection = coordinate[1][0]
                movement = coordinate[1][1]
                rgp = self.movementsrgps_dict[intersection][movement]
                #shifted_reds is a list, [(rs1,re1),....]
                shifted_reds = self.offset_RGP(rgp, offset = offset + traveltime)
                tmp_rgp.reset_signals(shifted_reds)
                coor_reds.append(tmp_rgp.reds)
                #print 'aggregate reds',time.time() - start
                #start = time.time()
                
                #compute the bandwidth using slef.bandwidth_among_reds()
                bands_coordinates.append(self.bandwidth_among_reds(coor_reds))
                #print 'bandwidth_among_reds',time.time()-start
                #start = time.time()
                
            bands_offset.append(np.mean(bands_coordinates))
        
        return bands_offset,offsets
        
    import time
    builtins.tmp = []
    def optimization_offset_2intersections0(self,coordinates, traveltime = 15):
        """
        OK.
        optimize the offset that obtain the maximum bandwidth by enumeration.  
        The input include the 
            - self.intersectiontraffics, the keys are the intersection label
                - self.intersectiontraffics[ii].IntersectionSignal.current is a pd.Series, the phases duration. The cycle can be calculated as sum(self.intersectiontraffics[ii].IntersectionSignal.current).
            - coordinates
                a list. Each element is a list:
                coordinate = coordinates[i]
                coordinate = [(label1:movement), (label2:movement)], a list containing the intersection label and the corresponding movement
        ----------------------------------------------------
        input: coordinates
            a list. Each element is a list:
                coordinate = coordinates[i]
                coordinate = [(label1:movement), (label2:movement)], a list containing the intersection label and the corresponding movement
        input: traveltime
            the traveltime between the two intersections. It will be convered when optimize the offset. 
        ---------------------------------------------------
        Steps:
            - first check the cycle length of the two intersections.
                - adopt the max cycle and revise all the signals.
            - then by emumeration optimize the offset(increment 1 by 1).
                - for each offset value o
                    - find the bandwith
                    - 
        """
        #check the cycle length
        #   find the maximum 
        cycle = max([1.0*sum(j.IntersectionSignal.current) for i,j in self.intersectiontraffics.iteritems()])
        for i,j in self.intersectiontraffics.iteritems():
            j.IntersectionSignal.current = j.IntersectionSignal.current*cycle/sum(j.IntersectionSignal.current)
            #convert to rgps.
            #   rgps is a dict, keys are 'east_left',.....
            j.IntersectionSignal.rgps = j.IntersectionSignal.To_rgps()
        
        #store the bandwidth of each offset
        bands_offset = []
        # for each offset
        for offset in range(int(cycle)):
            #store the band width of each coordinate
            #   len(bands_coordinate)=len(coordinates)
            bands_coordinates = []
            #coordinate is a list, 
            #   coordinate=[(intersectionlabel1,movement),()...]
            for coordinate in coordinates:
                #coor_reds store all the reds of the movement
                #   each reds is a pyinter.IntervalSet
                #start = time.time()
                intersection = coordinate[0][0]
                movement = coordinate[0][1]
                coor_reds =[self.intersectiontraffics[intersection].IntersectionSignal.rgps[movement].reds]
                #tmp reds
                tmp_rgp = RGP()
                intersection = coordinate[1][0]
                movement = coordinate[1][1]
                rgp = self.intersectiontraffics[intersection].IntersectionSignal.rgps[movement]
                #shifted_reds is a list, [(rs1,re1),....]
                shifted_reds = self.offset_RGP(rgp, offset = offset + traveltime)
                tmp_rgp.reset_signals(shifted_reds)
                coor_reds.append(tmp_rgp.reds)
                #print 'aggregate reds',time.time() - start
                #start = time.time()
                
                #compute the bandwidth
                bands_coordinates.append(self.bandwidth_among_reds(coor_reds))
                #print 'bandwidth_among_reds',time.time()-start
                #start = time.time()
                
            bands_offset.append(np.mean(bands_coordinates))
        
        return bands_offset
    
    
    def pso_init(self, w = .5, c1 = .5 , c2 = .5, popusize = 50, iterations = 1000, plan_ids = {0:0,1:0}, neighbores_percent = .7):
        """
        pso of the two intersection system.
        ---------------------------------------
        input: plan_id
            when defind the 
        """
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.neighbores_percent = neighbores_percent
        self.popusize = popusize
        
        self.population = []#length = popusize
        self.population_neighbores = {}#the neighbores of one individual.
        self.population_distances = pd.DataFrame()#the index is range(len(self.population))
        self.population_historybest = []#element is an individual
        self.population_historybest_fit = []#element is scalar.
        self.population_velocities = [{} for i in range(popusize)]#length = popusize
        self.population_fitnesses = []#changed after each iteration.
        #current best and fit, a list the same length with self.population, because there is a percentage of the neighbores
        self.current_best = [[]]*popusize
        self.current_best_fit = [[]]*popusize
        #global best and its fit.
        self.globalbest = 0#
        self.globalbest_fit = np.inf#a scalar
        #history trajectory and 
        self.his_best_fits = []##a list, each corresponding to a iteration.
        
        #initilaize the population
        somenode = np.random.choice(self.intersectiontraffics.keys())
        somemovement =  np.random.choice(self.intersectiontraffics[somenode].movements.keys())
        mt = self.intersectiontraffics[somenode].movements[somemovement]
        for i in range(popusize):
            signalsample = {}
            for node in self.intersectiontraffics.keys():
                #signalsample[node] is a pd.Series instance.
                signalsample[node] = mt.RandomSampleCyclePlan(plan_id = plan_ids[node])
                self.population_velocities[i][node]=0
            self.population.append(signalsample)
            
    
    def pso_get_neighbores(self,):
        """
        get the neighbores of each individuals.
        ----------------------------------
        input: 
            no input. Using the following attributes:
                - self.
        """
        self.population_neighbores = {}
        
        
        self.neighbores = {}
        #idxx is the rank that within neighbores. 
        idxx = max(1,self.neighbores_percent*self.popusize)
        for i in range(self.popusize):
            #find the neighbores. 
            #   first rank the distances. 
            #       rank is a series, the index is the same as self.populations_distances.loc[i,:]
            #       the value is from 1 to n
            rank = self.population_distances.loc[i,:].rank(ascending=True)
            
            #   get the index of neighbores.
            #       neighbores_idx is a pd.Index instance. 
            neighbores_idx = rank[rank<=idxx].index
            self.population_neighbores[i] = neighbores_idx
        
    
    def pso_distance_population(self, tpyee = 'cityblock'):
        """
        compute the distance between two individuals using scipy.spatial.distance (as scipydistance) function 
        -------------------------------------
        input: indi1, indi2
            both are dict. indi1[intersectionlabel]=pd.Series
        input: tpyee
            the type of the distance. 
        """
        for i in range(len(self.population)):
            for j in range(len(self.population)):
                if i==j:
                    self.population_distances.loc[i,j]=0
                    continue
                if tpyee=='cityblock':
                    ds = [scipydistance.cityblock(self.population[i][label], self.population[j][label]) for label in self.population[i].keys()]
                    self.population_distances.loc[i,j]=sum(ds)
                
    def pso_best_find(self, pi='delay'):
        """
        find the current best indiviual for each individual neighbores area. 
        The following arttributes are used:
            self.population_fitnesses, a list.
            self.population_distances, a pd.DataFrame
        ----------------------------
        input: pi
            no input, but using the 
        output: 
            no output. but the following are changed:
                - self.current_best 
                    (a list. eelement is int, the index of the current best neigbore of a specific individual)
                - self.current_best_fit
                    a list. correspond to the self.current_best
        -----------------------------
        steps:
            - calculate the distance between neighbor
            - for each individual
            - find all the neighbores in neighbores_idxxes
            - find the optimal one and its best
        """
        #current best among neighbores. stored in self.current_best
        #   get the idx of the minimal fit
        for i in range(self.popusize):
            #all neighbores, Index instance
            indexes = self.population_neighbores[i]
            #find the minimal, minidxx is one index
            minidxx = pd.Series(self.population_fitnesses)[indexes].idxmin()
            self.current_best[i] = copy.deepcopy(self.population[minidxx])
            self.current_best_fit[i] = self.population_fitnesses[minidxx]
        
        #indiviaual history best
        #   stored in self.population_historybest and self.population_historybest_fit
        if len(self.population_historybest)==0:
            self.population_historybest = copy.deepcopy(self.population)
            self.population_historybest_fit = copy.deepcopy(self.population_fitnesses)
        else:
            for i in range(self.popusize):
                if self.population_fitnesses[i]<self.population_historybest_fit[i]:
                    self.population_historybest[i]  = copy.deepcopy(self.population[i])
                    self.population_historybest_fit[i] = self.population_fitnesses[i]
        
        #global best and global fit
        #   stored in
        #   self.globalbest and self.globalbest_fit
        minidxx = pd.Series(self.population_fitnesses).idxmin()
        if self.population_fitnesses[minidxx]<self.globalbest_fit:
            self.globalbest = copy.deepcopy(self.population[minidxx])
            self.globalbest_fit = self.population_fitnesses[minidxx]
    
    def pso_velocity_move(self,):
        """
        compute the velocity and move the solution. 
        Based on 
            - self.
        
        -------------------------------------------
        
        
        
        """
        
        #calculate the velocity to self.populations_velocities
        for i in range(self.popusize):
            for intersectionlabel in self.population[i].keys():
                #get the distances to the current best 
                #d_best is a vector, should be a pd.Series   
                d_best = self.current_best[i][intersectionlabel]-self.population[i][intersectionlabel]
                #get the distance to the individual history best. 
                #   d_history_best is a vector, should be a pd.Series 
                d_history_best = self.population_historybest[i][intersectionlabel] -self.population[i][intersectionlabel]
                #calculate and assign the velocity
                self.population_velocities[i][intersectionlabel]  = self.w*self.population_velocities[i][intersectionlabel] + self.c1*self.pso_r1()*d_best + self.c2*self.pso_r2()*d_history_best
        
        #move
        for i in range(self.popusize):
            for intersectionlabel in self.population[i].keys():
                self.population[i][intersectionlabel] = self.population[i][intersectionlabel] + self.population_velocities[i][intersectionlabel]
                #confine the solution within gmin and gmax
                #   tmp_signal is a MD.SignalParameters() instance
                self.population[i][intersectionlabel][self.population[i][intersectionlabel]<tmp_signal.gmin] = tmp_signal.gmin
                self.population[i][intersectionlabel][self.population[i][intersectionlabel]>tmp_signal.gmax]=tmp_signal.gmax
    
    
    def pso_fit_individual(self,individual, pi = 'delay', T_horizon = 3600.0):
        """
        fitness of single indiviual
        -------------------------------
        input: individual
            a dict. individual[intersectionlabel] is a pd.Series data.
        output: scalar
        """
        self.intersectionssignals = individual
        self.run_trafficflow_oneshot(T_horizon=T_horizon)
        pi_value = self.get_pi(pi =  pi)
        return pi_value
        
        #clear the demand that need further input to null
        #   movement.arrival_t = [0]
        #   movement.N = [0]
        #   movement.N_modes = []
        self.reset_demand()
        
    def pso_fit_population(self, pi = 'delay', T_horizon = 3600.0):
        """
        get the fitness of all the individuals, which is stored in
            - self.population, a list. Each is an individual (a dict.)
        -------------------------------------
        output: 
            no output. the following are changed:
                - self.population_fitnesses
                - self.movementsrgps_dict
        """
        
        self.population_fitnesses = []
        for i,individual in enumerate(self.population):
            self.input_signals(individual)
            self.population_fitnesses.append(self.pso_fit_individual(individual=individual, pi = pi, T_horizon = T_horizon))
            
            
        
        
        
    def pso_r1(self,):
        return np.random.random()
    
    def pso_r2(self,):
        return np.random.random()
        
        
    def pso(self, w = .5, c1 = .5 , c2 = .5, popusize = 50, iterations = 1000, pi = 'delay', typee = 'maximize', neighborespercent = .7):
        """
        optimize the signal
        ---------------------------
        input: pi and typee
            the objective of the optimization.
         input: neighborespercent
        output: best, bestfit
            best is a dict.
            the history best is stored in self.his_best_fits, a list.
        -------------------------------
        Steps:
            - initialize
            - for each iteration
                - evaluate the populations. (self.pso_fit_population())
                - calculate the current best of each individual and global best (self.pso_)
                - 
        """
        
        
        pass

import geopy.distance
#   center lat and lon for HongLi rd-----XinZhou rd.
center_lat = 22.550621101980095
center_lon = 114.04421458411854
class data_pprocess():
    
    
    @classmethod
    def plotxy(self):
        """
        
        """
        
        
        pass
    
    @classmethod
    def radiusfilter_onevehicle(self, onevehiclexy, radius = 500):
        """
        distance between two geo points are calculated using:
            geopy.distance.vincenty((lat1,lon1), (lat2, lon2)).m
        --------------------------------
        input: center_lat and center_lon
            both are float. The latitude and longitude of the centerr point
        input: onevehiclexy
            pd.DataFrame data. the index are the moments and the columns are x and y. 
            somerow.name = moment
        input: radius
            the radius of the filtering. unit is meters. 
        output: filteredradius
            a pd.DataFrame. 
        
        """
        
        distance = pd.Series()
        for row in onevehiclexy.iterrows():
            #row[1].name is the moment
            distance.loc[row[1].name] = np.sqrt(row[1].x**2+row[1].y**2)
        
        
        return onevehiclexy.loc[distance.index[distance<=radius] , :]
    
    @classmethod
    def onedaygpsdata2xy(self,onedaygps):
        """
        convert one day gps data to xy. 
        -----------------------------------------
        input: onedaygps
            a dict. keys are the plate numbers. 
            onedaygps[plate] = pd.DataFrame. 
        output: converteddaygps
            a dict, keys are also plates. 
            converteddaygps[plate] is a pd.DataFrame, the index are moments (unique), the columns are 'x' and 'y'.
            
        
        """
        converteddaygps = {}
        for plate in onedaygps.keys():
            converteddaygps[plate] = self.gps_frame2xy(onedaygps[plate])
            print 'Completed ------>',1.0*len(converteddaygps)/len(onedaygps)
        
        return converteddaygps
    
    
    @classmethod
    def gps_frame2xy(self,gpsdataframe):
        """
        convert the gps series into x and y,
        ---------------------------
        input:  gpsdataframe
            pd.DataFrame, that convert the lon lat to x and y
            The columns fields are      ['date','time','firm','plate','lon','lat','speed','orient','state','valid']
        output: filtereddata
            a dataframe. The index are the moments, or corresponding to the gpsdataframe['time']. Columns are 'x' and 'y'
            
        """
        
        filtereddata = pd.DataFrame(columns = ['x','y'])
        for row in gpsdataframe.iterrows():
            if row[1]['time'] in filtereddata.index:continue
            x,y = self.latlon2xy(center_lat, center_lon, row[1]['lat'], row[1]['lon'])
            filtereddata.loc[row[1]['time'],['x','y']] = x,y
        
        return filtereddata
    
    
    @classmethod
    def latlon2xy(self,center_lat, center_lon, lat, lon):
        """
        convert the lat, lon to x and y both of which are in meters
        NOTE that when ploting, y corresponding to longitude and x correspond to lattitude!!
        --------------------------------------------
        input:center_lat, center_lon
            both are float.
        input: lat and lon
            both are float.
        
        """
        
        #center of the intersection: HongLi RD----XinZHou RD
        center_coor  = (center_lat, center_lon)
        
        #positive or negative
        a  = 1.0 if lat>center_lat else -1.0
        x = a*geopy.distance.vincenty(center_coor, (lat, center_lon)).m
        
        a  = 1.0 if lon>center_lon else -1.0
        y = a*geopy.distance.vincenty(center_coor, (center_lat, lon)).m
        return y,x
        pass
    
    pass
