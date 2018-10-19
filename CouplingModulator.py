"""
Created on Fri Feb  9 20:32:56 2018
@author: M.H. Tahersima
Data analysis of Experimental ring transfer functions 
"""
#import os
#clear = lambda: os.system('cls')
#clear()
import pandas as pd
import matplotlib.pyplot as plt

import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (5, 5),
         'axes.labelsize': 'x-large',
         'axes.titlesize':'x-large',
         'xtick.labelsize':'medium',
         'ytick.labelsize':'medium'}
pylab.rcParams.update(params)
from matplotlib.pyplot import cm
import numpy as np
import argparse
from sklearn.preprocessing import MinMaxScaler
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(description = 'DNN')
parser.add_argument('--Data', '-d', default = 'EO_A', help = 'Data to be plotted')
# radius: various passive rings; process: effect of process steps on transfer function
# final: final dveice; EO: Electro optic modulation; IV: the current vs voltage plot from negative to positive bias
# EO_A: Electro optic modulation data new data
parser.add_argument('--inDB', '-db', default = 'True', help = 'Should plot be in dB')
parser.add_argument('--pos', '-p', default = True, help = 'Is bias voltage positive?')
args = parser.parse_args()
##############################################################################
# READ Data
##############################################################################
xl=pd.ExcelFile("4003_measurments.xlsx")
print("Sheet names are:", xl.sheet_names)
##############################################################################
# Functions
##############################################################################
def Range(arr):
    array_start=398 #from 1547 nm
#    array_start=248 #from 1547 nm
    array_end=482 #to 1553 nm
    arr = arr[array_start:array_end]
    return arr

def norm (arr,arr0):
    normalize =MinMaxScaler(feature_range=(0,1))
    arr = normalize.fit_transform(np.divide(Range(arr),Range(arr0)))
#    arr = arr + 10**-2.5 #fudge factor not to get zero minima
    if args.inDB == 'True': # if want to show it in dB
        arr = 10*np.log10(arr)
    return arr

def getArray (D1, D2, sheet, pos):
    T_arr = np.zeros ((D1, D2))
    for vv in range(T_arr.shape[0]):
        if pos:
            arr = np.array(sheet[['V'+str(vv)]])*10**9
        else:
            arr = np.array(sheet[['V-'+str(vv)]])*10**9
        arr = np.squeeze (arr)
        T_arr[vv, :]= arr
    return T_arr

def getArray2 (dev, ite, vv, Length, sheet):
    T_arr = np.zeros ((dev, ite, vv, Length))
    for ii in range (dev):
        for jj in range (ite):
            for kk in range(vv):
                arr = np.array(sheet[[str(ii+1)+str(jj+1)+str(kk+1)]])*10**9
                arr = np.squeeze (arr)
                T_arr[ii, jj, kk, :]= arr
    return T_arr

def ER(arr, arr0):
    ERdB = 10*np.log10(arr/arr0)
    return ERdB


def lorentz(x, *p):
    I, gamma, x0 = p
    return I * gamma**2 / ((x - x0)**2 + gamma**2)
def fit(p, x, y):
    return curve_fit(lorentz, x, y, p0 = p)
def calc_r2(y, f):
    avg_y = y.mean()
    sstot = ((y - avg_y)**2).sum()
    ssres = ((y - f)**2).sum()
    return 1 - ssres/sstot


# find local minimas and local maximas
#minima = (np.diff(np.sign(np.diff(r80_norm))) > 0).nonzero()[0] + 1 # local min
#maxima = (np.diff(np.sign(np.diff(r80_norm))) < 0).nonzero()[0] + 1 # local max
##############################################################################
# Data Manipulation
##############################################################################
if args.Data == 'radius': ## Various ring radius
    sheet = xl.parse(1)
    WL=np.array(sheet['WL'])
    WL = Range (WL)
    EDFA=np.array(sheet[['EDFA']])*10**9
    r80=np.array(sheet[['r80']])*10**9
    r60=np.array(sheet[['r60-1']])*10**9
    r50=np.array(sheet[['r50-1']])*10**9
    
    spec1, spec2, spec3 = norm (r80, EDFA), norm (r60, EDFA), norm (r50, EDFA)
    
    plt.plot(WL,spec1,'#2471A3', label='r80',linewidth=1)
    plt.plot(WL,spec2,'#FF0000', label='r60',linewidth=1)
    plt.plot(WL,spec3,'#000000', label='r50',linewidth=1)
#    plt.axis([1536,1542,-2,0])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.legend(loc="upper right")
    plt.show()

##############################################################################           
if args.Data == 'process': ## Process steps
    sheet = xl.parse(0)
    WL=np.array(xl.parse(0)[['WL[nm]']])
    WL = Range (WL)    
    EDFA=np.array(xl.parse(0)[['EDFA[W]']])*10**9
    ring=np.array(xl.parse(0)[['step0[W]']])*10**9
    passiOx=np.array(xl.parse(0)[['step1-11nmAl2O3[W]']])*10**9
    ITO1=np.array(xl.parse(0)[['step2-ITO1[W]']])*10**9
    gateOx=np.array(xl.parse(0)[['step3-gateOxide[W]']])*10**9
    ITO2=np.array(xl.parse(0)[['Step4-ITO2[W]']])*10**9 


    spec1, spec2, spec3, spec4, spec5 = norm (ring, EDFA), norm (passiOx, EDFA), norm (ITO1, EDFA), norm (gateOx, EDFA), norm (ITO2, EDFA)

    plt.plot(WL,spec1,'#2471A3', label='ring',linewidth=1)
    plt.plot(WL,spec2,'#FF0000', label='passiOx',linewidth=1)
    plt.plot(WL,spec3,'#000000', label='ITO1',linewidth=1)
    plt.plot(WL,spec2,'#000000', label='gateOx',linewidth=1)
    plt.plot(WL,spec3,'#0000FF', label='ITO2',linewidth=1)
#    plt.axis([1536,1542,-2,0])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.legend(loc="upper right")
    plt.show()

##############################################################################   
if args.Data == 'final': ## Process steps
    sheet = xl.parse(2)
    WL=Range(np.array(sheet[['WL[nm]']]))
    EDFA=np.array(sheet[['EDFA[W]']])*10**9
    A3=np.array(sheet[['A3[W]']])*10**9
    B3=np.array(sheet[['B3[W]']])*10**9
    C3=np.array(sheet[['C3[W]']])*10**9
    D3=np.array(sheet[['D3[W]']])*10**9
    E3=np.array(sheet[['E3[W]']])*10**9
    F3=np.array(sheet[['F3[W]']])*10**9

    spec1, spec2, spec3, spec4, spec5, spec6 = norm (A3, EDFA), norm (B3, EDFA), norm (C3, EDFA), norm (D3, EDFA), norm (E3, EDFA), norm (F3, EDFA)

    plt.figure()
    plt.plot()
    plt.plot(WL,spec1,'#2471A3', label='A3',linewidth=2)
    plt.plot(WL,spec2,'#FF0000', label='B3',linewidth=2)
    plt.plot(WL,spec3,'#000000', label='C3',linewidth=2)
    plt.plot(WL,spec4,'#228B22', label='D3',linewidth=2)
    plt.plot(WL,spec5,'#0000FF', label='E3',linewidth=2)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.legend(loc="upper right")
    plt.show()

    plt.plot(WL,spec5,'#C0C0C0', label='16um device',linewidth=2)
    plt.plot(WL,spec6,'#000000', label='passive device',linewidth=2)
#    plt.axis([1536,1542,-2,0])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.legend(loc="upper right")
    plt.show()

##############################################################################
if args.Data == 'EO': ## Process steps
    sheet = xl.parse(4)
    WL=np.array(sheet[['WL[nm]']])
    WL = Range (WL)
    EDFA=np.array(sheet[['EDFA[W]']])*10**9
    T_arr = getArray(41, 851, sheet, args.pos)
    
    color=iter(cm.viridis(np.linspace(0,1,T_arr.shape[0]//5+1)))
    for ii in range(0,T_arr.shape[0],5):
        spec = T_arr[ii]#norm (T_arr[ii], EDFA)
        spec = spec.reshape((851,1))
        spec = norm (spec, EDFA)
        plt.plot (WL,spec,c = next(color), label='V'+str(ii//5),linewidth=2)
#    plt.axis([1536,1542,-2,0])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.legend(loc="upper right")
    plt.show()

    color=iter(cm.viridis(np.linspace(0,1,T_arr.shape[0]//5+1)))
    spec0 = Range(T_arr[0].reshape((851,1)))
    for ii in range(5,T_arr.shape[0],5):        
        spec = T_arr[ii]#norm (T_arr[ii], EDFA)
        spec = spec.reshape((851,1))
        spec = Range (spec)
        plt.plot (WL,ER(spec, spec0),c = next(color), label='V'+str(ii//5),linewidth=2)
#    plt.axis([1530,1560,-4,4])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("ER [dB]")
    plt.legend(loc="upper right")
    plt.show()
##############################################################################
if args.Data == 'IV': ## Process steps
    sheet = xl.parse(4)
    V=np.array(sheet[['V']])
    I=np.array(sheet[['I']])
    I=np.absolute(I)
    plt.semilogy (V/5,I, label='IV curve', marker='o', color='#000000',linestyle='none')
    plt.xlabel("Voltage [V]")
    plt.ylabel("Current [nA]")
    plt.show()

##############################################################################
# Resonance spectra of a single MRR is Lorentzian function; Photonic Microresonator Research and Applications, Ioannis Chremmos
# Get the fitting parameters for the best lorentzian
#solp, ier = fit(p1, WL, spec)
#r2 = calc_r2(volts1, lorentz(time1, *solp1))
##############################################################################
if args.Data == 'EO_A': ## Process steps
    sheet = xl.parse(7)
    WL=np.array(sheet[['WL[nm]']])
    WL = Range (WL)
    EDFA=np.array(sheet[['EDFA[W]']])*10**9
#    T_arr = getArray(9, 851, sheet, args.pos)
    DeviceLength = 5
    sampleNum = 3
    T_arr = getArray2(DeviceLength, 3, 9, 851, sheet)
    
    print ("\n transmission vs wavelength")
    color=iter(cm.viridis(np.linspace(0,1,T_arr.shape[2])))
    for ii in range(0,T_arr.shape[2],1):
        spec = T_arr[0,0,ii]#norm (T_arr[ii], EDFA)
        spec = spec.reshape((851,1))
        spec = norm (spec, EDFA)
        plt.plot (WL,spec,c = next(color), label='V'+str(ii),linewidth=2)
#    plt.axis([1536,1542,-2,0])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("Transmission")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    

    
    print ("\n The extinsion rati for negative voltage bias")
    color=iter(cm.viridis(np.linspace(0,1,5)))
    spec0 = Range(T_arr[DeviceLength-1,sampleNum-1,4].reshape((851,1)))
    for ii in [4, 3, 2, 1, 0]:        
        spec = T_arr[DeviceLength-1,sampleNum-1,ii]#norm (T_arr[ii], EDFA)
        spec = spec.reshape((851,1))
        spec = Range (spec)
        plt.plot (WL,ER(spec, spec0),c = next(color), label='V'+str(ii),linewidth=2)
#    plt.axis([1530,1560,-4,4])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("ER [dB]")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    print ("\n The extinsion rati for positive voltage bias")
    color=iter(cm.viridis(np.linspace(0,1,5)))
    spec0 = Range(T_arr[DeviceLength-1,sampleNum-1,4].reshape((851,1)))
    for ii in [4, 5, 6, 7, 8]:        
        spec = T_arr[DeviceLength-1,sampleNum-1,ii]#norm (T_arr[ii], EDFA)
        spec = spec.reshape((851,1))
        spec = Range (spec) 
        plt.plot (WL,ER(spec, spec0),c = next(color), label='V'+str(ii),linewidth=2)
#    plt.axis([1530,1560,-4,4])
    plt.xlabel("Wavelength [nm]")
    plt.ylabel("ER [dB]")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    print ("\n Extinsion ratio voltage vias for a particular device")
    ER_STD, ER_AVG = np.zeros(0), np.zeros(0)  
    spec0 = Range(T_arr[DeviceLength-1,sampleNum-1,4].reshape((851,1)))
    for ii in range (0,9,1):
        spec = T_arr[DeviceLength-1,sampleNum-1,ii]
        spec = spec.reshape((851,1))
        spec = Range (spec)
        ER_AVG = np.append (ER_AVG, np.average (ER(spec, spec0)))
        ER_STD = np.append( ER_STD, np.std (ER(spec, spec0)))        
    bias = np.linspace(-4,4,9)
    plt.errorbar(bias, np.absolute(ER_AVG), ER_STD, fmt = 'o', color='black', ecolor='lightgray', elinewidth=2, capsize=5)
    plt.xlabel("Bias Voltage [V]")
    plt.ylabel("ER [dB]")
    plt.show()
    
    print ("\n Extinsion ratio voltage vias for a range of device")
    bias = np.linspace(-4,4,9)
    color=iter(cm.viridis(np.linspace(0,1,3)))
    for dd in [3,4,5]:
        ER_STD, ER_AVG = np.zeros(0), np.zeros(0)  
        spec0 = Range(T_arr[dd-1,sampleNum-1,4].reshape((851,1)))        
        for ii in range (0,9,1):
            spec = T_arr[dd-1,sampleNum-1,ii]
            spec = spec.reshape((851,1))
            spec = Range (spec)
            ER_AVG = np.append (ER_AVG, np.average (ER(spec, spec0)))
            ER_STD = np.append( ER_STD, np.std (ER(spec, spec0)))        
        plt.errorbar(bias, np.absolute(ER_AVG), ER_STD, fmt = 'o', c = next(color), label = str(dd)+' micron')
    plt.xlabel("Bias Voltage [V]")
    plt.ylabel("ER [dB]")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()
    
    print ("\n Extinsion ratio vs device length")
    bias = np.linspace(-4,4,9)
    color=iter(cm.viridis(np.linspace(0,1,4)))
    for dd in [1,2,3,4,5]:
        ER_STD, ER_AVG = np.zeros(0), np.zeros(0)  
        spec0 = Range(T_arr[dd-1,sampleNum-1,4].reshape((851,1)))        
        for ii in range (0,9,1):
            spec = T_arr[dd-1,sampleNum-1,ii]
            spec = spec.reshape((851,1))
            spec = Range (spec)
            ER_AVG = np.append (ER_AVG, np.average (ER(spec, spec0)))
            ER_STD = np.append( ER_STD, np.std (ER(spec, spec0)))        
        plt.errorbar(bias, np.absolute(ER_AVG), ER_STD, fmt = 'o', c = next(color), label = str(dd)+' micron')
    plt.xlabel("Bias Voltage [V]")
    plt.ylabel("ER [dB]")
    plt.legend(bbox_to_anchor=(1.04,0.5), loc="center left")
    plt.show()