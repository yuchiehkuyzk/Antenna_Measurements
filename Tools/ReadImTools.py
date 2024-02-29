import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d

def read_Imp(freq, filename):
    df=pd.read_csv(filename,header=None)
    mask = (df[0] < freq+0.001) & (df[0] > freq) 
    #print(df[mask])
    return df[mask].values.ravel()[1:3]

def read_VSWR(freq, filename):
    df=pd.read_csv(filename,header=None)
    mask = (df[0] < freq+0.001) & (df[0] > freq)
    #print(df[mask])
    return df[mask].values.ravel()[1:2]

#return gain_phi, gain_theta in unit of power ratio 


def read_gain(freq, theta, phi, filename):
    df=pd.read_csv(filename,header=None)
    mask = (df[1] < theta+1) & (df[1] > theta-1) & (df[2] < phi+1) & (df[2] > phi-1) & (df[0] < freq+0.001) & (df[0] > freq-0.001)
    #print(df[mask])
    return df[mask].values.ravel()[3:5]


def read_gain_Interpolate(freq, theta, phi, filename):
    data = np.loadtxt(filename,delimiter=',')
    mask = (data[:,1] < theta+1) & (data[:,1] > theta-1) & (data[:,2] < phi+1) & (data[:,2] > phi-1) 
    data_theta_phi = data[mask]
    Freq = data_theta_phi[:,0]
    G_phi = data_theta_phi[:,3]
    G_theta = data_theta_phi[:,4]
    
    f_phi = interp1d(Freq, G_phi,'linear')
    f_theta = interp1d(Freq, G_theta,'linear')

    return f_phi(freq), f_theta(freq)

# return H component gain in the H port in unit of dBi
def read_Hpol_Hgain(freq, theta, phi, filenameZ):
    Zport = read_gain_Interpolate(freq, theta, phi, filenameZ)
    return 10*np.log10(Zport[0])

def read_Hpol_Vgain(freq, theta, phi, filenameZ):
    Zport = read_gain_Interpolate(freq, theta, phi, filenameZ)
    return 10*np.log10(Zport[1])


def read_Vpol_Vgain(freq, theta, phi, filenameZ):
    Zport = read_gain(freq, theta, phi, filenameZ)
    return 10*np.log10(Zport[1])

def read_Vpol_Hgain(freq, theta, phi, filenameZ):
    Zport = read_gain(freq, theta, phi, filenameZ)
    return 10*np.log10(Zport[0])

def Cal_RefCoeff(freq, Z0, Imp_file):
    Imp=read_Imp(freq, Imp_file)
    ZA=complex(Imp[0],Imp[1])
    RCoeff=(ZA-Z0)/(ZA+Z0)
    return RCoeff

def Cal_VSWR(freq, Z0, Imp_file):
    RCoeff=Cal_RefCoeff(freq,Z0,Imp_file)
    VSWR=(1+abs(RCoeff))/(1-abs(RCoeff))
    return VSWR

def read_S11(freq, filename):
    df=pd.read_csv(filename,header=None)
    mask = (df[0] < freq+0.001) & (df[0] > freq)
    #print(df[mask])
    return df[mask].values.ravel()[1:3]

def read_S11_measured(freq, filename):
    df=pd.read_csv(filename,header=None)
    mask = (df[0] < freq+50000) & (df[0] >= freq)
    #print(df[mask])
    return df[mask].values.ravel()[0:2]

def read_S11_measured_interpolate(freq, filename):
    data=np.loadtxt(filename,delimiter = ",")
    f=interp1d(data[:,0],data[:,1],'linear')
    #print(df[mask])
    #print(data[:,0])
    return f(freq)


def read_S11_Transformer(freq, filename):
    data=np.loadtxt(filename)
    f=interp1d(data[:,0],data[:,1],'linear')
    #print(df[mask])
    #print(data[:,0])
    return f(freq)



def read_ILoss(freq, filename):
    data=np.loadtxt(filename)
    f=interp1d(data[:,0],data[:,1],'linear')
    #print(df[mask])
    #print(data[:,0])
    return f(freq)

def read_UnBalance(freq, filename):
    data=np.loadtxt(filename)
    f=interp1d(data[:,0],data[:,1],'linear')
    #print(df[mask])
    #print(data[:,0])
    return f(freq)


#r1,i1=read_S11_measured(1619750000,"../s11/Copy_File_0SINUOUS1.dat")
#print(r1,i1)


    

if __name__ == "__main__":
    #read_gain_Interpolate(0.1, 90, 0, "/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
    G = read_Hpol_Hgain(0.105, 90, 0, "/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
    print(G)
