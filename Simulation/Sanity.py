import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d
import scipy.signal
import os


dataWarm=np.loadtxt("Waveform_Warminster_rotator_Theta90Phi0.csv",skiprows=1,delimiter = ",")
Sim_time_Warm = dataWarm[:,0]
Sim_sinuous_Warm = dataWarm[:,1]

data=np.loadtxt("Waveform_Warminster_rotator_Theta90Phi0.csv",skiprows=1,delimiter = ",")
Sim_time = data[:,0]
Sim_sinuous = data[:,1]


threshold = 0.15


idx_v1 = np.where(abs(Sim_sinuous_Warm) > threshold)[0][0]
idx_v2 = np.where(abs(Sim_sinuous) > threshold)[0][0]
time_shift = Sim_time_Warm[idx_v1] - Sim_time[idx_v2]

plt.plot(Sim_time_Warm + time_shift, Sim_sinuous_Warm)
plt.plot(Sim_time + time_shift, Sim_sinuous)
plt.show()




