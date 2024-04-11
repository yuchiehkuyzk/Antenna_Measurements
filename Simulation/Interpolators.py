import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d
import csv
import scipy.interpolate

data=np.loadtxt(f"External_Excitation_V_Hpo_Warminster_rotatordetailsl_Theta90_Phi0.csv",skiprows=1,delimiter = ",")

f=interp1d(data[:,0],data[:,1],'linear')
#f = scipy.interpolate.interp1d(data[:,0], data[:,1], kind='cubic', bounds_error=False, fill_value="extrapolate")


time=np.arange(0,80,0.1)

#V_interpolate=f(time)
V_interpolate=f(time)
output_file = f'Waveform_Warminster_rotator_Theta90Phi0.csv'
print(output_file)
# Save the array values to a text file without brackets

data = list(zip(time, V_interpolate))

with open(output_file, 'w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerows(data)

plt.plot(time,V_interpolate)

plt.show()

