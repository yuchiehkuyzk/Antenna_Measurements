import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d
import scipy.signal
import os

import sys

import ImpulseResponse as IR

sys.path.append('../VNA')
import plotBoreSightGain as VNA

sys.path.append('../../rftoolkit')
from rftoolkit import waveform, tx2rx, vna

sys.path.append('../../rftoolkit/rftoolkit')
from waveform import Waveform

from scipy.constants import c

sys.path.append('/home/yzk5303/Trigger/LF_Trigger/src')
import waveform_t as wt

def read_ILoss(freq, filename): # GHz
    data=np.loadtxt(filename)
    
    lower_boundary = data[0, 1]
    upper_boundary = data[-1, 1]
    
    f=interp1d(data[:,0],data[:,1],'linear', bounds_error=False, fill_value="extrapolate")
    
    interpolated=f(freq)
    interpolated[freq < data[0, 0]] = lower_boundary
    interpolated[freq > data[-1, 0]] = upper_boundary

    return interpolated

def balun_correction():
    #SIMULATION
    c=3e8
    
    g_from_impresp = lambda freq, vfft: 4 * np.pi * ((freq/c)**2) * np.abs(vfft)**2
    
    to_dB = lambda lin: 10 * np.log10(np.abs(lin))
    
    to_linear = lambda freq, db: np.sqrt(10**(0.1*db)/( 4 * np.pi * ((freq/c)**2)))

    data=np.loadtxt("../Simulation/Waveform_Theta90Phi0.csv",skiprows=1,delimiter = ",")
    Sim_time = data[:,0]
    Sim_sinuous = data[:,1]
    print(len(Sim_time))
    # convert to wavefrom object
    sim_wave = Waveform(data=[Sim_time, Sim_sinuous])
    
    (freq_sim, Afft_sim, Afft_db_test) = sim_wave.calc_fft(rfft=True)
    
    gain_impulse_response_Sinuous_Simulation = to_dB(g_from_impresp(freq_sim, Afft_sim))

    plt.plot(freq_sim, gain_impulse_response_Sinuous_Simulation)
    
    gain_impulse_response_Sinuous_Simulation_phase = np.unwrap(np.angle(Afft_sim), period=2*np.pi)
    
    #print("sim_wave.calc_fft[0]=",freq_sim) 
    TX4_IL = read_ILoss(freq_sim, "../data/TX4_62HP_ILoss.dat")
    
    #print("TX4_IL=",TX4_IL)
    
    gain_impulse_response_Sinuous_Simulation_balun = gain_impulse_response_Sinuous_Simulation - TX4_IL
    
    plt.plot(freq_sim, gain_impulse_response_Sinuous_Simulation_balun)
    plt.xlim(0,0.7)
    plt.ylim(-180,None)
    plt.grid()
    plt.savefig("../../test_sim_gain.png")
    plt.close()

    for_convert_linear_balun = to_linear(freq_sim,gain_impulse_response_Sinuous_Simulation_balun)
    for_convert_linear = to_linear(freq_sim,gain_impulse_response_Sinuous_Simulation)

    complexWaveform_freq_balun = for_convert_linear_balun * np.exp(1j * gain_impulse_response_Sinuous_Simulation_phase)
    complexWaveform_freq = for_convert_linear * np.exp(1j * gain_impulse_response_Sinuous_Simulation_phase)
    
    # pad zero at 0 Hz
    complexWaveform_freq_balun = np.pad(complexWaveform_freq_balun, (1, 0), 'constant')
    complexWaveform_freq = np.pad(complexWaveform_freq, (1, 0), 'constant')

    recon_waveform_balun = np.real(np.fft.irfft(complexWaveform_freq_balun))
    recon_waveform = np.real(np.fft.irfft(complexWaveform_freq))

    time_axis = np.arange(len(recon_waveform_balun)) 
    print(len(time_axis))
    plt.plot(time_axis, -1*recon_waveform_balun)
    plt.plot(time_axis, -1*recon_waveform,alpha=0.5)
    #plt.plot(Sim_time, Sim_sinuous)

    plt.savefig("../../testbalun.png")
    #RG_14[i] = RG_14[i] - TX4_IL
    
if __name__=="__main__":
    balun_correction()
