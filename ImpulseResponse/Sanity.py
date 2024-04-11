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

from scipy.constants import c

sys.path.append('/home/yzk5303/Trigger/LF_Trigger/src')
import waveform_t as wt

def Average_waveform(N_start,N_repeat_LPDA):

    #====================
    #Averaging waveform
    #====================
    Dir_name = '/data2/PUEO/Warminster/CSV_DATA/LPDA_LPDA/LPH_LPH_0/'
    # Dir_name = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/08_02_2024_impulse_measurements/LPH_LPH_0/'
    LPDA_t_v = []
    Pulse_t_v = []
    max_index=[]

    file_list = os.listdir(Dir_name)
    for i, filename in enumerate(file_list):
        if i>N_start and i<N_start+N_repeat_LPDA:
            Time, ch1, ch2, ch3=IR.ReadData(Dir_name+'/'+filename)

            if np.isnan(ch1).any() or np.isnan(ch2).any() or np.isnan(ch3).any() or np.isinf(ch1).any() or np.isinf(ch2).any() or np.isinf(ch3).any():
                print("nan of inf!")
                continue

            idx_v1 = np.where(ch2==np.max(ch2))
            max_index.append(idx_v1)
            LPDA_t_v.append(ch2)
            Pulse_t_v.append(ch1)

    LPDA_t_v_avg = np.mean(np.array(LPDA_t_v), axis = 0)
    Pulse_t_v_avg = np.mean(np.array(Pulse_t_v), axis = 0)

    rx_data = [Time, LPDA_t_v_avg]
    pulse_data = [Time, Pulse_t_v_avg]

    return rx_data, pulse_data

def IR_waveforms(rx_data, pulse_data):

    phaseLP = 56.5 / 2
    d_r = (43*12+1+3/4)*0.0254 # Distance btw backs of two LPDA
    d_r_LPDA = (43*12+1+3/4 - phaseLP*2)*0.0254 # Assume physical center as the Phase center
    d_r_sinuous = (54*12+2-(11*12+2+3/4))*0.0254 - phaseLP*0.0254#  Distance btw back of the tx (LPDA) to Sinuous plane - phaseLP

    c=3e8
    g_from_impresp = lambda freq, vfft: 4 * np.pi * ((freq/c)**2) * np.abs(vfft)**2
    to_dB = lambda lin: 10 * np.log10(np.abs(lin))

    addCompS21 = []
    subCompS21 = []
    s21_data_dir = '/data2/PUEO/Warminster/pulsingSetupS21s/'
    #s21_data_dir = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s'
    S21_pulse_List = [s21_data_dir+'S2A1.s2p', s21_data_dir+'S2A2.s2p', s21_data_dir+'A2C1.s2p']
    S21_Rx_V_List = [s21_data_dir+'S1T.s2p', s21_data_dir+'RAVC3.s2p']
    S21_Rx_H_List = [s21_data_dir+'S1T.s2p', s21_data_dir+'RBHC2.s2p']


    for s2p in S21_pulse_List:
        v = vna.S2P(s2p)
        v.smooth()
        addCompS21.append(v)
    for s2p in S21_Rx_H_List:
        v = vna.S2P(s2p)
        v.smooth()
        subCompS21.append(v)


    #=====================================
    # Use tx2rx func to calculate impulse
    #======================================

    impResp_t, impResp, impResp_f, impRespfft = tx2rx.impulseResponse(rx_data, pulse_data, d_r_LPDA, addComponentsS21=addCompS21, subtractComponentsS21=subCompS21,verbose=1,noise=(-150,-130),wienerFilter=False ,returnDomain='both')
    print(impResp_t*1e9, impResp*1e-9)
    plt.plot(impResp_t*1e9,impResp*1e-9)
    plt.xlim(0,200)
    plt.grid()
    plt.savefig("../../Sanity_testImpulseResponse.png")
    plt.close()

    print("len(impResp_f) = ",len(impResp_f))

    gain_impulse_response_LPDA = to_dB(g_from_impresp(impResp_f, impRespfft))
    gain_impulse_response_LPDA_phase = np.rad2deg(np.unwrap(np.angle(impRespfft), period=np.pi))
    #Tx_gain_complex = [impResp_f, gain_impulse_response_LPDA, gain_impulse_response_LPDA_phase]

    evalFreqHz, gain_LPDA = tx2rx.gain(rx_data, pulse_data, d_r_LPDA, gainTx=None, addComponentsS21=addCompS21, subtractComponentsS21=subCompS21, verbose=1)
    Tx_gain_complex = [impResp_f, gain_impulse_response_LPDA, gain_impulse_response_LPDA_phase]


    plt.plot(impResp_f, gain_impulse_response_LPDA, label='from Impulse')
    plt.plot(evalFreqHz, gain_LPDA)
    plt.xlim(0,300e6)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_testLPDA_gain.png")
    plt.close()
    
    return impResp_t, impResp, impResp_f, impRespfft

if __name__=="__main__":
    
    rx_data_1, pulse_data_1 = Average_waveform(0,300)
    rx_data_2, pulse_data_2 = Average_waveform(0,900)

    impResp_t_1, impResp_1, impResp_f_1, impRespfft_1 = IR_waveforms(rx_data_1, pulse_data_1)
    impResp_t_2, impResp_2, impResp_f_2, impRespfft_2 = IR_waveforms(rx_data_2, pulse_data_2)
    
    plt.plot(rx_data_1[0]*1e9,rx_data_1[1]*1e-9,label='0,300')
    plt.plot(rx_data_2[0]*1e9,rx_data_2[1]*1e-9,label='0,900',alpha=0.5)
    plt.xlim(-10,10)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_Rx.png")
    plt.close()
    
    plt.plot(rx_data_1[0]*1e9,rx_data_1[1]*1e-9,label='0,300')
    plt.plot(rx_data_2[0]*1e9,rx_data_2[1]*1e-9,label='0,900',alpha=0.5)
    plt.xlim(10,50)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_Rx_1.png")
    plt.close()

    plt.plot(rx_data_1[0]*1e9,rx_data_1[1]*1e-9,label='0,300')
    plt.plot(rx_data_2[0]*1e9,rx_data_2[1]*1e-9,label='0,900',alpha=0.5)
    plt.xlim(50,100)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_Rx_2.png")
    plt.close()

    plt.plot(rx_data_1[0]*1e9,rx_data_1[1]*1e-9,label='0,300')
    plt.plot(rx_data_2[0]*1e9,rx_data_2[1]*1e-9,label='0,900',alpha=0.5)
    plt.xlim(-50,0)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_Rx_3.png")
    plt.close()
    
    plt.plot(rx_data_1[0]*1e9,rx_data_1[1]*1e-9,label='0,300')
    plt.plot(rx_data_2[0]*1e9,rx_data_2[1]*1e-9,label='0,900',alpha=0.5)
    plt.xlim(-100,-50)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_Rx_4.png")
    plt.close()
    
    plt.plot(rx_data_1[0]*1e9,(rx_data_1[1]-rx_data_2[1])/rx_data_2[1],label='residuals')
    plt.grid()
    plt.legend()
    plt.ylim(-10,10)
    plt.xlim(0,200)
    plt.savefig("../../Sanity_Rx_Rsdiduals.png")
    plt.close()


    plt.plot(pulse_data_1[0]*1e9,pulse_data_1[1]*1e-9,label='0,300')
    plt.plot(pulse_data_2[0]*1e9,pulse_data_2[1]*1e-9,label='0,900',alpha=0.5)
    plt.xlim(-10,10)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_Pulse.png")
    plt.close()
    


    plt.plot(impResp_t_1*1e9,impResp_1*1e-9,label='0,300')
    plt.plot(impResp_t_2*1e9,impResp_2*1e-9,label='0,900')

    plt.xlim(0,200)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_testImpulseResponse.png")
    plt.close()

    plt.plot(impResp_f_1, impRespfft_1)
    plt.plot(impResp_f_2, impRespfft_2)
    plt.xlim(0,550e6)
    plt.grid()
    plt.legend()
    plt.savefig("../../Sanity_testLPDA_gain.png")
    plt.close()

