import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d
import scipy.signal
import os

import sys
sys.path.append('/home/yuchieh/Research/code/antenna/sinuous/NCS95R')
import ReadGainTools as rI
sys.path.append('/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster')
import plotBoreSightGain as VNA

sys.path.append('/home/yuchieh/Research/code/Trigger/Waveform')
import Waveforms as WaveformSIM 

sys.path.append('/home/yuchieh/Research/code/antenna/sinuous/AngularResolution')
import waveform_t as wt

sys.path.append('/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/Zack_package')
from waveform import Waveform 
#import tx2rx

def ReadLPDA_dimension(filename,freq): #freq in MHz
    data=np.loadtxt(filename,delimiter="\t",skiprows=1,dtype=float)
    
    f=interp1d(data[:,0],data[:,1],'linear') 
    min_freq = data[0, 0]  # Minimum frequency in the data
    max_freq = data[-1, 0]  # Maximum frequency in the data
    clipped_freq = np.clip(freq, min_freq, max_freq)
    return f(clipped_freq)

def ReadData(filename):
    data=np.loadtxt(filename,skiprows=13,delimiter=',',dtype=float)
    time=data[:,0]
    ch1=data[:,1]
    ch2=data[:,2]
    ch3=data[:,3]

    return time, ch1, ch2, ch3

def ReadLPDAspec(filename):
    data=np.loadtxt(filename,skiprows=1,delimiter=',',dtype=float)
    freq=data[:,0] #GHz
    gain=data[:,1] #dBi

    return freq, gain

def ReadS21(filename):
    file1 = open(filename, 'r+')
    data=np.loadtxt(filename,delimiter="\t",skiprows=13,dtype=str)
    data=data[:-1]
    Freq=np.array(data[:,0]).astype(float)
    S21=np.array(data[:,3]).astype(float)
    S21_phi=np.array(data[:,4]).astype(float)

    return Freq,S21,S21_phi


def Filter(Time,Wave,low,high):
    sp=np.fft.fft(Wave)
    freq=np.fft.fftfreq(Time.size,d=Time[1]-Time[0])
    cond1=np.where(np.logical_and(np.rint(freq/1e6)<low,np.rint(freq/1e6)>-1*low))
    cond2=np.where(np.logical_or(np.rint(freq/1e6)>high,np.rint(freq/1e6)<-1*high))
    sp[cond1]=0
    sp[cond2]=0
    return np.fft.ifft(sp)

def impulseReponse_gainAVG(N, d_r, f_S21_Receiver_V_att,f_S21_Pulser_att,Dir_name):

    file_list = os.listdir(Dir_name)

    file_list = [file for file in file_list if os.path.isfile(os.path.join(Dir_name, file))]

    LPDA_gain_values = []
    for i, filename in enumerate(file_list):
        if i>0 and i<N:
            Time, ch1, ch2, ch3=ReadData(Dir_name+'/'+filename)

            (freq, ch1psd) = scipy.signal.periodogram(ch1, 1/(Time[1]-Time[0]), scaling='density')
            (freq2, ch2psd) = scipy.signal.periodogram(ch2, 1/(Time[1]-Time[0]), scaling='density')
            (freq3, ch3psd) = scipy.signal.periodogram(ch3, 1/(Time[1]-Time[0]), scaling='density')
             
            freq = freq[1:]
            freq2 = freq2[1:]
            freq3 = freq3[1:]
            ch1psd = ch1psd[1:]
            ch2psd = ch2psd[1:]
            ch3psd = ch3psd[1:]

            LPDA_phase_center_offset = 56.6 - ReadLPDA_dimension('/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/LPDA_SAS517_dimension.dat',freq/1e6) # in inches
            d_r_cal = d_r*np.ones(len(freq))-LPDA_phase_center_offset*0.0254*2 # 2 for LPDA-LPDA
            Friis_distance_factor=20*np.log10((3e8/freq)/(4*np.pi*d_r_cal))
            
            At_RX = 10*np.log10(ch2psd)-f_S21_Receiver_V_att(freq2)
            At_Tx = 10*np.log10(ch1psd)-f_S21_Pulser_att(freq)
            
            #Firiis equation: G1 + G2 + Friis_distance_factor = S21 = At_RX - At_Tx
            
            LPDA_gain = 1/2*(At_RX-At_Tx-Friis_distance_factor)
            if np.isnan(LPDA_gain).any():
                 continue
            LPDA_gain_values.append(LPDA_gain)
            print(i,len(LPDA_gain_values))
            
            debug=False
            if i==1 and debug==True: 
                plt.figure(figsize=(30, 20))

                plt.plot(freq/1e6,10*np.log10(ch2psd), label = "10*np.log10(ch2_PSD)")
                plt.plot(freq/1e6, 10*np.log10(ch1psd), label='10*np.log10(ch1_PSD)')
                plt.plot(freq/1e6,f_S21_Receiver_V_att(freq2),label='S21_Receiver_Attenuation')
                plt.plot(freq/1e6, f_S21_Pulser_att(freq),label='S21_Pulser_Attenuation')
                plt.plot(freq/1e6, Friis_distance_factor,label='Friis_distance_factor')
                plt.plot(freq/1e6, 10*np.log10(ch2psd)-10*np.log10(ch1psd),label='10*np.log10(ch2_PSD)-10*np.log10(ch1_PSD)')
                plt.plot(freq/1e6, 10*np.log10(ch2psd)-f_S21_Receiver_V_att(freq2)-10*np.log10(ch1psd)+f_S21_Pulser_att(freq)-Friis_distance_factor, label='2*Gain_LPDA = 10*np.log10(ch2_PSD)-s21_Receiver_Attenuation-10*np.log10(ch1_PSD)+S21_Pulser_Attenuation-Friis_distance_factor')
                plt.plot(freq/1e6, 1/2*(10*np.log10(ch2psd)-f_S21_Receiver_V_att(freq2)-10*np.log10(ch1psd)+f_S21_Pulser_att(freq)-Friis_distance_factor), label='Gain_LPDA = 1/2*(10*np.log10(ch2_PSD)-s21_Receiver_Attenuation-10*np.log10(ch1_PSD)+S21_Pulser_Attenuation-Friis_distance_factor)')
                plt.xlim(0,1500)
                plt.xlabel("MHz")
                plt.ylabel("dB")
                plt.legend(fontsize=25)
                plt.savefig("LPDA_breakdown.png")
                plt.show()
                plt.close()
    
    
    LPDA_gain_arrays = np.array(LPDA_gain_values)
    average_LPDA_gain_array = np.mean(LPDA_gain_arrays, axis=0)
    return freq, average_LPDA_gain_array

def impulseReponse_SinuousgainAVG(N, d_r, f_S21_Receiver_V_att,f_S21_Receiver_H_att,f_S21_Pulser_att,LPDA_gain,Dir_name):

    file_list = os.listdir(Dir_name)

    file_list = [file for file in file_list if os.path.isfile(os.path.join(Dir_name, file))]

    SIN_gainH_values = []
    SIN_gainV_values = []

    for i, filename in enumerate(file_list):
        if i>0 and i<N:
            Time, ch1, ch2, ch3=ReadData(Dir_name+'/'+filename)

            (freq, ch1psd) = scipy.signal.periodogram(ch1, 1/(Time[1]-Time[0]), scaling='density')
            (freq2, ch2psd) = scipy.signal.periodogram(ch2, 1/(Time[1]-Time[0]), scaling='density')
            (freq3, ch3psd) = scipy.signal.periodogram(ch3, 1/(Time[1]-Time[0]), scaling='density')
            
            freq = freq[1:]
            freq2 = freq2[1:]
            freq3 = freq3[1:]
            ch1psd = ch1psd[1:]
            ch2psd = ch2psd[1:]
            ch3psd = ch3psd[1:]

            LPDA_phase_center_offset = 56.6 - ReadLPDA_dimension('/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/LPDA_SAS517_dimension.dat',freq/1e6) # in inches
            d_r_cal = d_r*np.ones(len(freq))-LPDA_phase_center_offset*0.0254 # 1 for one LPDA

            Friis_distance_factor_sinuous=20*np.log10((3e8/freq)/(4*np.pi*d_r_cal))
            At_RXH = 10*np.log10(ch2psd)-f_S21_Receiver_H_att(freq2)
            At_RXV = 10*np.log10(ch3psd)-f_S21_Receiver_V_att(freq3)
            At_Tx = 10*np.log10(ch1psd)-f_S21_Pulser_att(freq)
            
            sinuous_gainH = At_RXH-At_Tx-Friis_distance_factor_sinuous-LPDA_gain
            sinuous_gainV = At_RXV-At_Tx-Friis_distance_factor_sinuous-LPDA_gain
           
            SIN_gainH_values.append(sinuous_gainH)
            SIN_gainV_values.append(sinuous_gainV)

            print(i,len(SIN_gainH_values))

            debug = False
            if i==1 and debug==True:
                plt.figure(figsize=(30, 20))
                plt.plot(freq/1e6,f_S21_Receiver_V_att(freq2),label='S21_Receiver_Attenuation')
                plt.plot(freq/1e6,f_S21_Pulser_att(freq),label='S21_Pulser_Attenuation')
                
                plt.plot(freq/1e6,10*np.log10(ch3psd)-f_S21_Receiver_V_att(freq3),label='At_RX (power at Rx) = 10*np.log10(ch3psd)-S21_Receiver_Attenuation')
                plt.plot(freq/1e6,10*np.log10(ch1psd)-f_S21_Pulser_att(freq),label='At_TX (power at Tx) = 10*np.log10(ch1psd)-S21_Pulser_Attenuation')

                plt.plot(freq/1e6, Friis_distance_factor_sinuous,label=f'Friis_distance_factor_sinuous (d={d_r:2f} m)')
                
                plt.plot(freq/1e6, At_RXV-At_Tx-Friis_distance_factor_sinuous-LPDA_gain, label='$G_{sinuous}$ = At_RX-At_Tx-Friis_distance_factor-$G_{LPDA}$')

                plt.plot(freq/1e6, LPDA_gain,label='$G_{LPDA}$ (measured)')
                plt.title("LPV_LFBF_V_H_00, port V, breakdown")
                plt.xlim(0,1000)
                plt.xlabel("MHz")
                plt.ylabel("dB")
                plt.yticks([-160,-140,-120, -100, -80, -60, -40, -20,0,5,10,15])
                plt.grid()
                plt.legend(fontsize=12)

                plt.savefig("sinuous_breakdown.png")
                plt.show() 

    SIN_gainH_arrays = np.array(SIN_gainH_values)
    average_SIN_gainH_array = np.mean(SIN_gainH_arrays, axis=0)
    SIN_gainV_arrays = np.array(SIN_gainV_values)
    average_SIN_gainV_array = np.mean(SIN_gainV_arrays, axis=0)
    return freq, average_SIN_gainH_array, average_SIN_gainV_array


def ImpulseResponse():
    RAW_DATA_PLOT=False
    S21 = True
    LPDA = True
    N_LPDA = 500
    savetxt= False
    SINUOUS = True
    N_SINUOUS = 300
    SINUOUS_SIM = True
    SINUOUS_SIM_ROTATOR = False
    PLOT_VNA = True
    CAL_OPTION = 'spec'
    PLOT = True
    savefigname = f'plot/PHASE_CENTER_DetailCal_BF_VNA_all_{CAL_OPTION}.png'
    #Ch1: pulser-BarthAtt-splitter-long cable-scope
    #Ch2/3: pulser-BarthAtt-splitter-cable-LPDA-Air-RX-Cable-Scope
    if S21:
        freq_S2C1, s21_S2C1, s21_phi_S2C1 = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/S2C1.s2p")
        freq_S2A2, s21_S2A2, s21_phi_S2A2 = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/S2A2.s2p")
        freq_S2A1, s21_S2A1, s21_phi_S2A1 = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/S2A1.s2p")
        freq_A2C1, s21_A2C1, s21_phi_A2C1 = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/A2C1.s2p")
        freq_S1T, s21_S1T, s21_phi_S1T = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/S1T.s2p")
        freq_RBHC2, s21_RBHC2, s21_phi_RBHC2 = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/RBHC2.s2p")
        freq_RAVC3, s21_RAVC3, s21_phi_RAVC3 = ReadS21("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/pulsingSetupS21s/RAVC3.s2p")


        S21_Pulser_att = s21_S2A1 + s21_S2A2 + s21_A2C1
        S21_Receiver_V_att = s21_S1T + s21_RAVC3
        S21_Receiver_H_att = s21_S1T + s21_RBHC2

        
        #Interpolator
        f_S21_Pulser_att = interp1d(freq_S2C1,S21_Pulser_att,'linear', fill_value='extrapolate') 
        f_S21_Receiver_V_att = interp1d(freq_S2C1,S21_Receiver_V_att,'linear',  fill_value='extrapolate')
        f_S21_Receiver_H_att = interp1d(freq_S2C1,S21_Receiver_H_att,'linear',  fill_value='extrapolate')

        plt.plot(freq_S2C1/1e6,s21_S2C1,label='S2C1', alpha=0.4)
        plt.plot(freq_S2A1/1e6, S21_Pulser_att, label='CH1 Atten: S2A1 + S2A2 + A2C1')
        plt.plot(freq_S2A1/1e6, S21_Receiver_V_att, label='CH3 (V) Atten: s21_S1T + s21_RAVC3')
        plt.plot(freq_S2A1/1e6, S21_Receiver_H_att, label='CH2 (V) Atten: s21_S1T + s21_RBHC3')

        plt.xlim(50,550)
        plt.grid()
        plt.legend()
        #plt.savefig("S21_sanity.png")
        plt.close()
    
    if RAW_DATA_PLOT:
        file_list = os.listdir("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/LPH_LFBF_V_E_00/")
        for i,filename in enumerate(file_list):
            Time, ch1, ch2, ch3 = ReadData(f"/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/LPH_LFBF_V_E_00/"+filename)
            fig, ax = plt.subplots(3,sharex=True,figsize=(10,10))
            #plt.subplot(3,1,1)
            ax[0].plot(Time*1e9,ch1,label='CH1',color='gold')
            ax[0].tick_params(labelsize=15)
            #plt.subplot(3,1,2)
            ax[1].plot(Time*1e9,ch2,label='CH2',color='blue')
            ax[1].tick_params(labelsize=15)
        
            #plt.subplot(3,1,3)
            ax[2].plot(Time*1e9,ch3,label='CH3',color='red')
            ax[2].tick_params(labelsize=15)
            ax[2].set_xlabel('Time (ns)', fontsize=25)
            ax[1].set_ylabel('Voltage (V)', fontsize=25)
            ax[0].legend()
            ax[1].legend()
            ax[2].legend()
            if i==1:
                plt.savefig(f"Rawdata_{i}.png")
                plt.show()
    
    #LPDA spec sheet
    freq_LPDA_spec, gain_LPDA_spec = ReadLPDAspec("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/beampattern/UNCal/CAL/LP196(0.080-4.000GHz).tbl")
    
    

    # Pulsing
    d_r = (43*12+1+3/4)*0.0254 # Distance btw backs of two LPDA
    d_r_sinuous = (54*12+2-(11*12+2+3/4))*0.0254 # Distance btw back of the tx (LPDA) to Sinuous plane
    
    if LPDA:
        # Specify the directory path
        directory_LPDA_cal_VV = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/08_02_2024_impulse_measurements/LPV_LPV_0'  # Change this to the directory path you want to list
        directory_LPDA_cal_HH = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/08_02_2024_impulse_measurements/LPH_LPH_0' 
        freqLPDA_VV, gain_LPDA_VV = impulseReponse_gainAVG(N_LPDA, d_r,f_S21_Receiver_V_att ,f_S21_Pulser_att,directory_LPDA_cal_VV)
        freqLPDA_HH, gain_LPDA_HH = impulseReponse_gainAVG(N_LPDA, d_r, f_S21_Receiver_H_att ,f_S21_Pulser_att,directory_LPDA_cal_HH)
    
        plt.plot(freqLPDA_VV/1e6, gain_LPDA_VV,label='LPDA gain (V-V pulsing)',alpha=0.8)
        plt.plot(freqLPDA_HH/1e6, gain_LPDA_HH,label='LPDA gain (H-H pulsing)',alpha=0.8)
        plt.plot(freq_LPDA_spec*1e3, gain_LPDA_spec,label='LPDA spec sheet')
        plt.axvspan(50, 550,color = 'black', alpha=0.1, label='50-550 MHz')
        plt.axvspan(200, 1300,color = 'darkred', alpha=0.1, label='200-1300 MHz')

        plt.xlim(0,3000)
        plt.ylim(-5,10)
        plt.xlabel("MHz")
        plt.ylabel("dBi")
        plt.legend()
        plt.grid()
        plt.savefig("PHASE_CENTER_DetailCAL_wide_LPDAgain_from_ImpulseResponse_avg_VV_HH_spec.png")
        plt.close()
        
        if savetxt:
            LPDA_saveHH = np.column_stack((freqLPDA_HH/1e9, gain_LPDA_HH))
            LPDA_saveVV = np.column_stack((freqLPDA_VV/1e9, gain_LPDA_VV))
            np.savetxt('LPDA_HH_Measured.txt', LPDA_saveHH)
            np.savetxt('LPDA_VV_Measured.txt', LPDA_saveVV)


    
    # sinuous
    measurementList = ['LPH_LFBF_V_XE','LPH_LFBF_V_E','LPV_LFBF_V_H','LPV_LFBF_V_XH']
    #measurementList = ['LPH_LFBB_V_XE','LPH_LFBB_V_E','LPV_LFBB_V_H','LPV_LFBB_V_XH']
    colorList = ['darkred','red','rosybrown','sienna']
    if SINUOUS:
        plt.figure(figsize=(30, 20))
        for i,measurement in enumerate(measurementList):
            directory_sinuous = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/'+measurement+'_00'  
            
            if 'LPH' in measurement:
                LPDA_gain_used = gain_LPDA_HH
            else:
                LPDA_gain_used = gain_LPDA_VV

            freq_sinuous, average_SIN_gain_ch2, average_SIN_gain_ch3 = impulseReponse_SinuousgainAVG(N_SINUOUS, d_r_sinuous, f_S21_Receiver_V_att,f_S21_Receiver_H_att,f_S21_Pulser_att,LPDA_gain_used,directory_sinuous)
            
            if measurement=='LPH_LFBF_V_XE' or measurement == 'LPV_LFBF_V_XH' or measurement=='LPH_LFBB_V_XE' or measurement == 'LPV_LFBB_V_XH':
                copol_gain = average_SIN_gain_ch2
                xpol_gain = average_SIN_gain_ch3

            else:
                copol_gain = average_SIN_gain_ch3
                xpol_gain = average_SIN_gain_ch2

            plt.plot(freq_sinuous/1e6, copol_gain, color=plt.get_cmap('gist_heat')(1-float(i)/float(len(measurementList)+2)-0.2),label='Warminster Pulsing: '+measurement,linewidth=3)
            plt.plot(freq_sinuous/1e6, xpol_gain, color=plt.get_cmap('gist_heat')(1-float(i)/float(len(measurementList)+2)-0.2), alpha=0.8, linestyle='--')

            #plt.plot(freq/1e6, average_sinuous_gain_arrayX,color='darkred',linestyle='--')


    # Simulation:
    if SINUOUS_SIM:
        freqepd600 = np.linspace(0.05, 0.60 , num=56)
        RG_14=np.zeros(56)
        RG_14_cross=np.zeros(56)
        RG_Warminster=np.zeros(56)
        RG_Warminster_cross=np.zeros(56)
        fixedphi=0
        fixedtheta=90
        for i in range(0,56):
            RG_14[i]=rI.read_Hpol_Hgain(freqepd600[i],fixedtheta,fixedphi,"/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
            RG_14_cross[i]=rI.read_Hpol_Vgain(freqepd600[i],fixedtheta,fixedphi,"/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
            RG_Warminster[i]=rI.read_Hpol_Hgain(freqepd600[i],fixedtheta,fixedphi,"/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_Warminster_rotators_details.csv")
            RG_Warminster_cross[i]=rI.read_Hpol_Vgain(freqepd600[i],fixedtheta,fixedphi,"/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_Warminster_rotators_details.csv")


        plt.plot(freqepd600*1e3*65/60, RG_14, label='simulation',color='indigo',linewidth=3)
        plt.plot(freqepd600*1e3*65/60, RG_14_cross,color='indigo',linestyle='--')
        
        if SINUOUS_SIM_ROTATOR:
            plt.plot(freqepd600*1e3, RG_Warminster, label='simulation with Rotator details',color='darkorange')
            plt.plot(freqepd600*1e3, RG_Warminster_cross, label='Xpol',color='darkorange',linestyle='--')

    
    # VNA:
    measurementList_VNA = ['LPH_LFBF_H_E','LPH_LFBF_V_E','LPV_LFBF_V_H','LPV_LFBF_H_H']
    colorList_VNA = ['darkgreen','olive','darkseagreen','lightseagreen']
    if PLOT_VNA:
        for i, measurement in enumerate(measurementList_VNA):
            freq_array, boresight_gain = VNA.freq_spectrum('/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/beampattern/UNCal/LFBF/'+measurement+'.lst', 0, 181, CAL_OPTION)
            plt.plot(freq_array*1000, boresight_gain, color=plt.get_cmap('viridis')(1-float(i)/float(len(measurementList_VNA)+2)-0.2),label='Warminster VNA setup: '+ measurement, linewidth=3)

    plt.axvspan(85, 500,color = 'black', alpha=0.2,label = '85(Turn on)-500 MHz')

    plt.xlim(0,700)
    plt.ylim(-15,10)
    plt.xlabel("MHz",fontsize=50)
    plt.ylabel("dBi",fontsize=50)
    plt.xticks(fontsize=35)
    plt.yticks(fontsize=35)
    #plt.title("Tx:LPDA V; Rx: sinuous: V")
    #plt.title("Sinuous gain at boresight")
    plt.legend(fontsize=25)
    plt.grid(True)
    if PLOT:
        plt.savefig(savefigname)
    plt.show()

def LPDA_IR_Waveform(Dir_name):

    LPDA_IR_values = []
    LPDA_t_v = []
    file_list = os.listdir(Dir_name)
    for i, filename in enumerate(file_list):
        if i>0 and i<20:
            Time, ch1, ch2, ch3=ReadData(Dir_name+'/'+filename)

            if np.isnan(ch1).any() or np.isnan(ch2).any() or np.isnan(ch3).any():
                print("nan!")
                continue
            print(len(LPDA_t_v))
            LPDA_t_v.append(ch2)


    #LPDA_IR_values_arrays = np.array(LPDA_IR_values)
    #LPDA_IR_values_avg = np.mean(LPDA_IR_values_arrays, axis=0)
    LPDA_t_v_avg=np.mean(np.array(LPDA_t_v),axis=0)
    plt.plot(Time*1e9, LPDA_t_v_avg)
    plt.title("LPDA conv LPDA impulse response")
    plt.xlim(-50,200)
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")
    plt.grid()
    plt.savefig("plot/LPDA_impulseResponse.png")
    plt.show()
    plt.close()

    #pulseWave = Waveform(data=[Time,ch1])
    RxWave = Waveform(data=[Time,LPDA_t_v_avg])

    #LPDA_t_v.append(ch2)

    (freq, Afft, Afft_db)=RxWave.calc_fft(rfft=True)

    # deconvolution

    phase = np.unwrap(np.angle(Afft))

    phase_deconv = phase/2

    Amp_deconv = np.sqrt(abs(Afft))

    A_complex_deconv = Amp_deconv * np.exp(1j * phase_deconv)
    
    # plot
    plt.plot(freq/1e6, phase, label='phase')
    plt.plot(freq/1e6, phase_deconv, label='phase/2')
    plt.xlabel("Freq (MHz)")
    plt.ylabel("Phase (rad)")
    plt.title("unwrapped phase")
    plt.grid()
    plt.legend()
    plt.savefig("LPDA_phase.png")
    plt.close()

    LPDA_v_t = np.fft.irfft(A_complex_deconv)
    t_len=len(LPDA_v_t)
    plot_time = np.linspace(0, (t_len - 1) * (Time[1]-Time[0])*1e9, t_len)
    plt.plot(plot_time, LPDA_v_t)
    plt.xlim(50,300)
    plt.title("LPDA impulse response")
    plt.xlabel("Time (ns)")
    plt.ylabel("Voltage (V)")
    plt.grid()
    plt.savefig("plot/LPDA_deconv_impulseResponse.png")

    return A_complex_deconv

def Sinuous_IR_Waveform():

    Dir_name = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/08_02_2024_impulse_measurements/LPH_LPH_0'

    A_complex_deconv = LPDA_IR_Waveform(Dir_name)

    FilterHi = 0.5 #GHz
    FilterLo = 0.05 #GHz
    
    # LPDA conv SIN
    run='LPH_LFBF_V_E_00' # LPH_LFBF_V_XE_00, LPH_LFBF_V_E_00, LPV_LFBF_V_H_00, LPV_LFBF_V_XH_00 
    Flip = -1

    Dir_name_LP_SIN = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/'+run

    file_list_sin = os.listdir(Dir_name_LP_SIN)
    LPDA_conv_SIN_v_t = []

    for i, filename in enumerate(file_list_sin):
        if i>0 and i<20:

            Time_SIN, ch1_SIN, ch2_SIN, ch3_SIN = ReadData(Dir_name_LP_SIN+'/'+os.listdir(Dir_name_LP_SIN)[1])

            LPDA_conv_SIN_v_t.append(ch3_SIN)


    LPDA_conv_SIN_v_t_avg = np.mean(np.array(LPDA_conv_SIN_v_t),axis=0)

    LP_SIN_RX_Wave = Waveform(data=[Time_SIN,LPDA_conv_SIN_v_t_avg])

    (freq_sin, Afft_sin, Afft_db_sin)=LP_SIN_RX_Wave.calc_fft(rfft=True)

    Afft_sin = Afft_sin/A_complex_deconv

    Sinuous_v_t = np.fft.irfft(Afft_sin)
    t_len=len(Sinuous_v_t)
    plot_time = np.linspace(0, (t_len - 1) * (Time_SIN[1]-Time_SIN[0])*1e9, t_len)


    band_pass_Sinuous_v_t = wt.butter_bandpass_filter(Sinuous_v_t, FilterLo, FilterHi, 1.0 / (plot_time[1] - plot_time[0]), order=2)
    #plt.plot(plot_time,Sinuous_v_t)
    #plt.plot(plot_time, Sinuous_v_t, label='Sinuous Impulse Response (LPDA pulsing Sinuous)', color='darkred')
    
    band_pass_Sinuous_v_t = band_pass_Sinuous_v_t / max(abs(band_pass_Sinuous_v_t))

    plt.plot(plot_time, band_pass_Sinuous_v_t, label=f'Measured(pulsing setup): {int(FilterLo*1000)}-{int(FilterHi*1000)} MHz', linewidth=2,color='red')


    #SIMULATION

    data=np.loadtxt("../Simulation/Waveform_Theta90Phi0.csv",skiprows=1,delimiter = ",")
    Sim_time = data[:,0]
    Sim_sinuous = data[:,1]
    #Sim_time, Sim_sinuous = WaveformSIM.IR_waveform(90,0,3)
    Sim_sinuous = wt.butter_bandpass_filter(Sim_sinuous, FilterLo, FilterHi, 1.0 / (Sim_time[1] - Sim_time[0]), order=2)
    
    Sim_sinuous = Sim_sinuous / max(abs(Sim_sinuous))* Flip 

    threshold = 0.2
    idx_v1 = np.where(band_pass_Sinuous_v_t > threshold)[0][0]
    idx_v2 = np.where(Sim_sinuous > threshold)[0][0]
    time_shift = plot_time[idx_v1] - Sim_time[idx_v2]
    

    plt.plot(Sim_time+time_shift, Sim_sinuous, linestyle='--',color='darkblue',label=f'Simulation :{int(FilterLo*1000)}-{int(FilterHi*1000)} MHz')
    
    plt.grid()
    plt.title('Sinuous Antenna Impulse Response:'+run)
    plt.xlabel("Time (ns)")
    plt.ylabel("Normalized Voltage")
    plt.legend(fontsize=9)
    plt.xlim(50,140)
    plt.savefig(f"plot/ImpulseResponse/Sinuous_IR_{run}_Filter{FilterLo}_{FilterHi}.png")


    #plt.close()
    DeconWave = Waveform(data=[plot_time/1e9,Sinuous_v_t])

    (freq_test, Afft_test, Afft_db_test)=DeconWave.calc_fft(rfft=True)
    #plt.plot(freq_test,Afft_db_test)


    plt.show()


    
if __name__=="__main__":
    #Sinuous_IR_Waveform()
    ImpulseResponse()
    #Dir_name = '/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/data/08_02_2024_impulse_measurements/LPH_LPH_0'

    #A_complex_deconv = LPDA_IR_Waveform(Dir_name)

