import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmath
from scipy.interpolate import interp1d
import scipy.signal
import sys
sys.path.append('../Tools')
import ReadImTools as rI

def ReadWarminsterData(filename, freq, Nangle, CAL):
    # Open the file in read mode
    with open(filename, "r") as file:
        # Flag to indicate whether we've found the target line
        found_target_line = False
        # Counter to keep track of the lines read after the target line
        lines_after_target = 0
        # List to store the lines in "number number number" format
        numbers_array = []

        # Iterate through each line in the file
        for line in file:
            # Check if the target line is found
            #if f"Frequency:     {freq:.5f} GHz" in line: #5
            if f"Freq:    {freq:.3f} GHz" in line or f"Frequency:     {freq:.5f} GHz" in line:
                found_target_line = True
                # Read the next 20 lines
                for _ in range(7):
                    next(file)
                for _ in range(Nangle): #19 (test),181, 21(cal)
                    next_line = next(file, None)  # Read the next line
                    if next_line is not None:
                        # Convert the line to a list of numbers
                        numbers = [float(num) for num in next_line.split()]
                        numbers_array.append(numbers)
                    else:
                        break  # Break if there are no more lines in the file
                break  # Break out of the loop since we've found and processed the target line
            else:
                continue

        # Check if the target line was not found
        if not found_target_line:
            print("Target line not found in the file.")

    # Print the array of numbers
    numbers_array = np.array(numbers_array)
    # Split the array into separate arrays for each column
    column_a = numbers_array[:, 0]
    angle = numbers_array[:, 1]
    gain = numbers_array[:, 2]

    # Calibration
    if CAL!=False:
        print("Apply Calibration!")
        if 'LPH' in filename:
            CAL_POL='CAL_LPDA_HPol_80MHz.lst'
        elif 'LPV' in filename:
            CAL_POL = 'CAL_LPDA_VPol_80MHz.lst'
        else:
            print("CAL not found!")
        with open("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/beampattern/UNCal/CAL/"+CAL_POL, "r") as file:
            # Flag to indicate whether we've found the target line
            found_target_line = False
            # Counter to keep track of the lines read after the target line
            lines_after_target = 0
            # List to store the lines in "number number number" format
            numbers_array = []

            # Iterate through each line in the file
            for line in file:
                # Check if the target line is found
                #if f"Frequency:     {freq:.5f} GHz" in line: #5
                if f"Freq:    {freq:.3f} GHz" in line or f"Frequency:     {freq:.5f} GHz" in line:
                    found_target_line = True
                    # Read the next 20 lines
                    for _ in range(7):
                        next(file)
                    for _ in range(21): 
                        next_line = next(file, None)  # Read the next line
                        if next_line is not None:
                            # Convert the line to a list of numbers
                            numbers = [float(num) for num in next_line.split()]
                            numbers_array.append(numbers)
                        else:
                            break  # Break if there are no more lines in the file
                    break  # Break out of the loop since we've found and processed the target line
                else:
                    continue
        CAL_array = np.array(numbers_array)
        LPDA_CAL_angle = CAL_array[:,1]
        LPDA_CAL_S21 = CAL_array[:,2]
        LPDA_CAL_S21_Boresight = LPDA_CAL_S21[np.where(LPDA_CAL_angle==0)]
        
        #Read Spec sheet
        if CAL == 'spec':
            data = np.loadtxt("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/beampattern/UNCal/CAL/LP196(0.080-4.000GHz).tbl",skiprows=1,delimiter=',',dtype=float)
        else:
            data = np.loadtxt("/home/yuchieh/Research/code/antenna/sinuous/Measurement/Warminster/ImpulseResponse/LPDA_HH_Measured.txt",skiprows=1,delimiter=' ',dtype=float)
        spec_freq=data[:,0] #GHz
        spec_gain=data[:,1] #dBi
        f_spec = interp1d(spec_freq,spec_gain,'linear')
        CAL_spec_gain = f_spec(freq)

        gain = gain - LPDA_CAL_S21_Boresight + CAL_spec_gain


    return angle, gain

def freq_spectrum(filename, targetangle, Nangle, CAL):


    freq_array = np.arange(0.08, 0.555, 0.005)
    boresight_gain = np.zeros(len(freq_array))*-100

    for i, freq in enumerate(freq_array):
        angle, gain = ReadWarminsterData(filename, freq, Nangle, CAL)
        #print(angle,gain)
        boresight_gain[i] = gain[np.where(abs(angle-targetangle)<0.1)]
        #print(boresight_gain[i]) 
    return freq_array, boresight_gain

def plotBoresightGain(MeasurementList, directory):
    for measurement in MeasurementList:
        if 'X' in measurement:
            style='--'
        else:
            style = 'solid'
        Nangle=19
        if measurement == 'LPH_LFBF_H_E':
            Nangle=181
        #directory =  measurement.split('_', 2)[1]
        labelname = measurement.split('_', 3)[3]
        if measurement == 'LPH_LFBF_H_E':
            labelname = 'Current plan (vertical+horizontal to a corner)'

        freq_array, boresight_gain = freq_spectrum(str(directory)+'.lst', angle, Nangle, CAL)
        ax.plot(freq_array*1000, boresight_gain,linestyle=style,label=labelname)
        print(measurement)
    plt.title(f'angle={angle}')
    plt.xlabel('Frequency (MHz)', fontsize=18)
    plt.ylabel('Gain (dBi)', fontsize=18)
    plt.grid()
    plt.legend(fontsize=8)
    #plt.savefig(f"plot/test/Test_short_Angle{angle}.png")

    plt.show()


if __name__=="__main__":
    #plot()
    Boresightgain = True
    SINUOUS_SIM = True
    SINUOUS_SIM_ROTATOR = True
    RadiationPattern = False
    RadiationPattern_SIM = False
    RESIDUALS = False

    ListLFBF = ['LPH_LFBF_H_E','LPH_LFBF_H_XE','LPV_LFBF_H_H','LPV_LFBF_H_XH','LPH_LFBF_V_E','LPH_LFBF_V_XE','LPV_LFBF_V_H','LPV_LFBF_V_XH']
    ListLFBFCol = ['LPH_LFBF_H_E','LPV_LFBF_H_H','LPH_LFBF_V_E','LPV_LFBF_V_H']
    ListLFBFCol_1 = ['LPH_LFBF_H_E']
    ListLFAF = ['LPH_LFAF_H_E','LPH_LFAF_H_XE','LPV_LFAF_H_H','LPV_LFAF_H_XH','LPH_LFAF_V_E','LPH_LFAF_V_XE','LPV_LFAF_V_H','LPV_LFAF_V_XH']
    ListLFAFCol = ['LPH_LFAF_H_E','LPV_LFAF_H_H','LPH_LFAF_V_E','LPV_LFAF_V_H']
    CalList=['CAL_LPDA_HPol_80MHz','CAL_LPDA_VPol_80MHz']
    AllList = ListLFBF + ListLFAF
    AllCol = ListLFBFCol + ListLFAFCol
    TestList = ['LPH_LFBF_H_E','LPH_LFBF_H_CoPolDiagonalCable','LPH_LFBF_H_CoPolDualDiagFerriteSthCable','LPH_LFBF_H_StraightwAbsCable','LPH_LFBF_H_CoPoInvertedlDualDiagFerriteSthCable','LPH_LFBF_H_CoPolDiagonalFerrite2Cable','LPH_LFBF_H_CoPolStraightCable','LPH_LFBF_H_CoPolDiagFerriteSthCable','LPH_LFBF_H_CoPolDiagonalFerriteCable','LPH_LFBF_H_CoPolDiagFerritewAbsCable','LPH_LFBF_H_CoPolDiagonalSpaceCable','LPH_LFBF_H_CoPolDiagonal5inSpaceCable','LPH_LFBF_H_CoPolDualDiagCable']
    TestList_short = ['LPH_LFBF_H_E','LPH_LFBF_H_CoPolStraightCable']
    angle = 0
    CAL='spec'
    directory = '../../beampattern/UNCal/LFBF/'
    Nangle=181   
    MeasurementList = ['LPH_LFBF_H_E']
    if Boresightgain==True:
        fig, ax = plt.subplots()
        for i,measurement in enumerate(MeasurementList):
            if 'X' in measurement:
                style='--'
            else:
                style = 'solid'
            if measurement == 'LPH_LFBF_H_E':
                Nangle=181
            #directory =  measurement.split('_', 2)[1]
            labelname = measurement
            if measurement == 'LPH_LFBF_H_E':
                labelname = 'Current plan (vertical+horizontal to a corner)'
            X_measurement = measurement[:-1] + 'X' + measurement[-1]

            freq_array, boresight_gain = freq_spectrum(str(directory)+measurement+'.lst', angle, Nangle, CAL)    
            freq_arrayX, boresight_gainX = freq_spectrum(str(directory)+X_measurement+'.lst', angle, Nangle, CAL)

            ax.plot(freq_array*1000, boresight_gain,linestyle=style,label=measurement,color=plt.get_cmap('viridis')(1-float(i)/float(len(MeasurementList)+2)-0.2)) 
            ax.plot(freq_arrayX*1000, boresight_gainX,linestyle='--',color=plt.get_cmap('viridis')(1-float(i)/float(len(MeasurementList)+2)-0.2))

            print(measurement) 
    
    # Simulation:
    if SINUOUS_SIM:
        freqepd600 = np.linspace(0.05, 0.60 , num=56)
        RG_14=np.zeros(56)
        RG_14_cross=np.zeros(56)
        RG_Warminster=np.zeros(56)
        RG_Warminster_cross=np.zeros(56)
        fixedtheta=90
        for i in range(0,56):
            RG_14[i]=rI.read_Hpol_Hgain(freqepd600[i],fixedtheta,angle,"../data/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
            RG_14_cross[i]=rI.read_Hpol_Vgain(freqepd600[i],fixedtheta,angle,"../data/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
            RG_Warminster[i]=rI.read_Hpol_Hgain(freqepd600[i],fixedtheta,angle,"../data/RG_60in_to600MHz_Warminster_rotators_details_Hport_Eplane_Zport.csv")
            RG_Warminster_cross[i]=rI.read_Hpol_Vgain(freqepd600[i],fixedtheta,angle,"../data/RG_60in_to600MHz_Warminster_rotators_details_Hport_Eplane_Zport.csv")


        plt.plot(freqepd600*1e3*65/60, RG_14, label='simulation',color='indigo')
        plt.plot(freqepd600*1e3*65/60, RG_14_cross,color='indigo',linestyle='--')

        if SINUOUS_SIM_ROTATOR:
            plt.plot(freqepd600*1e3, RG_Warminster, label='simulation with Rotator details',color='darkorange')
            plt.plot(freqepd600*1e3, RG_Warminster_cross,color='darkorange',linestyle='--')
    
        plt.title(f'$\phi$ = {angle}')
        plt.xlabel('Frequency (MHz)', fontsize=18)
        plt.ylabel('Gain (dBi)', fontsize=18)
        plt.ylim(-15,10)
        plt.grid()
        plt.legend(fontsize=8)

        plt.savefig(f"../../plot/{MeasurementList}_Angle{angle}_rotator.png")

        plt.show()
        plt.close()


    # circular plot
    diretory_List = ['../../beampattern/UNCal/LFBF/', '../../beampattern/UNCal/LFBB/'] 
    freq_array = [0.08, 0.1,0.205,0.3,0.4,0.5]
    #freq_array = [0.08, 0.1, 0.205]
    #freq_array = [0.3,0.4,0.5]
    #colorList=["firebrick","darkorange","yellowgreen","darkgreen","steelblue","blueviolet","purple","darkblue"]
    measurementBF = 'LPV_LFBF_H_H'
    measurementBB = 'LPV_LFBB_H_H'
    BFgain = np.zeros((len(freq_array), Nangle))
    BBgain = np.zeros((len(freq_array), Nangle))
    plot_angle = np.zeros(Nangle)
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    if RadiationPattern:
        for directory in diretory_List:
            if directory=='beampattern/UNCal/LFBF/':
                measurement = measurementBF
            elif directory=='beampattern/UNCal/LFBB/':
                measurement = measurementBB
            for i,freq in enumerate(freq_array):
                angle, gain = ReadWarminsterData(str(directory)+measurement+'.lst', freq, Nangle, CAL)
                plot_angle = np.copy(angle)
                if measurement == measurementBB:
                    BBgain[i] = gain 
                    angle = angle-180
                    ax.plot(angle*np.pi/180, gain ,color=plt.get_cmap('gist_heat')(1-float(i)/float(len(freq_array)+2)-0.2),alpha=0.8,linewidth=2)
                else:
                    BFgain[i] = gain
                    ax.plot(angle*np.pi/180, gain ,color=plt.get_cmap('gist_heat')(1-float(i)/float(len(freq_array)+2)-0.2),label=str(int(freq*1000))+'MHz',alpha=0.8,linewidth=2)
        
        if RadiationPattern_SIM:
            fixedtheta = 90
            sim_angle = np.arange(0,356,5)
            for i, freq in enumerate(freq_array):
                Sim_gain_phi = np.zeros(len(sim_angle))
                for j, angle_phi in enumerate(sim_angle):
                    print(angle_phi)
                    #Sim_gain_phi[j] = rI.read_Hpol_Hgain(freq*60/65,fixedtheta,angle_phi,"/home/yuchieh/Research/code/antenna/sinuous/data/XF_Simulation/RG_65in_to600MHz_ReducedCAD_tau075_broadneck_V2_lowampa.csv")
                    Sim_gain_phi[j] = rI.read_Hpol_Hgain(freq,fixedtheta,angle_phi,"../data/RG_60in_to600MHz_Warminster_rotators_details_Hport_Eplane.csv")
                ax.plot(sim_angle*np.pi/180, Sim_gain_phi ,color=plt.get_cmap('gist_heat')(1-float(i)/float(len(freq_array)+2)-0.2),alpha=0.8,linewidth=2, linestyle='--')
        

        #ax.plot(np.arange(0,2*np.pi,0.01),3*np.ones_like(np.arange(0,2*np.pi,0.01)),"--",color="black",alpha=0.8,label='3dBi')
        ax.plot([], [], '--',color='black', label='Simulation')
        ax.tick_params(labelsize=13)
        ax.grid(True)
        ax.legend(bbox_to_anchor=(1,1),bbox_transform=fig.transFigure)
        title="Radiation Pattern:("+measurementBF+")"
        ax.set_title(title, fontsize=14)
        ax.grid(True)
        ax.tick_params(labelsize=9)
        ax.set_rmax(10)
        ax.set_rmin(-20)
        
        #plt.savefig(f"plot/{measurementBF}_CAL{CAL}_SIM_RotatorTower_{freq_array}.png")
        plt.show()
        
        # Residuals:
        if RESIDUALS:
            plt.close()
            print(BFgain[0]-BBgain[0])
            for i, freq in enumerate(freq_array):
                plt.plot(plot_angle, BFgain[i]-BBgain[i] ,color=plt.get_cmap('gist_heat')(1-float(i)/float(len(freq_array)+2)-0.2),alpha=0.8, label=f"{int(freq*1e3)} MHz")
            plt.xlabel("$\phi$($^\circ$)")
            plt.ylabel("Front (dB)-Back (dB)")
            plt.xlim(-60,60)
            plt.ylim(-10,10)
            plt.legend()
            plt.savefig(f"../../plot/Residuals_{measurementBF}_CAL{CAL}_{freq_array}.png")
            plt.show()

