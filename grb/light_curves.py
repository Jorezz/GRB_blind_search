import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime
import numpy as np
from .config import ACS_DATA_PATH
from astropy.io import fits
import matplolib.pyplot as plt
from .time import get_ijd_from_utc, get_utc_from_ijd

class LightCurve():
    '''
    Base class for light curves from different sources
    '''
    def __init__(self, event_time,duration=None):
        self.event_time = event_time
        self.duration = duration

        self.times = None
        self.times_err = None
        self.signal = None
        self.signal_err = None

    def plot(self,kind='plot',figsize=(6.4, 4.8),logx=None,logy=None):
        '''
        Plot the light curve
        '''
        fig = plt.figure(figsize = figsize)
        if kind == 'plot':
            plt.plot(self.times, self.signal)
        elif kind == 'errorbar':
            plt.errorbar(self.times, self.signal, xerr=self.times_err, yerr=self.signal_err, fmt = 'o')
        elif kind == 'scatter':
            plt.scatter(self.times, self.signal)
        elif kind == 'step':
            plt.step(self.times, self.signal)
        else:
            raise NotImplementedError(f'Plotting method {kind} not supported')

        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')
        plt.show()

    def bin_photons(data,bin_dur,mas,bining):
        bin_time = []
        bin_time_err = []
        bin_counts = []
        if mas is None:
            mas = np.linspace(data[0,0],data[-1,0],num=int((data[-1,0]-data[0,0])/bining)+1)
        for time in mas:
            bin_time.append(time)
            bin_time_err.append(bin_dur/2)
            bin_counts.append(np.sum((data[:,0]>time-bin_dur/2)&(data[:,0]<=time+bin_dur/2)))
            
        return bin_time,bin_time_err,bin_counts

    def rebin(self,bin_duration):
        """
        Rebin light curve
        bin_duration >= 0.05
        """
        self.binning=bin_duration
        N_bef=int(self.binning*20)
        temp_time=[]
        temp_time_err=[]
        temp_signal=[]
        temp_signal_err=[]
        bin_signal=[]
        bin_time=[]
        for i in range(len(self.original_times)):
            if (len(bin_signal)<N_bef):
                bin_signal.append(self.original_signal[i])
                bin_time.append(self.original_times[i])
            if (len(bin_signal)>=N_bef):
#                  and (self.original_times[i+1]-self.original_times[i])>100*(self.original_times[i]-self.original_times[i-1])
                temp_time.append(np.mean(bin_time))
                temp_time_err.append((max(bin_time)-min(bin_time))/2)
                temp_signal.append(sum(bin_signal))
                temp_signal_err.append(np.sqrt(sum(bin_signal)*1.35))
                bin_signal=[]
                bin_time=[]
        else:
            if len(bin_time):
                temp_time.append(np.mean(bin_time))
                temp_time_err.append((max(bin_time)-min(bin_time))/2)
                temp_signal.append(sum(bin_signal))
                temp_signal_err.append(np.sqrt(sum(bin_signal)*1.35))
        self.times=temp_time
        self.times_err=temp_time_err
        self.signal=temp_signal
        self.signal_err=temp_signal_err
        return self

    
class SPI_ACS_LightCurve(LightCurve):
    '''
    Class for light curves from SPI-ACS/INTEGRAL
    '''
    def __init__(self,loading_method='local',scale='utc',*args,**kwargs):
        '''
        Args:
            loading_method (str): 'local' or 'remote
            scale (str): 'utc' or 'ijd'
        '''
        super().__init__(*args,**kwargs)
        if self.duration is None:
            raise ValueError('Duration is required')

        self.__acs_scw_df = None
        if loading_method == 'local':
            self.original_times,self.original_signal = self.get_light_curve_from_file(scale = scale)
        elif loading_method =='web':
            self.original_times,self.original_signal = self.get_light_curve_from_web(scale = scale)
        else:
            raise ValueError(f'Loading method {loading_method} not supported')

        self.times = self.original_times
        avg_time_delta = np.mean((self.times[1:] - self.times[:-1])/2) # determine size of time window
        self.times_err = np.full(self.original_times.shape[0], avg_time_delta)
        self.signal = self.original_signal
        self.signal_err = np.sqrt(self.original_signal)

    @property
    def acs_scw_df(self):
        if self.__acs_scw_df is None:
            acs_scw_df= []
            with open(f'{ACS_DATA_PATH}swg_infoc.dat','r') as f:
                for line in f:
                    acs_scw_df.append(line.split())

            acs_scw_df = pd.DataFrame(acs_scw_df,columns=['scw_id','obt_start','obt_finish','ijd_start','ijd_finish','scw_duration','x','y','z','ra','dec'])
            acs_scw_df['scw_id'] = acs_scw_df['scw_id'].astype(str)
            acs_scw_df['ijd_start'] = acs_scw_df['ijd_start'].astype(float)
            acs_scw_df['ijd_finish'] = acs_scw_df['ijd_finish'].astype(float)
            self.__acs_scw_df = acs_scw_df

        return self.__acs_scw_df

    def get_light_curve_from_file(self,scale):
        center_time = datetime.datetime.strptime(self.event_time,'%Y-%m-%d %H:%M:%S')
        left_time = float(get_ijd_from_utc((center_time - datetime.timedelta(seconds=self.duration)).strftime('%Y-%m-%d %H:%M:%S')))
        right_time = float(get_ijd_from_utc((center_time + datetime.timedelta(seconds=self.duration)).strftime('%Y-%m-%d %H:%M:%S')))
        scw_needed = self.scw_info[((self.scw_info['ijd_start']>left_time)&(self.scw_info['ijd_start']<right_time))|((self.scw_info['ijd_finish']>left_time)&(self.scw_info['ijd_finish']<right_time))|((self.scw_info['ijd_start']<left_time)&(self.scw_info['ijd_finish']>left_time))|((self.scw_info['ijd_start']<right_time)&(self.scw_info['ijd_finish']>right_time))]
        if scw_needed.shape[0]==0:
            raise ValueError(f'No data found for {self.event_time}')

        current_data = []
        for _,row in scw_needed.iterrows():
            for i in range(5):
                try:
                    with open(f'{ACS_DATA_PATH}0.05s/{row["scw_id"][:4]}/{row["scw_id"]}.00{i}.dat','r') as f:
                        for line in f:
                            line = line.split()
                            if float(line[0]) > 0:
                                current_data.append([row['ijd_start']+float(line[0])/(24*60*60),float(line[2])])
                    break
                except FileNotFoundError:
                    continue
            
        current_data = np.asarray(current_data)
        
        current_data[:,0] = (current_data[:,0] - (left_time+self.duration/(24*60*60)))*(24*60*60)
        current_data = current_data[np.abs(current_data[:,0])<self.duration]

        if scale == 'ijd':
            current_data[:,0] = current_data[:,0]/(24*60*60) + (left_time+self.duration/(24*60*60))

        return current_data[:,0],current_data[:,1]

    def get_light_curve_from_web(self,scale = 'ijd'):
        #todo
        pass

class ACS_Light_Curve():
    def __init__(self,grb_time,duration,source=None):
        self.time=grb_time
        self.duration=duration
        self.binning=0.05
        if source==None:
            self.download_lc(self.time,self.duration)
        else:
            t=Time(self.time,scale='utc')
            t.format = 'jd'
            t = t.value-2451544.5
            data=np.loadtxt(source)
            self.times=list((data[:,0]-t)*24*60*60)
            self.times_err=list(data[:,1]*24*60*60)
            self.signal=list(data[:,2])
            self.signal_err=list(np.sqrt(data[:,2]*1.35))

            self.original_times=self.times
            self.original_signal=self.signal


    def load_from_file(self,grb_time,duration,binning):
#         print('loaded from file')
        path_time=grb_time.replace(':','-')
        data=np.loadtxt(f'astr_triggers/{path_time}_{duration}_{binning}.txt')
        self.times=list(data[:,0])
        self.times_err=list(data[:,1])
        self.signal=list(data[:,2])
        self.signal_err=list(data[:,3])

        self.original_times=self.times
        self.original_signal=self.signal

    def download_lc(self,grb_time,duration):
#         print('downloaded')
        data=get_acs_lc(grb_time,duration) #Download then clear data
        times=data[:,0]
        signal=data[:,1]
        gap=0.05
        temp_times=[]
        temp_signal=[]
        temp_times.append(times[0])
        if (np.isfinite(signal[0])):
            temp_signal.append(signal[0])
        else:
            temp_signal.append(0)

        counter=1
        while counter<len(times):
            if times[counter]>1.5*gap+temp_times[-1]:
                temp_times.append(temp_times[-1]+gap)
                temp_signal.append(0)
            else:
                temp_times.append(times[counter])
                if np.isfinite(signal[counter]):
                    temp_signal.append(signal[counter])
                else:
                    temp_signal.append(0)
                counter += 1

        self.times=temp_times
        self.times_err=[0 for _ in range(len(temp_times))]
        self.signal=temp_signal
        self.signal_err=[np.sqrt(el*1.35) for el in temp_signal]

        self.original_times=self.times
        self.original_signal=self.signal

    def save_data(self):
        """
        Save light curve, if not saved already
        """
        path_time=self.time.replace(':','-')
        if not os.path.isfile(f'astr_triggers/{path_time}_{self.duration}_{self.binning}.txt'):
            data=np.vstack((self.times,self.times_err,self.signal,self.signal_err)).T
            np.savetxt(f'astr_triggers/{path_time}_{self.duration}_{self.binning}.txt',data)

    def get(self,*intervals):
        """
        Returns rebinned light curve between specific intervals
        """
        if len(intervals)==0:
            return self.times,self.times_err,self.signal,self.signal_err
        elif len(intervals)%2!=0:
            print('Wrong intervals')
        else:
            temp_time=[]
            temp_time_err=[]
            temp_signal=[]
            temp_signal_err=[]
            for j in range(0,len(intervals)//2):
                for i in range(0,len(self.times)):
                    if is_between(intervals[j*2],self.times[i],intervals[j*2+1]):
                        temp_time.append(self.times[i])
                        temp_time_err.append(self.times_err[i])
                        temp_signal.append(self.signal[i])
                        temp_signal_err.append(self.signal_err[i])
            return temp_time,temp_time_err,temp_signal,temp_signal_err

    def rebin(self,bin_duration):
        """
        Rebin light curve
        bin_duration >= 0.05
        """
        self.binning=bin_duration
        N_bef=int(self.binning*20)
        temp_time=[]
        temp_time_err=[]
        temp_signal=[]
        temp_signal_err=[]
        bin_signal=[]
        bin_time=[]
        for i in range(len(self.original_times)):
            if (len(bin_signal)<N_bef):
                bin_signal.append(self.original_signal[i])
                bin_time.append(self.original_times[i])
            if (len(bin_signal)>=N_bef):
#                  and (self.original_times[i+1]-self.original_times[i])>100*(self.original_times[i]-self.original_times[i-1])
                temp_time.append(np.mean(bin_time))
                temp_time_err.append((max(bin_time)-min(bin_time))/2)
                temp_signal.append(sum(bin_signal))
                temp_signal_err.append(np.sqrt(sum(bin_signal)*1.35))
                bin_signal=[]
                bin_time=[]
        else:
            if len(bin_time):
                temp_time.append(np.mean(bin_time))
                temp_time_err.append((max(bin_time)-min(bin_time))/2)
                temp_signal.append(sum(bin_signal))
                temp_signal_err.append(np.sqrt(sum(bin_signal)*1.35))
        self.times=temp_time
        self.times_err=temp_time_err
        self.signal=temp_signal
        self.signal_err=temp_signal_err
        return self

    def reset(self):
        """
        Return to original data
        """
        self.binning=0.05
        self.times=self.original_times
        self.signal=self.original_signal
        self.times_err=[0 for _ in range(len(self.times))]
        self.signal_err=[np.sqrt(el*1.35) for el in self.signal]
        return self

    def clear(self,threshold):
        """
        Filter light curve from excesses
        """
        med=np.median(self.signal)
        sign_array=[(el-med)/np.sqrt(el*1.35) for el in self.signal]
        wrong_array=np.array(sign_array)
        wrong_array=wrong_array[wrong_array<-threshold]
        wrong_array=list(wrong_array)
#         print(sign_array)
#         print(wrong_array)
        for el in wrong_array:
            wrong=sign_array.index(el)
            sign_array.pop(wrong)
            self.times.pop(wrong)
            self.times_err.pop(wrong)
            self.signal.pop(wrong)
            self.signal_err.pop(wrong)
        return self

    def plot(self,*intervals):
        fig=plt.figure(dpi=100)
        a,b,c,d=self.get(*intervals)
        plt.errorbar(a,c,xerr=b,yerr=d, fmt='o',color='black')
        plt.xlabel('Time since trigger , sec')
        plt.ylabel(f'ACS Counts \ {self.binning} sec')

    def cut_original(self, *intervals):
        temp_time=np.asarray(self.original_times)
        temp_time_err=np.asarray(self.times_err)
        temp_signal=np.asarray(self.original_signal)
        temp_signal_err=np.asarray(self.signal_err)

        for j in range(0,len(intervals)//2):
            bool_mask = tuple([np.logical_or(intervals[j*2]>temp_time,temp_time>intervals[j*2+1])])
            temp_time=temp_time[bool_mask]
            temp_time_err=temp_time_err[bool_mask]
            temp_signal=temp_signal[bool_mask]
            temp_signal_err=temp_signal_err[bool_mask]

        self.original_times=list(temp_time)
        self.times_err=list(temp_time_err)
        self.original_signal=list(temp_signal)
        self.signal_err=list(temp_signal_err)

        self.times=list(temp_time)
        self.signal=list(temp_signal)
        return self

    def cut(self, *intervals):
        temp_time=np.asarray(self.times)
        temp_time_err=np.asarray(self.times_err)
        temp_signal=np.asarray(self.signal)
        temp_signal_err=np.asarray(self.signal_err)

        for j in range(0,len(intervals)//2):
            bool_mask = tuple([np.logical_or(intervals[j*2]>temp_time,temp_time>intervals[j*2+1])])
            temp_time=temp_time[bool_mask]
            temp_time_err=temp_time_err[bool_mask]
            temp_signal=temp_signal[bool_mask]
            temp_signal_err=temp_signal_err[bool_mask]

        self.times=list(temp_time)
        self.times_err=list(temp_time_err)
        self.signal=list(temp_signal)
        self.signal_err=list(temp_signal_err)
        return self

class IREM_LightCurve(LightCurve):
    def __init__(self,center_time,duration,source='Nan'):
        self.time=center_time
        self.duration=duration
        if source == 'Nan':
            self.load_from_file()
        else:
            self.TC3=np.loadtxt(source+'irem_TC3_0040-2410_T1000.dat')
#             self.S14=np.loadtxt(source+'SF20211028_S14_60.txt')
#             self.S33=np.loadtxt(source+'SF20211028_S33_60.txt')
#             self.C2=np.loadtxt(source+'SF20211028_C2_60.txt')
#             self.C3=np.loadtxt(source+'SF20211028_C3_60.txt')
#             self.C4=np.loadtxt(source+'SF20211028_C4_60.txt')

    def substract_bkg(self,IREM_array,*interval):
        bkg=IREM_array[np.logical_and(interval[0]<IREM_array[:,0],IREM_array[:,0]<interval[1])]
        bkg=np.mean(bkg[:,2])
        IREM_array[:,2] = IREM_array[:,2] - bkg
        IREM_array[:,3] = np.sqrt(bkg)
        return IREM_array

    def cut(self,IREM_array,*interval):
        for i in range(len(interval)//2):
            IREM_array=IREM_array[np.logical_or(interval[2*i]>IREM_array[:,0],IREM_array[:,0]>interval[2*i+1])]
        return IREM_array


    def get():
        pass


    def load_from_file(self):
#         TC2=np.loadtxt('../Wavelet/IREM/lc_IREM_TC2_T1000.dat')
#         self.TC2=self.extract(TC2)
        TC3=np.loadtxt('E:\ACS\irem_TC3_0040-2410_T1000(1).dat')
        self.TC3=self.extract(TC3)
#         S14=np.loadtxt('../Wavelet/IREM/lc_IREM_S14_T1000.dat')
#         self.S14=self.extract(S14)
#         S33=np.loadtxt('../Wavelet/IREM/lc_IREM_S33_T1000.dat')
#         self.S33=self.extract(S33)
#         C3=np.loadtxt('../Wavelet/IREM/lc_IREM_C3_T1000.dat')
#         self.C3=self.extract(C3)
#         C4=np.loadtxt('../Wavelet/IREM/lc_IREM_C4_T1000.dat')
#         self.C4=self.extract(C4)

    def extract(self,IREM_array):
        """
        From 20 year interval get 12 ks of data
        """
        dt=self.time[0:10]+' '+self.time[11:]
        t=get_ijd_from_utc(dt)
        IREM_array[:,0]=IREM_array[:,0] - t
        IREM_array[:,0:2]=IREM_array[:,0:2]*24*60*60
        IREM_array=IREM_array[np.logical_and(-self.duration<IREM_array[:,0],IREM_array[:,0]<self.duration)]
        return IREM_array


class GBM_LightCurve(LightCurve):
    def __init__(self,code,redshift = None,dict_file = None, binning = None):
        self.binning = binning if binning else 0.01
        self.redshift = redshift if redshift else 0
        self.code = code
        if dict_file:
            if code in dict_file:
                self.lightcurve = LightCurve(time=dict_file[code]['time'],
                                             signal=dict_file[code]['signal'])
            else:
                self.lightcurve = LightCurve(time=self.get_light_curve()[0],
                                             signal=self.get_light_curve()[2])
                dict_file[code] = self.lightcurve
        else:
            dict_file = {}
            self.lightcurve = LightCurve(time=self.get_light_curve()[0],
                                         signal=self.get_light_curve()[2])
            dict_file[code] = self.lightcurve
             
    @staticmethod
    def filter_energy(data,low_en=6,high_en=850,detector='n0'):
        if detector[0] == 'b':
            low_en = low_en if low_en else 200
            high_en = high_en if high_en else 6500
        elif detector[0] == 'n':
            low_en = low_en if low_en else  6
            high_en = high_en if high_en else 850
            
        return data[np.logical_and(data[:,1]>low_en,data[:,1]<high_en)]
                
    @staticmethod
    def bin_photons(data,bin_dur,mas):
        bin_time = []
        bin_time_err = []
        bin_counts = []
        for time in mas:
            bin_time.append(time)
            bin_time_err.append(bin_dur/2)
            bin_counts.append(np.sum((data_temp[:,0]>time-bin_dur/2)&(data_temp[:,0]<=time+bin_dur/2)))
            
        return bin_time,bin_time_err,bin_counts
    
    @staticmethod
    def load_ligh_curve_url(detector,code):
        for i in range(5):
            try:
                return fits.open(f'https://heasarc.gsfc.nasa.gov/FTP/fermi/data/gbm/bursts/20{code[2]}{code[3]}/{code}/current/glg_tte_{detector}_{code}_v0{i}.fit')
            except requests.exceptions.HTTPError:
                pass
            
    @staticmethod
    def apply_redshift(data,redshift):
        data[:,1] = data[:,1] * (1 + redshift)
        data[:,0] = data[:,0] / (1 + redshift)
        return data
        
    @cache
    def get_light_curve(self):
        mas = None
        lumin_detectors = [detectors[i] for i,value in enumerate(list(gbm_cat[gbm_cat['grb_code']==self.code]['mask'].values[0])) if value == '1']
        for detector in lumin_detectors:
            hdul = self.load_ligh_curve_url(detector,self.code)
            ebounds = {line[0]:np.sqrt(line[1]*line[2]) for line in hdul[1].data}
            tzero = float(hdul[2].header['TZERO1'])
            kek = pd.DataFrame(hdul[2].data)
            kek['TIME'] = kek['TIME'] - tzero
            kek['PHA'] = kek['PHA'].replace(ebounds)
            data = kek.values
            data = self.apply_redshift(data,self.redshift)
            if mas is None:
                mas = np.linspace(data[0,0],data[-1,0],num=int((data[-1,0]-data[0,0])/bining)+1)
            
            return self.bin_photons(filter_energy(data,detector=detector),self.binning,mas)
        
    def get_hist(self):
        pass
        
    def get_bkg(self):
        pass