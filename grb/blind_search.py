import numpy as np
from .config import DATA_PATH, logging
from tqdm import tqdm
from .time import get_utc_from_ijd
from .utils import Chi2_polyval
import matplotlib.pyplot as plt

def filter_missing_data_and_flares(data):
    '''
    Load gaps in ligth curve and exclude from data
    '''
    # load hand-filtered ligth curve and extract gaps
    breaks_data = np.loadtxt(f'{DATA_PATH}spi-acs_filtered.txt')
    breaks = []
    avg_gap = np.median(breaks_data[:,1])
    for i in range(breaks_data.shape[0]-1):
        if (breaks_data[i+1,0] - breaks_data[i,0]) > 3 * avg_gap:
            breaks.append([breaks_data[i,0],breaks_data[i+1,0]])

    # iterate through data and mark gaps with zeros
    for gap in breaks:
        data[(data[:,0]>=gap[0])&(data[:,0]<=gap[1]),2] = 0
    
    return data[data[:,2]!=0]

def process_time_window(sub_time,sub_counts,center_bin,threshold, plot = False):
    '''
    For given time window (sub_time,sub_counts) find event in center_bin
    '''
    bkg_time = np.delete(sub_time,[center_bin])
    bkg_counts = np.delete(sub_counts,[center_bin])
    param = np.polyfit(bkg_time,bkg_counts,3)
    event_flux = sub_counts[center_bin] - np.polyval(param,sub_time[center_bin])
    overpuasson = np.var(bkg_counts)/np.mean(bkg_counts)
    sigma = np.sqrt(np.var(bkg_counts))
    if (event_flux/sigma) > threshold:
        utc_time = get_utc_from_ijd(sub_time[center_bin])
        chi_2 = Chi2_polyval(bkg_time,bkg_counts,np.sqrt(bkg_counts*overpuasson),param)
        logging.info(f'{utc_time=}, sigma = {round(event_flux/sigma,1)}, chi2 bkg = {round(chi_2,1)}')
        if plot:
            fig=plt.figure()
            plt.errorbar(bkg_time,bkg_counts,yerr=np.sqrt(bkg_counts*overpuasson),fmt='o',label='background')
            plt.errorbar(sub_time[center_bin],sub_counts[center_bin],yerr=sigma,fmt='o',c='red',label=f'event, {round(event_flux/sigma,1)} sigma')
            plt.plot(sub_time,np.polyval(param,sub_time),label=f'chi2 bkg={round(chi_2,1)}')
            plt.legend()
            plt.show()
        return (utc_time,round(event_flux/sigma,1),round(chi_2,1))
    else:
        return (None,None,None)

def recursive_process_event():
    pass