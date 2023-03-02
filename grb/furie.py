import numpy as np
import matplotlib.pyplot as plt
from .light_curves import LightCurve
from scipy.stats import chi2


def make_pds(signal, time_step):
    '''
    Get Power Density Spectrum from signal
    Args:
        signal (np.array): input signal
        time_step (float): time step
    '''
    freqs = np.fft.fftfreq(signal.shape[0], time_step)
    ps = 2*np.abs(np.fft.fft(signal - np.mean(signal)))**(2)/np.sum(signal)
    mask = (freqs>0)
    
    return freqs[mask], ps[mask]

def group_log_bins(freqs: np.array, ps: np.array, N_bins: int = 30, step: float = None, log_scale: np.array = None):
        '''
        Group bins in log scale
        Args:
            freqs (np.array): input frquencies
            ps (np.array): input power spectrum
            N_bins (int, optional): number of bins. Defaults to 30.
        '''
        time=[]
        time_err=[]
        flux=[]
        flux_err=[]
        log_x = np.log10(freqs)

        if step is None:
            step = (log_x[-1] - log_x[0])/(2*N_bins)
        if log_scale is None:
            log_scale = np.linspace(log_x[0]+step,log_x[-1]-step,N_bins)

        for i in range(0,N_bins):
            mask1=tuple([np.logical_and(log_x>=log_scale[i]-step,log_x<log_scale[i]+step)])
            time.append(np.mean(freqs[mask1]) if len(ps[mask1])!=0 else 10**log_scale[i])
            time_err.append((10**(log_scale[i]+step)-10**(log_scale[i]) + 10**(log_scale[i])-10**(log_scale[i]-step))/2)
            flux.append(np.mean(ps[mask1]) if len(ps[mask1])!=0 else 0)
            flux_err.append(chi2.ppf(0.67,2*len(ps[mask1]))/len(ps[mask1]) if len(ps[mask1])!=0 else 1)
        
        return np.array(time), np.array(time_err), np.array(flux), np.array(flux_err)

class FurieLightCurve():
    def __init__(self, light_curve: LightCurve, 
                       interval_t90: tuple = None,
                       bkg_substraction_resolution: float = 10,
                       bkg_polynom_degree: int = 3,
                       pad_size: int = None,
                       ):

        '''
        Args:
            light_curve (LightCurve): light curve object
            interval_t90 (tuple, optional): time interval for t90. Defaults to None
            bkg_substraction_resolution (float, optional): background substraction resolution, defaults to 10
            bkg_polynom_degree (int, optional): background polynom degree, defaults to 3
        '''
        self.light_curve = light_curve

        if interval_t90 is None:
            interval_t90 = (self.light_curve.times[0]/2,self.light_curve.times[-1]/2)
        
        bkg_intervals = [(self.light_curve.times[0]-self.light_curve.resolution,interval_t90[0]),
                         (interval_t90[1],self.light_curve.times[-1]+self.light_curve.resolution)]

        mean_bkg = np.mean(self.light_curve.rebin().set_intervals(*bkg_intervals).signal)
    
        rebined_param = np.polyfit(self.light_curve.rebin(bkg_substraction_resolution).set_intervals(*bkg_intervals).times,self.light_curve.rebin(bkg_substraction_resolution).set_intervals(*bkg_intervals).signal,bkg_polynom_degree)
        rebined_param = rebined_param * (self.light_curve.original_resolution/self.light_curve.resolution)

        signal = self.light_curve.rebin().substract_polynom(rebined_param).set_intervals(interval_t90).signal
        if pad_size:
            signal = np.pad(signal,(pad_size - signal.shape[0],0), 'constant')
        
        self.freqs, self.ps =  make_pds(signal+mean_bkg,self.light_curve.original_resolution)
        self.freqs_err, self.ps_err = np.full(self.freqs.shape[0], 0), np.sqrt(self.ps)

    def plot(self, kind: str = 'scatter', logx: bool = True, logy: bool = True, N_bins: int = 30, **kwargs):
        '''
        Plot PDS
        Args:
            kind (str, optional): plotting method
            logx (bool, optional): log x axis
            logy (bool, optional): log y axis
            N_bins (int, optional): number of bins, if None - plots without grouping
            kwargs (dict, optional): keyword arguments for plotting used by matplotlib.pyplot.plot
        '''
        if N_bins is None:
            x,x_err,y,y_err = self.freqs, self.freqs_err, self.ps, self.ps_err
        else:
            x,x_err,y,y_err = group_log_bins(self.freqs, self.ps, N_bins)

        if kind == 'plot':
            plt.plot(x, y, **kwargs)
        elif kind == 'errorbar':
            plt.errorbar(x, y, xerr=x_err, yerr=y_err, fmt = 'o', **kwargs)
        elif kind == 'scatter':
            plt.scatter(x, y, **kwargs)
        elif kind == 'step':
            plt.step(x, y, **kwargs)
        else:
            raise NotImplementedError(f'Plotting method {kind} not supported')

        if logx:
            plt.xscale('log')
        if logy:
            plt.yscale('log')


    
