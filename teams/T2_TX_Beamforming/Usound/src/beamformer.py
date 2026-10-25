import json
from pathlib import Path
import numpy as np
from .transducer_interface import TransducerConfig

def create_linear_array(n_elements,pitch_m):
    return (np.arange(n_elements,dtype=float)-(n_elements-1)/2)*pitch_m

def make_apodization(n_elements,kind='hann'):
    k=kind.lower(); w=np.ones(n_elements) if k in ('none','uniform','rect') else np.hanning(n_elements) if k in ('hann','hanning') else np.hamming(n_elements) if k=='hamming' else None
    if w is None: raise ValueError(f'Unsupported apodization: {kind}')
    return w/np.max(w) if np.max(w)>0 else w

def calculate_tx_delays(element_x_m,focus_x_m,focus_z_m,sound_speed_m_s,steering_angle_deg=0.0):
    if sound_speed_m_s<=0 or focus_z_m<=0: raise ValueError('sound speed and focus depth must be positive')
    if abs(steering_angle_deg)<1e-12:
        d=np.sqrt((focus_x_m-element_x_m)**2+focus_z_m**2); tau=(np.max(d)-d)/sound_speed_m_s
    else:
        theta=np.deg2rad(steering_angle_deg); tau=(element_x_m*np.sin(theta))/sound_speed_m_s; tau-=tau.min()
    return tau

def gaussian_pulse(t_s,f0_hz,sigma_cycles=1.0):
    sigma=sigma_cycles/(2*np.pi*f0_hz); return np.exp(-0.5*(t_s/sigma)**2)*np.sin(2*np.pi*f0_hz*t_s)

def generate_element_pulses(delays_s,f0_hz,weights,fs_hz=None,sigma_cycles=1.0):
    fs_hz=fs_hz or max(20*f0_hz,100e6); dur=max(6*sigma_cycles/f0_hz,2.5*2/f0_hz); t=np.arange(-dur,float(np.max(delays_s))+dur,1/fs_hz); p=np.array([w*gaussian_pulse(t-d,f0_hz,sigma_cycles) for d,w in zip(delays_s,weights)]); return t,p

def build_tx_package(transducer:TransducerConfig,beam_cfg):
    x=create_linear_array(transducer.n_elements,transducer.pitch_m); c=float(beam_cfg['sound_speed_m_s']); tau=calculate_tx_delays(x,float(beam_cfg['focus_x_m']),float(beam_cfg['focus_z_m']),c,float(beam_cfg.get('steering_angle_deg',0))); w=make_apodization(transducer.n_elements,beam_cfg.get('apodization','hann'))
    return {'element_x_m':x,'tx_delays_s':tau,'tx_weights':w,'tx_frequency_hz':float(transducer.center_frequency_hz),'sound_speed_m_s':c,'focus_x_m':float(beam_cfg['focus_x_m']),'focus_z_m':float(beam_cfg['focus_z_m']),'steering_angle_deg':float(beam_cfg.get('steering_angle_deg',0))}

def save_package(p,out): np.savez(out,**p)
def save_summary(p,t,b,out):
    s={'probe':t.probe_name,'n_elements':t.n_elements,'pitch_m':t.pitch_m,'center_frequency_hz':t.center_frequency_hz,'focus_x_m':b['focus_x_m'],'focus_z_m':b['focus_z_m'],'steering_angle_deg':b.get('steering_angle_deg',0),'sound_speed_m_s':b['sound_speed_m_s'],'delay_min_s':float(p['tx_delays_s'].min()),'delay_max_s':float(p['tx_delays_s'].max()),'delay_max_us':float(p['tx_delays_s'].max()*1e6)}; Path(out).write_text(json.dumps(s,indent=2))
