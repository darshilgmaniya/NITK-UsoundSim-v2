from pathlib import Path
import json, numpy as np
from src.transducer_interface import TransducerConfig
from src.beamformer import build_tx_package,save_package,save_summary,generate_element_pulses
R=Path(__file__).parent; O=R/'outputs'; O.mkdir(exist_ok=True)
def main():
 t=TransducerConfig.from_json(R/'config/transducer_interface.json'); b=json.loads((R/'config/beamforming.json').read_text()); p=build_tx_package(t,b); save_package(p,O/'tx_beamforming_package.npz'); save_summary(p,t,b,O/'tx_beamforming_summary.json'); ts,ps=generate_element_pulses(p['tx_delays_s'],p['tx_frequency_hz'],p['tx_weights'],sigma_cycles=b.get('pulse_duration_sigma_cycles',1)); np.savez(O/'tx_element_pulses.npz',time_s=ts,pulses=ps); print('TX beamforming completed'); print('Elements:',t.n_elements); print('Frequency MHz:',t.center_frequency_hz/1e6); print('Delay range us:',p['tx_delays_s'].min()*1e6,p['tx_delays_s'].max()*1e6); print('Handoff:',O/'tx_beamforming_package.npz')
if __name__=='__main__': main()
