from pathlib import Path
import json,numpy as np
from src.transducer_interface import TransducerConfig
from src.beamformer import create_linear_array,calculate_tx_delays,build_tx_package
R=Path(__file__).parents[1]
def test_shapes():
 t=TransducerConfig.from_json(R/'config/transducer_interface.json'); x=create_linear_array(128,t.pitch_m); assert x.shape==(128,); assert np.isclose(x.mean(),0); b=json.loads((R/'config/beamforming.json').read_text()); p=build_tx_package(t,b); assert p['tx_delays_s'].shape==(128,); assert p['tx_weights'].shape==(128,)
def test_delay():
 t=TransducerConfig.from_json(R/'config/transducer_interface.json'); x=create_linear_array(128,t.pitch_m); d=calculate_tx_delays(x,0,.04,1540); assert np.all(d>=0) and np.isclose(d.min(),0)
