import numpy as np
REQUIRED=('element_x_m','tx_delays_s','tx_weights','tx_frequency_hz','sound_speed_m_s')
def load_tx_package(path):
    with np.load(path) as d:
        miss=[k for k in REQUIRED if k not in d]
        if miss: raise ValueError(f'Missing TX fields: {miss}')
        p={k:np.asarray(d[k]) for k in d.files}
    for k in ('element_x_m','tx_delays_s','tx_weights'):
        if p[k].shape!=(128,): raise ValueError(f'{k} must have shape (128,)')
    return p
