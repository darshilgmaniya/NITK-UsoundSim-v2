from dataclasses import dataclass
from pathlib import Path
import json
@dataclass
class TransducerConfig:
    probe_name:str; n_elements:int; pitch_m:float; frequency_min_hz:float; frequency_max_hz:float; center_frequency_hz:float; element_width_m:float|None=None; element_height_m:float|None=None; material:str=""
    @classmethod
    def from_json(cls,path):
        d=json.loads(Path(path).read_text()); req=['probe_name','n_elements','pitch_m','frequency_min_hz','frequency_max_hz','center_frequency_hz']; miss=[x for x in req if x not in d]
        if miss: raise ValueError(f'Missing transducer fields: {miss}')
        c=cls(**{k:d[k] for k in cls.__dataclass_fields__ if k in d}); c.validate(); return c
    def validate(self):
        if self.n_elements!=128: raise ValueError(f'This interface requires 128 elements, got {self.n_elements}')
        if self.pitch_m<=0: raise ValueError('pitch_m must be positive')
        if self.frequency_min_hz<=0 or self.frequency_max_hz<self.frequency_min_hz: raise ValueError('Invalid frequency range')
        if not self.frequency_min_hz<=self.center_frequency_hz<=self.frequency_max_hz: raise ValueError('center_frequency_hz must be inside operating range')
