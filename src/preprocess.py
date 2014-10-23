import EEG
import numpy as np
import scipy as sp
from scipy import fftpack

p = EEG.EEG('Dog_2', 'interictal', 17)
#p = EEG.EEG('Patient_2', 'interictal', 17)
print p
#p.normalize_channels()
#p.normalize_overall()

#print np.shape(p.data)  == (16, ~240k)

data = p.data
#eeg = np.rollaxis(data, 1)

signal_time  = 15.0 # in secs
signal_length = int(p.sample_rate_in_hz * signal_time)

signal_start = 0
z = fftpack.rfft(p.data, n=signal_length, axis=1)

print np.shape(z)
