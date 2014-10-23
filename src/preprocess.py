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
#print p.data[0:1, 0:20]

signal_duration = 15.0 # in secs

sample_length  = int(p.sample_rate_in_hz * signal_duration)
sample_start   = int(p.sample_rate_in_hz * 0)

z = fftpack.rfft(p.data[:, sample_start:], n=sample_length, axis=1)
#print np.shape(z)
#print z[0:1, 0:20]

freq = fftpack.rfftfreq(n=sample_length, d=1./p.sample_rate_in_hz)
print freq[0:20]
