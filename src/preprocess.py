import EEG

import numpy as np
import scipy as sp
from scipy import fftpack

import math

p = EEG.EEG('Dog_2', 'interictal', 17)
#p = EEG.EEG('Patient_2', 'interictal', 17)
print p
#p.normalize_channels()
#p.normalize_overall()

#print np.shape(p.data)  == (16, ~240k)

data = p.data
#eeg = np.rollaxis(data, 1)
#print p.data[0:1, 0:20]

signal_duration_min = 9.0 # in secs
pow2 = math.log(p.sample_rate_in_hz * signal_duration_min)/math.log(2.)

sample_length  = int(2.0 ** (int(pow2)+1))   # Rounds up  
sample_start   = int(p.sample_rate_in_hz * 0)

signal_duration = sample_length/p.sample_rate_in_hz
print "Pow2: ", pow2
print "Signal duration : %6.2f" % (signal_duration,)

z = fftpack.rfft(p.data[:, sample_start:], n=sample_length, axis=1)
print np.shape(z)
#print z[0:1, 0:20]

freq = fftpack.rfftfreq(n=sample_length, d=1./p.sample_rate_in_hz)
print freq[0:20]
