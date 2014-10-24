import EEG

import numpy as np
#import scipy as sp
#from scipy import fftpack

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
signal_duration = sample_length/p.sample_rate_in_hz
print "Pow2: ", pow2
print "Signal duration : %6.2fsec " % (signal_duration,)

## Matrix that gathers FFT entries into buckets
## Want buckets to be [0-0.000001 - 1.5 - 2.5 - 3.5 - ... - 49.5] Hz
bin_array = np.linspace(0., 49., num=50) 

## http://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft

#freq = fftpack.rfftfreq(n=sample_length, d=1./p.sample_rate_in_hz)
freq = np.fft.rfftfreq(n=sample_length, d=1./p.sample_rate_in_hz)
print freq[0:100]

bin_fft = np.zeros( (len(freq), len(bin_array)) )
for i, bn in enumerate(bin_array):
  a = np.where( (freq>(bn-0.5)) & (freq<(bn+0.5)) , 1, 0)
  bin_fft[:,i] = a
  #print bn+0.5

#print bin_fft[0:20, 0:5]



sample_start   = int(p.sample_rate_in_hz * 0.)  # start time in seconds

#z = fftpack.rfft(p.data[:, sample_start:], n=sample_length, axis=1)
z = np.fft.rfft(p.data[:, sample_start:], n=sample_length, axis=1)
print np.shape(z)
print z[0:1, 0:20]

binned = np.dot(z,bin_fft)
#print binned[0:1, :]
print binned[0, 0]
print np.shape(binned)

#print np.sum(z[0,0:6]) # Works!
