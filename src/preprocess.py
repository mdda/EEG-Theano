import numpy as np
#import scipy as sp
#from scipy import fftpack

import hickle

import EEG

signal_duration_min = 9.0  # in secs
signal_period_step  = 5.0  # in secs

bin_fft=None

def preprocess(p):
  p.load()
  #p = EEG.EEG('Patient_2', 'interictal', 17)
  print p
  #p.normalize_channels()
  #p.normalize_overall()

  print np.shape(p.data), p.data.nbytes  # == (16, ~240k)

  data = p.data
  #eeg = np.rollaxis(data, 1)
  #print p.data[0:1, 0:20]

  global bin_fft
  if bin_fft is None :
    pow2 = np.log2(p.sample_rate_in_hz * signal_duration_min)

    sample_length  = int(2.0 ** (int(pow2)+1))   # in samples, rounds up  
    signal_duration = sample_length/p.sample_rate_in_hz
    print "Pow2: ", pow2
    print "Signal duration : %6.2fsec = %d samples " % (signal_duration, sample_length, )

    ## Matrix that gathers FFT entries into buckets
    ## Want buckets to be [0-0.000001 - 1.5 - 2.5 - 3.5 - ... - 49.5] Hz
    bin_array = np.linspace(0., 49., num=50) 

    ## http://docs.scipy.org/doc/numpy/reference/routines.fft.html#module-numpy.fft

    #freq = fftpack.rfftfreq(n=sample_length, d=1./p.sample_rate_in_hz)
    freq = np.fft.rfftfreq(n=sample_length, d=1./p.sample_rate_in_hz)
    #print freq[0:100]

    bin_fft = np.zeros( (len(freq), len(bin_array)) )
    for i, bn in enumerate(bin_array):
      bn_lower=(bin_array[i-1]+bin_array[i+0])/2. if i>0 else bn-0.5
      bn_upper=(bin_array[i+0]+bin_array[i+1])/2. if i<len(bin_array)-1 else bn+0.5
      a = np.where( (freq>bn_lower) & (freq<=bn_upper) , 1, 0)
      bin_fft[:,i] = a
      #print bn_lower, bn, bn_upper

    #print bin_fft[0:20, 0:5]
  
  ## Now, take whole period, and find the start times in seconds
  signal_period_starts = np.arange( start=0, stop=p.length_in_sec-signal_duration, step=signal_period_step )
  #print signal_period_starts

  param_length = p.n_channels * len(bin_array)
  all_params = np.zeros( (len(signal_period_starts), param_length), dtype=np.complex64 )

  for i, start_period in enumerate(signal_period_starts):
    sample_start   = int(p.sample_rate_in_hz * start_period)  # start time in seconds

    #z = fftpack.rfft(p.data[:, sample_start:], n=sample_length, axis=1)
    fft_raw = np.fft.rfft(p.data[:, sample_start:], n=sample_length, axis=1)
    #print np.shape(fft_raw)
    #print fft_raw[0:1, 0:20]

    binned = np.dot(fft_raw,bin_fft)
    #print np.shape(binned)
    #print binned[0:1, :] 
    #print binned[0, 0]           # Check that first bin is equal to first sums...
    #print np.sum(fft_raw[0,0:6]) # Works!

    params = np.log(binned.ravel())

    all_params[i,:]=params

  print np.shape(all_params), all_params.nbytes

  to_hickle = dict(
    features=all_params,
    signal_period_starts=signal_period_starts,
  )

  # Dump data, with compression
  f = "data/feat/%s/%s_%s_segment_%04d.hickle" % (p.patient, p.patient, p.desc, p.num)
  hickle.dump(to_hickle, f, mode='w', compression='gzip')

p = EEG.EEG('Dog_2', 'interictal', 17)

preprocess(p)
