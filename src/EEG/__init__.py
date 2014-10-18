import util
import numpy as np

class EEG:
  def __init__(self, _patient, _type, _num):
    d = util.load(_patient, _type, _num)
    self.data = d[0].astype(dtype=np.float32)
    
    #print d[1][0]
    #print type(d[0])
    
    # Unbox simple parameters
    self.length_in_sec = d[1][0][0]
    self.sample_rate_in_hz = d[2][0][0]
    
    electrode_arr = d[3][0]  # This is an array
    self.n_channels = len(electrode_arr)
    self.electrode = [ e[0] for e in electrode_arr ]

    self.timeperiod = None if _type == 'test' else d[4][0][0]

    
  def normalize_channels(self):
    ## This is the key that stops mean() and std() behaving as expected
    #print np.result_type(data)

    ## See :: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/data.py
    eeg = np.rollaxis(self.data, 1)

    means = eeg.mean(axis=0)
    stdev = eeg.std(axis=0)

    eeg -= means
    eeg /= stdev

  def __repr__(self):
    s = "%d Electrodes : Length: %f sec, SampleRate: %f Hz, TimePeriod: %d" % (
      len(self.electrode),
      self.length_in_sec,
      self.sample_rate_in_hz,
      self.timeperiod,
    )
    return s
