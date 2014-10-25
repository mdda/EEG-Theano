import numpy as np

import util

class EEG:
  def __init__(self, _patient, _desc, _num):
    d = util.load(_patient, _desc, _num)
    self.data = d[0].astype(dtype=np.float32)
    
    self.patient=_patient
    self.desc   =_desc
    self.num    =_num
    
    #print d[1][0]
    #print type(d[0])
    
    # Unbox simple parameters
    self.length_in_sec = d[1][0][0]
    self.sample_rate_in_hz = d[2][0][0]
    
    electrode_arr = d[3][0]  # This is an array
    self.n_channels = len(electrode_arr)
    self.electrode = [ e[0] for e in electrode_arr ]

    self.timeperiod = None if _desc == 'test' else d[4][0][0]

    
  def normalize_channels(self):
    ## This is the key that stops mean() and std() behaving as expected
    #print np.result_type(data)

    ## See :: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/data.py
    eeg = np.rollaxis(self.data, 1)

    means = eeg.mean(axis=0)
    stdev = eeg.std(axis=0)

    eeg -= means
    eeg /= stdev

  def normalize_overall(self):
    stdev = self.data.std() # Over all channels  -> scalar
    self.data /= stdev

  def __repr__(self):
    s = "%d Electrodes : Length: %f sec, SampleRate: %f Hz, TimePeriod: %d" % (
      len(self.electrode),
      self.length_in_sec,
      self.sample_rate_in_hz,
      self.timeperiod,
    )
    return s

  @classmethod
  def survey_header(_cls):
    return ','.join([
      'patient','desc','num',
      'period','length_in_sec','sample_rate_in_hz', 
      'is_test', 
      'est0', 'est1', 'est2', 
      'train_value0','train_value1','train_value2', 
    ])

  def survey_line(self):
    is_test  = 1 if self.desc=='test' else 0
    is_ictal = 1 if self.desc=='preictal' else 0
    return "%s,%s,%d,%d,%16.8f,%16.8f,%d,-1,-1,-1,%d,-1,-1,\n" % (
      self.patient, self.desc, self.num, 
      self.timeperiod, self.length_in_sec, self.sample_rate_in_hz, 
      is_test, is_ictal
    )
  
