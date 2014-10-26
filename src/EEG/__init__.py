import numpy as np

import util

class EEG:
  def __init__(self, _patient, _desc, _num):
    self.data=None
    
    self.patient=_patient
    self.desc   =_desc
    self.num    =_num
    
    self.is_test  = 1 if self.desc=='test'     else 0
    self.is_ictal = 1 if self.desc=='preictal' else 0
    
    self.est0 = self.est1 = self.est2 = -1.
    self.train0 = float(self.is_ictal)
    self.train1 = self.train2 = -1.
    
    self.electrode=[]
    
  def load(self):
    d = util.load(self.patient, self.desc, self.num)
    self.data = d[0].astype(dtype=np.float32)
    
    #print d[1][0]
    #print type(d[0])
    
    # Unbox simple parameters
    self.length_in_sec = d[1][0][0]
    self.sample_rate_in_hz = d[2][0][0]
    
    electrode_arr = d[3][0]  # This is an array
    self.n_channels = len(electrode_arr)
    self.electrode = [ e[0] for e in electrode_arr ]

    self.timeperiod = -1 if self.desc == 'test' else d[4][0][0]

    
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
    s = "%d num, %d Electrodes : Length: %f sec, SampleRate: %f Hz, TimePeriod: %d" % (
      self.num,
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
      'timeperiod','length_in_sec','sample_rate_in_hz', 
      'is_test', 
      'est0', 'est1', 'est2', 
      'train0','train1','train2', 
    ])

  def survey_line_write(self):
    return ("%s,%s,%d," +"%d,%16.8f,%16.8f,"+
            "%d,"+
            "%.8f,%.8f,%.8f,%.8f,%.8f,%.8f,\n") % (
      self.patient, self.desc, self.num, 
      self.timeperiod, self.length_in_sec, self.sample_rate_in_hz, 
      self.is_test, 
      self.est0, self.est1, self.est2,
      self.train0, self.train1,self.train2,
    )
  
  def survey_line_read(self, line):
    #print line 
    i = line.split(",")
    self.patient, self.desc, self.num = i[0], i[1], int(i[2])
    self.timeperiod, self.length_in_sec, self.sample_rate_in_hz = float(i[3]), float(i[4]), float(i[5])
    self.is_test = int(i[6])
    self.est0, self.est1, self.est2 = float(i[7]), float(i[8]), float(i[9])
    self.train0, self.train1, self.train2 = float(i[10]), float(i[11]), float(i[12])
