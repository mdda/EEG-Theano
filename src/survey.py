import os
import re 

import numpy as np
import hickle

import EEG

_patient = 'Dog_2'

d = "data/orig/%s/" % (_patient, )

csv=open("data/survey.%s.csv" % (_patient,), 'w')
csv.write(EEG.EEG.survey_header()+"\n")

f_match = re.compile(r'%s_(.*?)_segment_(\d*?)\.mat' % (_patient,) )
for f in sorted(os.listdir(d)):
  m = re.match(f_match, f)
  if m is None: continue
  desc, num = m.group(1), int(m.group(2))
  
  print desc, num
  
  p = EEG.EEG(_patient, desc, num)
  p.load()
  csv.write(p.survey_line_initial())

csv.close()
