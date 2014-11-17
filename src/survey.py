import os
import re 

import numpy as np
import hickle

import EEG

import argparse
parser = argparse.ArgumentParser(description='Survey the data')
parser.add_argument('--subject', type=str, required=True, help="Dog_{1,2,3,4,5}, Patient_{1,2}")
args = parser.parse_args()

# 'Dog_2', 'Patient_2'
_subject = args.subject

d = "data/orig/%s/" % (_subject, )

csv=open("data/survey.%s.csv" % (_subject,), 'w')
csv.write(EEG.EEG.survey_header()+"\n")

f_match = re.compile(r'%s_(.*?)_segment_(\d*?)\.mat' % (_subject,) )
for f in sorted(os.listdir(d)):
  m = re.match(f_match, f)
  if m is None: continue
  desc, num = m.group(1), int(m.group(2))
  
  print desc, num
  
  p = EEG.EEG(_subject, desc, num)
  p.load()
  csv.write(p.survey_line_write())

csv.close()
