#import pygtk

import EEG
import numpy as np

#p = EEG.EEG('Dog_2', 'interictal', 17)
p = EEG.EEG('Patient_2', 'interictal', 17)
print p
p.normalize_channels()

#print np.shape(p.data)  == (16, ~240k)

data = p.data
eeg = np.rollaxis(data, 1)

# see p797 of matplotlib pdf : matshow() for 2d colour-map of (say) correlation matrix

#import matplotlib
#matplotlib.use('Qt4Agg')  # Needs PySide - difficult to install, large
#matplotlib.use('GTKAgg')  # Needs pygtk   - found script on Gist.github.com
#matplotlib.use('TkAgg')  # Needs Tkinter - difficult within VirtualEnv
import matplotlib.pyplot as plt

#data = p.data
n = p.n_channels
spacing = 5
trace_levels = spacing*np.arange(n,0,-1)

t_start     =  500 # in seconds  * 0 - 595
t_duration  =   5  # in seconds

s_start  = int(p.sample_rate_in_hz * t_start)
s_length = int(p.sample_rate_in_hz * t_duration)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(eeg[s_start:(s_start+s_length), :] + trace_levels)
ax.plot(np.zeros((s_length, n)) + trace_levels, '--', color='gray')

plt.yticks(trace_levels)
ax.set_yticklabels(p.electrode)
ax.axis('tight')

#ax.legend(first['channels'])

plt.show()

print "DONE"
