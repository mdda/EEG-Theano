#import pygtk

import EEG
import numpy as np

p = EEG.EEG('Dog_2', 'interictal', 17)
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

t_start  = 0
t_length = 2000

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(eeg[t_start:(t_start+t_length), :] + trace_levels)
ax.plot(np.zeros((t_length, n)) + trace_levels, '--', color='gray')

plt.yticks(trace_levels)
ax.set_yticklabels(p.electrode)
ax.axis('tight')

#ax.legend(first['channels'])

plt.show()

print "DONE"
