import EEG
import numpy as np

p = EEG.EEG('Dog_2', 'interictal', 17)
p.normalize_channels()

#print np.shape(p.data)  == (16, ~240k)

data = p.data
eeg = np.rollaxis(data, 1)

# see p797 of matplotlib pdf : matshow() for 2d colour-map of (say) correlation matrix

import matplotlib
#matplotlib.use('Qt4Agg')  # Needs PySide
#matplotlib.use('GTKAgg')  # Needs pygtk
#matplotlib.use('TkAgg')  # Needs Tkinter
import matplotlib.pyplot as plt

#data = p.data
n = p.n_channels
spacing = 5
trace_levels = spacing*np.arange(n,0,-1)

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(eeg[0:2000, :] + trace_levels)

ax.plot(np.zeros((2000, n)) + trace_levels, '--', color='gray')

ax.set_yticklabels(p.electrode)
ax.axis('tight')

#ax.legend(first['channels'])

plt.show()

print "DONE"
