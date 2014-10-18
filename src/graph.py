import EEG
import numpy as np

p = EEG.EEG('Dog_2', 'interictal', 17)
p.normalize_channels()

#print np.shape(p.data)  == (16, ~240k)

data = p.data
eeg = np.rollaxis(data, 1)

# see p797 of matplotlib pdf : matshow() for 2d colour-map of (say) correlation matrix

import matplotlib.pyplot as plt

#data = p.data
n = p.n_channels
spacing = 3

fig = plt.figure()
ax = fig.add_subplot(111)

ax.plot(eeg[:, 0:20000] + spacing*np.arange(0,n,1))

#ax.plot(np.zeros((20000,p.n_channels)) + spacing*np.arange(0,n,1),'--',color='gray')
#ax.yticks([])
ax.axis('tight')
#ax.legend(first['channels'])

plt.show()

print "DONE"
