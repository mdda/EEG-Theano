import EEG
import numpy as np

p = EEG.EEG('Dog_2', 'interictal', 17)

#print np.shape(p.data)

data = p.data
norms = np.apply_along_axis(np.linalg.norm, 0, data)

#eeg = data / norms

exit(0)

# p797 of pdf : matshow()

import matplotlib.pyplot as plt

data = p.data

plt.figure(3);
plt.plot(filtered[:, 0:20000].T + 5*np.arange(7,-1,-1));

#plt.plot(np.zeros((512,8)) + 80*np.arange(7,-1,-1),'--',color='gray');
#plt.yticks([]);
#plt.axis('tight');
#plt.legend(first['channels']);
