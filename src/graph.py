import EEG
import numpy as np

p = EEG.EEG('Dog_2', 'interictal', 17)

#print np.shape(p.data)  == (16, ~240k)

data = p.data
#norms = np.apply_along_axis(np.linalg.norm, 0, data)

## This is the key that stops mean() and std() behaving as expected
#print np.result_type(data)

## See :: https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/preprocessing/data.py
eeg = np.rollaxis(data, 1)

means = eeg.mean(axis=0)
stdev = eeg.std(axis=0)

#print np.shape(means)

#print means
#print stdev

eeg -= means
eeg /= stdev

means = eeg.mean(axis=0)
stdev = eeg.std(axis=0)

#print np.shape(means)
#print np.shape(eeg)

#print means
#print stdev

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
