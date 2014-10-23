import EEG
import numpy as np
import scipy as sp

p = EEG.EEG('Dog_2', 'interictal', 17)
#p = EEG.EEG('Patient_2', 'interictal', 17)
print p
#p.normalize_channels()
#p.normalize_overall()

#print np.shape(p.data)  == (16, ~240k)

data = p.data
#eeg = np.rollaxis(data, 1)

