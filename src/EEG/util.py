

#Loading Data into Python :
#  https://www.kaggle.com/c/seizure-prediction/forums/t/10128/fields-in-matlab-file-can-t-find-data-etc

"""
data_struct = scipy.io.loadmat('Dog_1_interictal_segment_0008.mat')

Expanding on Claudiu1989's great response:

data_struct[sample][0][0] is your 'base' level to access the clip's information. From there:

data_struct[sample][0][0][0][x] - The series of electrode measurements corresponding to electrode x. For the example I was looking at from Dog_5, this = 239766 measurements (399 Hz * 600 seconds).
data_struct[sample][0][0][1][0] - The length (in seconds) of the clip. If they are all 10 minutes as described, this should be 600.
data_struct[sample][0][0][2][0] - The sampling rate in Hz (e.g. ~399 for Dog_5)
data_struct[sample][0][0][3][0][x] - The name of the xth electrode.
data_struct[sample][0][0][4][0] - The index of the clip's location within the hour (e.g. 4 = from minutes 40-50)

EDIT: For the last bullet, you should only expect to have a '4' value for the third index for preictal and interictal files. Since we are not given the test files in the context of an hour long series, they are not provided with this field. Thanks to Lawrence for pointing out the lack of clarity! :)

"""

