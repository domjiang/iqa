import scipy.io as sio
from numpy import genfromtxt
import numpy as np

paths = genfromtxt('path.csv', delimiter=' ', dtype='str')

train_data = []

for i in range(len(paths)):
    mat_fname = paths[i]

    img_data = sio.loadmat(mat_fname)

    train_data.append(img_data['patches'].T)

train_input = np.reshape(np.concatenate(train_data, axis=0), newshape=(1473 * 64, 192, 1))
print(train_input.shape)