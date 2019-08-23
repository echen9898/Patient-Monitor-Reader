import os, sys
import numpy as np
import scipy.io
import cv2
import imutils

def shuffle_in_unison(a, b):
    assert len(a) == len(b)
    shuffled_a = np.empty(a.shape, dtype=a.dtype)
    shuffled_b = np.empty(b.shape, dtype=b.dtype)
    permutation = np.random.permutation(len(a))
    for old_index, new_index in enumerate(permutation):
        shuffled_a[new_index] = a[old_index]
        shuffled_b[new_index] = b[old_index]
    return shuffled_a, shuffled_b

############################## Char74 Fonts ##############################
os.chdir('char74/fonts/')

x = list()
y = list()
for folder in os.listdir():
    if folder not in ['Sample051_copy', 'Sample051', '.DS_Store', 'x.npy', 'y.npy']:
        label = int(folder[-2:])-1
        os.chdir(folder)
        for file in os.listdir():
            if file != '.DS_Store':
                gray = cv2.imread(file, cv2.IMREAD_GRAYSCALE) # (128, 128, 3)
                resized = cv2.resize(gray, (28, 28))
                thresh, bw = cv2.threshold(resized, 128, 255, cv2.THRESH_BINARY)
                x.append(255-bw) # white digit on black background
                y.append(label)
        os.chdir('..')

x = np.array(x) # (11168, 28, 28)
y = np.array(y) # (11168, )
x, y = shuffle_in_unison(x, y) # shuffle together before saving

# for i in range(11168):
#     print(y[i])
#     cv2.imshow('b', x[i])
#     cv2.waitKey(0)

np.save('x.npy', x)
np.save('y.npy', y)
        
