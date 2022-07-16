# 2020.01.28
# @yifan
# color space conversion
#
import numpy as np
import copy
import cv2
from skimage.measure import block_reduce

def Clip(X):
    tmp = copy.deepcopy(X)
    tmp = tmp.astype('int16')
    tmp[tmp > 255] = 255
    tmp[tmp < 0] = 0
    return tmp

def BGR2RGB(X):
    R        = copy.deepcopy(X[:,:,2:])
    G        = copy.deepcopy(X[:,:,1:2])
    B        = copy.deepcopy(X[:,:,0:1])
    return np.concatenate((R, G, B), axis=-1)

def RGB2BGR(X):
    B        = copy.deepcopy(X[:,:,2:])
    G        = copy.deepcopy(X[:,:,1:2])
    R        = copy.deepcopy(X[:,:,0:1])
    return np.concatenate((B, G, R), axis=-1)

def RGB2YUV(X, doClip=False):
    X = X.astype('float32')-127
    K = np.array([[   0.299,    0.587,    0.114],
                  [-0.14713, -0.28886 ,    0.436],
                  [  0.615   , -0.51499   , -0.10001]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3,-1))
    X = np.moveaxis(X.reshape(S), 0, -1)
    if doClip == True:
        return np.round(X)
    return X

def YUV2RGB(X, doClip=False):
    K = np.array([[1,        0,  1.13983],
                  [1, - 0.39465 , - 0.58060],
                  [1,  2.03211,       0]])
    X = np.moveaxis(X, -1, 0)
    S = X.shape
    X = np.dot(K, X.reshape(3,-1))
    X = np.moveaxis(X.reshape(S), 0, -1)
    if doClip == True:
        return Clip(X)
    return X+127

def BGR2YUV(X, doClip=False):
    return RGB2YUV(BGR2RGB(X), doClip=doClip)

def YUV2BGR(X, doClip=False):
    return RGB2BGR(YUV2RGB(X, doClip))

def c444_to_c420(YUV):
    return [YUV[:,:,0], block_reduce(YUV[:,:,1], (2,2), np.mean), block_reduce(YUV[:,:,2], (2,2), np.mean)]

def c420_to_c444(X):
    S = X[0].shape
    tmp = [ X[0].reshape(S[0],S[1],1), cv2.resize(X[1], (S[1], S[0])).reshape(S[0],S[1],1), cv2.resize(X[2], (S[1], S[0])).reshape(S[0],S[1],1) ]
    return np.concatenate(tmp, axis=-1)
