from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import re
from scipy import linalg
import scipy.ndimage as ndi
from six.moves import range
import os
import threading
from PIL import Image
from glob import glob

def load_img(path, grayscale=False, target_size=None):
    '''Load an image into PIL format.
    # Arguments
        path: path to image file
        grayscale: boolean
        target_size: None (default to original size)
            or (img_height, img_width)
    '''
    
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:  # Ensure 3 channel even when loaded image is grayscale
        img = img.convert('RGB')
    if target_size:
        img = img.resize((target_size[1], target_size[0]))
    return img

def img_to_array(img, dim_ordering='tf'):
    
    if dim_ordering not in ['th', 'tf']:
        raise Exception('Unknown dim_ordering: ', dim_ordering)
    
    # image has dim_ordering (height, width, channel)
    x = np.asarray(img, dtype='float32')
    if len(x.shape) == 3:
        if dim_ordering == 'th':
            x = x.transpose(2, 0, 1)
    elif len(x.shape) == 2:
        if dim_ordering == 'th':
            x = x.reshape((1, x.shape[0], x.shape[1]))
        else:
            x = x.reshape((x.shape[0], x.shape[1], 1))
    else:
        raise Exception('Unsupported image shape: ', x.shape)
    return x

def array_to_img(x, dim_ordering='tf', scale=True):

    if dim_ordering == 'th':
        x = x.transpose(1, 2, 0)
    if scale:
        x += max(-np.min(x), 0)
        x_max = np.max(x)
        if x_max != 0:
            x /= x_max
        x *= 255
    if x.shape[2] == 3:
        # RGB
        return Image.fromarray(x.astype('uint8'), 'RGB')
    elif x.shape[2] == 1:
        # grayscale
        return Image.fromarray(x[:, :, 0].astype('uint8'), 'L')
    else:
        raise Exception('Unsupported channel number: ', x.shape[2])

def load_log_images(path, dim_ordering='tf', image_shape=(4, 60, 3), window_length=7, nb_sublog=10, top_n=-1):

	files = sorted(glob(path + '*.png'))

	images = np.zeros((len(files),) + image_shape)
	for i, file in enumerate(files):
		# print(file)
		img = load_img(file)
		x = img_to_array(img, dim_ordering)
		images[i] = x

		if (top_n > 0 and top_n >= i):
			break

	return images, files

def load_log_matrix(logfile, dictfile):
    
    fin = open(dictfile)
    logDict = {}
    for line in fin:
        tokens = line.split(',')
        idx = int(tokens[0])
        log = tokens[1]
        logDict[log] = idx

    fin = open(logfile)
    logList = []    
    for n, line in enumerate(fin):
        tokens = line.split(',')
        index = int(tokens[0])
        
        log_i = []        
        for i in range(1, len(tokens)):
            log_i.append(int(tokens[i]))

        logList.append(log_i)

    #null_index = logDict["0:0:0"]
    null_index = 4

    # get transition matrix
    max_freq = -1
    dim = len(logDict)
    Ts = get_transition_matrix(dim, logList, 5, null_index)

    return Ts

def load_log_coarse_matrix(logfile, dictfile):
    
    fin = open(dictfile)
    logDict = {}
    logList_coarse = []
    for line in fin:
        tokens = line.split(',')
        subtokens = tokens[1].split(':')
        #print(tokens[1])
        idx = int(tokens[0])
        log = tokens[1]
        logDict[idx] = log
        logList_coarse.append(subtokens[0] + ":" + subtokens[1])

    null_index = logList_coarse.index("0:0")
    print(null_index)

    fin = open(logfile)
    logList = []
    for n, line in enumerate(fin):
        tokens = line.split(',')
        index = int(tokens[0])
        
        log_i = []        
        for i in range(1, len(tokens)):
            subtokens = logDict[int(tokens[i].split('.')[0])].split(':')
            log_i.append(logList_coarse.index(subtokens[0] + ":" + subtokens[1]))

        logList.append(log_i)

    # get transition matrix
    dim = len(logList_coarse)
    Ts = get_transition_matrix(dim, logList, 5, null_index)

    return Ts

def get_transition_matrix(dim, logList, d, null_index=-1):

    max_freq = -1    
    Ts = np.zeros((len(logList), ) + (dim, dim))
    for n, log_i in enumerate(logList):
        Ti = np.zeros((dim,dim))

        for j in range(len(log_i)-1):
            u = log_i[j]

            Ti[u,u] += 1.0
            max_freq = Ti[u,u] if (Ti[u,u] > max_freq) else max_freq

            for k in range(j+1, min(j+d, len(log_i))):                
                v = log_i[k]

                if (u != null_index and v != null_index):
                    Ti[u,v] += 1.0

        # normalize
        for i in range(dim):
            Ti[i,i] /= max_freq
            max_freq_i = max(Ti[i])

            if (max_freq_i > 0):
                for j in range(dim):
                    if (i is not j):
                        Ti[i,j] /= max_freq_i

        Ts[n] = Ti

    return Ts

from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Activation, ZeroPadding2D, Flatten, Reshape, Dropout, LocallyConnected2D
from keras.models import Model, Sequential, load_model

max_trial = 5
index = '_f1_0.8_deeper'

# Load logimages
base_path = '/Users/sukim/Documents/keras/'
input_path = base_path + 'png4/'
output_path = base_path + 'result4_trans/'
X, files = load_log_images(input_path)
Ts = load_log_coarse_matrix(base_path + 'log4', base_path + 'log4_dict')
dim = Ts.shape[1]
#print(Ts.shape, Ts.shape[1])

# Model
isLocal = True
if (isLocal):
    input_img = Input(shape=(3, 60, 4))
    x = LocallyConnected2D(64, 7, 4, activation='relu')(input_img)
    x = Convolution2D(32, 2, 1, activation='relu')(input_img)
    x = Flatten()(x)
    x = Dense(dim*dim, activation='relu')(x)
    x = Dropout(0.8)(x)
    x = Dense(dim*dim, activation='relu')(x)
    x = Dropout(0.8)(x)
    encoded = Dense(dim, activation='relu')(x)
    x = Dense(dim*dim, activation='relu')(x)
    x = Dropout(0.8)(x)
    x = Dense(dim*dim, activation='sigmoid')(x)
    decoded = Reshape((dim, dim))(x)
else:
    input_img = Input(shape=(3, 60, 4))
    x = LocallyConnected2D(64, 7, 4, activation='relu')(input_img)
    x = Convolution2D(32, 2, 1, activation='relu')(input_img)
    x = Flatten()(x)
    x = Dense(dim*dim, activation='relu')(x)
    x = Dropout(0.2)(x)
    x = Dense(dim*dim, activation='relu')(x)
    x = Dropout(0.2)(x)    
    encoded = Dense(dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(dim, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(dim, activation='sigmoid')(x)    
    decoded = Reshape((dim, dim))(x)

autoencoder = Model(input_img, decoded)
encoder = Model(input_img, encoded)

autoencoder.compile(optimizer='rmsprop', loss='mse')

n = len(X)
X = np.reshape(X, (n, 3, 60, 4))
X = X / 255.0
for trial in range(0, max_trial):
    permuted_index = np.random.permutation(n)
    X_train = X[permuted_index[:int(n*0.7)], :, :, :]
    X_test = X[permuted_index[int(n*0.7):n], :, :, :]
    Ts_train = Ts[permuted_index[:int(n*0.7)], :, :]
    Ts_test = Ts[permuted_index[int(n*0.7):n], :, :]
    
    print(trial)

    autoencoder.fit(X_train, Ts_train,
        nb_epoch=30,
        batch_size=128,
        shuffle=True,
        validation_data=(X_test, Ts_test))

X_eval = np.reshape(X, (len(X), 3, 60, 4))

# write encoded images and files
fin_filelist = open(output_path + 'files', 'w')
fin_veclist = open(output_path + 'vecs' + index, 'w')
encoded_img = encoder.predict(X_eval)
np.save(fin_veclist, encoded_img)
