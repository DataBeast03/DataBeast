__author__ = 'mike.bowles'
import cPickle
import numpy as np
import os
from sklearn import preprocessing

def unpickle(file):
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

##http://blog.dominodatalab.com/gpu-computing-and-deep-learning/

def cifar(nData=1, Normalize=False):
    names = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    ##data_dir = '/home/ubuntu/Documents/galvanize/code/theano_exp1/tuftsCode'

    fd = open('data_batch_1')
    dictData = cPickle.load(fd)
    fd.close()
    xTr = np.array(dictData['data']).astype('float64')
    if Normalize: xTr = preprocessing.scale(xTr)
    yTr = np.array(dictData['labels']).astype('int32')

    for i in range(1, nData, 1):
        fd = open(names[1])
        dictData = cPickle.load(fd)
        fd.close()
        xTrTemp = np.array(dictData['data']).astype('float64')
        if Normalize: xTrTemp = preprocessing.scale(xTrTemp)
        yTrTemp = np.array(dictData['labels']).astype('int32')
        xTr = np.vstack((xTr, xTrTemp))
        yTr = np.vstack((yTr, yTrTemp))

    #
    fd = open('test_batch')
    dictData = cPickle.load(fd)
    fd.close()
    xTe = np.array(dictData['data']).astype('float64')
    if Normalize: xTe = preprocessing.scale(xTe)
    yTe = np.array(dictData['labels']).astype('int32')
    #
    yTe = one_hot(yTe, 10)
    yTr = one_hot(yTr, 10)
    #
    return xTr, yTr, xTe, yTe


