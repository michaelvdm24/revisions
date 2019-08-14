#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:30:48 2018

@author: dalezhang
"""

import numpy as np
import scipy.linalg
import scipy.signal
import scipy.io.wavfile
import matplotlib.pyplot as plt


N = 320
K = 10
Nb = 122

def LPC_encoding(y, L):

    # compute the temporal correlation function
    R_y = np.correlate(y, y, mode = 'full')

    # using the temporal correlation function to construct the 
    # Yule-Walker Matrix
    M = scipy.linalg.toeplitz(R_y[:K], R_y[:K])

    # Find the least square solution of the Yule-Walker Equation
    alpha = np.linalg.lstsq(M, R_y[1:K+1], rcond=None)[0]

    # Estimating the residue
    #e = scipy.signal.lfilter(np.concatenate([[1],-alpha]), [1], y)
    e = np.array([0] + [(1 - sum([alpha[k]*y[z]*z**(-(k+1)) for k in range(K)]))*y[z]\
            for z in range(1,len(y))])

    # split the original signal in to N segments
    # and quantize e_Q 
    segs = []
    for i in range(Nb):

        e_Q_seg = e[i*N:(i+1)*N]

        # Residue Quantization
        delta = (np.max(e_Q_seg) - np.min(e_Q_seg))/(2**L)
        e_Q_seg = np.round(e_Q_seg / delta) * delta
        segs.append(e_Q_seg)

    e_Q = np.concatenate(segs)

    #delta = (np.max(e) - np.min(e))/(2**L)
    #e_Q = np.round(e / delta) * delta
    print(alpha.shape, alpha)
    print(e_Q.shape, e_Q)

    return alpha, e_Q


def LPC_reconstruct(alpha, e_Q):

    # reconstruct the signal
    # y_LPC = scipy.signal.lfilter([1], np.concatenate([[1],-alpha]), e_Q)
    # y_LPC[0] is computed saparetely... 
    y_LPC = e_Q[0] + [e_Q[z]/(1-np.sum([alpha[k]*z**(-(k+1)) \
            for k in range(K)])) for z in range(1,len(e_Q))]
    #y_LPC = np.reshape(y_LPC, order = 'F')
    
    return y_LPC

if __name__ == '__main__':

    # loading data
    y = np.loadtxt('TestData2017.txt', delimiter = ',')

    # normalizing data
    y = y / np.max(np.abs(y))


    # encode signal
    alpha, e_Q = LPC_encoding(y, 1)

    # decode signal
    y_LPC = LPC_reconstruct(alpha, e_Q)

    # plotting
    plt.figure ()
    plt.plot (y[0:320], 'b',label='true sample')
    # plt.plot (e_Q[0:320], 'y*', label = 'e_Q')
    plt.plot (y_LPC[0:320], 'r:' ,label='reconstructed signal')
    plt.legend ()
    plt.show()


    scipy.io.wavfile.write('output' + str(1) + '.wav', 11025, y_LPC.astype(np.float32))