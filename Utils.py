from struct import *
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import scipy.io

def bitflip(x, pos, type=np.float64):
	t = 'f' if type is np.float32 else 'd'
	fs = pack(t, x)
	if (type is np.float64):
		s = 'BBBBBBBB'
		p = 63
	if (type is np.float32):
		s = 'BBBB'	
		p = 31
	bval = list(unpack(s,fs))
	[q,r] = divmod(p-pos,8)
	bval[q] ^= 1 << r
	
	fs = pack(s, *bval)
	fnew=unpack(t, fs)
	return fnew[0]


def load_mat(path='./gre_216a.mat'):
    
    try:
        A = scipy.io.loadmat(path)['Problem'][0][0][1].toarray()
    except:
        A = scipy.io.loadmat(path)['Problem'][0][0][2].toarray()
    n = A.shape[0]
    x = np.ones((n, 1))
    b = A.dot(x)
    x = np.zeros((n, 1))
    return {"A":A, "x0":x, "b":b}

def plot_2D(X, Y, title = '', grid = True, label = "", log=False, logX=False, linestyle = None, xlabel = None, ylabel = None, xlim = None, ylim = None, bbox_to_anchor=(1, 0.5)):

    if log:
        p = plt.semilogy(X, Y[:len(X)], label=label)
    else:
        if logX:
            p = plt.semilogx(X, Y[:len(X)], label=label)
        else:
            p = plt.plot(X, Y[:len(X)], label=label)
    if linestyle:
        plt.setp(p, linestyle=linestyle) 
    plt.title(title)
    plt.legend(loc='center left', bbox_to_anchor=bbox_to_anchor)
    if xlim:
        plt.xlim(xlim)
    if ylim:
        plt.ylim(ylim)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if grid:
        plt.grid(True)

