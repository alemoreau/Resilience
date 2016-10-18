from struct import *
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
import scipy.io
import base64

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


def load_mat(path='./gre_216a.mat', sparse = False):
    with open(path, 'rb') as myfile:
        data_file=base64.b64encode(myfile.read())
    problem = scipy.io.loadmat(path)['Problem']
    A = problem["A"][0][0]
    if sparse:
        A = A.tocsr()
    else:
        A = A.toarray()
    name = problem["name"][0][0][0]

    n = A.shape[0]
    x = np.ones(n)
    b = A.dot(x)
    x = np.zeros(n)
    return {"A":A, "x0":x, "b":b, "problem": problem, "file": data_file, "name": name}


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



def has_converged(data, epsilon = 1.e-12):
    return len(filter (lambda d: d[0] < epsilon and d[1] < epsilon, zip(data["true_residuals"], data["residuals"]))) > 0

def when_has_converged(data, epsilon = 1.e-12):
    iteration_residual = filter (lambda d: d[1][0] < epsilon and d[1][1] < epsilon, 
                                 enumerate(zip(data["true_residuals"], data["residuals"])))
    if iteration_residual:
        return min(iteration_residual, key=lambda d:d[0])[0]
    else:
        return None


def false_detection(data, c = 0.5, epsilon = 1.e-12, key_checksum="checksum", key_threshold="threshold"):
    m = when_has_converged(data, epsilon = (1-c)*epsilon)
    for i, (checksum, threshold) in enumerate(zip(data[key_checksum][:m], data[key_threshold][:m])):
        if checksum > c * threshold:
            if data["faults"][0]["timer"] != i:
                return True
    return False

def no_impact_fault_detection(data, c = 0.5, epsilon = 1.e-12, key_checksum="checksum", key_threshold="threshold"):
    m = when_has_converged(data, epsilon = (1-c)*epsilon)
    f = data["faults"][0]["timer"]
    if f >= len(data[key_threshold]) or f >= len(data[key_checksum]):
	return False
    return (has_converged(data, epsilon = epsilon) and 
            data[key_checksum][f] > c * data[key_threshold][f])
        
def no_impact_fault_no_detection(data, c = 0.5, epsilon = 1.e-12, key_checksum="checksum", key_threshold="threshold"):
    f = data["faults"][0]["timer"]
    if f >= len(data[key_threshold]) or f >= len(data[key_checksum]):
	return False
    return (has_converged(data, epsilon = epsilon) and
            data[key_checksum][f] < c * data[key_threshold][f])
        
def fault_no_detection(data, c = 0.5, epsilon = 1.e-12, key_checksum="checksum", key_threshold="threshold"):
    f = data["faults"][0]["timer"]
    if f >= len(data[key_threshold]) or f >= len(data[key_checksum]):
	return False
    return (not has_converged(data, epsilon = epsilon)  and
            data[key_checksum][f] < c * data[key_threshold][f])
        
def fault_detection(data, c = 0.5, epsilon = 1.e-12, key_checksum="checksum", key_threshold="threshold"):
    f = data["faults"][0]["timer"]
    if f >= len(data[key_threshold]) or f >= len(data[key_checksum]):
	return False
    return (not has_converged(data, epsilon = epsilon) and
            data[key_checksum][f] > c * data[key_threshold][f])
               
