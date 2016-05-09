from struct import *
import numpy as np

def bitflip(x, pos, type=np.float64):
	t = 'f' if type is np.float32 else 'd'
	fs = pack(t, x)
	if (type is np.float64):
		s = 'BBBBBBBB'
	if (type is np.float32):
		s = 'BBBB'	
	bval = list(unpack(s,fs))
	[q,r] = divmod(pos,8)
	bval[q] ^= 1 << r
	
	fs = pack(s, *bval)
	fnew=unpack(t, fs)
	return fnew[0]


