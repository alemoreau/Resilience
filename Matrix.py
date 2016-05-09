# -*- coding: utf-8 -*-
import numpy as np
import Utils as U

from Parameters import *


class Matrix():
	def __init__(self, numpy_matrix, info = {}, dtype='d', vulnerable= False, vulnerability_parameters=Parameters()):
		self._data = numpy_matrix
		self.shape = numpy_matrix.shape
		self.info = info 
                self.dtype = dtype
		self.vulnerability_parameters = vulnerability_parameters
		self.vulnerable = vulnerable 
		self.faults = []
                self.timer = 0

        def set_data(self, data):
                self._data = data
                self.shape = data.shape

	def get_info(self):
		return self.info

	def dot(self, matrix):
		return Matrix(self._data.dot(matrix), dtype=self.dtype)
	def norm(self, ord=None):
		return np.linalg.norm(self._data, ord=ord)
	def transpose(self):
		return Matrix(self._data.transpose(), dtype=self.dtype)
	def ravel(self):
                return Matrix(np.ravel(self._data), dtype = self.dtype)
	def dot(self, other):
		s = 0.
		for a, b in zip(self._data, other._data):
			if not np.isscalar(a):
				a = a[0]
			if not np.isscalar(b):
				b = b[0]
			s += a * b			
		return	s
	def svd(self):
		return np.linalg.svd(self._data)
	def det(self):
		return np.linalg.det(self._data)
	def eig(self):
		return np.linalg.eig(self._data)

	def __abs__(self):
                return Matrix(abs(self._data), dtype=self.dtype)
        
	def __add__(self, other):
		return Matrix(self._data + other._data, dtype=self.dtype)
	def __div__(self, other):
		return Matrix(self._data / other, dtype=self.dtype)

	def __mul__(self, other):
		if np.isscalar(other):
			return Matrix(self._data * other, dtype=self.dtype)
		else:
                        if (type(self.timer) is int) and self.vulnerable:
                                self.timer = self.timer + 1
                        if isinstance(other, Matrix) and type(other.timer) is int and other.vulnerable:
                                other.timer = other.timer + 1

                        
			perform_safe_product = not self.is_vulnerable()
                        if (isinstance(other, Matrix)):
				perform_safe_product &= not other.is_vulnerable()

			product = Matrix(np.dot(self._data, other._data), dtype=self.dtype)
                        if perform_safe_product:
                                return product
			
			else:
				
				vulnerable_self = self.is_vulnerable()
				vulnerable_other = other.is_vulnerable()
				if vulnerable_self or vulnerable_other:
					(i,j,k) = self.vulnerability_parameters.get("fault_indices", (int(self.shape[0]), 0, int(self.shape[1] * np.random.rand())))
					# looking for the closest non-zero couple (self[i, k], other[k, j])

					(n, m) = self.shape
					(m, l) = other.shape

					increment_i = 1
					oscillation_i = 1			
					fault_occured = False
					for s in xrange(2 * max(n-i, i)):
						#next i index
						increment_j = 1
						oscillation_j = 1
						for t in xrange(2 * max(l-j, j)):
							#next j index
							increment_k = 1
							oscillation_k = 1
							for u in xrange(2 * max(m-k, k)):
								#next k index
								if (i < 0 or i >= n):
									break
								if (k >= 0 and k < m):
									if self.__get__((i, k)) != 0. and other[k, j] != 0.:
										x_self = self.__get__((i, k))
										x_other = other[k, j]
										x_product = x_self * x_other
								
										if vulnerable_self:
											bits = self.vulnerability_parameters.get("vulnerable_bits", [i for i in xrange(64)])
											callback = self.vulnerability_paramaters("fault_callback", None)
										else:
											bits = other.vulnerability_parameters.get("vulnerable_bits", [i for i in xrange(64)])
											callback = other.vulnerability_paramaters("fault_callback", None)
										
										bit_flipped = bits[int(np.floor(np.random.rand() * len(bits)))]
										which_register = np.random.rand()
										
										fault_self = {}
										fault_self["loc"] = (i, j, k)

										fault["value_before"] = x_product
										if which_register < 1./3:
											fault["register"] = "left"
											fault["register_before"] = x_self
											x_self = U.bitflip(x_self, bit_flipped, type=self.dtype)
											fault["register_after"] = x_self
											fault["timer"] = self.timer
											x_product = x_self * x_other
										elif which_register < 2./3:
											fault["register"] = "right"
											fault["register_before"] = x_other
											x_other = U.bitflip(x_other, bit_flipped, type=other.dtype)
											fault["register_after"] = x_other
											x_product = x_self * x_other
										else:
											fault["register"] = "middle"
											fault["register_before"] = x_product
											x_product = U.bitflip(x_product, bit_flipped, type=self.dtype)
											fault["register_after"] = x_product
										fault["value_after"] = x_product


										fault = {"loc":(i, j, k), "value_before": M[i,k], "value_after" : xm, "bit" :bit_flipped, "origin":"left", "without_fault" : no_fault, "with_fault":xn*xm, "timer":M.timer}
										R.faults += [fault]
										fault = {"loc":(i, j, k), "value_before": M[i,k], "value_after" : xm, "bit" :bit_flipped, "origin":"self", "without_fault" : no_fault, "with_fault":xn*xm, "timer":M.timer}
										M.faults += [fault]
										callback_self = self.vulnerability_paramaters("fault_callback", None)
										if self_callback:
											fault_callback_m(fault)

										fault_occured = True
										break
								k += oscillation_k * increment_k
								oscillation_k *= -1
								increment_k += 1
							if fault_occured:
								break
							j += oscillation_j * increment_j
							oscillation_j *= -1
							increment_j += 1
						if fault_occured:
							break
						i += oscillation_i * increment_i
						oscillation_i *= -1
						increment_i += 1
					return product
						
					
			

	def __sub__(self, other):
		return Matrix(self._data - other._data, dtype=self.dtype)
	def __getitem__(self, key):
		if type(key[0]) is int:
			if type(key[1]) is int:
				return self._data[key]
			else:
				return Matrix(self._data[key], dtype=self.dtype)
		else:
			if type(key[1]) is int:
				return Matrix(self._data[key[0], key[1]:key[1]+1], dtype=self.dtype)
			else:
				return Matrix(self._data[key], dtype=self.dtype)
	def __setitem__(self, key, item): #bad
		if type(key[0]) is int:
			if type(key[1]) is int:
				if np.isscalar(item):
					self._data[key] = item
				else:
					self._data[key] = item[0, 0]
			else:
				self._data[key] = item._data
		else:
			if type(key[1]) is int:
				self._data[key[0], key[1]:key[1]+1] = item._data
			else:
				self._data[key] = item._data
	
	def __str__(self):
		return str(self._data)

	def qr(self, qrMethod = np.linalg.qr):
		Q, R = qrMethod(self._data)
		return Matrix(Q, dtype=self.dtype), Matrix(R, dtype=self.dtype)

	def reshape(self, i, j):
		return Matrix(self._data.reshape(i, j), dtype=self.dtype)

	def lstsq(self, x):
		y, a, b, c = np.linalg.lstsq(self._data, x._data)
		return Matrix(y, dtype=self.dtype), a, b, c	
	def is_vulnerable(self)
                self_vulnerable = self.vulnerable
                self_timer = ((self.vulnerability_parameters.get("timer", None) == None and np.random.rand() < self.vulnerability_parameters.get("p", 0)) or
                              (self.timer == self.vulnerability_parameters.get("timer", None))) 
                self_fault_count = len(self.faults) < self.vulnerability_parameters.get("max_fault_count", 1)
                        
                return self_vulnerable and self_timer and self_fault_count


class Zeros(Matrix):
	def __init__(self, shape, dtype='d'):
		Matrix.__init__(self, np.zeros(shape, dtype=dtype))
class Ones(Matrix):
	def __init__(self, shape, dtype='d', immune=True, immunity_parameters=Parameters()):
		Matrix.__init__(self, np.ones(shape, dtype=dtype))


class Matrix_(Matrix):
	" Matrix vulnerable to transient faults "
	def __init__(self, numpy_matrix, vulnerable = False, vulnerability_parameters = Parameters(), info = {}, dtype='d'):
		Matrix.__init__(self, numpy_matrix, info, dtype=dtype)
		self._p = vulnerability_parameters
		self.vulnerable = vulnerable 
		self.faults = []
                self.timer = 0

	def set_parameters(self, p):
		self._p = p
        def inherit_parameters(self, mother):
		if (mother._p["inherit"]):
			self.set_parameters(mother._p)

		if mother._p["inherited_parameters"]:
			self.set_parameters(mother._p["inherited_parameters"])

	def __getitem__(self, key):
		if type(key[0]) is int:
			if type(key[1]) is int:
				return self._data[key]
		else:
			if type(key[1]) is int:
				return Matrix_(self._data[key[0], key[1]:key[1]+1], self.vulnerable, self._p, dtype=self.dtype)
			else:
				return Matrix_(self._data[key], self.vulnerable, self._p, dtype=self.dtype)

        def __abs__(self):
                return Matrix_(abs(self._data), dtype=self.dtype)
        
	def __mul__(self, other):
		if np.isscalar(other):
			M = Matrix_(self._data * other, dtype=self.dtype)
                        M.inherit_parameters(self)
                        return M
		else:
			#####################################################
			## Temporary version

                        if (type(self.timer) is int) and self.vulnerable:
                                self.timer = self.timer + 1
                        if isinstance(other, Matrix_) and type(other.timer) is int and other.vulnerable:
                                other.timer = other.timer + 1

                        vulnerable_ok = self.vulnerable
                        timer_ok = (self._p["timer"] == None or 
                                    (self.timer == self._p["timer"])) 
                        fault_count_ok = len(self.faults) < self._p.get("max_fault_count", 1)
                        
                        perform_safe_multiplication = (not vulnerable_ok or 
                                                       not timer_ok or
                                                       not fault_count_ok)

                        if (isinstance(other, Matrix_)):
                                vulnerable_ok = other.vulnerable
                                timer_ok = (other._p["timer"] == None or 
                                            (other.timer == other._p["timer"])) 

                                fault_count_ok = len(other.faults) < other._p.get("max_fault_count", 1)
                                
                                perform_safe_multiplication &= (not vulnerable_ok or
                                                                not timer_ok or
                                                                not fault_count_ok)


                        if perform_safe_multiplication:
                                return Matrix_(np.dot(self._data, other._data), dtype=self.dtype)
			if self._p["fault_type"] == 0:
				i_fault, j_fault, t_fault = self._get_fault_info()
				M = Matrix_(np.dot(self._data, other._data), dtype=self.dtype)		
                                M.inherit_parameters(self)
                                return M

			if self._p["fault_type"] == 1: # To be defined
				if self._p["p"]:
					return self.mul_type1(other, self._p["p"])
				else:
					return self.mul_type1(other)
			if (self._p["fault_type"] == None):
				M = Matrix_(np.dot(self._data, other._data), dtype=self.dtype)
                                M.inherit_parameters(self)
                                return M
			####################################################
			## True version (outdated :(  check parameters)
			if self.vulnerable:
				if "fault_type" in self._p:
					if self._p["fault_type"] == 0: # To be defined
						i_fault, j_fault, t_fault = self._get_fault_info()
						return Matrix_(np.dot(self._data, other._data), dtype=self.dtype)		
					if self._p["fault_type"] == 1: # To be defined
						if "p" in self._p:
							return self.mul_type1(other, self._p["p"])
						else:
							return self.mul_type1(other)
						
				return Matrix_(np.dot(self._data, other._data), dtype=self.dtype)		
			else:
				return Matrix_(np.dot(self._data, other._data), dtype=self.dtype)	

	def mul_type1(self, other, p = 1.e-6):
		# Type 1 vulnerable multiplication : Fault (bitflip) may occur on each operation, with probability p
		# Slow ! Full python implementation
		# M x N
		M = self
		if isinstance(other, Matrix_):
			N = other
		elif isinstance(other, Matrix):
			N = Matrix_(other._data, dtype=self.dtype)
		else :
			N = Matrix_(other, dtype=self.dtype)
		(mi, mj) = M.shape
		(ni, nj) = N.shape
		# if mj != ni ... todo

		max_fault_count_m = M._p.get("max_fault_count", mi*mj)
		max_fault_count_n = N._p.get("max_fault_count", ni*nj)
		
		bits_m = M._p.get("vulnerable_bits", [i for i in xrange(64)]) # TODO: check if double precision
		bits_n = N._p.get("vulnerable_bits", [i for i in xrange(64)])

		fault_indices_m = M._p.get("fault_indices", [])
		fault_indices_n = N._p.get("fault_indices", [])
		
		random_m = M._p.get("random", False)
		random_n = N._p.get("random", False)

		p_m = M._p.get("p", p)
		p_n = N._p.get("p", p)

		fault_callback_m = M._p["fault_callback"]
		fault_callback_n = N._p["fault_callback"]

		R = Matrix_(np.zeros((mi, nj)), dtype=self.dtype)


		# TODO: Priority system for parameters inheritance (Here priority(M) > priority(N))
		if (N._p["inherit"]):
			R.set_parameters(N._p)
		if (M._p["inherit"]):
			R.set_parameters(M._p)

		if N._p["inherited_parameters"]:
			R.set_parameters(N._p["inherited_parameters"])
		if M._p["inherited_parameters"]:
			R.set_parameters(M._p["inherited_parameters"])


		for i in xrange(mi):
			for j in xrange(nj):
				v = 0.
				for k in xrange(mj):
					xm = M[i, k]
					xn = N[k, j]
					if (xm != 0. and xn != 0.):
                                                no_fault = xm * xn
						### TODO: change to make it more generic
						if (M.vulnerable and 
                                                    max_fault_count_m > len(M.faults) and
                                                    ((random_m and np.random.rand() < p_m) or
                                                     ((i,j,k) in fault_indices_m)) and
                                                    (M._p["timer"] == M.timer or M._p["timer"] == None)):
							bit_flipped = bits_m[int(np.floor(np.random.rand() * len(bits_m)))]
							xm = U.bitflip(xm, bit_flipped, type=self.dtype)

							fault = {"loc":(i, j), "k": k, "value_before": M[i,k], "value_after" : xm, "bit" :bit_flipped, "origin":"left", "without_fault" : no_fault, "with_fault":xn*xm, "timer":M.timer}
							R.faults += [fault]
							fault = {"loc":(i, j), "k": k, "value_before": M[i,k], "value_after" : xm, "bit" :bit_flipped, "origin":"self", "without_fault" : no_fault, "with_fault":xn*xm, "timer":M.timer}
							M.faults += [fault]
							if fault_callback_m:
								fault_callback_m(fault)

						if (N.vulnerable and 
                                                    max_fault_count_n > len(N.faults) and 
                                                    ((random_n and np.random.rand() < p_n) or
                                                     ((i,j,k) in fault_indices_n)) and
                                                    (N._p["timer"] == None or N.timer == N._p["timer"])):
							bit_flipped = bits_n[int(np.floor(np.random.rand() * len(bits_n)))]
							xn = U.bitflip(xn, bit_flipped, type=self.dtype)

							fault = {"loc":(i, j), "k": k, "value_before": N[k, j], "value_after" : xn, "bit" :bit_flipped, "origin":"right",  "without_fault" : no_fault, "with_fault":xn*xm, "timer":N.timer}
							R.faults += [fault]
							fault = {"loc":(i, j), "k": k, "value_before" : N[k,j], "value_after" : xn, "bit" :bit_flipped, "origin":"self",  "without_fault" : no_fault, "with_fault":xn*xm, "timer":N.timer}
							N.faults += [fault]
							if fault_callback_n:
								fault_callback_n(fault)
						v += xm * xn
				R[i, j] = v
		return R


        def mul_fault(self, other, fault):
                (mi, mj) = self.shape
                M = self
		if isinstance(other, Matrix_):
		        N = other
		elif isinstance(other, Matrix):
		        N = Matrix_(other._data, dtype=self.dtype)
		else :
			N = Matrix_(other, dtype=self.dtype)
		(mi, mj) = M.shape
		(ni, nj) = N.shape

                R = Matrix_(np.zeros((mi, nj)), dtype=self.dtype)

                # TODO: finish
                for i in xrange(mi):
                        for j in xrange(nj):
                                v = 0.
                                for k in xrange(mj):
                                        xm = M[i, k]
                                        xn = N[k, j]
                                        if (xm != 0. and xn != 0.):
                                                no_fault = xm * xn
                                                if (fault["loc"] == (i, j) and fault["k"] == k and fault["origin"] == "self"):
                                                        xm = U.bitflip(xm, fault["bit"], type=self.dtype)
                                                v += xm * xn
                                R[i, j] = v
                return R
        
	def _get_fault_info(self):
		i_fault, j_fault = 0 # fault indices
		if ("fault_location" not in self._p):
			i_fault = int(np.floor(np.random.rand() * self.shape[0]))
			j_fault = int(np.floor(np.random.rand() * self.shape[1]))
		else:				
			if (self._p["fault_location"][0] < 0 or
			    self._p["fault_location"][0] >= self.shape[0]): #random
				i_fault = int(np.floor(np.random.rand() * self.shape[0]))
			else:
				i_fault = self._p["fault_location"][0]
			if (self._p["fault_location"][1] < 0 or
			    self._p["fault_location"][1] >= self.shape[1]): #random
				j_fault = int(np.floor(np.random.rand() * self.shape[1]))
			else:
				j_fault = self._p["fault_location"][1]

		t_fault = 0 # time when the fault occurs (%) 0 = beginning, 100 = end (not effect)
		if "fault_time" not in self._p:
			t_fault = int(np.floor(np.random.rand() * self.shape[1]))
		else:
			if self._p["fault_time"] < 0 or \
			   self._p["fault_time"] > 100:
				t_fault = int(np.floor(np.random.rand() * self.shape[1]))
			else:
				t_fault = self._p["fault_time"]
		return i_fault, j_fault, t_fault	


	def norm(self, ord=None):
		return np.linalg.norm(self._data, ord=ord)
	def transpose(self):
		A = Matrix_(self._data.transpose(), dtype=self.dtype)
                A.inherit_parameters(self)
                return A

	def dot(self, other):
		s = 0.
		for a, b in zip(self._data, other._data):
			if not np.isscalar(a):
				a = a[0]
			if not np.isscalar(b):
				b = b[0]
			s += a * b			
		return	s

	def det(self):
		return np.linalg.det(self._data)
	def eig(self):
		return np.linalg.eig(self._data)
	
	def __add__(self, other):
		A = Matrix_(self._data + other._data, dtype=self.dtype)
                A.inherit_parameters(self)
                return A
	def __div__(self, other):
                A = Matrix_(self._data / other, dtype=self.dtype)
                A.inherit_parameters(self)
                return A
                
	def __sub__(self, other):
		A = Matrix_(self._data - other._data, dtype=self.dtype)
                A.inherit_parameters(self)
                return A

	def __getitem__(self, key):
		if type(key[0]) is int:
			if type(key[1]) is int:
				return self._data[key]
			else:
				A = Matrix_(self._data[key], dtype=self.dtype)
                                A.inherit_parameters(self)
                                return A
		else:
			if type(key[1]) is int:
				A = Matrix_(self._data[key[0], key[1]:key[1]+1], dtype=self.dtype)
                                A.inherit_parameters(self)
                                return A
			else:
				A = Matrix_(self._data[key], dtype=self.dtype)
                                A.inherit_parameters(self)
                                return A

	def __setitem__(self, key, item): #bad
		if type(key[0]) is int:
			if type(key[1]) is int:
				if np.isscalar(item):
					self._data[key] = item
				else:
					self._data[key] = item[0, 0]
			else:
				self._data[key] = item._data
		else:
			if type(key[1]) is int:
				self._data[key[0], key[1]:key[1]+1] = item._data
			else:
				self._data[key] = item._data
	
	def __str__(self):
		return str(self._data)

	def qr(self, qrMethod = np.linalg.qr):
		Q, R = qrMethod(self._data)
		Q_ = Matrix_(Q, dtype=self.dtype)
                R_ = Matrix_(R, dtype=self.dtype)
                Q_.inherit_parameters(self)
                R_.inherit_parameters(self)
                return Q_, R_

	def reshape(self, i, j):
		M = Matrix_(self._data.reshape(i, j), dtype=self.dtype)
                M.inherit_parameters(self)
                return M

	def lstsq(self, x):
		y, a, b, c = np.linalg.lstsq(self._data, x._data)
		Y = Matrix_(y, dtype=self.dtype)
                Y.inherit_parameters(self)
                return Y, a, b, c	


