import Utils as U
import numpy as np

from Parameters import *

class Fault():
    def __init__(self, parameters=Parameters()):
        self.parameters = parameters
        self.faults = []
        self.timer = 0
        
    def product(self, M, N):
        """
        Compute the product of two matrices
        A fault may occur according to the parameters
        """
        if np.isscalar(N):
            #todo
	    return M * N
	else:
            # Non-faulty product
            product =M.dot(N)

            fault = self.parameters.get("fault", None)
            if fault:
                loc = fault["loc"]
                i, j, k = loc["i"], loc["j"], loc["k"]
                bit_flipped = fault["bit"]
                timer = fault["timer"]
                register = fault["register"]
            else:
                fault_indices = self.parameters.get("fault_indices", None)
                if fault_indices != None and "i" in fault_indices:
                    i = fault_indices["i"]
                else:
                    i = int(M.shape[0] * np.random.rand())

                if fault_indices != None and "j" in fault_indices:
                    j = fault_indices["j"]
                else:
                    j = int(N.shape[1] * np.random.rand()) if len(N.shape) == 2 else 0

                if fault_indices != None and "k" in fault_indices:
                    k = fault_indices["k"]
                else:
                    k = int(M.shape[1] * np.random.rand()) if len(M.shape) == 2 else 0


                register = self.parameters.get("register", None)
                if (register == None):
                    register = np.random.choice(["left", "middle", "right"])

                bits = self.parameters.get("vulnerable_bits", [bit for bit in xrange(64)])
                bit_flipped = bits[int(np.floor(np.random.rand() * len(bits)))]
                timer = self.parameters.get("timer", None)


            callback = self.parameters.get("fault_callback", None)			    
            max_fault = self.parameters.get("max_fault", 1)

            safe_product_because_not_vulnerable = not (self.parameters.get("vulnerable", True))
            safe_product_because_max_fault = (max_fault <= len(self.faults))
            safe_product_because_timer = (np.random.rand() > self.parameters.get("p", 1./(M.shape[0]))) if (timer == None) else (self.timer != timer)
            perform_safe_product = (safe_product_because_max_fault or
                                    safe_product_because_timer or
                                    safe_product_because_not_vulnerable)

            if perform_safe_product:
                self.timer += 1
                return product
				        
	    fault = {}
	    fault["register"] = register
            fault["bit"] = bit_flipped
            fault["timer"] = self.timer              

            # looking for the closest non-zero couple (M[i, k], N[k, j])
            n = M.shape[0]
            m = N.shape[0]
            if len(N.shape) == 2:
                l = N.shape[1]
            else:
                l = 1


	    increment_i = 1
	    oscillation_i = 1			
	    fault_occured = False
	    for s in xrange(2 * max(n-i, i)):
		#next i index
                if len(self.faults) >= max_fault:
		    break
                if (i >= 0 and i < n):
		    increment_j = 1
		    oscillation_j = 1
		    for t in xrange(2 * max(l-j, j)):
		        #next j index
                        if len(self.faults) >= max_fault:
			    break
                        if (j >= 0 and j < n):
		            increment_k = 1
		            oscillation_k = 1
		            for u in xrange(2 * max(m-k, k)):
			        #next k index
			        if (k >= 0 and k < m):
			            if (M[i, k] != 0. and 
                                        ((len(N.shape) == 2 and N[k, j] != 0.) or
                                        (len(N.shape) == 1 and N[k] != 0.))):
                                        
                                        fault["loc"] = {"i":i, "j":j, "k": k}
                                        
				        xm_before = M[i, k]
				        xn_before = N[k, j] if len(N.shape) == 2 else N[k]
				        xp_before = xm_before * xn_before
                                        
                                        fault["value_before"] = product[i, j] if len(product.shape) == 2 else product[i]
                                        if (register == "left"):    
                                            xm_after = U.bitflip(xm_before, bit_flipped, type=M.dtype.type)
                                            xn_after = xn_before
                                            fault["register_before"] = xm_before
                                            fault["register_after"] = xm_after
                                            
                                            xp_after = xm_after * xn_after
                                    
                                        if (register == "right"):
                                            xm_after = xm_before
                                            xn_after = U.bitflip(xn_before, bit_flipped, type=M.dtype.type)
                                            fault["register_before"] = xn_before
                                            fault["register_after"] = xn_after

                                            xp_after = xm_after * xn_after
                                    
                                        if (register == "middle"):
                                            xm_after = xm_before
                                            xn_after = xn_before
                                            xp_after = U.bitflip(xp_before, bit_flipped, type=M.dtype.type)
                                            fault["register_before"] = xp_before
                                            fault["register_after"] = xp_after
                                            
                                        self.faults += [fault]

                                        difference = xp_after - xp_before

                                        if len(product.shape) == 2:
                                            product[i, j] += difference
                                        if len(product.shape) == 1:
                                            product[i] += difference
                                        
                                        fault["value_after"] = product[i, j] if len(product.shape) == 2 else product[i]
                                        
                                        if callback:
                                            callback(fault)
                                        break
				k += oscillation_k * increment_k
				oscillation_k *= -1
				increment_k += 1
			j += oscillation_j * increment_j
			oscillation_j *= -1
			increment_j += 1
		i += oscillation_i * increment_i
                oscillation_i *= -1
		increment_i += 1
            self.timer += 1
	    return product
