import Utils as U
import numpy as np
from Parameters import *

class Fault():
    def __init__(self, parameters=Parameters()):
	if parameters:
            self.parameters = parameters
        else:
	    self.parameters = Parameters()
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
            product = M.dot(N)
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
                    register = np.random.choice([1, 2, 3])

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
	    fault["loc"] = {"i":i, "j":j, "k": k}
	    fault["value_before"] = product[i, j] if len(product.shape) == 2 else product[i]
	    xm_before = M[i, k] if register < 4 else M[:, k].toarray()
	    xn_before = N[k, j] if len(N.shape) == 2 else N[k]
	    xp_before = xm_before * xn_before
	    if (register == 1):    
                xm_after = U.bitflip(xm_before, bit_flipped, type=M.dtype.type)
                xn_after = xn_before
                fault["register_before"] = xm_before
                fault["register_after"] = xm_after
                xp_after = xm_after * xn_after
                                    
            if (register == 2 or register == 4):
                xm_after = xm_before
                xn_after = U.bitflip(xn_before, bit_flipped, type=M.dtype.type)
                fault["register_before"] = xn_before
                fault["register_after"] = xn_after
                xp_after = xm_after * xn_after

            if (register == 3):
                xm_after = xm_before
                xn_after = xn_before
                xp_after = U.bitflip(xp_before, bit_flipped, type=M.dtype.type)
                fault["register_before"] = xp_before
                fault["register_after"] = xp_after
                
            self.faults += [fault]
            if register < 4:
                difference = np.zeros(product.shape)
                difference[i] = xp_after-xp_before
            else:
                difference = xp_after - xp_before
            fault["difference"] = difference
            if len(product.shape) == 2: #TODO register4
                product[i, j] += difference
            if len(product.shape) == 1:
                product += difference.reshape(product.shape)

                                        
            fault["value_after"] = product[i, j] if len(product.shape) == 2 else product[i]
                                        
            if callback:
                callback(fault)

            self.timer += 1
	    return product
