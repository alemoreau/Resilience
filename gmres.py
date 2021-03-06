from Parameters import *
from Fault import *
import numpy as np
from scipy.sparse.linalg import spilu, spsolve, inv
from scipy.sparse import tril
from scipy.linalg import solve_triangular

def implementation(self, input, algorithm_parameters, experiment_parameters, save_data=True, display=False):
    # todo: check parameters
    A = input["A"]
    n = A.shape[0]
    
    b = input["b"]
    x0 = input["x0"]

    parameters = algorithm_parameters.copy()
    parameters.update(experiment_parameters)

    m = parameters.get("m", n)
    tol = parameters.get("tol", 1.e-12)
    max_iter = parameters.get("iterMax", m)
    dtype = parameters.get("type", 'd')
    vulnerable = parameters.get("vulnerable", True)
    orthogonalization = parameters.get("orthMethod", classical_gramschmidt)
    save_data = parameters.get("save_data", None)
    full = parameters.get("full", False)
    faulty = Fault(Parameters(parameters.get("fault_parameters", {"max_fault_count":1})))
 
    
    normb = np.linalg.norm(b, ord=2)
 
    x = np.zeros((n, 1), dtype=dtype)
    x[:, 0] = x0[:, 0]
    
    r = b - (np.dot(A, x))
    beta = np.linalg.norm(r)

    sum_A = np.dot(np.transpose(A), np.ones(n))

    if (normb == 0.0):
        normb = 1.
  
    resid = np.linalg.norm(r) / normb
    if save_data:
	if "iteration_count" in save_data:
	    self.data["iteration_count"] = 0
	if "residual" in save_data:
	    self.data["residual"] = resid
	if "residuals" in save_data:
	    self.data["residuals"] = [resid]
	if "true_residual" in save_data:
	    self.data["true_residual"] = resid
	if "true_residuals" in save_data:
            self.data["true_residuals"] = [resid]
	if "H_rank" in save_data:
	    self.data["H_rank"] = [0]
	if "orthogonality" in save_data:
	    self.data["orthogonality"] = [0.]
	if "arnoldi" in save_data:
	    self.data["arnoldi"] = [0.]
	if "y" in save_data:
	    self.data["y"] = []
	if "checksum" in save_data:
	    self.data["checksum"] = []
	if "criteria" in save_data:
	    self.data["criteria"] = []
	if "threshold" in save_data:
	    self.data["threshold"] = []
	if "delta" in save_data:
	    self.data["delta"] = [0.]
	if "breakdown" in save_data:
	    self.data["breakdown"] = False
    if (resid <= tol):
        return x

    # V : Krylov basis
    V = np.zeros((n, m+1), dtype=dtype)
    # H : upper Hessenberg
    H = np.zeros((m+1, m), dtype=dtype)

    j = 0
    while (j < max_iter):
        V[:, 0:1] = r * (1.0 / beta)
        
        s = np.zeros((m+1, 1), dtype=dtype)
        s[0, 0] = beta
        
        cs, sn = [], []
        
        i = 0
        while (i < m and j < max_iter):
            
            if vulnerable:
                w = faulty.product(A, V[:, i])
            else:
		w = np.dot(A, V[:, i])
            
	    if "checksum" in save_data:
	    	checksum_Av = np.dot(w, np.ones((n, 1)))
	    	checksum_A  = np.dot(sum_A, V[:, i])
	    	checksum = abs(checksum_Av - checksum_A)[0]
		

            orthogonalization(w, V, H, i)
  
            # Happy breakdown
            if (H[i+1, i] < 1.e-20):
		if not full:
		    try:
			if i > 0:
		            y = solve_triangular(H[:i, :i], s[:i])
			else:
			    return x
		    except:
			if "breakdown" in save_data:
			    self.data["breakdown"] = True
			vulnerable = False
			return x

                    return x + np.dot(V[:, :i], y) # a verifier
        
            V[:, i+1] = w * (1.0 / H[i+1, i])
            
  
            # Previous plane rotations
            for k, (cs_k, sn_k) in enumerate(zip(cs, sn)):
                ApplyGivens(H, k, i, cs_k, sn_k)
        
            # Current plane rotation
            mu = np.sqrt(H[i, i]**2 + H[i+1, i]**2)
            cs_i = H[i, i] / mu
            sn_i = -H[i+1, i] / mu
            cs.append(cs_i)
            sn.append(sn_i)        
            
            # rotation on H
            H[i  , i] = cs_i * H[i, i] - sn_i * H[i+1, i]
            H[i+1, i] = 0.
            # rotation on right hand side
            ApplyGivens(s, i, 0, cs_i, sn_i)
            
            resid = abs(s[i+1, 0]) / normb
            
            if save_data:
		if "iteration_count" in save_data:
		    self.data["iteration_count"] = j+1
		if "residual" in save_data:
		    self.data["residual"] = resid
		if "residuals" in save_data:
		    self.data["residuals"] += [resid]
		if ("true_residual" in save_data or 
		    "true_residuals" in save_data or
		    "y" in save_data):
			try:
		            y = solve_triangular(H[:i+1, :i+1], s[:i+1])
		        except:
			    if "breakdown" in save_data:
			        self.data["breakdown"] = True
			    vulnerable = False
			    return x			
			#y = Solve_y(i, H, s)
                	#xc = x + Update(i, H, s, V)
			xc = x + np.dot(V[:, :i+1], y)
                	true_resid = np.linalg.norm(b - (np.dot(A, xc)))
                	true_resid_ = true_resid / normb
			if "y" in save_data:
			    self.data["y"] += [y]
			if "true_residual" in save_data:
                	    self.data["true_residual"] = true_resid_
			if "true_residuals" in save_data:
                	    self.data["true_residuals"] += [true_resid_]
		if "faults" in save_data:
                    self.data["faults"] = faulty.faults
                if "H" in save_data:
		    self.data["H"] = H
                if "V" in save_data:
		    self.data["V"] = V
		if "H_rank" in save_data:
		    self.data["H_rank"] += [np.linalg.matrix_rank(H[:i+1, :i+1])-(i+1)]
		if "orthogonality" in save_data:
		    self.data["orthogonality"] += [np.linalg.norm(np.dot(V[:, :i+2].T, V[:,:i+2]) - np.eye(i+2),ord='fro')/np.linalg.norm(np.eye(i+2))]
		if "arnoldi" in save_data:
		    self.data["arnoldi"] += [np.linalg.norm(np.dot(A, V[:,:i+1]) - np.dot(V[:,:i+2], H[:i+2, :i+1]),ord='fro') / np.linalg.norm(np.dot(V[:,:i+2], H[:i+2, :i+1]), ord='fro')]
		if "delta" in save_data:
		    if len(self.data["faults"]) > 0:
			Ekvk = abs(self.data["faults"][0]["value_after"] - self.data["faults"][0]["value_before"])
			k = self.data["faults"][0]["timer"]
			self.data["delta"] += [(abs(y[k]) * Ekvk) / normb]
		    else:
			self.data["delta"] += [0]
                if "checksum" in save_data: 
		    self.data["checksum"] += [checksum]
		if "threshold" in save_data:
		    self.data["threshold"] += [(tol * normb)/abs(y[i])]
		if "criteria" in save_data: #TODO: bug if y not in save_data
		    self.data["criteria"] += [(tol * normb)/abs(y[i])]
                if (true_resid_ < tol):
		    if not full:
                    	return xc
            if (not save_data or not "true_residual" in save_data or not "true_residual" in save_data):
                if resid < tol:
		    if not full:
                    	return x + Update(i, H, s, V)
            #if (resid < tol):
            #    if save_data:
            #        return xc
            #    else:
            #        return x + Update(i, H, s, V)    

            i += 1
            j += 1
            
        xc = x + Update(i - 1, H, s, V)
        r = b - np.dot(A, xc)
        beta = np.linalg.norm(r)
        resid = beta / normb
        #if save_data:
        #    self.data["iteration_count"] = j
        #    self.data["residual"] = resid
        #    self.data["residuals"] += [self.data["residual"]]
        #    self.data["true_residual"] = resid
        #    self.data["true_residuals"] += [self.data["true_residual"]]        
        if (resid < tol):
            return xc

    tol = resid
    return x





def precond(self, input, algorithm_parameters, experiment_parameters, save_data=True, display=False):
    # todo: check parameters
    A = input["A"]
    n = A.shape[0]
    b = input["b"]
    x0 = input["x0"]

    parameters = algorithm_parameters.copy()
    parameters.update(experiment_parameters)
    m = parameters.get("m", n)
    tol = parameters.get("tol", 1.e-12)
    max_iter = parameters.get("iterMax", m)
    dtype = parameters.get("type", 'd')
    vulnerable = parameters.get("vulnerable", True)
    orthogonalization = parameters.get("orthMethod", classical_gramschmidt)
    save_data = parameters.get("save_data", None)
    full = parameters.get("full", False)
    faulty = Fault(Parameters(parameters.get("fault_parameters", {"max_fault_count":1})))
 

    normb = np.linalg.norm(b, ord=2)
    
    x = np.zeros((n, 1), dtype=dtype)
    x[:, 0] = x0[:, 0]
    
    r = b - A.dot(x)
    beta = np.linalg.norm(r)

   
    M = spilu(A)
    
    sum_A = A.sum(axis=0)

    if (normb == 0.0):
        normb = 1.
    
    resid = np.linalg.norm(r) / normb
    if save_data:
	if "iteration_count" in save_data:
	    self.data["iteration_count"] = 0
	if "residual" in save_data:
	    self.data["residual"] = resid
	if "residuals" in save_data:
	    self.data["residuals"] = [resid]
	if "true_residual" in save_data:
	    self.data["true_residual"] = resid
	if "true_residuals" in save_data:
            self.data["true_residuals"] = [resid]
	if "H_rank" in save_data:
	    self.data["H_rank"] = [0]
	if "orthogonality" in save_data:
	    self.data["orthogonality"] = [0.]
	if "arnoldi" in save_data:
	    self.data["arnoldi"] = [0.]
	if "y" in save_data:
	    self.data["y"] = []
	if "checksum" in save_data:
	    self.data["checksum"] = []
	if "criteria" in save_data:
	    self.data["criteria"] = []
	if "threshold" in save_data:
	    self.data["threshold"] = []
	if "delta" in save_data:
	    self.data["delta"] = [0.]
	if "breakdown" in save_data:
	    self.data["breakdown"] = False
    if (resid <= tol):
        return x

    # V : Krylov basis
    V = np.zeros((n, m+1), dtype=dtype)
    # H : upper Hessenberg
    H = np.zeros((m+1, m), dtype=dtype)

    j = 0
    while (j < max_iter):
        V[:, 0:1] = r * (1.0 / beta)
        
        s = np.zeros((m+1, 1), dtype=dtype)
        s[0, 0] = beta
        
        cs, sn = [], []
        
        i = 0
        while (i < m and j < max_iter):
            
	    z  = M.solve(V[:, i])

            if vulnerable:
                w = faulty.product(A, z)
            else:
		w = A.dot(z)
            
	    if "checksum" in save_data:
	    	checksum_Av = w.sum(axis = 0)
	    	checksum_A  = np.dot(sum_A, z) if np.isscalar(np.dot(sum_A, z)) else np.dot(sum_A, z)[0, 0]
	    	checksum = abs(checksum_Av - checksum_A)
		

            orthogonalization(w, V, H, i)
  
            # Happy breakdown
            if (H[i+1, i] < 1.e-20):
		try:
		    if i > 0:
                        y = solve_triangular(H[:i, :i], s[:i])
		        xc = x + M.solve(np.dot(V[:, :i], y))
			return xc
		    else:
			return x
		except:
		    if "breakdown" in save_data:
			self.data["breakdown"] = True
		    if i > 0:
                        y = solve_triangular(H[:i, :i], s[:i])
		        xc = x + M.solve(np.dot(V[:, :i], y))
			return xc
		    else:
			return x
        
            V[:, i+1] = w * (1.0 / H[i+1, i])
            
  
            # Previous plane rotations
            for k, (cs_k, sn_k) in enumerate(zip(cs, sn)):
                ApplyGivens(H, k, i, cs_k, sn_k)
        
            # Current plane rotation
            mu = np.sqrt(H[i, i]**2 + H[i+1, i]**2)
	    try:
                cs_i = H[i, i] / mu
                sn_i = -H[i+1, i] / mu
                cs.append(cs_i)
                sn.append(sn_i) 
            except:
		if "breakdown" in save_data:
			self.data["breakdown"] = True
		if i > 0:
		    return x + xc # a verifier
		else:
		    return x
            
            # rotation on H
            H[i  , i] = cs_i * H[i, i] - sn_i * H[i+1, i]
            H[i+1, i] = 0.
            # rotation on right hand side
            ApplyGivens(s, i, 0, cs_i, sn_i)
            
            resid = abs(s[i+1, 0]) / normb
            
            if save_data:
		if "iteration_count" in save_data:
		    self.data["iteration_count"] = j+1
		if "residual" in save_data:
		    self.data["residual"] = resid
		if "residuals" in save_data:
		    self.data["residuals"] += [resid]
		if ("true_residual" in save_data or 
		    "true_residuals" in save_data or
		    "y" in save_data):
			try:
		            y = solve_triangular(H[:i+1, :i+1], s[:i+1])
			    xc = x + M.solve(np.dot(V[:, :i+1], y))
		        except:
			    if "breakdown" in save_data:
			        self.data["breakdown"] = True
			    vulnerable = False
			    return
			#y = Solve_y(i, H, s)
			#y = solve_triangular(H[:i+1, :i+1], s[:i+1])
                	#xc = x + Update(i, H, s, V)
			#xc = x + solve_triangular(A, np.dot(V[:, :i+1], y), lower=True, check_finite=True)
			#xc = x + np.dot(V[:, :i+1], y)
			xc  = x + M.solve(np.dot(V[:, :i+1], y))
			# M xc = V[:,:i+1] * y
			#xc  = x + spsolve(M, np.dot(V[:, :i+1], y))
                	true_resid = np.linalg.norm(b - A.dot(xc))
                	true_resid_ = true_resid / normb
			if "y" in save_data:
			    self.data["y"] += [y]
			if "true_residual" in save_data:
                	    self.data["true_residual"] = true_resid_
			if "true_residuals" in save_data:
                	    self.data["true_residuals"] += [true_resid_]
		if "faults" in save_data:
                    self.data["faults"] = faulty.faults
                if "H" in save_data:
		    self.data["H"] = H
                if "V" in save_data:
		    self.data["V"] = V
		if "H_rank" in save_data:
		    self.data["H_rank"] += [np.linalg.matrix_rank(H[:i+1, :i+1])-(i+1)]
		if "orthogonality" in save_data:
		    self.data["orthogonality"] += [np.linalg.norm(np.dot(V[:, :i+2].T, V[:,:i+2]) - np.eye(i+2),ord='fro')/np.linalg.norm(np.eye(i+2))]
		if "arnoldi" in save_data:
		    self.data["arnoldi"] += [np.linalg.norm(np.dot(A, V[:,:i+1]) - np.dot(V[:,:i+2], H[:i+2, :i+1]),ord='fro') / np.linalg.norm(np.dot(V[:,:i+2], H[:i+2, :i+1]), ord='fro')]
		if "delta" in save_data:
		    if len(self.data["faults"]) > 0:
			Ekvk = abs(self.data["faults"][0]["value_after"] - self.data["faults"][0]["value_before"])
			k = self.data["faults"][0]["timer"]
			self.data["delta"] += [(abs(y[k]) * Ekvk) / normb]
		    else:
			self.data["delta"] += [0]
                if "checksum" in save_data: 
		    self.data["checksum"] += [checksum]
		if "threshold" in save_data:
		    self.data["threshold"] += [(tol * normb)/abs(y[i])]
		if "criteria" in save_data: #TODO: bug if y not in save_data
		    self.data["criteria"] += [(tol * normb)/abs(y[i])]
                if (true_resid_ < tol):
		    if not full:
                    	return xc
            if (not save_data or not "true_residual" in save_data or not "true_residual" in save_data):
                if resid < tol:
		    if not full:
                    	return x + Update(i, H, s, V)
            #if (resid < tol):
            #    if save_data:
            #        return xc
            #    else:
            #        return x + Update(i, H, s, V)    

            i += 1
            j += 1
            
        xc = x + Update(i - 1, H, s, V)
        r = b - A.dot(xc)
        beta = np.linalg.norm(r)
        resid = beta / normb
        #if save_data:
        #    self.data["iteration_count"] = j
        #    self.data["residual"] = resid
        #    self.data["residuals"] += [self.data["residual"]]
        #    self.data["true_residual"] = resid
        #    self.data["true_residuals"] += [self.data["true_residual"]]        
        if (resid < tol):
            return xc

    tol = resid
    return x


def pipelined_p1(self, input, algorithm_parameters, experiment_parameters, display=False):

    A = input["A"]
    n = A.shape[0]
    
    b = input["b"]
    x0 = input["x0"]
    
    parameters = algorithm_parameters.copy()
    parameters.update(experiment_parameters)

    m = parameters.get("m", n)
    tol = parameters.get("tol", 1.e-12)
    max_iter = parameters.get("iterMax", 10*m)
    dtype = parameters.get("type", 'd')
    vulnerable = parameters.get("vulnerable", True)
    orthogonalization = parameters.get("orthMethod", classical_gramschmidt)
    save_data = parameters.get("save_data", None)

    faulty = Fault(Parameters(parameters.get("fault_parameters", {"max_fault_count":1})))

 
    normb = np.linalg.norm(b, ord=2)
    
    x = np.zeros((n, 1), dtype=dtype)
    x[:, 0] = x0[:, 0]
    
    r = b - (np.dot(A, x))
    beta = np.linalg.norm(r)
  
    if (normb == 0.0):
        normb = 1.
  
    resid = np.linalg.norm(r) / normb
    if save_data:
	if "iteration_count" in save_data:
	    self.data["iteration_count"] = 0
	if "residual" in save_data:
	    self.data["residual"] = resid
	if "residuals" in save_data:
	    self.data["residuals"] = [resid]
	if "true_residual" in save_data:
	    self.data["true_residual"] = resid
	if "true_residuals" in save_data:
            self.data["true_residuals"] = [resid]
	if "H_rank" in save_data:
	    self.data["H_rank"] = [0]
	if "orthogonality" in save_data:
	    self.data["orthogonality"] = [0.]
	if "arnoldi" in save_data:
	    self.data["arnoldi"] = [0.]

    if (resid <= tol):
        return x

    # V : Krylov basis
    V = np.zeros((n, m+1 +1), dtype=dtype)
    # H : upper Hessenberg
    H = np.zeros((m+1+1+1, m+1+1), dtype=dtype)
    # Z 
    Z = np.zeros((n, m+1 +1 +1), dtype=dtype)
    
    iteration = 0
    while (iteration < max_iter and resid > tol):
        
	V[:, 0:1] = r * (1.0 / beta)
        Z[:, 0:1] = r * (1.0 / beta)

        for i in xrange(m+2):
            
	    if vulnerable:
                w = faulty.product(A, Z[:, i])
            else:
		w = np.dot(A, Z[:, i])

            if i > 1:
                
                h = H[i-1, i-2]
                if (h == 0.):
                    print "Happy breakdown : " + str(i)
		    break
                
                V[:, i-1]    /= h
                Z[:, i]      /= h
                w            /= h
                H[:i-1, i-1] /= h
                H[i-1 , i-1] /= h * h
            
            Z[:, i+1] = w
            for j in xrange(i):
                Z[:, i+1] -= Z[:, j+1] * H[j, i-1] 
            
            if i > 0:
                V[:, i] = Z[:, i]
                for j in xrange(i):
                    V[:, i] -= V[:, j] * H[j, i-1]
                H[i, i-1] = np.linalg.norm(V[:, i])

            #for j in xrange(i+1):
            #    H[j, i] = Z[:, i+1].dot(V[:, j])
            
            #t = Z[:, i+1]
            #for j in xrange(i+1):
            #    H[j, i] = np.dot(t, V[:, j])
            #    t -= V[:, j] * H[j, i]
	    orthogonalization(np.copy(Z[:, i+1]), V, H, i)
	    
	    
	    if i > 1:
		e1 = np.zeros((i+1, 1), 'd') 
		e1[0] = 1.0
		(Q, R) = np.linalg.qr(H[:i+1, :i])
		y, _, _, _ = np.linalg.lstsq(R, np.dot(np.transpose(Q), e1))
		y = y * beta			
		resid = np.linalg.norm(np.dot(H[:i+1, :i], y) - beta * e1) / normb
		xc = x + np.dot(V[:, :i], y)
                true_resid = np.linalg.norm(b - (np.dot(A, xc)))
                true_resid_ = true_resid / normb
                if save_data:
		    if "iteration_count" in save_data:
		        self.data["iteration_count"] = iteration+1
		    if "residual" in save_data:
		        self.data["residual"] = resid
		    if "residuals" in save_data:
		        self.data["residuals"] += [resid]
		    if "true_residual" in save_data or "true_residuals" in save_data:
		        if "true_residual" in save_data:
                	    self.data["true_residual"] = true_resid_
		        if "true_residuals" in save_data:
                	    self.data["true_residuals"] += [true_resid_]
		    if "faults" in save_data:
                        self.data["faults"] = faulty.faults
                    if "H" in save_data:
		        self.data["H"] = H
                    if "V" in save_data:
		        self.data["V"] = V
		    if "H_rank" in save_data:
		        self.data["H_rank"] += [np.linalg.matrix_rank(H[:i+1, :i+1])-(i+1)]
		    if "orthogonality" in save_data:
		        self.data["orthogonality"] += [np.linalg.norm(np.dot(V[:, :i].T, V[:, :i]) - np.eye(i),ord='fro')/np.linalg.norm(np.eye(i+1))]
		    if "arnoldi" in save_data:
		        self.data["arnoldi"] += [np.linalg.norm(np.dot(A, Z[:,:i]) - np.dot(Z[:,:i+1], H[:i+1, :i]),ord='fro') / np.linalg.norm(A, ord='fro')]

               

                    if (save_data and "true_residual" in save_data and true_resid_ < tol):
                        return xc
                if (not save_data or not "true_residual" in save_data or not "true_residual" in save_data):
                    if resid < tol:
                        return xc
   
            iteration += 1
                   
	e1 = np.zeros((m+2, 1), 'd') 
	e1[0] = 1.0
	(Q, R) = np.linalg.qr(H[:m+2, :m+1])
	y, _, _, _ = np.linalg.lstsq(R, np.dot(Q.T, e1))
	y = y * beta			
	resid = np.linalg.norm(np.dot(H[:m+2, :m+1], y) - beta * e1) / normb
	xc = x + np.dot(V[:, :m+1], y)
        true_resid = np.linalg.norm(b - (np.dot(A, xc)))
        true_resid_ = true_resid / normb

        if resid < tol:
            return xc

    return x



def classical_gramschmidt(w, V, H, i, reorth=False, alpha=0.001):    
    for k in xrange(i+1):
        H[k, i] = w.dot(V[:, k])
        
    for k in xrange(i+1):
        w -= V[:, k] * H[k, i]
    if reorth:
	norm_before = np.linalg.norm(V[:,i+1])
	if norm_before + alpha * H[i+1, i] == norm_before:
	    Hr = []
	    for k in xrange(i+1):
		Hr += [w.dot(V[:, k])]
    	    
	    for k in xrange(i+1):
                H[k, i] += Hr[k]
        
            for k in xrange(i+1):
                w -= V[:, k] * Hr[k]
	    
    H[i+1, i] = np.linalg.norm(w)

def modified_gramschmidt(w, V, H, i, reorth=False, alpha=0.001):
    for k in xrange(i+1):
        H[k, i] = w.dot(V[:, k])
        w -= V[:, k] * H[k, i]
    if reorth:
	norm_before = np.linalg.norm(V[:,i+1])
	if norm_before + alpha * H[i+1, i] == norm_before:
    		for k in xrange(i+1):
			Hr = w.dot(V[:, k])
			H[k, i] += Hr
			w -= V[:, k] * Hr
    H[i+1, i] = np.linalg.norm(w)

def Update(k, H, s, V):
    y = Solve_y(k, H, s)
    return np.dot(V[:, :k+1], y)

def Solve_y(k, H, s):
    y = np.zeros((k+1, 1), dtype=H.dtype)
    y[:, 0] = s[:k+1, 0]
        
    for i in xrange(k, -1, -1):
        if y[i, 0] != 0.:
            y[i, 0] /= H[i, i];
            for j in xrange(i-1, -1, -1):
                y[j, 0] -= H[j,i] * y[i, 0]
    return y

def Solve(H, V, x, beta, m):
    
    s = np.zeros((m+1, 1))
    s[0, 0] = beta

    y, _, _ ,_ = np.linalg.lstsq(H[:m+1, :m], s)
    return x + np.dot(V[:, :m], y[:m, :])



def ApplyGivens(H, k, i, c, s):
    tmp1 = c * H[k, i] - s * H[k+1, i]
    tmp2 = s * H[k, i] + c * H[k+1, i]
    H[k  , i] = tmp1
    H[k+1, i] = tmp2
