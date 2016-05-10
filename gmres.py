from Parameters import *
from Fault import *
import numpy as np

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
    max_iter = parameters.get("iterMax", 10*m)
    dtype = parameters.get("type", 'd')
    vulnerable = parameters.get("vulnerable", True)
    orthogonalization = parameters.get("orthMethod", classical_gramschmidt)
    faulty = Fault(Parameters(parameters.get("fault_parameters", {"max_fault_count":1})))
    
    # To remove
    normA = np.linalg.norm(A, ord=2)
    u, s, v = np.linalg.svd(A)
    s_min = s[-1]
    s_max = s[0]
    K_A = s_max / s_min
    # To remove
    
    normb = np.linalg.norm(b, ord=2)
    
    x = np.zeros((n, 1), dtype=dtype)
    x[:, 0] = x0[:, 0]
    
    r = b - (np.dot(A, x))
    beta = np.linalg.norm(r)
  
    if (normb == 0.0):
        normb = 1.
  
    resid = np.linalg.norm(r) / normb
    if save_data:
        self.data["residual"] = resid
        self.data["residuals"] = [resid]
        self.data["true_residual"] = resid
        self.data["true_residuals"] = [resid]
        self.data["iteration_count"] = 0
	self.data["orthogonality"] = [0.] 
	self.data["arnoldi"] = [0.]
        
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
            
            orthogonalization(w, V, H, i)
  
            # Happy breakdown
            if (H[i+1, i] == 0.):
                return x + Update(i - 1, H, s, V); # a verifier
        
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
                xc = x + Update(i, H, s, V)
                true_resid = np.linalg.norm(b - (np.dot(A, xc)))
                true_resid_ = true_resid / normb
                self.data["iteration_count"] = j+1
                self.data["residual"] = resid
                self.data["residuals"] += [resid]
                self.data["true_residual"] = true_resid_
                self.data["true_residuals"] += [true_resid_]
                self.data["faults"] = faulty.faults
                self.data["H"] = H
                self.data["V"] = V
		self.data["orthogonality"] += [np.linalg.norm(np.dot(V[:, :i+2].T, V[:,:i+2]) - np.eye(i+2),ord='fro')/np.linalg.norm(np.eye(i+2))]
		self.data["arnoldi"] += [np.linalg.norm(np.dot(A, V[:,:i+1]) - np.dot(V[:,:i+2], H[:i+2, :i+1]),ord='fro') / np.linalg.norm(np.dot(V[:,:i+2], H[:i+2, :i+1]), ord='fro')]

                if (faulty.faults and faulty.faults[-1]["register"] == "left" and 'check' not in faulty.faults[-1]):
                    Ej = abs(faulty.faults[-1]['register_before'] - faulty.faults[-1]['register_after'])
                    
                    gamma = normb
                    if (Ej < (s_min / (4 * n) * min(1., (3*gamma)/(2*true_resid)*tol))):
                        faulty.faults[-1]['check'] = True
                    else:
                        gamma = 1./(4. + tol * K_A) * normA * np.sqrt(n) + normb
                        if (Ej < (s_min / (4 * n) * min(1., (3*gamma)/(2*true_resid)*tol))):
                            faulty.faults[-1]['check'] = False
                        else:
                            faulty.faults[-1]['check'] = None
                
                if ( faulty.faults and faulty.faults[-1]["register"] == "right" and 'check' not in faulty.faults[-1]):
                    row = faulty.faults[-1]['loc']["i"]
                    error = faulty.faults[-1]['value_before'] - faulty.faults[-1]['value_after']
                    Ej = 0.
                    minimum = 0.
                    v_min = max(abs(V[:, i]))
                    
                    for k in xrange(n):
                        if (A[row, k] != 0. and v[k, 0] != 0. and abs(v[k, 0]) < v_min ):
                            v_min = abs(v[k, 0])
                            
                    
                    Ej = abs(error / v_min)
                    
                    gamma = normb
                                    
                    if (Ej < (s_min / (4 * n) * min(1., (3*gamma)/(2*true_resid)*tol))):
                        faulty.faults[0]['check'] = True
                    else:
                        gamma = 1./(4. + tol * K_A) * normA * np.sqrt(n) + normb
                        if (Ej < (s_min / (4 * n) * min(1., (3*gamma)/(2*true_resid)*tol))):
                            faulty.faults[0]['check'] = False
                        else:
                            faulty.faults[0]['check'] = None

                if (true_resid_ < tol):
                    return xc
            if (not save_data):
                if resid < tol:
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

def classical_gramschmidt(w, V, H, i):    
    for k in xrange(i+1):
        H[k, i] = w.dot(V[:, k])
        
    for k in xrange(i+1):
        w -= V[:, k] * H[k, i]
        
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
    y = np.zeros((k+1, 1), dtype=H.dtype)
    y[:, 0] = s[:k+1, 0]
        
    for i in xrange(k, -1, -1):
        if y[i, 0] != 0.:
            y[i, 0] /= H[i, i];
            for j in xrange(i-1, -1, -1):
                y[j, 0] -= H[j,i] * y[i, 0]
    
    return np.dot(V[:, :k+1], y)


def ApplyGivens(H, k, i, c, s):
    tmp1 = c * H[k, i] - s * H[k+1, i]
    tmp2 = s * H[k, i] + c * H[k+1, i]
    H[k  , i] = tmp1
    H[k+1, i] = tmp2
