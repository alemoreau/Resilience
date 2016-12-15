
# coding: utf-8

# # Criterion for convergence in backward error after fault


import numpy as np
from scipy.sparse.linalg import LinearOperator, norm
from scipy.linalg import solve_triangular
from sys import stdout
from scipy.sparse.linalg import spilu
import math

from Experiment import *
from Parameters import *
from Fault import *
from Algorithm import *
from Utils import *


from gmres import *
import gmres_display




def faulty_product(A, Faulty = None):
    if Faulty:
        return lambda v: Faulty.product(A, v)
    else:
        return lambda v: A.dot(v)



def implementation(self, input, algorithm_parameters, experiment_parameters):
    A = input["A"]
    b = input["b"]
    x0= input["x0"]

    parameters = algorithm_parameters.copy()
    parameters.update(experiment_parameters)
    
    n = A.shape[0]
    normb = np.linalg.norm(b)
    maxiter = parameters.get("maxiter", n)
    tol = parameters.get("tol", 1.e-10)
    restart = parameters.get("restart", None)
    m = restart if restart else maxiter
    precond = parameters.get("precond", None)
    Checksum = parameters.get("checksum", None)
    left = parameters.get("left", not parameters.get("right", False)) #TODO
    save_data = parameters.get("save_data", {})
    A1T = A.sum(axis=0)
    
    if "residuals" in save_data:
        self.data["residuals"] = [1.]
    if "true_residuals" in save_data:    
        self.data["true_residuals"] = [1.]
    if "checksum" in save_data:
        self.data["checksum"] = []
    if "true_delta" in save_data:
        self.data["true_delta"] = [0.]
    if "Ekvk" in save_data:
        self.data["Ekvk"] = 0.
    if "delta" in save_data:
        self.data["delta"] = [0.]
    if "y" in save_data:
        self.data["y"] = []
    if "w" in save_data:
        self.data["w"] = []

    if parameters.get("vulnerable", True):
        faulty = Fault(parameters.get("fault_parameters", Parameters()))
    else:
        faulty = None
    A_ = LinearOperator((n, n), 
                        matvec=faulty_product(A, faulty),
                        dtype=A.dtype)


    if precond:
        import scipy.sparse.linalg as spla
        P = precond(A)
        
        M_x = lambda x: P.solve(x)
        M = spla.LinearOperator((n, n), matvec=M_x, dtype=A.dtype)
        if left:
            normb = np.linalg.norm(M_x(b))

        if (not Checksum):
            if left:
                MA1T = M_x(A.A).sum(axis=0)
            else:
                if (checksum_mode):
                    AM1T = (A.dot(M_x(np.eye(n)))).sum(axis=0)
            
    else:
        M = None

    if (Checksum):
        checksum = Checksum(A, M, left)

    def callback(iter_num, resid, work, work2, ijob):
        if ijob != -1:
            iter_num = iter_num - 1
        R = work[0*n:1*n]
        S = work[1*n:2*n]
        W = work[2*n:3*n]
        Y = work[3*n:4*n]
        AV= work[4*n:5*n]
        V = work[5*n:(5+iter_num)*n]
        v = work[(5+iter_num) * n:(6 + iter_num)*n]
        H = work2[:iter_num*(m+1)]
        c = work2[m*(m+1):m*(m+2)]
        s = work2[m*(m+2):m*(m+3)]
        
        if faulty and "faults" in save_data:
            self.data["faults"] = faulty.faults
            
        if "checksum" in save_data:
            self.data["checksum"] += [checksum(AV, v, W)]
            # if precond:
            #     if left:
            #         self.data["checksum"] += [np.linalg.norm(np.sum(W) - np.dot(MA1T, v))]

            #     else:
            #         # old way : 
            #         # self.data["checksum"] += [np.linalg.norm(np.sum(W) - np.dot(A1T, AV))]
            #         # new way :
            #         if checksum_mode:
            #             self.data["checksum"] += [np.linalg.norm(np.sum(W) - np.dot(AM1T, v))]
            #         else:
            #             self.data["checksum"] += [np.linalg.norm(np.sum(W) - np.dot(A1T, AV))]
            
            # else:
            #     self.data["checksum"] += [np.linalg.norm(np.sum(W) - np.dot(A1T, v))]


        if iter_num == 0 and ijob == 2:
            return
        
        if ("H" in save_data or 
            "V" in save_data or
            "y" in save_data or 
            "delta" in save_data or
            "true_delta" in save_data or
            "true_residual" in save_data or
            "true_residuals" in save_data):
            H = H.reshape((iter_num, m+1))[:,:iter_num]
            V = V.reshape((iter_num, n))
            S = S[:iter_num]
            if ("y" in save_data or 
                "true_residuals" in save_data or 
                "true_delta" in save_data or 
                "delta" in save_data):
                try:
                    y = solve_triangular(H.T, S)
                except:
                    return True
                x_ = np.dot(V.T, y)
                if M:
                    if left:
                        x = x0 + x_
                        true_resid = np.linalg.norm(M.matvec(A * x - b)) / normb
                        resid = resid * np.linalg.norm(b) / normb
                    else:
                        x = x0 + M.matvec(x_)
                        true_resid = np.linalg.norm(A * x - b) / normb
                else:
                    x = x0 + x_
                    true_resid = np.linalg.norm(A * x - b) / normb

        if "faults" in save_data and faulty and len(faulty.faults) == 1 and faulty.timer == iter_num:
            self.data["faults_i"] = faulty.faults[0]["loc"]["i"] if isinstance(faulty.faults[0]["loc"]["i"],int) else np.asscalar(faulty.faults[0]["loc"]["i"])
            self.data["faults_k"] = faulty.faults[0]["loc"]["k"] if isinstance(faulty.faults[0]["loc"]["k"],int) else np.asscalar(faulty.faults[0]["loc"]["k"])
            self.data["faults_f"] = faulty.faults[0]["timer"]
            self.data["faults_b"] = faulty.faults[0]["bit"]
            self.data["faults_r"] = faulty.faults[0]["register"]
            
        if "Ekvk" in save_data:
            if faulty and len(faulty.faults) == 1 and self.data["Ekvk"] == 0.:
                error = faulty.faults[0]["difference"]
                if M:
                    if left:
                        self.data["Ekvk"] = np.linalg.norm(M.matvec(error))
                        #self.data["Ekvk"] = np.linalg.norm(error)
                    else:
                        self.data["Ekvk"] = np.linalg.norm(error)
                else:
                    self.data["Ekvk"] = np.linalg.norm(error)
        if "y" in save_data:
            self.data["y"] += [y]
        if "iteration_count" in save_data:
            self.data["iteration_count"] = iter_num 
        if "l" in save_data:
            self.data["l"] = iter_num 
    
        if "residual" in save_data:
            self.data["residual"] = resid
        if "residuals" in save_data:
            self.data["residuals"] += [resid]
        if "true_residual" in save_data:
            self.data["true_residual"] = true_resid
        if "true_residuals" in save_data:
            self.data["true_residuals"] += [true_resid]
        if "delta" in save_data:
            self.data["delta"] += [abs(y[-1] * self.data["Ekvk"]) /  normb] #TODO
        if "true_delta" in save_data:
            self.data["true_delta"] += [abs(resid - true_resid)] 

              
        if "H" in save_data:
            self.data["H"] = H.T
        if "V" in save_data:
            self.data["V"] = V.T
        if "givens" in save_data:
            self.data["givens_c"] = c[:iter_num]
            self.data["givens_s"] = s[:iter_num]


        
        if "w" in save_data:
            self.data["w"] += [W]
            
        if true_resid < tol:
            return True
        else:
            return False
            
    x = gmres(A_, 
              b, 
              x0, 
              tol = 0 if left else tol,
              maxiter = maxiter,
              restart = restart,
              callback = callback if parameters.get("save_data", None) else None,
              M = M,
              left = left
             )
    return x







# In[ ]:




# In[9]:


# ## Experiment initialization

def experiment_parameters_generator(vulnerable = True,
                                    tol = 1.e-10,
                                    bits = [20], 
                                    iterations = [17], 
                                    registers = [4], 
                                    maxiter = 100,
                                    restart = 100,
                                    indices = [{"i":0, "j":0, "k":0}],
                                    fault_callback = None
                                    #lambda fault: (stdout.write(" %s " % str(fault)))
                                   ):
    parameters = Parameters({})
    parameters["vulnerable"] = vulnerable
    parameters["tol"] = tol
    parameters["maxiter"] = maxiter
    parameters["restart"] = restart
    fault_parameters = {}
    fault_parameters["fault_callback"] = fault_callback
    fault_parameters["vulnerable"] = vulnerable
    while (True):

        parameters["fault_parameters"] = fault_parameters

        for bit in bits:
            for iteration in iterations:
                for reg in registers:
                    for indice in indices:
                        fault_parameters["vulnerable_bits"] = [bit]
                        fault_parameters["timer"] = iteration
                        fault_parameters["register"] = reg
                        fault_parameters["fault_indices"] = indice
                        yield parameters
def download(url):
    import urllib2
    file_name = url.split('/')[-1]
    u = urllib2.urlopen(url)
    f = open(file_name, 'wb')
    meta = u.info()
    file_size = int(meta.getheaders("Content-Length")[0])
    print "Downloading: %s Bytes: %s" % (file_name, file_size)
    
    file_size_dl = 0
    block_sz = 8192
    while True:
        buffer = u.read(block_sz)
        if not buffer:
            break
        
        file_size_dl += len(buffer)
        f.write(buffer)
        status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
        status = status + chr(8)*(len(status)+1)
        print status,
        
        f.close()
    return file_name


def update_data(Experiment, epsilon = 1.e-10, c = 0.5):
    
    Experiment.data = map(lambda d: dict(zip(d.keys(), d)) if isinstance(d, sqlite3.Row) else d, Experiment.get_data())
    normb = np.linalg.norm(Experiment.inputs[0]["b"])
    #A = Experiment.inputs[0]["A"]
    #A1T = A.sum(axis=0)
    #A1T2 = []
    #for i in xrange(A.shape[1]):
    #    A1T2 += [norm(A[:,i])]
        
    for data in Experiment.get_data():
        data["checksum"] = map(lambda d: abs(d), data["checksum"])
        k = data["faults"][0]["loc"]["k"]
        data["corrected_checksum"] = data["checksum"] #abs(A1T2[k] / A1T[0, k]) * data["checksum"]
        data["true_checksum"] = [0. if i != data["faults"][0]["timer"] else abs(data["Ekvk"]) for i in xrange(len(data["checksum"]))]
        data["threshold"] = map(lambda d: c * epsilon * normb / abs(d), data["y"][(when_has_converged(data, epsilon) - 1) if when_has_converged(data, epsilon) else (data["l"] - 1)])
        data["computed_threshold"] = [abs(c * epsilon * normb / d[-1]) for d in data["y"]]
        data["delta"] = map(lambda d: abs((data["Ekvk"]) * d) / normb, map(lambda d: 0 if len(d) <= data["faults"][0]["timer"] else d[data["faults"][0]["timer"]], data["y"]))

def run(algorithm_parameters, A,  npoints = 20):
    try:
        n = A.shape[0]
        x = np.ones(n)
        b = A.dot(x)
        normb = np.linalg.norm(b)
        x0= np.zeros(n)
        input = {"A":A, "x0":x0, "b":b}
        inputs = [input]
    except:
        print "Problème avec la matrice A !"
        
    print "Running the reference experiment (no fault) ..."
    
    epsilon = algorithm_parameters["tol"]
    algorithm_parameters["restart"] = None
    algorithm_parameters["save_data"] ={ "l",
                                         "residual",
                                         "residuals",
                                         "true_residual",
                                        "true_residuals",
                                         #"orthogonality",
                                         #"arnoldi",
                                         "faults",
                                         "H",
                                         "V",
                                         "givens",
                                         "y",
                                         "checksum",
                                         "Ekvk",
                                         #"threshold",
                                         #"criteria",
                                         "delta",
                                         "true_delta"
    }

    experiment_parameters = Parameters({})

    experiment_parameters["vulnerable"] = False
    experiment_parameters["tol"] = 0
    experiment_parameters["full"] = False
    
    gmres_algorithm = Algorithm(algorithm_parameters, implementation)
    F = Experiment(experiment_parameters, gmres_algorithm)
    F.set_inputs([input])


    F.run(show_progress=True, n=1)

    minimum = min(filter(lambda d: d[1] < epsilon, enumerate(F.get_data("residuals")[0])),
                  key=lambda d: d[0])
    min_iteration = minimum[0]
    
    gmres_display.convergence_history(F.get_data()[0], xlim = (0, 2*min_iteration), ylim = (epsilon / 1000000, 10), computed_residual = True, true_residual = True, checksum = False, threshold = False, c = 1, delta = False)
    plt.plot([0, n], [epsilon, epsilon], linestyle="--", color="orange")
    plt.show()

    print "Running the full experiment..."
    
    step_ite = int(math.ceil(float(min_iteration) / npoints))
    step_bit = int(math.ceil(float(64) / npoints))
    
    non_zero = A.nonzero()
    index = int(np.random.rand()*len(non_zero[0]))
    indices = [{"i": np.asscalar(non_zero[0][index]), 
               "j": 0, 
               "k": np.asscalar(non_zero[1][index])}]
    register = algorithm_parameters.get("register", 4)
    experiment_parameters = experiment_parameters_generator(bits = np.arange(63, -1, -step_bit),
                                                            iterations = np.arange(0, min_iteration, step_ite),
                                                            registers = [register],
                                                            tol = epsilon / 100,
                                                            restart = 2 * min_iteration,
                                                            maxiter = 2 * min_iteration,
                                                            indices = indices)
    
    I = Experiment(experiment_parameters, gmres_algorithm, callback=None, save_data=True)

    I.set_inputs([input])
    
    I.run(show_progress=True, n=int(math.ceil(float(min_iteration)/step_ite)) * int(math.ceil(64./step_bit)))
    
    I.min_iteration = min_iteration
    update_data(I, epsilon = epsilon, c = 0.5)
    return I
    #I.save_database("pores_2_1.e-5_right_csc.sqlite3")
    #I.load_database("pores_2_1.e-5_right_csc.sqlite3")

    
def convergence_results(I):
    if (I.results):
        parameters = I.algorithm.parameters
    else:
        print "L'experience ne contient pas de données à afficher"
        return
    fig = plt.figure()
    epsilon = parameters["tol"]
    min_iteration = I.min_iteration

    gmres_display.scatter_bit_injection(I.get_data(), 
                                        min_iteration = min_iteration, 
                                        color=lambda d : gmres_display.convergence_outcome_color(d, min_iteration, epsilon),
                                        epsilon = epsilon)
    plt.show()
    # gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
    # idx += 1


# # In[23]:

# fig = plt.figure()
# epsilon = 1.e-10
# gmres_display.scatter_bit_injection(J.get_data(), 
#                                     min_iteration = min_iteration_J, 
#                                     color=lambda d : gmres_display.convergence_outcome_color(d, min_iteration_J, epsilon),
#                                     epsilon = epsilon)
# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # import os
# # import sys
# # #import mysql.connector as mariadb
# # import pymysql as mariadb
# # mariadb_connection = mariadb.connect(user='resilience', 
# #                                      #password='resilience_password', 
# #                                      passwd='resilience_password_',
# #                                      #host='192.168.1.48',
# #                                      database='gmres')
# # cursor = mariadb_connection.cursor()

# # def database_generator(cursor, data, n = None, show_progress = True):
# # 
# #     cursor.execute("SELECT COUNT(*) from Experiment")
# #     if not n:
# #         n_rows = cursor.fetchone()[0]
# #     else:
# #         n_rows = n
# #     c_rows = 0
# #     
# #     request = "SELECT"
# #     first = True
# #     for key in data:
# #         if not first:
# #             request += ","
# #         else:
# #             first = False
# #         request += " "
# #         if key == "faults":
# #             request += "fault_i, fault_k, fault_f, fault_bit, fault_reg"
# #         else:
# #             request += key
# #     request += " FROM Experiment"
# #     
# #     first = True
# #     for key in data:
# #         if data[key]:
# #             if first:
# #                 first = False
# #                 request += " WHERE "
# #             else:
# #                 request += " AND "
# #             if key == "faults":
# #                 key = "fault_i"
# #             request += key + " = " + str(data[key])
# #         
# #     
# #     print request    
# #     cursor.execute(request)
# #     row = cursor.fetchone()
# #     while row is not None:
# #         c_row = {}
# #         k = 0
# #         for key in data:
# #             if key == "faults":
# #                 fault = {}
# #                 fault["loc"] = {"i":int(row[k]), "j":0, "k":int(row[k+1])}
# #                 fault["timer"] = int(row[k+2])
# #                 fault["bit"] = int(row[k+3])
# #                 fault["register"] = int(row[k+4])
# #                 c_row["faults"] = [fault] #TODO
# #                 k += 4
# #             elif key == "l" or key == "m":
# #                 c_row[key] = int(row[k])
# #             elif key == "residuals" or key == "true_residuals":
# #                 c_row[key] = np.fromstring(row[k])
# #             else:
# #                 c_row[key] = row[k]
# #             k += 1
# #         c_rows += 1
# #         if show_progress:
# #             new_progress = (100 * c_rows) / n_rows
# #             stdout.write("\r%d %c " % (new_progress, '%'))
# #             stdout.flush()
# #         yield c_row
# #         if (c_rows == n):
# #             break
# #         row = cursor.fetchone()
# #     

# # for d in database_generator(cursor, {"A":None, "m":None, "tol":None, "residual":None, "residuals":None, "true_residual":None, "true_residuals": None, "faults":None, "l":None}, 10000):
# #     cursor2 = mariadb_connection.cursor()
# #     cursor2.execute("SELECT MIN(l) FROM Experiment WHERE A = '" + d["A"] + "'")
# #     L = cursor2.fetchone()[0]
# #     
# #     if d["true_residual"] <= epsilon:
# #         if L < d["l"]:
# #             c = tuple(0, 255 * (float(d["l"]) / d["m"]) , 255 * (1 - (float(L)/d["l"])), 0.5)
# #         else:
# #             c = "green"
# #     else:
# #         c = "red"
# #     
# #     plt.scatter(d["faults"][0]["bit"], (100*d["faults"][0]["timer" ]) / L, alpha = 0.5, c = c)
# #     plt.xlim(0, 64)
# #     plt.ylim(0, 100)

# # mariadb_connection.close()

# # In[24]:



# # ## Fault impact prediction

# # In[25]:

# epsilon = 1.e-10
# fig = plt.figure()
# data = J.get_data()[582]
# gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], threshold = False, delta = False, true_delta = True, true_delta_label = "Measured residual gap", checksum = False, Ekvk=False, Ekvk_label = "Error", computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon), xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
# plt.legend(numpoints=1)
# plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1
# plt.show()


# # In[26]:

# fig = plt.figure()
# data = J.get_data()[582]
# gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], threshold = False, delta = True, true_delta = True, true_delta_label = "Measured residual gap", checksum = False, Ekvk=False, Ekvk_label = "Error", computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon), xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
# plt.legend(numpoints=1)
# plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1
# plt.show()


# # In[27]:

# fig = plt.figure()
# data = J.get_data()[368]
# gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], threshold = True, delta = False, true_delta = False, true_delta_label = "Measured residual gap", checksum = False, Ekvk=True, Ekvk_label = "Error", computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon), xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
# plt.legend(numpoints=1)
# plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1
# plt.show()


# # In[28]:

# fig = plt.figure()
# data = J.get_data()[668]
# gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], threshold = True, delta = False, true_delta = False, true_delta_label = "Measured residual gap", checksum = False, Ekvk=True, Ekvk_label = "Error", computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon), xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
# plt.legend(numpoints=1)
# plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1
# plt.show()


# # In[29]:

def prediction_result(I, c = 0.5, epsilon = 1.e-10):
    
    update_data(I, c = c, epsilon = epsilon)
    normb = np.linalg.norm(I.inputs[0]["b"])
    
    def color_prediction_outcome(data, epsilon):
        l = when_has_converged(data, epsilon)
        if l is None:
            l = data["l"]
        f = data["faults"][0]["timer"]
        if (l == f):
            return "red"
        ylf = data["y"][l-1][f]
        #checksum = data["checksum"][f]
        Ekvk = data["Ekvk"]
        threshold = (c * epsilon * normb) / abs(ylf)
        if (Ekvk < threshold):
            return "green"
        else:
            return "red"

    print "Executions qui ont convergée sans delai"
    fig = plt.figure()
    prediction = gmres_display.scatter_bit_injection(filter(lambda d: color_prediction_outcome(d, epsilon) == "green" and has_converged(d, epsilon) and when_has_converged(d, epsilon) == I.min_iteration, I.get_data()), 
                                                     min_iteration = I.min_iteration, 
                                                     color="green")
    noprediction = gmres_display.scatter_bit_injection(filter(lambda d: color_prediction_outcome(d,epsilon) == "red" and has_converged(d, epsilon) and when_has_converged(d, epsilon) == I.min_iteration, I.get_data()), 
                                                       min_iteration = I.min_iteration, 
                                                       color="red")
    plt.legend([prediction, noprediction], ["Predicted to converge", "No prediction"])
    # gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
    
    # idx += 1
    plt.show()

    print "Executions qui ont convergée avec delai"
    fig = plt.figure()
    prediction = gmres_display.scatter_bit_injection(filter(lambda d: color_prediction_outcome(d, epsilon) == "green" and has_converged(d, epsilon) and when_has_converged(d, epsilon) > I.min_iteration, I.get_data()), 
                                                     min_iteration = I.min_iteration, 
                                                     color="green")
    noprediction = gmres_display.scatter_bit_injection(filter(lambda d: color_prediction_outcome(d, epsilon) == "red" and has_converged(d, epsilon) and when_has_converged(d, epsilon) > I.min_iteration, I.get_data()), 
                                                       min_iteration = I.min_iteration, 
                                                       color="red")
    plt.legend([prediction, noprediction], ["Predicted to converge", "No prediction"])
    #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
    
    # idx += 1
    plt.show()

    print "Executions qui n'ont pas convergée"
    fig = plt.figure()
    prediction = gmres_display.scatter_bit_injection(filter(lambda d: color_prediction_outcome(d, epsilon) == "green" and not has_converged(d, epsilon), I.get_data()), 
                                                     min_iteration = I.min_iteration, 
                                                     color="green")
    noprediction = gmres_display.scatter_bit_injection(filter(lambda d: color_prediction_outcome(d, epsilon) == "red" and not has_converged(d, epsilon) , I.get_data()), 
                                                       min_iteration = I.min_iteration, 
                                                       color="red")
    plt.legend([prediction, noprediction], ["Predicted to convrege", "No prediction"])
    #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
    
    # idx += 1
    plt.show()

def detection_results(I, key_checksum = "checksum", key_threshold = "computed_threshold", c=0.5, epsilon = 1.e-10):
    #fig = plt.figure()
    update_data(I, epsilon = epsilon, c = 1)
    gmres_display.show_test_result(I.get_data(), I.min_iteration, c=c, epsilon = epsilon, key_checksum=key_checksum, key_threshold=key_threshold)
    plt.show()
    #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
    #idx += 1
# # 
# # 
# #     
# # for data in I.get_data()[282:283]:
# #     fig = plt.figure()
# #     data = dict(zip(data.keys(), data))
# #     data["checksum"] = map(lambda d: abs(np.linalg.norm(d)), data["checksum"])
# #     data["threshold"] = map(lambda d: c * epsilon * normb / abs(d), data["y"][when_has_converged(data, epsilon) if when_has_converged(data, epsilon) else data["l"]])
# #     data["delta"] = map(lambda d: abs((data["Ekvk"]) * d) / normb, map(lambda d: 0 if len(d) <= data["faults"][0]["timer"] else d[data["faults"][0]["timer"]], data["y"]))
# #     #data["delta"] = [data["Ekvk"] * data["y"][data["l"]][data["faults"][0]["timer"]] / normb for i in xrange(data["l"])]
# #     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], threshold = True, delta = False, true_delta = False, true_delta_label = "Measured residual gap", checksum = False, Ekvk=True, Ekvk_label = "Error", computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=gmres_display.convergence_outcome_color(data, min_iteration, epsilon), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
# #     plt.plot([0, int(1.5 * min_iteration)], [epsilon, epsilon], linestyle="--", color="orange")
# #     plt.legend(numpoints=1)
# #     ##gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(i) + ".pdf")
# #     #i += 1
# #     plt.show()

# # In[ ]:




# # ## Fault detection

# # ### Qualitative results

# # def criterion(data, c = 0.25, epsilon = epsilon):
# #     m = when_has_converged(data, epsilon = (1-c)*epsilon)
# #     if len(data["checksum"]) >= m and len(data["threshold"]) >= m:
# #         return reduce(lambda d, e: d and (e[0] < c * e[1]), zip(data["checksum"][:m], data["threshold"][:m])) and data["checksum"][0] < c * data["threshold"][0]
# #     else:
# #         return True

# # In[30]:


# key_threshold = "threshold"
# key_checksum = "true_checksum"


# # epsilon = 1.e-10
# # c = 0.5
# # experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J / 3], bits = [30], tol = epsilon/100)
# # 
# # E = Experiment(experiment_parameters, gmres_algorithm)
# # E.set_inputs(inputs)
# # E.run(show_progress=True, n=1)
# # 
# # update_data(E, epsilon, c = c)
# # for data in E.get_data():
# #     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
# #     fig = plt.figure()
# #     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", threshold = True, checksum = False, computed_threshold = False, Ekvk=True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 1000000, 1), xytext = (1, 2 * epsilon / 1000000), bbox_to_anchor=(0.38, 0.85))
# #     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
# #     #fig.show()
# #     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# #     idx += 1
# #     
# # epsilon = 1.e-10
# # c = 0.5
# # experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J / 3], bits = [24], tol = epsilon/100)
# # 
# # E = Experiment(experiment_parameters, gmres_algorithm)
# # E.set_inputs(inputs)
# # E.run(show_progress=True, n=1)
# # 
# # update_data(E, epsilon, c = c)
# # for data in E.get_data():
# #     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
# #     fig = plt.figure()
# #     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", threshold = True, checksum = False, computed_threshold = False, Ekvk=True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 1000000, 1), xytext = (1, 2 * epsilon / 1000000), bbox_to_anchor=(0.38, 0.85))
# #     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
# #     #fig.show()
# #     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# #     idx += 1
# # 

# # In[31]:

def detection_results_c(I, key_checksum, key_threshold, epsilon):
    X = [1.1**(-50+i) for i in xrange(51)]
    test_orange = []
    test_red = []
    test_green = []
    test_gray1 = []
    test_gray2 = []
    
    update_data(I, epsilon, c = 1)
    data = filter(lambda d: len(d["faults"]) > 0, I.get_data())

    for x in X:
        #test_red += [(100. * len(filter_data(data, {'converged':False, 'criterion':True}, c=x)))/len(data)]
        #test_green += [(100. * len(filter_data(data, {'converged':False, 'criterion':False}, c=x)))/len(data)]
        #test_gray1 += [(100. * len(filter_data(data, {'converged':True, 'criterion':True}, c=x)))/len(data)]
        #test_gray2 += [(100. * len(filter_data(data, {'converged':True, 'criterion':False}, c=x)))/len(data)]
        
        
    
        test_orange += [(100. * len(filter(lambda d: false_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
        test_red += [(100. * len(filter(lambda d: fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
        test_green += [(100. * len(filter(lambda d: fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
        test_gray1 += [(100. * len(filter(lambda d: no_impact_fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
        test_gray2 += [(100. * len(filter(lambda d: no_impact_fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
        
        
    Y = np.row_stack((test_gray1, test_gray2, test_green, test_red))   
    Y = np.cumsum(Y, axis=0)  
        
    fig = plt.figure()
    ax = fig.add_subplot(111)
    
    #ax.plot(X, test_orange, c="orange", linestyle="--")
    ax.fill_between(X, 0, Y[0,:], facecolor="lightgray", alpha=.7, label="No impact fault not detected")
    ax.fill_between(X, Y[0,:], Y[1,:], facecolor="darkgray", alpha=.7, label="No impact fault detected")
    ax.fill_between(X, Y[1,:], Y[2,:], facecolor="green", alpha=.7, label="Critical fault detected")
    ax.fill_between(X, Y[2,:], Y[3,:], facecolor="red",alpha=.7, label="Critical fault not detected")
    ax.set_xscale('log')
    ax.set_ylim([0, 100])
    ax.set_xlim([0.01, 1])
    ax.set_xlabel("c")
    ax.set_ylabel("%")
    ax.set_title("Test result repartition")
    
    p1 = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=.7)
    p2 = plt.Rectangle((0, 0), 1, 1, fc="lightgray", alpha=.7)
    p3 = plt.Rectangle((0, 0), 1, 1, fc="darkgray", alpha=.7)
    p4 = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=.7)
    plt.legend([p1, p2, p3, p4], ["Critical fault detected", "No impact fault ignored", "No impact fault detected", "Critical fault ignored"], loc=4)
    plt.plot([0.5, 0.5], [0, 100], c = "red", linestyle = "--")
    plt.text(0.45, -7, "0.5")
    plt.show()

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[32]:


# X = [1.1**(-50+i) for i in xrange(51)]
# test_orange = []
# test_red = []
# test_green = []
# test_gray1 = []
# test_gray2 = []

# epsilon = 1.e-10
# update_data(J, epsilon, c = 1)
# data = filter(lambda d: len(d["faults"]) > 0, J.get_data())

# for x in X:
#     #test_red += [(100. * len(filter_data(data, {'converged':False, 'criterion':True}, c=x)))/len(data)]
#     #test_green += [(100. * len(filter_data(data, {'converged':False, 'criterion':False}, c=x)))/len(data)]
#     #test_gray1 += [(100. * len(filter_data(data, {'converged':True, 'criterion':True}, c=x)))/len(data)]
#     #test_gray2 += [(100. * len(filter_data(data, {'converged':True, 'criterion':False}, c=x)))/len(data)]

#     test_orange += [(100. * len(filter(lambda d: false_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_red += [(100. * len(filter(lambda d: fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_green += [(100. * len(filter(lambda d: fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_gray1 += [(100. * len(filter(lambda d: no_impact_fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_gray2 += [(100. * len(filter(lambda d: no_impact_fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
    

# Y = np.row_stack((test_gray1, test_gray2, test_green, test_red))   
# Y = np.cumsum(Y, axis=0)  

# fig = plt.figure()
# ax = fig.add_subplot(111)

# #ax.plot(X, test_orange, c="orange", linestyle="--")
# ax.fill_between(X, 0, Y[0,:], facecolor="lightgray", alpha=.7, label="No impact fault not detected")
# ax.fill_between(X, Y[0,:], Y[1,:], facecolor="darkgray", alpha=.7, label="No impact fault detected")
# ax.fill_between(X, Y[1,:], Y[2,:], facecolor="green", alpha=.7, label="Critical fault detected")
# ax.fill_between(X, Y[2,:], Y[3,:], facecolor="red",alpha=.7, label="Critical fault not detected")
# ax.set_xscale('log')
# ax.set_ylim([0, 100])
# ax.set_xlim([0.01, 1])
# ax.set_xlabel("c")
# ax.set_ylabel("%")
# ax.set_title("Test result repartition")

# p1 = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=.7)
# p2 = plt.Rectangle((0, 0), 1, 1, fc="lightgray", alpha=.7)
# p3 = plt.Rectangle((0, 0), 1, 1, fc="darkgray", alpha=.7)
# p4 = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=.7)
# plt.legend([p1, p2, p3, p4], ["Critical fault detected", "No impact fault ignored", "No impact fault detected", "Critical fault ignored"], loc=4)
# plt.plot([0.5, 0.5], [0, 100], c = "red", linestyle = "--")
# plt.text(0.45, -7, "0.5")
# plt.show()

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[33]:

# fig = plt.figure()
# epsilon = 1.e-5
# c = 0.5
# update_data(I, epsilon, c = 1)
# gmres_display.show_test_result(I.get_data(), min_iteration_I, c=c, epsilon = epsilon, key_checksum=key_checksum, key_threshold=key_threshold)    
# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[34]:

# fig = plt.figure()
# epsilon = 1.e-10
# c = 0.5
# update_data(J, epsilon, c = 1)
# gmres_display.show_test_result(J.get_data(), min_iteration_J, c=c, epsilon = epsilon, key_checksum=key_checksum, key_threshold=key_threshold)    
# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[35]:

# epsilon = 1.e-5
# c = 0.5

# experiment_parameters = experiment_parameters_generator(vulnerable = False, tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=1)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_I, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual = False, true_residual = False, true_residual_label="True residual", threshold = True, computed_threshold = True, Ekvk_label="True error", fault = False, fault_color=color, delta = False, xlim = (0, int(min_iteration_I * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.43, 0.15))
#     plt.plot([0, int(min_iteration_I * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
    
# epsilon = 1.e-10
# c = 0.5

# experiment_parameters = experiment_parameters_generator(vulnerable = False, tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=1)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual = False, true_residual = False, true_residual_label="True residual", threshold = True, computed_threshold = True, Ekvk_label="True error", fault = False, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.43, 0.15))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
    


# # In[36]:

# epsilon = 1.e-10
# c = 0.5
# experiment_parameters = experiment_parameters_generator(iterations = [10], bits = [24], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=1)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", threshold = False, checksum = True, computed_threshold = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 1000000, 1), xytext = (1, 2 * epsilon / 1000000), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
    
# epsilon = 1.e-10
# c = 0.5
# experiment_parameters = experiment_parameters_generator(iterations = [10], bits = [30], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=1)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", threshold = False, checksum = True, computed_threshold = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 1000000, 1), xytext = (1, 2 * epsilon / 1000000), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
       


# # In[37]:

# epsilon = 1.e-10
# c = 0.5
# experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J/2], bits = [27, 34], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=2)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", threshold = False, checksum = True, computed_threshold = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 1000000, 1), xytext = (1, 2 * epsilon / 1000000), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
    


# # In[38]:

# key_threshold = "computed_threshold"
# key_checksum = "checksum"


# # In[39]:


# X = [1.1**(-50+i) for i in xrange(51)]
# test_orange = []
# test_red = []
# test_green = []
# test_gray1 = []
# test_gray2 = []

# epsilon = 1.e-5
# update_data(I, epsilon, c = 1)
# data = filter(lambda d: len(d["faults"]) > 0, I.get_data())

# for x in X:
#     #test_red += [(100. * len(filter_data(data, {'converged':False, 'criterion':True}, c=x)))/len(data)]
#     #test_green += [(100. * len(filter_data(data, {'converged':False, 'criterion':False}, c=x)))/len(data)]
#     #test_gray1 += [(100. * len(filter_data(data, {'converged':True, 'criterion':True}, c=x)))/len(data)]
#     #test_gray2 += [(100. * len(filter_data(data, {'converged':True, 'criterion':False}, c=x)))/len(data)]
    

    
#     test_orange += [(100. * len(filter(lambda d: false_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_red += [(100. * len(filter(lambda d: fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_green += [(100. * len(filter(lambda d: fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_gray1 += [(100. * len(filter(lambda d: no_impact_fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_gray2 += [(100. * len(filter(lambda d: no_impact_fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]


# Y = np.row_stack((test_gray1, test_gray2, test_green, test_red))   
# Y = np.cumsum(Y, axis=0)  

# fig = plt.figure()
# ax = fig.add_subplot(111)

# #ax.plot(X, test_orange, c="orange", linestyle="--")
# ax.fill_between(X, 0, Y[0,:], facecolor="lightgray", alpha=.7, label="No impact fault not detected")
# ax.fill_between(X, Y[0,:], Y[1,:], facecolor="darkgray", alpha=.7, label="No impact fault detected")
# ax.fill_between(X, Y[1,:], Y[2,:], facecolor="green", alpha=.7, label="Critical fault detected")
# ax.fill_between(X, Y[2,:], Y[3,:], facecolor="red",alpha=.7, label="Critical fault not detected")
# ax.set_xscale('log')
# ax.set_ylim([0, 100])
# ax.set_xlim([0.01, 1])
# ax.set_xlabel("c")
# ax.set_ylabel("%")
# ax.set_title("Test result repartition")

# p1 = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=.7)
# p2 = plt.Rectangle((0, 0), 1, 1, fc="lightgray", alpha=.7)
# p3 = plt.Rectangle((0, 0), 1, 1, fc="darkgray", alpha=.7)
# p4 = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=.7)
# plt.legend([p1, p2, p3, p4], ["Critical fault detected", "No impact fault ignored", "No impact fault detected", "Critical fault ignored"], loc=4)
# plt.plot([0.5, 0.5], [0, 100], c = "red", linestyle = "--")
# plt.text(0.45, -7, "0.5")
# plt.show()

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[40]:


# X = [1.1**(-50+i) for i in xrange(51)]
# test_orange = []
# test_red = []
# test_green = []
# test_gray1 = []
# test_gray2 = []

# epsilon = 1.e-10
# update_data(J, epsilon, c = 1)
# data = filter(lambda d: len(d["faults"]) > 0, J.get_data())

# for x in X:
#     #test_red += [(100. * len(filter_data(data, {'converged':False, 'criterion':True}, c=x)))/len(data)]
#     #test_green += [(100. * len(filter_data(data, {'converged':False, 'criterion':False}, c=x)))/len(data)]
#     #test_gray1 += [(100. * len(filter_data(data, {'converged':True, 'criterion':True}, c=x)))/len(data)]
#     #test_gray2 += [(100. * len(filter_data(data, {'converged':True, 'criterion':False}, c=x)))/len(data)]

#     test_orange += [(100. * len(filter(lambda d: false_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_red += [(100. * len(filter(lambda d: fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_green += [(100. * len(filter(lambda d: fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_gray1 += [(100. * len(filter(lambda d: no_impact_fault_no_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
#     test_gray2 += [(100. * len(filter(lambda d: no_impact_fault_detection(d, epsilon = epsilon, c=x, key_checksum=key_checksum, key_threshold=key_threshold), data)))/len(data)]
    

# Y = np.row_stack((test_gray1, test_gray2, test_green, test_red))   
# Y = np.cumsum(Y, axis=0)  

# fig = plt.figure()
# ax = fig.add_subplot(111)

# #ax.plot(X, test_orange, c="orange", linestyle="--")
# ax.fill_between(X, 0, Y[0,:], facecolor="lightgray", alpha=.7, label="No impact fault not detected")
# ax.fill_between(X, Y[0,:], Y[1,:], facecolor="darkgray", alpha=.7, label="No impact fault detected")
# ax.fill_between(X, Y[1,:], Y[2,:], facecolor="green", alpha=.7, label="Critical fault detected")
# ax.fill_between(X, Y[2,:], Y[3,:], facecolor="red",alpha=.7, label="Critical fault not detected")
# ax.set_xscale('log')
# ax.set_ylim([0, 100])
# ax.set_xlim([0.01, 1])
# ax.set_xlabel("c")
# ax.set_ylabel("%")
# ax.set_title("Test result repartition")

# p1 = plt.Rectangle((0, 0), 1, 1, fc="green", alpha=.7)
# p2 = plt.Rectangle((0, 0), 1, 1, fc="lightgray", alpha=.7)
# p3 = plt.Rectangle((0, 0), 1, 1, fc="darkgray", alpha=.7)
# p4 = plt.Rectangle((0, 0), 1, 1, fc="red", alpha=.7)
# plt.legend([p1, p2, p3, p4], ["Critical fault detected", "No impact fault ignored", "No impact fault detected", "Critical fault ignored"], loc=4)
# plt.plot([0.5, 0.5], [0, 100], c = "red", linestyle = "--")
# plt.text(0.45, -7, "0.5")
# plt.show()

# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[41]:

# fig = plt.figure()
# epsilon = 1.e-5
# c = 0.5
# update_data(I, epsilon, c = 1)
# gmres_display.show_test_result(I.get_data(), min_iteration_I, c=c, epsilon = epsilon, key_checksum=key_checksum, key_threshold=key_threshold)    
# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[42]:

# fig = plt.figure()
# epsilon = 1.e-10
# c = 0.5
# update_data(J, epsilon, c = 1)
# gmres_display.show_test_result(J.get_data(), min_iteration_J, c=c, epsilon = epsilon, key_checksum=key_checksum, key_threshold=key_threshold)    
# #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
# idx += 1


# # In[43]:

# for data in filter(lambda d: fault_no_detection(d, epsilon = epsilon, c=c, key_checksum=key_checksum, key_threshold=key_threshold), J.get_data())[:10]:
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", threshold = False, checksum = True, computed_threshold = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 1000000, 1), xytext = (1, 2 * epsilon / 1000000), bbox_to_anchor=(0.38, 0.85))
#     plt.show()
#     print data["faults"]


# # In[ ]:




# # ### bits

# # In[44]:

# experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J/2], bits = [0, 25, 50], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=3)

# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1


# # ### Iterations

# # In[45]:

# experiment_parameters = experiment_parameters_generator(iterations = [5, 10, 15], bits = [25], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=3)

# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1


# # In[46]:

# experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J / 2], bits = [25], registers=[1, 2, 3], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=3)

# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, F.get_data()[0], epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1


# # In[47]:

# non_zero = A.nonzero()
# indices = np.random.rand(3)*len(non_zero[0])
# indices = map(lambda i: {"i":non_zero[0][int(i)], "k":non_zero[1][int(i)]}, indices)

# experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J / 2], bits = [25], indices = indices, tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=3)

# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, F.get_data()[0], epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)),  ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")

#     idx += 1


# # In[61]:

# experiment_parameters = experiment_parameters_generator(iterations = [min_iteration_J / 3], bits = [20, 35, 50], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=3)

# epsilon = 1.e-10
# c = 0.5
# update_data(E, epsilon, c = c)



# A1T = A.sum(axis=0)
# A1T2 = []
# for i in xrange(A.shape[1]):
#     A1T2 += [norm(A[:,i])]

# for data in E.get_data():
#     k = data["faults"][0]["loc"]["k"]
#     print k
#     data["checksum"] = A1T2[k] * abs(data["checksum"] / A1T[0,k])
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual_label="Computed residual (fault)", true_residual_label="True residual (fault)", checksum = True, Ekvk = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 10000000, 1), xytext = (1, 2 * epsilon / 10000000), bbox_to_anchor=(0.38, 0.85))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1


# # In[ ]:

# epsilon = 1.e-10
# c = 0.5

# experiment_parameters = experiment_parameters_generator(bits = [7], iterations=[26], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=1)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual = False, true_residual = False, true_residual_label="True residual", threshold = True, computed_threshold = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.43, 0.15))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
    
# epsilon = 1.e-10
# c = 0.5

# experiment_parameters = experiment_parameters_generator(bits = [8], iterations=[26], tol = epsilon/100)

# E = Experiment(experiment_parameters, gmres_algorithm)
# E.set_inputs(inputs)
# E.run(show_progress=True, n=1)

# update_data(E, epsilon, c = c)
# for data in E.get_data():
#     color = gmres_display.convergence_outcome_color(data, min_iteration_J, epsilon)
#     fig = plt.figure()
#     gmres_display.convergence_history(data, data_no_fault = F.get_data()[0], computed_residual = False, true_residual = False, true_residual_label="True residual", threshold = True, computed_threshold = True, Ekvk_label="True error", fault = True, fault_color=color, delta = False, xlim = (0, int(min_iteration_J * 1.2)), ylim = (epsilon / 100, 1), xytext = (1, 2 * epsilon / 100), bbox_to_anchor=(0.43, 0.15))
#     plt.plot([0, int(min_iteration_J * 1.2)], [epsilon, epsilon], linestyle="--", color="orange")
#     #fig.show()
#     #gmres_display.save_fig_pdf(fig, "./report/figures/pores_2/right/csc/figure" + str(idx) + ".pdf")
#     idx += 1
    

