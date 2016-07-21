from Parameters import *

import numpy as np

import matplotlib.image as mpimg
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LogNorm
from IPython.display import set_matplotlib_formats
set_matplotlib_formats('png', 'pdf')

# Define basic display

def plot_2D(X, Y, title = '', grid = True, label = "", log=False, logX=False, linestyle = None, xlabel = None, ylabel = None, xlim = None, ylim = None, bbox_to_anchor=(0.5, 0.88)):

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

        
# Define basic functions

def test_result_to_color(result):
    if result == "Correct":
        return "green"
    if result == "False positive":
        return "red"
    if result == "False negative":
        return "blue"
    return "black"

def criterion_valid(data, c = 0.25, oracle={"Ekvk":True, "ylk":True}, epsilon = 1.e-12, normb = 1. ):
    l = 0
    for k, true_residual in enumerate(data["true_residuals"][1:]):
        l = k
        if true_residual < (1-c) * epsilon:
            break

    if ("ylk" in oracle and oracle["ylk"]):
        print len(data["y"]), " / ", l
        if data["faults"][0]["timer"] < len(data["y"][l]):
            # Fault occurs before convergence
            ylk = data["y"][l][data["faults"][0]["timer"], 0]
        else:
            # Fault occurs after convergence
            return True
            
    for k in xrange(l):
        if "ylk" in oracle and not oracle["ylk"]:
            ylk = data["y"][k][k, 0]
            
        if ("Ekvk" in oracle and oracle["Ekvk"]):
            if k == data["faults"][0]["timer"]:
                Ekvk = abs(data["faults"][0]["value_after"] - data["faults"][0]["value_before"])
            else:
                Ekvk = 0
        else:
            Ekvk = data["checksum"][k]

            
        criterion = c * abs((epsilon * normb) / ylk)
        if Ekvk > criterion:
            return False
    return True

def filter_data(data, dictionnary, c = 0.25, epsilon = 1.e-12, normb = 1.):
    result = data
    if 'converged' in dictionnary:
        if dictionnary['converged']:
            result = filter(lambda d: d["true_residual"] < (1-c)*epsilon, result)
        else:
            result = filter(lambda d: d["true_residual"] > (1-c)*epsilon, result)
    if 'criterion' in dictionnary:
        if dictionnary['criterion']:
            result = filter(lambda d: criterion_valid(d, c=c, epsilon=epsilon, normb=normb), result)
        else:
            result = filter(lambda d: not criterion_valid(d, c=c, epsilon=epsilon, normb=normb), result)     
            
    return result


def draw_lines(data, c=0.25, epsilon=1.e-12, normb=1.):
    max_iteration = max(map(lambda f: f['faults'][0]["timer"], data))
    X_line = [63-i for i in xrange(64)]
    Y_line_1 = []
    Y_line_2 = []
    for x in X_line:
        mini = filter(lambda f: f['faults'][0]["bit"] == x, filter_data(data, Parameters({'criterion':True}), c = c, epsilon = epsilon, normb = normb))
        maxi = filter(lambda f: f['faults'][0]["bit"] == x, filter_data(data, Parameters({'criterion':False}), c = c, epsilon = epsilon, normb = normb))
        if mini:
            mini = map(lambda f: f['faults'][0]["timer"], mini)
            Y_line_2 += [min(mini)]
        else:
            Y_line_2 += [max_iteration]
            if maxi:
                maxi = map(lambda f: f['faults'][0]["timer"], maxi)
                Y_line_1 += [max(maxi)]
            else:
                Y_line_1 += [-1]
    return X_line, Y_line_1, Y_line_2

def smooth_lines(X_line, Y_line_1, Y_line_2):
    X_smooth = []
    Y_smooth_1 = []
    Y_smooth_2 = []
    for i in xrange(len(Y_line_1)):
        if (Y_line_1[i] > 0 and Y_line_2[i] > 0):
            X_smooth += [i]
            Y_smooth_1 += [Y_line_1[i]]
            Y_smooth_2 += [Y_line_2[i]]
    return X_smooth, Y_smooth_1, Y_smooth_2

def classification_criterion(data, c = 0.25, oracle={"Ekvk":False, "ylk":False}, epsilon = 1.e-12, normb = 1. ):
    # if algorithm converged (true_residual < epsilon) then we should have checksum < criterion
    # if algorithm did not converge, then we should have checksum < criterion 
        # except for the iteration where the fault occured (checksum[k] > criterion[k])
    converged = False
    l = 0
    for k, true_residual in enumerate(data["true_residuals"]):
        l = k
        if true_residual < (1-c) * epsilon:
            converged = True
            break

    if converged:
        for k in xrange(l):
            ykk = data["y"][k][k, 0]
            if ("Ekvk" in oracle and oracle["Ekvk"]):
                if k == data["faults"][0]["timer"]:
                    Ekvk = abs(data["faults"][0]["value_after"] - data["faults"][0]["value_before"])
                else:
                    Ekvk = 0
            else:
                Ekvk = data["checksum"][k]
            if ("ylk" in oracle and oracle["ylk"]):
                ylk = data["y"][l-1][data["faults"][0]["timer"], 0]
            else:
                ylk = ykk
            
            criterion = c * abs((epsilon * normb) / ylk)
            if Ekvk > criterion:
                return "False positive" # False positive
    else:
        for k in xrange(l):
            ykk = data["y"][k][k, 0]
            if ("Ekvk" in oracle and oracle["Ekvk"]):
                if k == data["faults"][0]["timer"]:
                    Ekvk = abs(data["faults"][0]["value_after"] - data["faults"][0]["value_before"])
                else:
                    Ekvk = 0
            else:
                Ekvk = data["checksum"][k]
            if ("ylk" in oracle and oracle["ylk"]):
                ylk = data["y"][l-1][data["faults"][0]["timer"], 0]
            else:
                ylk = ykk
            criterion = c * abs((epsilon * normb) / ylk)
                                
            if k == data["faults"][0]["timer"]: 
                if (Ekvk < criterion): 
                    return "False negative" # False negative
            else: 
                if (Ekvk > criterion):
                    return "False positive" # Weird case, False positive (kind of)
    return "Correct"

# Define display functions
def convergence_history(data, data_no_fault = None, computed_residual = True, computed_residual_label="Computed residual", true_residual = True, true_residual_label = "True residual", delta = False, delta_label="Delta",
delta_linestyle = '-', checksum = False, checksum_label = "Check-sum", checksum_linestyle = '-', threshold = False, threshold_label = "Threshold", threshold_linestyle = '-', c = 0.25, fault = False, arrow = False, xlim = None, ylim = None, xytext=(0, 0), log = True, bbox_to_anchor=(0.5, 0.88), title = 'Convergence History', xlabel="iteration", ylabel = "error", linestyle='-'):
    
    if data_no_fault:
	convergence_history(data_no_fault, linestyle = '--', true_residual = False, computed_residual_label = "residual (no fault)", xlim=xlim,ylim=ylim, title=title)
    
    X = np.arange(0, data['iteration_count']+1)
 
    if computed_residual:
	Y = data['residuals']
    	plot_2D(X, Y, log=log, title=title, linestyle = linestyle, label = computed_residual_label, xlabel= xlabel, ylabel = ylabel, bbox_to_anchor= bbox_to_anchor)
    if true_residual:
	Y = data['true_residuals']
	plot_2D(X, Y, log=log, title=title, linestyle = linestyle, label = true_residual_label, xlabel= xlabel, ylabel = ylabel, bbox_to_anchor= bbox_to_anchor)

    if delta:
	Y = data['delta']
	plot_2D(X, Y, log=log, title=title, 
		linestyle = delta_linestyle, 
		label = delta_label, 
		xlabel= xlabel, 
		ylabel = ylabel, 
		bbox_to_anchor= bbox_to_anchor)
    if fault:
	x = (data['faults'][0]['timer'])
        y = data['true_residuals'][x]
	color = "red"
        plt.plot([x], [y], 'ro', c=color)

        if data['faults'][0]['register'] == "left":
	    register = "reg1"
        if data['faults'][0]['register'] == "right":
	    register = "reg2"
        if data['faults'][0]['register'] == "middle":
	    register = "reg3"
        annotation = "iteration : %d \n location : (%d, %d) \n bit : %d \n register : %s \n " % (data['faults'][0]['timer'], 
 		      data['faults'][0]['loc']['i'], 
		      data['faults'][0]['loc']['k'],
                      data['faults'][0]['bit'],
		      register)
        
        if arrow:
        	plt.annotate(annotation, xy=(x, y), xytext=xytext,
                     	arrowprops=dict(facecolor="black", shrink=0.05),)
	else:
		plt.annotate(annotation, xy=(x, y), xytext=xytext)

    if checksum:
	Y = data['checksum']
	plot_2D(X, Y, log=log, title=title, 
		linestyle = checksum_linestyle, 
		label = checksum_label, 
		xlabel= xlabel, 
		ylabel = ylabel, 
		bbox_to_anchor= bbox_to_anchor)

    if threshold:
	Y = map(lambda d: c * d, data['threshold'])
	plot_2D(X, Y, log=log, title=title, 
		linestyle = threshold_linestyle, 
		label = threshold_label, 
		xlabel= xlabel, 
		ylabel = ylabel, 
		bbox_to_anchor= bbox_to_anchor)

    if xlim:
	plt.xlim(xlim)
    if ylim:
	plt.ylim(ylim)


def convergence_bit_iteration(data, parameters):

    # Define constants
    normb = parameters.get("normb", 1.) 
    c = parameters.get("c", 0.25)
    m = parameters["m"]
    oracle = parameters.get("oracle", Parameters({"Ekvk":False, "ylk":False}))
    image = parameters.get("image", mpimg.imread("/home/alemoreau/Pictures/float.png"))
    lines = parameters.get("lines", True)
    separated = parameters.get("separated", True)
    converged = parameters.get("converged", True)
    not_converged = parameters.get("not_converged", True)
    criterion = parameters.get("criterion", True)
    not_criterion = parameters.get("not_criterion", True)
    epsilon = parameters.get("epsilon", 1.e-12)
    
    xlim = parameters.get("xlim", None)
    ylim = parameters.get("ylim", None)
    
    # Filter data
    data = filter(lambda d: len(d["faults"]) > 0 and d["faults"][0]["timer"] < m, data)

    if converged:
        data_converged = filter_data(data, Parameters({'converged':True}), c = c, epsilon = epsilon, normb = normb)
        if criterion:
            data_converged_criterion = filter_data(data, Parameters({'converged':True, 'criterion':True}), c = c, epsilon = epsilon, normb = normb)
        if not_criterion:
            data_converged_not_criterion = filter_data(data, Parameters({'converged':True, 'criterion':False}), c = c, epsilon = epsilon, normb = normb)
    if not_converged:
        data_not_converged = filter_data(data, Parameters({'converged':False}), c = c, epsilon = epsilon, normb = normb)
        if criterion:
            data_not_converged_criterion = filter_data(data, Parameters({'converged':False, 'criterion':True}), c = c, epsilon = epsilon, normb = normb)
        if not_criterion:
            data_not_converged_not_criterion = filter_data(data, Parameters({'converged':False, 'criterion':False}), c = c, epsilon = epsilon, normb = normb)
        

    if not_converged:
        if criterion:
            X_not_converged_criterion = map(lambda d: 63-d['faults'][0]['bit'],
                                            data_not_converged_criterion)
            Y_not_converged_criterion = map(lambda d: d['faults'][0]['timer'],
                                            data_not_converged_criterion)
            
            C_not_converged_criterion = map(lambda d: test_result_to_color(classification_criterion(d, c = c, epsilon = epsilon, normb = normb, oracle = oracle)),
                                            data_not_converged_criterion)
        if not_criterion:
            X_not_converged_not_criterion = map(lambda d: 63-d['faults'][0]['bit'],
                                            data_not_converged_not_criterion)
            Y_not_converged_not_criterion = map(lambda d: d['faults'][0]['timer'],
                                            data_not_converged_not_criterion)
            
            C_not_converged_not_criterion = map(lambda d: test_result_to_color(classification_criterion(d, c = c, epsilon = epsilon, normb = normb, oracle = oracle)),
                                                data_not_converged_not_criterion)
    if converged:
        if criterion:
            X_converged_criterion = map(lambda d: 63-d['faults'][0]['bit'],
                                        data_converged_criterion)
            Y_converged_criterion = map(lambda d: d['faults'][0]['timer'],
                                        data_converged_criterion)
            
            C_converged_criterion = map(lambda d: test_result_to_color(classification_criterion(d, c = c, oracle=oracle)),
                                        data_converged_criterion)
        if not_criterion:
            X_converged_not_criterion = map(lambda d: 63-d['faults'][0]['bit'],
                                            data_converged_not_criterion)
            Y_converged_not_criterion = map(lambda d: d['faults'][0]['timer'],
                                            data_converged_not_criterion)
            
            C_converged_not_criterion = map(lambda d: test_result_to_color(classification_criterion(d, c = c, oracle=oracle)),
                                            data_converged_not_criterion)




    if separated:
        f, axarr = plt.subplots(3, sharex=False)
        f.set_size_inches(6.5, 6.5)

        # First plot : execution did not converge
        
        axarr[0].scatter(X_not_converged_criterion,
                         Y_not_converged_criterion,
                         marker='^',
                         c=C_not_converged_criterion,
                         alpha=0.5)
        axarr[0].scatter(X_not_converged_not_criterion,
                         Y_not_converged_not_criterion,
                         marker='o',
                         c=C_not_converged_not_criterion,
                         alpha=0.5)
        
        axarr[0].set_title('Execution did not converge to the desired accuracy')
        if xlim:
            axarr[0].set_xlim(xlim[0])
        if ylim:
            axarr[0].set_ylim(ylim[0])
        axarr[0].spines['top'].set_visible(False)
        axarr[0].spines['right'].set_visible(False)
        axarr[0].spines['bottom'].set_visible(False)
        axarr[0].xaxis.set_visible(False)

        # Second plot : image
        axarr[1].imshow(image)
        axarr[1].axis('off')
        if xlim:
            axarr[1].set_xlim(xlim[1])
        
        # Third plot : execution did converge

        
        axarr[2].scatter(X_converged_criterion,
                         Y_converged_criterion,
                         marker='^',
                         c=C_converged_criterion,
                         alpha=0.5)
        axarr[2].scatter(X_converged_not_criterion,
                         Y_converged_not_criterion,
                         marker='o',
                         c=C_converged_not_criterion,
                         alpha=0.5)

        if xlim:
            axarr[2].set_xlim(xlim[0])
        if ylim:
            axarr[2].set_ylim(ylim[0])

        axarr[2].spines['top'].set_visible(False)
        axarr[2].spines['right'].set_visible(False)
        axarr[2].spines['bottom'].set_visible(False)
        axarr[2].xaxis.set_visible(False)
        axarr[2].set_title('Execution converged to the desired accuracy', position=(0.5,-0.2))
        
        #cax = f.add_axes([0.9, 0.1, 0.03, 0.8])
        #f.colorbar(scatter, cax = cax)
        f.suptitle("Influence of the bit flipped and the iteration when the fault occured \n on the convergence and the iteration count (color)", fontsize=14, fontweight='bold', position=(0.5, 0.02))
        plt.subplots_adjust(hspace = -0.26)

        if lines:
            if not_converged:
                X_line, Y_line_1, Y_line_2 = draw_lines(data_not_converged, c = c, epsilon = epsilon, normb = normb)
                X_smooth, Y_smooth_1, Y_smooth_2 = smooth_lines(X_line, Y_line_1, Y_line_2)
            
                axarr[0].plot(X_smooth, Y_smooth_1, c="orange")
                axarr[0].plot(X_smooth, Y_smooth_2, c="red")
            if converged:
                X_line, Y_line_1, Y_line_2 = draw_lines(data_converged, c = c, epsilon = epsilon, normb = normb)
                X_smooth, Y_smooth_1, Y_smooth_2 = smooth_lines(X_line, Y_line_1, Y_line_2)
            
                axarr[2].plot(X_smooth, Y_smooth_1, c="orange")
                axarr[2].plot(X_smooth, Y_smooth_2, c="red")
        
    else:
        if converged:
            if criterion:
                plt.scatter(X_converged_criterion,
                            Y_converged_criterion,
                            marker='^',
                            c=C_converged_criterion,
                            alpha=0.5)
            if not_criterion:
                plt.scatter(X_converged_not_criterion,
                            Y_converged_not_criterion,
                            marker='o',
                            c=C_converged_not_criterion,
                            alpha=0.5)
            if lines:
                X_line, Y_line_1, Y_line_2 = draw_lines(data_converged, c = c, epsilon = epsilon, normb = normb)
                X_smooth, Y_smooth_1, Y_smooth_2 = smooth_lines(X_line,
                                                                Y_line_1,
                                                                Y_line_2)
            
                plt.plot(X_smooth, Y_smooth_1, c="orange")
                plt.plot(X_smooth, Y_smooth_2, c="red")
            plt.xlim(xlim)
            
        if not_converged:
            if criterion:
                plt.scatter(X_not_converged_criterion,
                            Y_not_converged_criterion,
                            marker='^',
                            c=C_not_converged_criterion,
                            alpha=0.5)
            if not_criterion:
                plt.scatter(X_not_converged_not_criterion,
                            Y_not_converged_not_criterion,
                            marker='o',
                            c=C_not_converged_not_criterion,
                            alpha=0.5)

            if lines:
                X_line, Y_line_1, Y_line_2 = draw_lines(data_not_converged, c = c, epsilon = epsilon, normb = normb)
                X_smooth, Y_smooth_1, Y_smooth_2 = smooth_lines(X_line,
                                                                Y_line_1,
                                                                Y_line_2)
            
                plt.plot(X_smooth, Y_smooth_1, c="orange")
                plt.plot(X_smooth, Y_smooth_2, c="red")
            plt.xlim(xlim)
        plt.show()

Display_gmres = Display()
Display_gmres.set_display("convergence_bit_iteration",
                          convergence_bit_iteration)
