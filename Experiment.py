from Parameters import *
import numpy as np
import types
import sqlite3
import io
import csv

class Experiment:
	"General experiment"
	def __init__(self, parameters = Parameters(), algorithm=None):
		self.parameters = parameters
		self.results = []
		self.inputs = []
		self.data = []
                self.seed = None
                self.algorithm = algorithm
		self.database = None

        def set_parameters(self, parameters):
                """
                parameters : Parameters object, or Parameters generator
                """
                self.parameters = parameters
                
	def set_inputs(self, inputs):
		""" 
		inputs : iterable (list, generator...) of dictionnaries representing inputs
			(ex: [{"A":A, "b": b,"x0":x0}]
			representing the equation to be solved Ax=b)
		"""
		self.inputs = inputs

        def set_algorithm(self, algorithm):
		"""
		algorithm : instance of Algorithm class to be used in the experiment
		"""
                self.algorithm = algorithm

	def apply_procedure(self, parameters, input):
		""" 
		Apply test procedure with given parameters on the given samples
		samples are deep copied in case the algorithm modifies them
		"""
                import copy
                input = copy.deepcopy(input)
                output = self.algorithm.run(input, parameters)
		self.results += [(input, output)]
                self.data += [copy.copy(self.algorithm.get_data())]

	def run(self, n=1, show_progress = False, seed=None):
		"""
		Run the experiment over all inputs, n times 
		"""
		if show_progress:
			from sys import stdout
                if seed:
                        self.seed = seed
                        np.random.seed(seed)
		progress = 0
		if (self.inputs):
			for i in xrange(n):
				if isinstance(self.parameters, types.GeneratorType):
					for (k, _input), _p in zip(enumerate(self.inputs), self.parameters):
                                                new_progress = int((100 * (i * len(self.inputs) + k)) / len(self.inputs * n))
                                                if (show_progress and (new_progress > progress)):
							stdout.write("\r%d %c " % (new_progress, '%'))
						        stdout.flush()
						progress = new_progress
						self.apply_procedure(_p, _input)
				else:
		
					for k, _input in enumerate(self.inputs):
						new_progress = int((100 * (i * len(self.inputs) + k)) / len(self.inputs * n))
						if (show_progress and (new_progress > progress)):
							stdout.write("\r%d %c " % (new_progress, '%'))
							stdout.flush()
						progress = new_progress
						self.apply_procedure(self.parameters, _input)
			if (show_progress):			
				stdout.write("\rComplete ! \n")
				stdout.flush()

		else :
			print "Use the set_inputs(inputs) method to add new test cases"

	
	def get_results(self):
		"""
		return results
		"""
		return self.results
	
	def clear_results(self):
		""" 
		Delete all results
		"""
		self.results = []
	
        def show_results(self):
		"""
		Show brut results 
		"""
                if (not self.results):
                        print "There are no results yet"
		else:
			print "Results are : "
			for result in self.results:
				self.show_result(result)

	def show_result(self, result):
		"""
		Show result
		"""
		print result

	def clear_data(self):
		self.data = []
	
	def get_data(self, key=None):
		if key:
			return map(lambda d: d[key], self.data)
		else:
			return self.data
				
	def save_data_database(self, database_filename):
	    self.database = database_filename
	    if len(self.data) > 0:
	    	conn = sqlite3.connect(database_filename, detect_types=sqlite3.PARSE_DECLTYPES)
	    	c = conn.cursor()

		# Converts np.array to TEXT when inserting
		sqlite3.register_adapter(np.ndarray, adapt_array)

		# Converts TEXT to np.array when selecting
		sqlite3.register_converter("array", convert_array)

		experiment = []
	    	for key in self.data[0]:
		    var = self.data[0][key]
		    if isinstance(var, ( int, long ) ):
			experiment += [(key, "integer")]
		    elif isinstance(var, float): 
			experiment += [(key, "real")]
		    elif isinstance(var, np.ndarray):
			experiment += [(key, "array")]
		    else:
			experiment += [(key, "text")]

		sorted(experiment, key=lambda data: data[0])
		
		sql = "CREATE TABLE experiments ("
		for (key, t) in experiment:
		    sql += str(key) + " " + t + ", "
		sql = sql[:-2]
		sql += ")"
            	# Create table
            	c.execute(sql)

            	# Insert a row of data
            	#c.execute("INSERT INTO stocks VALUES ('2006-01-05','BUY','RHAT',100,35.14)")

            	# Save (commit) the changes
            	conn.commit()

            	# We can also close the connection if we are done with it.
            	# Just be sure any changes have been committed or they will be lost.

            	conn.close()

	def save_data_file(self, filename):
    	    f=open(filename, "wb")
    	    w = csv.writer(f)
	    for experiment in self.data:
    	    	for key, val in experiment.items():
        	    w.writerow([key, val])
    	    f.close()
     
	def read_data_file(self, filename):
    	    f=open(fn,'rb')
    	    dict_rap={}
     
    	    for key, val in csv.reader(f):
        	dict_rap[key]=eval(val)
    	    f.close()
    	    return(dict_rap)

def adapt_array(arr):
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)


 



