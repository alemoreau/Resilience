class Algorithm():
	"General algorithm"
	def __init__(self, parameters, implementation=None):
		self.parameters = parameters
		self.data = {}
		self.result = None
                self.implementation = implementation

	def check_parameters(self):
		pass
	def check_input(self):
		pass
	def check_implementation(self):
		pass

        def set_implementation(self, implementation):
		"""
		Set the algorithm implementation
		implementation : function
			args : (self, input, algorithm_parameters, experiment_parameters)
			return : result
		"""
                self.implementation = implementation
        
	def implementation(self, parameters, save_data=True):
                pass
                #if self.implementation == None:
                #        pass
                #else:
                #        self.implementation
	
	def run(self, input, experiment_parameters=None):
		""" 
		Run the algorithm on the given input with 
		the algorithm parameters and experiment parameters
		"""
		if (self.implementation):
			self.result = self.implementation(self, input, self.parameters, experiment_parameters)
		else:
			print "Use set_implementation to set the algorithm implementation, or extend the Algorithm class"
		return self.result

	def get_data(self):
		"""
		Return recorded data 
		"""
		return self.data

	def get_result(self):
		""""
		Return algorithm result
		"""
		return self.result
