class Parameters():
	"Class representing and manipulating parameters data"
	def __init__(self, dictionnary={}):
		self._data = dictionnary

	def __setitem__(self, key, value):
		self._data[key] = value
                
	def __getitem__(self, key):
		if key in self._data:
			return self._data[key]
		else:
			return None
	def get(self, key, default):
		if key in self._data:
			return self._data[key]
		else:
			return default
        def __iter__(self):
                return self._data.__iter__()
        def next(self):
                return self._data.next()
	def __str__(self):
		return str(self._data)

	def copy(self):
		return Parameters(self._data.copy())

	def update(self, parameters):
		self._data.update(parameters._data)
		
	# TODO: xml parsing
