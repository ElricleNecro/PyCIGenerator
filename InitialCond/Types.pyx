from libc.stdlib cimport malloc, free
#from Types cimport _particule_data, Particule

cimport Types

cdef class Array2DWrapper:
	"""Get from : http://gael-varoquaux.info/blog/?p=157
	"""
	cdef void* data_ptr
	cdef int size[2]

	cdef set_data(self, int size[2], void* data_ptr):
		""" Set the data of the array
		This cannot be done in the constructor as it must recieve C-level
		arguments.

		Parameters:
		-----------
		size: int
		Length of the array.
		data_ptr: void*
		Pointer to the data            
		"""
		self.data_ptr = data_ptr
		self.size[0] = size[0]
		self.size[1] = size[1]

cdef class Array1DWrapper:
	"""Get from : http://gael-varoquaux.info/blog/?p=157
	"""
	cdef void* data_ptr
	cdef int size

	cdef set_data(self, int size, void* data_ptr):
		""" Set the data of the array
		This cannot be done in the constructor as it must recieve C-level
		arguments.

		Parameters:
		-----------
		size: int
		Length of the array.
		data_ptr: void*
		Pointer to the data            
		"""
		self.data_ptr = data_ptr
		self.size = size

	#def __array__(self):
		#""" Here we use the __array__ method, that is called when numpy
		#tries to get an array from the object."""
		#cdef np.npy_intp shape[2]
		#shape[0] = <np.npy_intp> self.size
		## Create a 1D array, of length 'size'
		#ndarray = np.PyArray_SimpleNewFromData(2, shape,

		#return ndarray

cdef Particules FromPointer(Types.Particule p, int N):
	tmp = Particules()

	tmp.set_data(p, N)

	return tmp

cdef Particules Single(Types._particule_data p):
	cdef Types.Particule tmp = NULL
	res = Particules()

	tmp = <Types._particule_data *>malloc( sizeof(Types._particule_data))
	if tmp is NULL:
		raise MemoryError()

	tmp[0] = p

	res.set_data(tmp, 1)

	return res

cpdef FromPyData(lst, colType=None, colm=None, colId=None):
	cdef Types.Particule tmp = NULL

	tmp = <Types._particule_data *>malloc(len(lst)* sizeof(Types._particule_data))
	if tmp is NULL:
		raise MemoryError()

	for i, a in enumerate(lst):
		for j, b in enumerate(a):
			if j < 3:
				tmp[i].Pos[j] = float(b)
			elif j < 6:
				tmp[i].Vit[j-3] = float(b)
			elif colType is not None and j == colType:
				tmp[i].Type = int(b)
			elif colm is not None and j == colm:
				tmp[i].m = float(b)
			if colId is not None and colId == j:
				tmp[i].Id = int(b)
			else:
				tmp[i].Id = i

	res = Particules()
	res.set_data(tmp, len(lst))

	return res

cdef class Particules:
	cdef set_data(self, Particule p, int N):
		self.N        = N
		self.ptr_data = p

	def __dealloc__(self):
		if self.ptr_data is not NULL:
			free(self.ptr_data)
			self.ptr_data = NULL

	cpdef Types.Particules Add(self, Types.Particules p):
		res = FromPointer( Types.Concat(self.ptr_data, self.N, p.ptr_data, p.N), self.N + p.N )
		if res.ptr_data is NULL:
			raise MemoryError
		return res

	def __add__(self, p):
		return self.Add(p)

	property Velocities:
		def __get__(self):
			cdef unsigned int i, j
			cdef list res, tmp

			res = list()
			for i in range(self.N):
				tmp = list()
				for j in range(3):
					tmp.append(self.ptr_data[i].Vit[j])
				res[i].append(tmp)

			return res

	property Positions:
		def __get__(self):
			cdef unsigned int i, j
			cdef list res, tmp

			res = list()
			for i in range(self.N):
				tmp = list()
				for j in range(3):
					tmp.append(self.ptr_data[i].Pos[j])
				res.append(tmp)

			return res

	def __repr__(self):
		ret = "<Particules Object %p"%id(self)
		if self.N > 0 or self.ptr_data is not NULL:
			ret += " from id=%d"%self.ptr_data[0].Id + " to %d>"%self.ptr_data[self.N-1].Id
		else:
			ret += " uninitialized>"
		return ret

	def __str__(self):
		return self.__repr__()

	#FromPointer = staticmethod(FromPointer)
	#Single      = staticmethod(Single)
	#FromPyData  = staticmethod(FromPyData)

