from libc.stdlib cimport malloc, free
from cTypes cimport _particule_data, Particule

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

cdef class Particules:
	#cdef Particule ptr_data
	#cdef readonly int N

	cdef set_data(self, Particule p, int N):
		self.N        = N
		self.ptr_data = p

	def __dealloc__(self):
		if self.ptr_data is not NULL:
			free(self.ptr_data)

	FromPointer = staticmethod(FromPointer)
	Single      = staticmethod(Single)
	FromPyData  = staticmethod(FromPyData)

cdef Particules FromPointer(Particule p, int N):
	tmp = Particules()

	tmp.set_data(p, N)

	return tmp

cdef Particules Single(_particule_data p):
	cdef Particule tmp = NULL
	res = Particules()

	tmp = <_particule_data *>malloc( sizeof(_particule_data))
	if tmp is NULL:
		raise MemoryError()

	tmp[0] = p

	res.set_data(tmp, 1)

	return res

cpdef FromPyData(lst, colType=None, colm=None, colId=None):
	cdef Particule tmp = NULL

	tmp = <_particule_data *>malloc(len(lst)* sizeof(_particule_data))
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
