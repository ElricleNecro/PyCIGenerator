import numpy as np
cimport numpy as np

cimport cython

cimport King_Bind as kb
cimport Gene_Bind as gb
#cimport King      as kg

from libc.stdlib cimport malloc, free

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

cdef class King:
	cdef kb.King _obj
	cdef int N
	cdef double r_grand
	cdef gb.Particule part

	def __cinit__(self, double w0, double rc, double sv, double G = 6.67e-11):
		kb.King_SetG(G)
		self.part = NULL
		self._obj = kb.King_New(w0, rc, sv)
		if self._obj.don is not NULL:
			raise MemoryError()

	@cython.boundscheck(False)
	cdef np.ndarray _get_data(self):

		cdef np.ndarray res = np.zeros([self._obj.lig, self._obj.col], dtype=np.float64)
		cdef i, j

		for i in range(0, self._obj.lig):
			for j in range(0, self._obj.col):
				res[i, j] = self._obj.don[i][j]
		return res

	property data:
		def __get__(self):
			if self._obj.don is not NULL:
				return self._get_data()

	property Mtot:
		def __get__(self):
			return self._obj.amas.Mtot

	property rho0:
		def __get__(self):
			return self._obj.amas.rho0

	property rmax:
		def __get__(self):
			return self._obj.amas.rmax

	property vmax:
		def __get__(self):
			return self._obj.amas.vmax

	property El:
		def __get__(self):
			return self._obj.amas.El

	property m:
		def __get__(self):
			return self._obj.amas.m

	property W0:
		def __get__(self):
			return self._obj.amas.W0

	property sigma2:
		def __get__(self):
			return self._obj.amas.sigma2

	property rc:
		def __get__(self):
			return self._obj.amas.rc

	property N:
		def __get__(self):
			return self.N

	cpdef Solve(self):
		kb.King_gud(&self._obj)

	cpdef SolveAll(self, int N):
		kb.King_gud      ( &self._obj    )
		kb.King_CalcR    ( &self._obj    )
		kb.King_CalcRho  ( &self._obj    )
		kb.King_CalcMtot ( &self._obj    )
		self.N = N
		kb.King_SetM     ( &self._obj, N )
		kb.King_CalcSig2 ( &self._obj    )
		kb.King_CalcEl   ( &self._obj    )
		kb.King_CalcPot  ( &self._obj    )
		kb.King_CalcDPot ( &self._obj    )
		kb.King_CalcMu   ( &self._obj    )
		kb.King_CalcVMax ( &self._obj    )

	cpdef CalcR(self):
		kb.King_CalcR(&self._obj)

	cpdef CalcRho(self):
		kb.King_CalcRho(&self._obj)

	cpdef CalcMtot(self):
		kb.King_CalcMtot(&self._obj)

	cpdef SetM(self, int N):
		self.N = N
		kb.King_SetM(&self._obj, N)

	cpdef CalcSig2(self):
		kb.King_CalcSig2(&self._obj)

	cpdef CalcEl(self):
		kb.King_CalcEl(&self._obj)

	cpdef CalcPot(self):
		kb.King_CalcPot(&self._obj)

	cpdef CalcDPot(self):
		kb.King_CalcDPot(&self._obj)

	cpdef CalcMu(self):
		kb.King_CalcMu(&self._obj)

	cpdef CalcVMax(self):
		kb.King_CalcVMax(&self._obj)

	def __dealloc__(self):
		if self._obj.don is not NULL:
			kb.King_free(&self._obj)
		if self.part is not NULL:
			free(self.part)

	cpdef long Generate(self, long seed):
		if self.N == 0:
			raise ValueError("You should call setM or SolveAll before calling this method.")

		self.part = <gb._particule_data *>malloc( self.N * sizeof(gb._particule_data))

		gb.King_Generate(self._obj, self.N, &self.r_grand, self.part, &seed)

		return seed

	property Particule:
		def __get__(self):
			if self.part is not NULL:
				ret = Array1DWrapper()
				ret.set_data(self.N, self.part)
				return ret

cdef class Object:
	cdef gb.Particule part
	cdef int N

	def __cinit__(self, int N):
		self.N    = N
		if self.N <= 0:
			raise ValueError("N cannot be negative or null.")
		self.part = <gb._particule_data *>malloc( self.N * sizeof(gb._particule_data))
		if self.part is NULL:
			raise MemoryError()

	def __dealloc__(self):
		if self.part is not NULL:
			free(self.part)


cdef class Sphere:
	#gauss = staticmethod(gauss)
	#homo  = staticmethod(homo)

	@cython.boundscheck(False)
	cpdef homo(self, double r_m, long seed, pos=True):
		cdef double **res
		cdef unsigned int i, j

		res = gb.sphere_homo(r_m, self.N, &seed)

		if pos:
			for i in range(self.N):
				for j in range(3):
					self.part[i].Pos[j]
		else:
			for i in range(self.N):
				for j in range(3):
					self.part[i].Vit[j]

		free(res[0])
		free(res)

		return seed

	@cython.boundscheck(False)
	cpdef gauss(self, double sig, long seed, pos=False):
		cdef double **res
		cdef unsigned int i, j

		res = gb.gauss(sig, self.N, &seed)

		if pos:
			for i in range(self.N):
				for j in range(3):
					self.part[i].Pos[j]
		else:
			for i in range(self.N):
				for j in range(3):
					self.part[i].Vit[j]

		free(res[0])
		free(res)

		return seed

	@cython.boundscheck(False)
	cpdef gauss_limited(self, double sig, double broke, long seed, pos=False):
		cdef double **res
		cdef unsigned int i, j

		res = gb.gauss_limit(sig, broke, self.N, &seed)

		if pos:
			for i in range(self.N):
				for j in range(3):
					self.part[i].Pos[j]
		else:
			for i in range(self.N):
				for j in range(3):
					self.part[i].Vit[j]

		free(res[0])
		free(res)

		return seed

#, pos=False, Type=0, Id=True, m=0.

	#ret = Array2DWrapper()
	#size[0] = Nb
	#size[1] = 3
	#ret.set_data(size, res)

	#return ret

	#part = <gb._particule_data *>malloc( self.N * sizeof(gb._particule_data))

