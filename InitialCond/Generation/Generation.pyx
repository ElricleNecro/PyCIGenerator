cimport cython
import  numpy as np
cimport numpy as np

from libc.stdlib cimport malloc, free

cimport Generation as gb

import  King #as toto
cimport King #as kb

import  InitialCond.Types as Types
cimport InitialCond.Types as Types

cdef class GKing(King.KingModel):
	cdef double r_grand
	cdef Types.Particule part
	def __dealloc__(self):
		super().__dealloc__()
		#if self._obj.don is not NULL:
			#kb.King_free(&self._obj)
		if self.part is not NULL:
			free(self.part)

	cpdef long Generate(self, long seed):
		if self.N == 0:
			raise ValueError("You should call setM or SolveAll before calling this method.")

		self.part = <Types._particule_data *>malloc( self.N * sizeof(Types._particule_data))

		gb.King_Generate(self._obj, self.N, &self.r_grand, self.part, &seed)

		return seed

	cdef Types.Particules _get_part(self):
		#ret = cTypes.Particules()
		#ret.set_data(self.part, self.N)
		cdef Types.Particules ret
		ret = Types.FromPointer(self.part, self.N)
		return ret

	property Part:
		def __get__(self):
			if self.part is not NULL:
				return self._get_part()

cdef class Object:
	cdef Types.Particule part
	cdef int N

	def __cinit__(self, int N):
		self.N    = N
		if self.N <= 0:
			raise ValueError("N cannot be negative or null.")
		self.part = <Types._particule_data *>malloc( self.N * sizeof(Types._particule_data))
		if self.part is NULL:
			raise MemoryError()

	def __dealloc__(self):
		if self.part is not NULL:
			free(self.part)


cdef class Sphere(Object):
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

