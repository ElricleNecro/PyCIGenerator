import numpy as np
cimport numpy as np

cimport cython

cimport King_Bind as kb
cimport cGeneration as gb
#cimport King      as kg

from libc.stdlib cimport malloc, free

cimport cTypes
import  cTypes

cdef class King:
	cdef kb.King _obj
	cdef int N
	cdef double r_grand
	cdef cTypes.Particule part

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

		self.part = <cTypes._particule_data *>malloc( self.N * sizeof(cTypes._particule_data))

		gb.King_Generate(self._obj, self.N, &self.r_grand, self.part, &seed)

		return seed

	cdef cTypes.Particules _get_part(self):
		#ret = cTypes.Particules()
		#ret.set_data(self.part, self.N)
		cdef cTypes.Particules ret
		ret = cTypes.FromPointer(self.part, self.N)
		return ret

	property Part:
		def __get__(self):
			if self.part is not NULL:
				return self._get_part()

cdef class Object:
	cdef cTypes.Particule part
	cdef int N

	def __cinit__(self, int N):
		self.N    = N
		if self.N <= 0:
			raise ValueError("N cannot be negative or null.")
		self.part = <cTypes._particule_data *>malloc( self.N * sizeof(cTypes._particule_data))
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

