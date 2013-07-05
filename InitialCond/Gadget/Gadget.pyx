cimport Gadget as g

import  InitialCond.Types as Types
cimport InitialCond.Types as Types

from libc.stdlib cimport free

cdef class Gadget:
	#cdef Types.Particules part
	#cdef g.Header header

	def __cinit__(self, filename):
		cdef unsigned int i
		self.filename = filename
		
		for i in range(6):
			self.header.npart[i]              = 0
			self.header.mass[i]               = 0.
			self.header.npartTotalHighWord[i] = 0

		self.header.time                   = 0.
		self.header.num_files              = 1
		self.header.redshift               = 0.
		self.header.flag_sfr               = 0
		self.header.flag_feedback          = 0
		self.header.flag_cooling           = 0
		self.header.BoxSize                = 0.
		self.header.Omega0                 = 0.
		self.header.OmegaLambda            = 0.
		self.header.HubbleParam            = 0.
		self.header.flag_stellarage        = 0
		self.header.flag_metals            = 0
		self.header.flag_entropy_instead_u = 0
	#cpdef Set_particules(self, Types.Particules part)
		#self.part = part

	#def __dealloc__(self):
		#free(self.filename)

	def __repr__(self):
		return  """Gadget file '{0}'. Header informations are:
Simulation parameters:
	Time       : {1}
	num_files  : {2}
	redshift   : {3}
	BoxSize    : {4}
	Particules : {5}
	Masses     : {6}
Physical parameters:
	Omega0      : {7}
	OmegaLambda : {8}
	HubbleParam : {9}
Simulation flags:
	flag_sfr               : {10}
	flag_feedback          : {11}
	flag_cooling           : {12}
	flag_stellarage        : {13}
	flag_metals            : {14}
	flag_entropy_instead_u : {15}
The only gadget file format supported is the gadget 1.
			""".format(
				self.filename,
				self.header.time,
				self.header.num_files,
				self.header.redshift,
				self.header.BoxSize,
				self.npart,
				self.mass,
				self.header.Omega0,
				self.header.OmegaLambda,
				self.header.HubbleParam,
				self.header.flag_sfr,
				self.header.flag_feedback,
				self.header.flag_cooling,
				self.header.flag_stellarage,
				self.header.flag_metals,
				self.header.flag_entropy_instead_u,
			)

	def __str__(self):
		return self.__repr__()

	cpdef int Write(self):
		cdef int res
		cdef unsigned int i
		cdef char *fname = <bytes>self.filename.encode()
		if self.part.ptr_data == NULL:
			raise MemoryError("Particules array not allocate.")
		for i in range(6):
			self.header.npartTotal[i] = self.header.npart[i]
		res = g.Gadget_Write(fname, self.header, self.part.ptr_data)
		return res

	cpdef Read(self, int num_files, bint bpot=0, bint bacc=0, bint bdadt=0, bint bdt=0):
		cdef int N = 0
		cdef unsigned int i
		cdef char *fname = <bytes>self.filename.encode()
		cdef Types.Particule part

		part = g.Gadget_Read(fname, &self.header, num_files, bpot, bacc, bdadt, bdt)
		if part is NULL:
			raise MemoryError
		
		for i in range(6):
			N += self.header.npart[i]
		self.part = Types.FromPointer(part, N)

	property Part:
		def __get__(self):
			if self.part.ptr_data is not NULL:
				return self.part
			else:
				return None
		def __set__(self, value):
			if isinstance(value, Types.Particules):
				self.part = value
			else:
				raise TypeError("You must passed a InitialCond.Types.Particules!")

	property npartTotalHighWord:
		def __get__(self):
			res = [0]*6
			for i in range(6):
				res[i] = self.header.npartTotalHighWord[i]
			return res
		def __set__(self, value):
			if len(value) != 6:
				raise ValueError("You should past a list of 6 integers!")
			for i in range(6):
				self.header.npartTotalHighWord[i] = value[i]

	property npart:
		def __get__(self):
			res = [0]*6
			for i in range(6):
				res[i] = self.header.npart[i]
			return res
		def __set__(self, value):
			if len(value) != 6:
				raise ValueError("You should past a list of 6 integers!")
			for i in range(6):
				self.header.npart[i] = value[i]

	property mass:
		def __get__(self):
			res = [0]*6
			for i in range(6):
				res[i] = self.header.mass[i]
			return res
		def __set__(self, value):
			if len(value) != 6:
				raise ValueError("You should past a list of 6 floats!")
			for i in range(6):
				self.header.mass[i] = value[i]

	property time:
		def __get__(self):
			return self.header.time
		def __set__(self, value):
			self.header.time = value

	property redshift:
		def __get__(self):
			return self.header.redshift
		def __set__(self, value):
			self.header.redshift = value

	property flag_sfr:
		def __get__(self):
			return self.header.flag_sfr
		def __set__(self, value):
			self.header.flag_sfr = value

	property flag_feedback:
		def __get__(self):
			return self.header.flag_feedback
		def __set__(self, value):
			self.header.flag_feedback = value

	property flag_cooling:
		def __get__(self):
			return self.header.flag_cooling
		def __set__(self, value):
			self.header.flag_cooling = value

	property BoxSize:
		def __get__(self):
			return self.header.BoxSize
		def __set__(self, value):
			self.header.BoxSize = value

	property Omega0:
		def __get__(self):
			return self.header.Omega0
		def __set__(self, value):
			self.header.Omega0 = value

	property OmegaLambda:
		def __get__(self):
			return self.header.OmegaLambda
		def __set__(self, value):
			self.header.OmegaLambda = value

	property HubbleParam:
		def __get__(self):
			return self.header.HubbleParam
		def __set__(self, value):
			self.header.HubbleParam = value

	property flag_stellarage:
		def __get__(self):
			return self.header.flag_stellarage
		def __set__(self, value):
			self.header.flag_stellarage = value

	property flag_metals:
		def __get__(self):
			return self.header.flag_metals
		def __set__(self, value):
			self.header.flag_metals = value

	property flag_entropy_instead_u:
		def __get__(self):
			return self.header.flag_entropy_instead_u
		def __set__(self, value):
			self.header.flag_entropy_instead_u = value

