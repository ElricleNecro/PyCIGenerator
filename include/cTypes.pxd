# cython: language_level=3
cdef extern from "types.h":
	cdef struct _particule_data:
		double Pos[3]
		double Vit[3]
		double m
		int Id
		int Type
	ctypedef _particule_data* Particule

#cdef class Particules:
	#cdef Particule ptr_data
	#cdef readonly int N

#cdef Particules FromPointer(Particule p, int N)
#cdef Particules Single(_particule_data p)
#cpdef FromPyData(lst, colType, colm, colId)


