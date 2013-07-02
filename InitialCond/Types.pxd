cdef extern from "types.h":
	cdef struct _particule_data:
		double Pos[3]
		double Vit[3]
		double m
		int Id
		int Type
	ctypedef _particule_data* Particule
	void Echange(Particule a, Particule b)
	Particule Concat(const Particule a, const int Na, const Particule b, const int Nb)

cdef Particules FromPointer(Particule p, int N)
cdef Particules Single(_particule_data p)
cpdef FromPyData(lst, colType=?, colm=?, colId=?)

cdef class Particules:
	cdef Particule ptr_data
	cdef readonly int N
	cdef set_data(self, Particule p, int N)
	cpdef Particules Add(self, Particules b)

