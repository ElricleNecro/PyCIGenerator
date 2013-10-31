cdef extern from "IOGadget/types.h":
	cdef struct _particule_data:
		float Pos[3]
		float Vit[3]
		float m
		float Pot
		float Acc[3]
		float dAdt
		float ts
		float Rho
		float U
		float Ne
		int Id
		int Type
	ctypedef _particule_data* Particule

cdef extern from "types.h":
	void Echange(Particule a, Particule b)
	Particule Concat(const Particule a, const int Na, const Particule b, const int Nb)
	void sort_by_id(Particule tab, const int N)

cdef Particules FromPointer(Particule p, int N)
cdef Particules Single(_particule_data p)
cpdef FromPyData(lst, colType=?, colm=?, colId=?)

cdef class Particules:
	cdef Particule ptr_data
	cdef readonly int N
	cdef bint b_potential, b_acceleration, b_rate_entropy, b_timestep
	cdef set_data(self, Particule p, int N)
	cdef _translate(self, double x, double y, double z)
	cpdef Particules Add(self, Particules b)
	cpdef SortById(self)

