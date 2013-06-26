from Gene_Bind cimport _particule_data, Particule

cdef extern from "tree.h":
	cdef struct _tnoeud:
		int    N
		int    level
		double x, y, z
		double cote
		Particule first
		_particule_data cm
		double CM
		_tnoeud* parent
		_tnoeud* frere
		_tnoeud* fils
	
	ctypedef _tnoeud* TNoeud

	void Tree_SetG(double nG)
	double Tree_GetG()
	TNoeud Create_Tree(Particule posvits, const int NbParticule, const int NbMin, const _particule_data center, const double taille)
	double Tree_CalcPot(TNoeud root, const Particule part, const double accept, const double soft)

