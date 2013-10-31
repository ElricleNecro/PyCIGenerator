cimport King
from InitialCond.Types cimport Particule

cdef extern from "generation.h":
	double** carree_smooth(const double rmax, const double smoothing, const int NbPart, long *seed) nogil
	double** carree_homo(const double rmax, const int NbPart, long *seed) nogil

	double** sphere_smooth(const double rmax, const double smoothing, const int NbPart, long *seed) nogil
	double** sphere_homo(const double rmax, const int NbPart, long *seed) nogil

	double** gauss(const double sig, const int NbPart, long *seed) nogil

	double** gauss_limit(const double sig, const double broke, const int NbPart, long *seed) nogil

	void	 King_gene(const King.King Amas, const int Nb_part_t1, double *r_grand, double **king_pos, double **king_vit, long *seed) nogil
	void	 King_Generate(const King.King Amas, const int Nb_part_t1, double *r_grand, Particule king, long *seed) nogil

	void	 Homo_Generate(const double rmax, const double vmax, const double m, const double WVir, const int NbPart, Particule res, long *seed) nogil
	void	 HomoGauss_Generate(const double rmax, const double sig, const double m, const double WVir, const int NbPart, Particule res, long *seed) nogil
	void	 HomoGaussLimited_Generate(const double rmax, const double sig, const double broke, const double m, const double WVir, const int NbPart, Particule res, long *seed) nogil

	void Fuji_Generate(const int Nb_part_t1, const double r_max, const double v_max, const double sig_v, const double rho_0, Particule king, double *r_grand, long *seed) nogil
