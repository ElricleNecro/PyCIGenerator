cimport King_Bind as kb
from cTypes cimport Particule

cdef extern from "generation.h":
	double** carree_homo(const double rmax, const int NbPart, long *seed)

	double** sphere_homo(const double rmax, const int NbPart, long *seed)

	double** gauss(const double sig, const int NbPart, long *seed)

	double** gauss_limit(const double sig, const double broke, const int NbPart, long *seed)

	void	 King_gene(const kb.King Amas, const int Nb_part_t1, double *r_grand, double **king_pos, double **king_vit, long *seed)
	void	 King_Generate(const kb.King Amas, const int Nb_part_t1, double *r_grand, Particule king, long *seed)

	void	 Homo_Generate(const double rmax, const double vmax, const double m, const double WVir, const int NbPart, Particule res, long *seed)
	void	 HomoGauss_Generate(const double rmax, const double sig, const double m, const double WVir, const int NbPart, Particule res, long *seed)
	void	 HomoGaussLimited_Generate(const double rmax, const double sig, const double broke, const double m, const double WVir, const int NbPart, Particule res, long *seed)

