cimport InitialCond.Types as Types

cdef extern from "gadget.h":
	cdef struct io_header:
		int		npart[6]
		double		mass[6]
		double		time
		double		redshift
		int		flag_sfr
		int		flag_feedback
		unsigned int	npartTotal[6]
		int		flag_cooling
		int		num_files
		double		BoxSize
		double		Omega0
		double		OmegaLambda
		double		HubbleParam
		int		flag_stellarage
		int		flag_metals
		unsigned int	npartTotalHighWord[6]
		int		flag_entropy_instead_u
		char		fill[60]
	ctypedef io_header Header

	#bool  write_gadget_file( const char *fname,
				 #const Particule part1, const double mass_t1, const int Nb_part_t1, const int index_t1,
				 #const Particule part2, const double mass_t2, const int Nb_part_t2, const int index_t2,
				 #const double BoxSize,
				 #const double LongFact,
				 #const double VitFact)

	#bool write_gadget_conf( const char *filename, const char *ci_file,
				#const double Tmax,
				#const int    periodic,
				#const double BS,
				#const double LongConv,
				#const double VitConv,
				#const double MConv,
				#const double Soft)
	
	Types.Particule Gadget_Read(const char *name, Header *header, int files, int b_potential, int b_acceleration, int b_rate_entropy, int b_timestep)
	int Gadget_Write(const char *name, const Header header, const Types.Particule part)

cdef class Gadget:
	cdef Types.Particules part
	cdef Header header
	cdef filename
	cpdef int Write(self)
	cpdef Read(self, int num_files, bint bpot=?, bint bacc=?, bint bdadt=?, bint bdt=?)

