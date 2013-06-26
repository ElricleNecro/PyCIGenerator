#include "generation.h"

void Homo_Generate(const double rmax, const double vmax, const double m, const double WVir, const int NbPart, Particule res, long *seed)
{
	printf("Viriel Voulu : \033[31m%g\033[00m\n", WVir);
	double **tmp = NULL;
	Particule tmp2 = NULL;
	tmp2 = (Particule) malloc(NbPart * sizeof(struct _particule_data));

	tmp = sphere_homo(2.0*rmax, NbPart, seed);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Pos[0] = tmp[i][0];
		tmp2[i].Pos[1] = tmp[i][1];
		tmp2[i].Pos[2] = tmp[i][2];
		tmp2[i].m = m;
		tmp2[i].Id = i;
	}
	double2d_libere(tmp), tmp=NULL;

	tmp = sphere_homo(2.0*vmax, NbPart, seed);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Vit[0] = tmp[i][0];
		tmp2[i].Vit[1] = tmp[i][1];
		tmp2[i].Vit[2] = tmp[i][2];
	}
	double2d_libere(tmp), tmp=NULL;

	TNoeud root = Create_Tree(tmp2, NbPart, 15, (struct _particule_data){ .Pos[0] = 0.0, .Pos[1]= 0.0, .Pos[2]=0.0}, 100.0 * rmax);

	double v = 0.0, pot = 0.0;
	for(int i = 0; i < NbPart; i++)
	{
		pot += m*Tree_CalcPot(root, &tmp2[i], 0.5, 0.0);
		v   += 0.5 * m * ( tmp2[i].Vit[0]*tmp2[i].Vit[0] + tmp2[i].Vit[1]*tmp2[i].Vit[1] + tmp2[i].Vit[2]*tmp2[i].Vit[2] );
	}
	pot /= 2.0;
	double vir = 2*v/pot;
	double fact = WVir/vir;

	printf("Viriel de l'objet central : \033[36m%g\033[00m (%g, %g)\n", vir, pot, v);
	printf("facteur : \033[31m%g\033[00m\n", fact);

	printf("%g %g %g\n", tmp2[NbPart-1].Vit[0], tmp2[NbPart-1].Vit[1], tmp2[NbPart-1].Vit[2]);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Vit[0] *= sqrt(fact);
		tmp2[i].Vit[1] *= sqrt(fact);
		tmp2[i].Vit[2] *= sqrt(fact);
	}
	printf("%g %g %g\n", tmp2[NbPart-1].Vit[0], tmp2[NbPart-1].Vit[1], tmp2[NbPart-1].Vit[2]);

	v = 0.0, pot = 0.0;
	for(int i = 0; i < NbPart; i++)
	{
		pot += m*Tree_CalcPot(root, &tmp2[i], 0.5, 0.0);
		v   += 0.5*m*(tmp2[i].Vit[0]*tmp2[i].Vit[0] + tmp2[i].Vit[1]*tmp2[i].Vit[1] + tmp2[i].Vit[2]*tmp2[i].Vit[2]);
	}
	pot/=2.0;
	printf("After, New Viriel : \033[36m%g\033[00m (%g, %g)\n", 2.0*v/pot, pot, v);
	Tree_Free(root);

	for(int i = 0; i < NbPart; i++)
	{
		res[i].Pos[0] = tmp2[i].Pos[0];
		res[i].Pos[1] = tmp2[i].Pos[1];
		res[i].Pos[2] = tmp2[i].Pos[2];

		res[i].Vit[0] = tmp2[i].Vit[0]; // * sqrt(fact);
		res[i].Vit[1] = tmp2[i].Vit[1]; // * sqrt(fact);
		res[i].Vit[2] = tmp2[i].Vit[2]; // * sqrt(fact);

		res[i].m      = m;
	}
	free(tmp2);
}

void HomoGauss_Generate(const double rmax, const double sig, const double m, const double WVir, const int NbPart, Particule res, long *seed)
{
	printf("Viriel Voulu : \033[31m%g\033[00m\n", WVir);
	double **tmp = NULL;
	Particule tmp2 = NULL;
	tmp2 = (Particule) malloc(NbPart * sizeof(struct _particule_data));

	tmp = sphere_homo(2.0*rmax, NbPart, seed);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Pos[0] = tmp[i][0];
		tmp2[i].Pos[1] = tmp[i][1];
		tmp2[i].Pos[2] = tmp[i][2];
		tmp2[i].m = m;
		tmp2[i].Id = i;
	}
	double2d_libere(tmp), tmp=NULL;

	tmp = gauss(sig, NbPart, seed);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Vit[0] = tmp[i][0];
		tmp2[i].Vit[1] = tmp[i][1];
		tmp2[i].Vit[2] = tmp[i][2];
	}
	double2d_libere(tmp), tmp=NULL;

	TNoeud root = Create_Tree(tmp2, NbPart, 15, (struct _particule_data){ .Pos[0] = 0.0, .Pos[1]= 0.0, .Pos[2]=0.0}, 100.0 * rmax);

	double v = 0.0, pot = 0.0;
	for(int i = 0; i < NbPart; i++)
	{
		pot += m*Tree_CalcPot(root, &tmp2[i], 0.5, 0.0);
		v   += 0.5 * m * ( tmp2[i].Vit[0]*tmp2[i].Vit[0] + tmp2[i].Vit[1]*tmp2[i].Vit[1] + tmp2[i].Vit[2]*tmp2[i].Vit[2] );
	}
	pot /= 2.0;
	double vir = 2*v/pot;
	double fact = WVir/vir;

	printf("Viriel de l'objet central : \033[36m%g\033[00m (%g, %g)\n", vir, pot, v);
	printf("facteur : \033[31m%g\033[00m\n", fact);

	//printf("%g %g %g\n", tmp2[NbPart-1].Vit[0], tmp2[NbPart-1].Vit[1], tmp2[NbPart-1].Vit[2]);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Vit[0] *= sqrt(fact);
		tmp2[i].Vit[1] *= sqrt(fact);
		tmp2[i].Vit[2] *= sqrt(fact);
	}
	//printf("%g %g %g\n", tmp2[NbPart-1].Vit[0], tmp2[NbPart-1].Vit[1], tmp2[NbPart-1].Vit[2]);

	v = 0.0, pot = 0.0;
	for(int i = 0; i < NbPart; i++)
	{
		pot += m*Tree_CalcPot(root, &tmp2[i], 0.5, 0.0);
		v   += 0.5*m*(tmp2[i].Vit[0]*tmp2[i].Vit[0] + tmp2[i].Vit[1]*tmp2[i].Vit[1] + tmp2[i].Vit[2]*tmp2[i].Vit[2]);
	}
	pot/=2.0;
	printf("After, New Viriel : \033[36m%g\033[00m (%g, %g)\n", 2.0*v/pot, pot, v);
	Tree_Free(root);

	for(int i = 0; i < NbPart; i++)
	{
		res[i].Pos[0] = tmp2[i].Pos[0];
		res[i].Pos[1] = tmp2[i].Pos[1];
		res[i].Pos[2] = tmp2[i].Pos[2];

		res[i].Vit[0] = tmp2[i].Vit[0]; // * sqrt(fact);
		res[i].Vit[1] = tmp2[i].Vit[1]; // * sqrt(fact);
		res[i].Vit[2] = tmp2[i].Vit[2]; // * sqrt(fact);

		res[i].m      = m;
	}

	free(tmp2);
}

void HomoGaussLimited_Generate(const double rmax, const double sig, const double broke, const double m, const double WVir, const int NbPart, Particule res, long *seed)
{
	printf("Viriel Voulu : \033[31m%g\033[00m\n", WVir);
	double **tmp = NULL;
	Particule tmp2 = NULL;
	tmp2 = (Particule) malloc(NbPart * sizeof(struct _particule_data));

	tmp = sphere_homo(2.0*rmax, NbPart, seed);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Pos[0] = tmp[i][0];
		tmp2[i].Pos[1] = tmp[i][1];
		tmp2[i].Pos[2] = tmp[i][2];
		tmp2[i].m = m;
		tmp2[i].Id = i;
	}
	double2d_libere(tmp), tmp=NULL;

	tmp = gauss_limit(sig, broke, NbPart, seed);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Vit[0] = tmp[i][0];
		tmp2[i].Vit[1] = tmp[i][1];
		tmp2[i].Vit[2] = tmp[i][2];
	}
	double2d_libere(tmp), tmp=NULL;

	TNoeud root = Create_Tree(tmp2, NbPart, 15, (struct _particule_data){ .Pos[0] = 0.0, .Pos[1]= 0.0, .Pos[2]=0.0}, 100.0 * rmax);

	double v = 0.0, pot = 0.0;
	for(int i = 0; i < NbPart; i++)
	{
		pot += m*Tree_CalcPot(root, &tmp2[i], 0.5, 0.0);
		v   += 0.5 * m * ( tmp2[i].Vit[0]*tmp2[i].Vit[0] + tmp2[i].Vit[1]*tmp2[i].Vit[1] + tmp2[i].Vit[2]*tmp2[i].Vit[2] );
	}
	pot /= 2.0;
	double vir = 2*v/pot;
	double fact = WVir/vir;

	printf("Viriel de l'objet central : \033[36m%g\033[00m (%g, %g)\n", vir, pot, v);
	printf("facteur : \033[31m%g\033[00m\n", fact);

	printf("%g %g %g\n", tmp2[NbPart-1].Vit[0], tmp2[NbPart-1].Vit[1], tmp2[NbPart-1].Vit[2]);
	for(int i = 0; i < NbPart; i++)
	{
		tmp2[i].Vit[0] *= sqrt(fact);
		tmp2[i].Vit[1] *= sqrt(fact);
		tmp2[i].Vit[2] *= sqrt(fact);
	}
	printf("%g %g %g\n", tmp2[NbPart-1].Vit[0], tmp2[NbPart-1].Vit[1], tmp2[NbPart-1].Vit[2]);

	v = 0.0, pot = 0.0;
	for(int i = 0; i < NbPart; i++)
	{
		pot += m*Tree_CalcPot(root, &tmp2[i], 0.5, 0.0);
		v   += 0.5*m*(tmp2[i].Vit[0]*tmp2[i].Vit[0] + tmp2[i].Vit[1]*tmp2[i].Vit[1] + tmp2[i].Vit[2]*tmp2[i].Vit[2]);
	}
	pot/=2.0;
	printf("After, New Viriel : \033[36m%g\033[00m (%g, %g)\n", 2.0*v/pot, pot, v);
	Tree_Free(root);

	for(int i = 0; i < NbPart; i++)
	{
		res[i].Pos[0] = tmp2[i].Pos[0];
		res[i].Pos[1] = tmp2[i].Pos[1];
		res[i].Pos[2] = tmp2[i].Pos[2];

		res[i].Vit[0] = tmp2[i].Vit[0]; // * sqrt(fact);
		res[i].Vit[1] = tmp2[i].Vit[1]; // * sqrt(fact);
		res[i].Vit[2] = tmp2[i].Vit[2]; // * sqrt(fact);

		res[i].m      = m;
	}
	free(tmp2);
}

double** carree_homo(const double rmax, const int NbPart, long *seed)
{
	if( NbPart == 0 )
		return NULL;

	//long   seed = -32;
	double **pos= NULL;
	double xx   = 0.0,
	       yy   = 0.0,
	       zz   = 0.0,
	       xm   = 0.0,
	       ym   = 0.0,
	       zm   = 0.0;
	       //pi   = 4.0*atan(1.0);

	pos = double2d(NbPart, 3);

//	FILE *fich = NULL;
//	fich = fopen("Seed_verif", "w");
	for (int i = 0; i < NbPart; i++)
	{

		xx = 1.0-2.0*ran2(seed);
//		fprintf(fich, "%ld\t", seed);
		yy = 1.0-2.0*ran2(seed);
//		fprintf(fich, "%ld\t", seed);
		zz = 1.0-2.0*ran2(seed);
//		fprintf(fich, "%ld\n", seed);
		pos[i][0] = xx*rmax/2.0;
		pos[i][1] = yy*rmax/2.0;
		pos[i][2] = zz*rmax/2.0;
	}
//	fclose(fich);

	xm = 0.0;
	ym = 0.0;
	zm = 0.0;

	for (int i = 0; i < NbPart; i++)
	{
		xm = xm + pos[i][0];
		ym = ym + pos[i][1];
		zm = zm + pos[i][2];
	}

	xm = xm/(float)(NbPart);
	ym = ym/(float)(NbPart);
	zm = zm/(float)(NbPart);

	for (int i = 0; i < NbPart; i++)
	{
		pos[i][0] = pos[i][0] - xm;
		pos[i][1] = pos[i][1] - xm;
		pos[i][2] = pos[i][2] - xm;
	}
//	fich=fopen("Stat_homo.res", "w");
//	for (int i = 0; i < NbPart; i++) {
//		for (int j = 0; j < 3; j++) {
//			fprintf(fich, "%g\t", pos[i][j]);
//		}
//		fprintf(fich, "\n");
//	}
//	fclose(fich);

	return pos;
}

double** sphere_homo(const double rmax, const int NbPart, long *seed)
{
	if( NbPart == 0 )
		return NULL;

	double **pos= NULL;
	double xx   = 0.0,
	       yy   = 0.0,
	       zz   = 0.0,
	       rr   = 0.0,
	       xm   = 0.0,
	       ym   = 0.0,
	       zm   = 0.0;
	int i, d;

	pos = double2d(NbPart, 3);

//	FILE *fich = NULL;
//	fich = fopen("Seed_verif", "w");
	for (i = 0, d=0; i < NbPart; i++)
	{

		xx = 1.0-2.0*ran2(seed);
//		fprintf(fich, "%ld\t", seed);
		yy = 1.0-2.0*ran2(seed);
//		fprintf(fich, "%ld\t", seed);
		zz = 1.0-2.0*ran2(seed);
//		fprintf(fich, "%ld\n", seed);
		rr = sqrt(xx*xx + yy*yy + zz*zz);
		if(rr >=  1.0)
		{
			i--;
			continue;
		}
		pos[i][0] = xx*rmax/2.0;
		pos[i][1] = yy*rmax/2.0;
		pos[i][2] = zz*rmax/2.0;
		d++;
	}
//	fclose(fich);

	xm = 0.0;
	ym = 0.0;
	zm = 0.0;

	for (i = 0; i < NbPart; i++)
	{
		xm = xm + pos[i][0];
		ym = ym + pos[i][1];
		zm = zm + pos[i][2];
	}

	xm = xm/(float)(NbPart);
	ym = ym/(float)(NbPart);
	zm = zm/(float)(NbPart);

	for (i = 0; i < NbPart; i++)
	{
		pos[i][0] = pos[i][0] - xm;
		pos[i][1] = pos[i][1] - xm;
		pos[i][2] = pos[i][2] - xm;
	}
//	fich=fopen("Stat_homo.res", "w");
//	for (int i = 0; i < NbPart; i++) {
//		for (int j = 0; j < 3; j++) {
//			fprintf(fich, "%g\t", pos[i][j]);
//		}
//		fprintf(fich, "\n");
//	}
//	fclose(fich);

	return pos;
}

double** gauss_limit(const double sig, const double broke, const int NbPart, long *seed)
{
	if( NbPart == 0 )
		return NULL;

	double **vit = NULL;
	vit = double2d(NbPart, 3);

	for (int i = 0; i < NbPart; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			do
			{
				vit[i][j] = sqrt(-2.0*sig*log(ran2(seed)))*cos(2.0*acos(-1.0)*ran2(seed));
			} while(vit[i][j] >= broke*sig);
		}
	}

	return vit;
}

double** gauss(const double sig, const int NbPart, long *seed)
{
	if( NbPart == 0 )
		return NULL;

	double **vit = NULL;
	vit = double2d(NbPart, 3);

	for (int i = 0; i < NbPart; i++)
	{
		for (int j = 0; j < 3; j++)
		{
			vit[i][j] = sqrt(-2.0*sig*log(ran2(seed)))*cos(2.0*acos(-1.0)*ran2(seed));
		}
	}

	return vit;
}

void King_gene(const King Amas, const int Nb_part_t1, double *r_grand, double **king_pos, double **king_vit, long *seed)
{
	double normal    = King_distrib(&Amas, Amas.amas.m * King_don_pot(&Amas, 0));
	//long   seed      = -32;
	int    rejet     = 0,
	       i	 = 0;

	double x, y, z, vx, vy, vz;
	x = y = z = vx = vy = vz = 0.0;

        printf("Normalisation de f(E) = %g\n\n", normal);

	*r_grand = 0.0;

	fprintf(stderr, "\r\033[01m%03.3f%%\033[00m", ( (double)i ) / ( (double) Nb_part_t1 ) * 100.0 );
	while( i < Nb_part_t1 )
	{
		Coord tmp;

		tmp.x    = -Amas.amas.rmax + 2.0*Amas.amas.rmax*ran2(seed);
		tmp.y    = -Amas.amas.rmax + 2.0*Amas.amas.rmax*ran2(seed);
		tmp.z    = -Amas.amas.rmax + 2.0*Amas.amas.rmax*ran2(seed);
		tmp.vx   = -Amas.amas.vmax + 2.0*Amas.amas.vmax*ran2(seed);
		tmp.vy   = -Amas.amas.vmax + 2.0*Amas.amas.vmax*ran2(seed);
		tmp.vz   = -Amas.amas.vmax + 2.0*Amas.amas.vmax*ran2(seed);

		double v = sqrt(tmp.vx*tmp.vx + tmp.vy*tmp.vy + tmp.vz*tmp.vz),
		       r = sqrt(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);

		double E = Amas.amas.m * v*v /2.0 + Amas.amas.m * King_don_pot(&Amas, r);

		if( r <= Amas.amas.rmax && v <= Amas.amas.vmax && ran2(seed) <= King_distrib(&Amas, E)/normal )
		{
			if( *r_grand < r )
				*r_grand = r;
			king_pos[i][0]  = tmp.x;
			king_pos[i][1]  = tmp.y;
			king_pos[i][2]  = tmp.z;
			king_vit[i][0]  = tmp.vx;
			king_vit[i][1]  = tmp.vy;
			king_vit[i][2]  = tmp.vz;
			x+=tmp.x;
			y+=tmp.y;
			z+=tmp.z;
			vx+=tmp.vx;
			vy+=tmp.vy;
			vz+=tmp.vz;
			i++;
			fprintf(stderr, "\r\033[01m%03.3f%%\033[00m", ( (double)i ) / ( (double) Nb_part_t1 ) * 100.0 );
		}
		else
			rejet++;
	}
	fputs("\n", stderr);

	x/=Nb_part_t1;
	y/=Nb_part_t1;
	z/=Nb_part_t1;
	vx/=Nb_part_t1;
	vy/=Nb_part_t1;
	vz/=Nb_part_t1;

	printf("%g\t%g\t%g\n%g\t%g\t%g\n", x, y, z, vx, vy, vz);

	printf("\033[36mPour le King :\n\tParticule accepté : %d\n\tParticule rejeté : %d\n\t%g %% particules rejetées et %g %% d'accepté.\n\tRayon de l'amas %g (%g)\n\tParametre de Softening conseillé : %g pc (%g)\n\033[00m",
		i, rejet, (float)(rejet)/(float)(rejet + i)*100.0, (float)(i)/(float)(rejet + i)*100.0, *r_grand, Amas.amas.rmax, 2.0 * (*r_grand  / 3.086e16) / pow(Nb_part_t1, 1.0/3.0) * 0.05, 2.0 * (*r_grand  / 3.086e16) / pow(Nb_part_t1, 1.0/3.0) * 0.05);
}

void King_Generate(const King Amas, const int Nb_part_t1, double *r_grand, Particule king, long *seed)
{
	double normal    = King_distrib(&Amas, Amas.amas.m * King_don_pot(&Amas, 0));
	//long   seed      = -32;
	long   rejet     = 0;
	int    i	 = 0;

	double x, y, z, vx, vy, vz;
	x = y = z = vx = vy = vz = 0.0;

        printf("Normalisation de f(E) = %g\n\n", normal);

	*r_grand = 0.0;

#ifdef SEED_WRITING
	FILE *seed_file = NULL;
	long total = 0;
	seed_file = fopen("seed.dat", "w");
	/*if( seed_file != NULL && total % SEED_FREQ == 0  )*/ fprintf(seed_file, "%ld\n", *seed);
#endif
	fprintf(stderr, "\r\033[01m%03.3f%%\033[00m", ( (double)i ) / ( (double) Nb_part_t1 ) * 100.0 );
	while( i < Nb_part_t1 )
	{
		Coord tmp;

		tmp.x    = -Amas.amas.rmax + 2.0*Amas.amas.rmax*ran2(seed);
//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif
		tmp.y    = -Amas.amas.rmax + 2.0*Amas.amas.rmax*ran2(seed);
//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif
		tmp.z    = -Amas.amas.rmax + 2.0*Amas.amas.rmax*ran2(seed);
//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif
		tmp.vx   = -Amas.amas.vmax + 2.0*Amas.amas.vmax*ran2(seed);
//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif
		tmp.vy   = -Amas.amas.vmax + 2.0*Amas.amas.vmax*ran2(seed);
//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif
		tmp.vz   = -Amas.amas.vmax + 2.0*Amas.amas.vmax*ran2(seed);
//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif

		double v = sqrt(tmp.vx*tmp.vx + tmp.vy*tmp.vy + tmp.vz*tmp.vz),
		       r = sqrt(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);

		double E = Amas.amas.m * v*v /2.0 + Amas.amas.m * King_don_pot(&Amas, r);

		if( r <= Amas.amas.rmax && v <= Amas.amas.vmax && ran2(seed) <= King_distrib(&Amas, E)/normal )
		{
			if( *r_grand < r )
				*r_grand = r;
			king[i].Pos[0]   = tmp.x;
			king[i].Pos[1]   = tmp.y;
			king[i].Pos[2]   = tmp.z;
			king[i].Vit[0]   = tmp.vx;
			king[i].Vit[1]   = tmp.vy;
			king[i].Vit[2]   = tmp.vz;

			x		+=tmp.x;
			y		+=tmp.y;
			z		+=tmp.z;
			vx		+=tmp.vx;
			vy		+=tmp.vy;
			vz		+=tmp.vz;

			i++;
			fprintf(stderr, "\r\033[01m%03.3f%%\033[00m", ( (double)i ) / ( (double) Nb_part_t1 ) * 100.0 );
#ifdef SEED_WRITING
			total++;
			if( total == NB_BY )
			{
				fprintf(seed_file, "%ld\n", *seed);
				total = 0;
			}
			//if( i % NB_BY == 0 ) fprintf(seed_file, "%ld\n", *seed);
#endif
		}
		else
			rejet++;

//#ifdef SEED_WRITING
		//total++;
		//if( seed_file != NULL && total % SEED_FREQ == 0  ) fprintf(seed_file, "%ld\n", *seed);
//#endif
	}
	fputs("\n", stderr);

#ifdef SEED_WRITING
	if( seed_file != NULL ) fclose(seed_file);
	//printf("\033[31mNb total de tirage : %ld\033[00m\n", total);
#endif

	x  /= Nb_part_t1;
	y  /= Nb_part_t1;
	z  /= Nb_part_t1;
	vx /= Nb_part_t1;
	vy /= Nb_part_t1;
	vz /= Nb_part_t1;

	printf("%g\t%g\t%g\n%g\t%g\t%g\n", x, y, z, vx, vy, vz);
	for(int j = 0; j < Nb_part_t1; j++)
	{
		king[j].Pos[0] -= x;
		king[j].Pos[1] -= y;
		king[j].Pos[2] -= z;
	}

	printf("\033[36mPour le King :\n\tParticule accepté : %d\n\tParticule rejeté : %ld\n\t%g %% particules rejetées et %g %% d'accepté.\n\tRayon de l'amas %g (%g)\n\tParametre de Softening conseillé : %g pc (%g)\n\033[00m",
		i, rejet, (float)(rejet)/(float)(rejet + i)*100.0, (float)(i)/(float)(rejet + i)*100.0, *r_grand, Amas.amas.rmax, 2.0 * (*r_grand  / 3.086e16) / pow(Nb_part_t1, 1.0/3.0) * 0.05, 2.0 * (*r_grand  / 3.086e16) / pow(Nb_part_t1, 1.0/3.0) * 0.05);
}

