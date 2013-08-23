#include "generation.h"

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
	       xm   = 0.0,
	       ym   = 0.0,
	       zm   = 0.0;
	int i;

	pos = double2d(NbPart, 3);

	for (i = 0; i < NbPart; i++)
	{
		xx = pow(ran2(seed), 1.0/3.0) * rmax;	// Rayon
		yy = 2.0*M_PI * ran2(seed);		// Phi entre 0, 2\pi
		zz = acos(1. - 2.0*ran2(seed));		// Theta entre 0, \pi

		pos[i][0] = xx * sin(zz) * cos(yy);
		pos[i][1] = xx * sin(zz) * sin(yy);
		pos[i][2] = xx * cos(zz);

		xm = xm + pos[i][0];
		ym = ym + pos[i][1];
		zm = zm + pos[i][2];

		//xx = 1.0-2.0*ran2(seed);
		//yy = 1.0-2.0*ran2(seed);
		//zz = 1.0-2.0*ran2(seed);
		//rr = sqrt(xx*xx + yy*yy + zz*zz);
		//if(rr >=  1.0)
		//{
			//i--;
			//continue;
		//}
		//pos[i][0] = xx*rmax/2.0;
		//pos[i][1] = yy*rmax/2.0;
		//pos[i][2] = zz*rmax/2.0;
		//d++;
	}

	//xm = 0.0;
	//ym = 0.0;
	//zm = 0.0;

	//for (i = 0; i < NbPart; i++)
	//{
		//xm = xm + pos[i][0];
		//ym = ym + pos[i][1];
		//zm = zm + pos[i][2];
	//}

	xm = xm/(float)(NbPart);
	ym = ym/(float)(NbPart);
	zm = zm/(float)(NbPart);

	for (i = 0; i < NbPart; i++)
	{
		pos[i][0] = pos[i][0] - xm;
		pos[i][1] = pos[i][1] - xm;
		pos[i][2] = pos[i][2] - xm;
	}

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
		}
		else
			rejet++;

	}
	fputs("\n", stderr);

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

static inline double Fuji_distrib(const double r, const double u, const double j, const double sig_v, const double rho_0)
{
	return rho_0 * pow( 2.*M_PI*sig_v*sig_v, -3.0/2.0) * exp(- (u*u + j*j/(r*r)) / (2.*sig_v*sig_v) );
}

void Fuji_Generate(const int Nb_part_t1, const double r_max, const double v_max, const double sig_v, const double rho_0, Particule king, double *r_grand, long *seed)
{
	unsigned int i = 0,
		     k = 0;
	unsigned int rejet = 0;
	//double normal = Fuji_distrib(0.1, 0., 0., sig_v, rho_0);
	double normal = 4./3. * M_PI * r_max * r_max * r_max; //Fuji_distrib(0.1, 0., 0., sig_v, rho_0);
	double v, r, j, u;
	double x = 0.,
	       y = 0.,
	       z = 0.,
	       vx = 0.,
	       vy = 0.,
	       vz = 0.;

	while( i < Nb_part_t1 )
	{
		for(k=0; k<3; k++)
			king[i].Pos[k] = ran2(seed) * r_max;
		for(k=0; k<3; k++)
			king[i].Vit[k] = ran2(seed) * v_max;

		v = sqrt(king[i].Vit[0]*king[i].Vit[0] + king[i].Vit[1]*king[i].Vit[1] + king[i].Vit[2]*king[i].Vit[2]);
		r = sqrt(king[i].Pos[0]*king[i].Pos[0] + king[i].Pos[1]*king[i].Pos[1] + king[i].Pos[2]*king[i].Pos[2]);
		j = sqrt(	  (king[i].Vit[1] * king[i].Pos[2] - king[i].Pos[1] * king[i].Vit[2])*(king[i].Vit[1] * king[i].Pos[2] - king[i].Pos[1] * king[i].Vit[2])
				+ (king[i].Pos[0] * king[i].Vit[2] - king[i].Pos[2]*king[i].Vit[0])*(king[i].Pos[0] * king[i].Vit[2] - king[i].Pos[2]*king[i].Vit[0])
				+ (king[i].Pos[1] * king[i].Vit[0] - king[i].Pos[0] * king[i].Vit[1])*(king[i].Pos[1] * king[i].Vit[0] - king[i].Pos[0] * king[i].Vit[1])
			);
		u = ( king[i].Pos[0]*king[i].Vit[0] + king[i].Pos[1]*king[i].Vit[1] + king[i].Pos[2]*king[i].Vit[2] ) / r;

		if( r <= r_max && v <= v_max && ran2(seed) <= Fuji_distrib(r, u, j, sig_v, rho_0)/normal )
		{
			if( *r_grand < r )
				*r_grand = r;

			x		+= king[i].Pos[0];
			y		+= king[i].Pos[1];
			z		+= king[i].Pos[2];
			vx		+= king[i].Vit[0];
			vy		+= king[i].Vit[1];
			vz		+= king[i].Vit[2];

			i++;
		}
		else
			rejet++;

	}

	x  /= Nb_part_t1;
	y  /= Nb_part_t1;
	z  /= Nb_part_t1;
	vx /= Nb_part_t1;
	vy /= Nb_part_t1;
	vz /= Nb_part_t1;

	for(int j = 0; j < Nb_part_t1; j++)
	{
		king[j].Pos[0] -= x;
		king[j].Pos[1] -= y;
		king[j].Pos[2] -= z;
	}
}
