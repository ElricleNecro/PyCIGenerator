#ifdef PERIODIC
void Tree_Voisin(TNoeud root, Part *Tab, int NbVois, const Part *part, const double BS)
#else
void Tree_Voisin(TNoeud root, Part *Tab, int NbVois, const Part *part)
#endif
{
#ifdef __DEBUG_CALCVOIS_TREECODE_P1__
	fprintf(stderr, "%s:: level -> %d ; indice -> %d\n", __func__, root->level, 0);
#endif
	/************************************************************************\
	 *	1ére étape : Vérifier qu'au moins une particule est candidate	*
	\************************************************************************/
#ifdef PERIODIC
	if( Tree_Dist(root, part, BS) > Tab[NbVois-1].r )
#else
	if( Tree_Dist(root, part) > Tab[NbVois-1].r )
#endif
	{
#ifdef __DEBUG_CALCVOIS_TREECODE_P2__
		fprintf(stderr, "\033[37m%d::Inutile de descendre plus bas (level : %d) !!!\033[00m\n", part->id, root->level);
#endif
		return ;
	}
#ifdef __DEBUG_CALCVOIS_TREECODE_P3__
	fprintf(stderr, "\033[37mDescendre plus bas (level : %d) !!!\033[00m\n", root->level);
#endif


	if(root->fils != NULL)
	{
		TNoeud t1 = root->fils;
		do
		{
#ifdef PERIODIC
			Tree_Voisin(t1, Tab, NbVois, part, BS);
#else
			Tree_Voisin(t1, Tab, NbVois, part);
#endif
			t1 = t1->frere;
		}
		while(t1 != NULL);
	}
	else if( root->N !=0 )
	{
#ifdef __DEBUG_CALCVOIS_TREECODE_P4__
		fprintf(stderr, "\033[35mParcours (level : %d) (%p, %d, %d)!!!\033[00m\n", root->level, root->first, root->N, NbVois);
#endif
#ifdef PERIODIC
		CalcVois(root->first, root->N, Tab, NbVois, part, BS);
#else
		CalcVois(root->first, root->N, Tab, NbVois, part);
#endif
	}

	return ;
}



#ifdef PERIODIC
void CalcVois(Part *insert, const int N, Part *Tab, const int NbVois, const Part *part, const double BS)
#else
void CalcVois(Part *insert, const int N, Part *Tab, const int NbVois, const Part *part)
#endif
{
#ifdef DOUBLE_BOUCLE
#	ifdef __DEBUG_CALCVOIS_TREECODE_P__
	fprintf(stderr, "\033[35m%s:: Vérification :: (%d, %d)\033[00m\n", __func__, N, NbVois);
#	endif
	Part *di = NULL;
	di       = Part1d(N);
	//Calcul des distances :
	for(int i=0; i<N; i++)
	{
#	ifdef PERIODIC
		di[i].r  = sqrt(  pow( (NEAREST( (insert[i].x - part->x), (BS/2.0), BS )), 2.0 ) +
				  pow( (NEAREST( (insert[i].y - part->y), (BS/2.0), BS )), 2.0 ) +
				  pow( (NEAREST( (insert[i].z - part->z), (BS/2.0), BS )), 2.0 )
			    );
#	else
		di[i].r  = sqrt(  pow( (insert[i].x - part->x), 2.0 ) +
				  pow( (insert[i].y - part->y), 2.0 ) +
				  pow( (insert[i].z - part->z), 2.0 )
			    );
#	endif
		di[i].id = insert[i].id;
#	ifdef __DEBUG_CALCVOIS_TREECODE_P__
		fprintf(stderr, "\033[36m%s::di :: %.16g (%.16g)\033[00m\n", __func__, di[i].r, Tab[NbVois - 1].r);
#	endif
		// Il faut conserver le tableau ordonné, ou on fait un qsort après la boucle :
#	ifndef USE_VOIS_QSORT
		for (int j = i-1/*N-2*/; j >= 0; j--)
		{
			if( di[j].r > di[j+1].r )
			{
				Echange(&di[j], &di[j+1]);
			}
			else
				break;
		}
#	endif
	}
#	ifdef USE_VOIS_QSORT
#		warning "Use of qsort in neighbourhood research : performance will be reduced."
	qsort(Tab, (size_t)NbVois, sizeof(Part), qsort_partstr);
#	endif

	for(int i=0; i<N; i++)
	{
		if( di[i].r < Tab[NbVois - 1].r && di[i].id != part->id && NotIn(di[i], Tab, NbVois) )
		{
			Tab[NbVois - 1].id = di[i].id;
			Tab[NbVois - 1].r  = di[i].r;
#	ifdef __DEBUG_CALCVOIS_TREECODE_P__
			fprintf(stderr, "\033[38m%s::selection :: %.16g\033[00m\n",
					__func__,
					Tab[NbVois - 1].r);
#	endif
			//On garde le tableau des voisins ordonné :
#	ifdef USE_VOIS_QSORT
#		warning "Use of qsort in neighbourhood research : performance will be reduced."
			qsort(Tab, (size_t)NbVois, sizeof(Part), qsort_partstr);
#	else
			for (int j = NbVois-2; j >= 0; j--)
			{
				if( Tab[j].r > Tab[j+1].r )
				{
					Echange(&Tab[j], &Tab[j+1]);
				}
				else
					break;
			}
#	endif
		}
		else if( di[i].r > Tab[NbVois - 1].r )
			break;
	}
	free(di);
#else
	double di = 0.0;
	for(int i = 0; i < N; i++)
	{
#	ifdef PERIODIC
		di  = sqrt(  pow( (NEAREST( (insert[i].x - part->x), (BS/2.0), BS )), 2.0 ) +
				  pow( (NEAREST( (insert[i].y - part->y), (BS/2.0), BS )), 2.0 ) +
				  pow( (NEAREST( (insert[i].z - part->z), (BS/2.0), BS )), 2.0 )
			    );
#	else
		di  = sqrt(  pow( (insert[i].x - part->x), 2.0 ) +
				  pow( (insert[i].y - part->y), 2.0 ) +
				  pow( (insert[i].z - part->z), 2.0 )
			    );
#	endif
		if( di < Tab[NbVois - 1].r && insert[i].id != part->id && NotIn(insert[i], Tab, NbVois) )
		{
			Tab[NbVois - 1].id = insert[i].id;
			Tab[NbVois - 1].r  = di;
			for (int j = NbVois-2; j >= 0; j--)
			{
				if( Tab[j].r > Tab[j+1].r )
				{
					Echange(&Tab[j], &Tab[j+1]);
				}
				else
					break;
			}
		}
	}
#endif
}

