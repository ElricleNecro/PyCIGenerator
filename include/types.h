#ifndef TYPES_H_GENE

#define TYPES_H_GENE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/**
 * \struct Part
 * Contient toutes les informations relatives à une particule.
 * Sert pour le Tree-Code, mais aussi pour d'autre traitement.
 */
struct _part {
	unsigned int id;		/*< Identifiant de la particule.*/
	double x;			/*< Abscisse de la particule.*/
	double y;			/*< Ordonnée de la particule.*/
	double z;			/*< Côte de la particule.*/
	double r;			/*< Distance de la particule à un objet (ie : centre de l'amas, particule, ...).*/
	double vx;			/*< Vitesse selon x de la particule.*/
	double vy;			/*< Vitesse selon y de la particule.*/
	double vz;			/*< Vitesse selon z de la particule.*/
	double v;			/*< Vitesse de la particule.*/
	double m;			/*< Masse de la particule.*/
};
typedef struct _part Part;

typedef struct _particule_data {
	double Pos[3];
	double Vit[3];
	double m;
	int Id;
	int Type;
}*Particule;

/**
 * Fonction s'occupant d'allouer un tableau de type Part.
 * @param n Taille du tableau.
 * @return Adresse du premier élément du tableau.
 */
Part* Part1d(const int n) __attribute__ ((__const__));

/**
 * Fonction additionnant 2 particules.
 * @param a Particule a.
 * @param b Particule b.
 * @return La somme de a et b.
 */
Part  Part_add(Part a, Part b) __attribute ((__const__));

/**
 * Fonction libérant le tableau alloué par \ref Part1d.
 * @param *ptf Adresse du premier élément du tableau à désallouer;
 */
void Part1d_libere(Part *ptf);

/**
 * Fonction adaptée à qsort pour le trie des tableaux de particules selon la distance à (0, 0, 0).
 * @param *a Premier objet à comparer.
 * @param *b Second objet à comparer.
 * @return -1 si a < b, 0 si a == b, 1 si a > b.
 */
int    qsort_partstr(const void *a, const void *b) __attribute__ ((__const__));

/**
 * Fonction adaptée à qsort pour le trie des tableaux de particules selon l'axe x.
 * @param *a Premier objet à comparer.
 * @param *b Second objet à comparer.
 * @return -1 si a < b, 0 si a == b, 1 si a > b.
 */
int    qsort_partaxe(const void *a, const void *b) __attribute__ ((__const__));

/**
 * Fonction se chargeant d'échanger 2 tableaux entre eux. Utilisé dans la fonction de construction de l'arbre.
 * @param *a Pointeur sur le premier élément du tableau ligne à échanger
 * @param *b Pointeur sur le premier élément du tableau ligne avec qui échanger
 */
void   Echange(Particule a, Particule b);

/**
 * Fonction concaténant 2 tableaux de type Particule et retourne le nouveau tableau.
 * @param a Premier tableau à concaténer.
 * @param Na Taille du premier tableau.
 * @param b Second tableau à concaténer.
 * @param Nb Taille du second tableau.
 * @return NULL si l'allocation échoue, le nouveau pointeur sinon.
 */
Particule Concat(const Particule a, const int Na, const Particule b, const int Nb);

#endif /* end of include guard: TYPES_H */
