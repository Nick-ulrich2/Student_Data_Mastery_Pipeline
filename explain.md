# JOUR 3: ANALYSE DU DATASET 

## Etape 1: AUDIT DE LA QUALITE DES DONNEES

Elle consiste a repondre a ces questions:

1) Combien de lignes ?
2) Combien et quelles colonnes ?
3) Quels types ? 
4) Combien de valeurs manquantes ? 
5) Combien et quels doublons ? 
6) Quelles catégories (Numerique ou Categorielle) ? 
7) Quelle plage pour la cible ?


## Etape 2 : RESOLUTION DES PROBLEMES LIES AUX DONNEES

1) Presence des valeurs manquantes (Je connais 3 methodes de gestion de ces valeurs)
- Remplacement de NaN par le mode (j'applique celle ci car la NaN s'eleve a !.9%)
- Creation d'une nouvelle categorie si la variable est categorielle 
- Suppression des lignes contenant des NaN

2) Maximun de la colonne Exam_Score a 101 au lieu de 100

## Etape 3: Configuration de notre projet 

cette etape consiste a creer un fichier (pour nous "configurations/config.py") qui contiendra des details de sur notre dataset

## Etape 4: Statistiques descriptives et première visualisation

Elles consiste a lancer une description sur notre cible (Exam_Score) puis de faire un graphique. 
De la description il en ressort que :
- La moyenne du dataset est environ 67,24 
- La médiane 67 
- La moitié centrale des scores se situe entre 65 et 69.
Pour le graphique il faut importer:
- matplotlib.pyplot as plt
- seaborn as sns

## Etape 5: Matrice de correlation
La matrice ne se fait qu'avec des variables numeriques.
Etant donne que notre dataset a des numeriques et des categorielles, nous pouvons:
- Soit faire cette matrice avec les variables numeriques uniquement 
- Soit encoder les variables categorielles avant de faire la correlation

