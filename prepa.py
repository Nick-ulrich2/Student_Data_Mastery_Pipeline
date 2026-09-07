# importons les librairies necessaires 
import numpy as np
import pandas as pd
from config import *
import matplotlib.pyplot as plt
import seaborn as sns
from themes import configure_matplotlib_theme, theme_chart_colors
from translations import t

# importation du dataset 
SPP = pd.read_csv("data/SPP.csv")

# 1) Combien de lignes et de colonnes ?
SPP.shape

# 2) Quelles colonnes avons nous ?
SPP.columns

# 3) Quels sont les types de mes colonnes ? 
SPP.dtypes

# 4) Pourcentage des valeurs manquantes ? 
SPP.isnull().mean()*100

# 5) Combien et quels doublons ? 
SPP.duplicated().sum()

# 7) Quelle plage pour la cible ? La cible est Exam_Score
# print(f"Le score minimal est {SPP['Exam_Score'].min()}\n\nLe score maximal {SPP['Exam_Score'].max()}")


# RESOLUTION DES PROBLEMES 

# # Remplacement des NaN (valeurs manquantes) par le mode 
# missing_values_cols = ["Distance_from_Home","Parental_Involvement"]
# for col in missing_values_cols:
#     SPP.loc[SPP[col].isnull() == True, col] = SPP[col].mode()[0]
    # print(f"Nombre de valeurs manquantes dans '{col}': {SPP[col].isnull().sum()}") 
    
# le max etant a 101, mettons le a 100
SPP.loc[SPP[TARGET] > 100, TARGET] = 100
# print(SPP['Exam_Score'].max())

DATA = SPP

# print(DATA.describe())

# Distribution de la cible en fonction du nombre total des lignes du dataset 
configure_matplotlib_theme()
chart_colors = theme_chart_colors()
sns.histplot(DATA[TARGET], bins=100, kde=True, color=chart_colors["blue"])
plt.title(t("Distribution des notes d'examens en fonction de la quantite", "Exam score distribution by quantity"))
plt.xlabel(t("Notes d'examen", "Exam score"))
plt.ylabel(t("Total des lignes", "Total rows"))
# plt.savefig("Graph/target_distribution.png", dpi=160)
# plt.show()

# Graphe de la matrice de correlation
plt.figure(figsize=(10,8))
sns.heatmap(
    DATA[NUMERIC_FEATURES].corr(),
    annot=True,
    annot_kws={"color": chart_colors["text"]},
    cmap=chart_colors["heatmap_cmap"],
    mask=np.triu(DATA[NUMERIC_FEATURES].corr()),
)
plt.title(t("Matrice de correlation", "Correlation matrix"))
# plt.savefig("Graph/corr.png")
# plt.show()