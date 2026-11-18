<p align="center">
  <a href="README.md"><b>English</b></a>
  &nbsp;·&nbsp;
  <a href="README.fr.md">Français</a>
  &nbsp;·&nbsp;
  <a href="README.es.md">Español</a>
</p>

# Triagegeist — Aide à la décision de triage aux urgences par prédiction de l’acuité

---

## Nom du projet et cadre

**Triagegeist** est le nom de la compétition Kaggle organisée par la **Laitinen-Fredriksson Foundation** (Helsinki), dédiée à la recherche en médecine d’urgence et en soins aigus. Le nom joue sur *Triage* (répartition des priorités aux urgences) et *Geist* (esprit, en allemand) : l’idée est d’explorer si un « esprit » algorithmique peut **soutenir** les décisions de triage sans remplacer le jugement clinique.

Ce dépôt contient une **solution baseline** : un modèle qui prédit le **niveau d’acuité du triage** (1 à 5) à partir des données d’admission, avec une démarche reproductible et documentée.

- **Compétition** : [Kaggle — Triagegeist](https://www.kaggle.com/competitions/triagegeist)  
- **Données** : jeu fourni dans le cadre de la compétition (ou compatibles avec le format décrit ci‑dessous).  
- **Livrable** : notebook exécutable + fichier de prédictions `submission.csv` + ce README.

---

## Ce que l’on résout (formulation du problème)

### Problème clinique

En salle d’urgence, l’infirmier ou le médecin de triage doit, en peu de temps et avec des informations partielles, attribuer un **niveau d’acuité** à chaque patient. Ce niveau détermine l’ordre de prise en charge et le type de filière (réanimation, box, attente). Les systèmes en vigueur (ESI, Manchester MTS, etc.) reposent sur des arbres décisionnels et le jugement humain. La **variabilité inter‑évaluateurs** et le **sous‑triage** (classer trop peu urgent un patient qui s’aggrave) sont documentés dans la littérature et ont un impact sur la sécurité des soins.

La question posée par Triagegeist est : **peut‑on utiliser l’IA pour aider à cette décision** (sans la prendre à la place du soignant) ?

### Ce que fait ce projet (et ce qu’il ne fait pas)

**Ce que l’on résout ici :**

- **Entrée** : pour chaque passage aux urgences, les données structurées d’admission disponibles au moment du triage : identifiants de site/infirmier, mode et horaire d’arrivée, âge, sexe, langue, assurance, origine du transport, localisation de la douleur, état de conscience, catégorie du motif (chief complaint), antécédents de recours et d’hospitalisation, médicaments, comorbidités, constantes vitales (TA, FC, FR, température, SpO₂, GCS, douleur, poids/taille/IMC), **shock_index** et **news2_score**.
- **Sortie** : une prédiction du **niveau d’acuité** codé 1 à 5 (1 = le plus urgent, 5 = le moins urgent), cohérente avec les conventions type ESI/MTS.
- **Objectif** : fournir une **estimation reproductible** de l’acuité à partir de ces seules variables, utilisable comme **aide à la décision** (alerte, cohérence, formation), pas comme décision automatique.

**Ce que l’on ne résout pas :**

- Pas de décision clinique finale : le modèle ne remplace pas le triage humain.
- Pas d’explication des prédictions (modèle boîte noire dans cette version).
- Pas de garantie de généralisation à d’autres hôpitaux ou pays sans ré‑évaluation.
- Pas d’utilisation du texte libre des motifs de recours ni d’autres sources non structurées dans ce baseline.

En résumé : on résout un **problème de prédiction supervisée ordinale** (acuité 1–5) à partir de données structurées d’admission, dans un cadre de compétition et de reproductibilité.

---

## Données

Tout est dans le dossier **`data/`** :

| Fichier | Description |
|--------|-------------|
| `train.csv` | ~80 000 passages avec **triage_acuity** (1–5) et variables d’admission |
| `test.csv` | ~20 000 passages sans étiquette (pour prédiction) |
| `chief_complaints.csv` | Motif de recours en texte libre + catégorie (optionnel pour NLP) |
| `patient_history.csv` | Antécédents binaires (comorbidités) par patient |
| `sample_submission.csv` | Exemple de format de soumission attendu |

**Variables utilisées (train/test)**  
Identifiants (site, infirmier), mode/heure/jour d’arrivée, âge, sexe, langue, assurance, origine du transport, localisation de la douleur, état de conscience au triage, catégorie du motif (chief_complaint_system), nombre de passages/hospitalisations antérieurs, médicaments, comorbidités, **constantes vitales** (TA, FC, FR, température, SpO₂, GCS, douleur, poids, taille, IMC), **shock_index**, **news2_score**.  
En train uniquement : disposition, durée de séjour, **triage_acuity** (cible).

---

## Démarche (pipeline)

1. **Chargement**  
   Lecture de `train`, `test`, `chief_complaints`, `patient_history`. Détection automatique de l’environnement (Kaggle vs local) pour les chemins des données.

2. **Exploration (EDA)**  
   - Taux de valeurs manquantes (surtout constantes dérivées : TA, shock_index, etc.).  
   - Lien entre **news2_score** et **triage_acuity** (plus le score est élevé, plus l’acuity est basse = plus urgent).

3. **Prétraitement**  
   - Colonnes exclues : `patient_id`, `triage_acuity`, `disposition`, `ed_los_hours`.  
   - Variables catégorielles : encodage par **LabelEncoder** (train) ; en test, modalités inconnues → `-1`.  
   - Variables numériques : valeurs manquantes imputées par la **médiane**.  
   - Toute colonne encore non numérique est convertie en numérique ; matrices finales en **float32** pour LightGBM.

4. **Modèle**  
   - **LightGBM** (LGBMClassifier) en classification multiclasse (classes 1–5), avec **StratifiedKFold** 5 folds.  
   - Si LightGBM absent : RandomForest en secours.  
   - Métrique : **Cohen’s Kappa quadratique** (pénalise davantage les erreurs d’un niveau qu’entre niveaux adjacents, adapté à une échelle ordinale).

5. **Sortie**  
   - Prédictions sur le test → entiers 1–5, bornés par `clip(1, 5)`.  
   - Fichier **`submission.csv`** : colonnes **`patient_id`** et **`triage_acuity`**, prêt pour soumission Kaggle.

---

## Performances obtenues — lecture critique et scientifique

### Métrique rapportée

- **Quadratic Cohen’s Kappa en validation croisée (5-fold stratifiée)** : **≈ 0,925**.

Le Kappa quadratique mesure l’accord entre prédictions et étiquettes en tenant compte du hasard, en pénalisant plus les erreurs « loin » sur l’échelle (ex. prédire 1 au lieu de 5) que les erreurs entre niveaux voisins (ex. 2 au lieu de 3). Une valeur de 0,925 place l’accord dans la zone dite « presque parfait » (convention Landis–Koch : > 0,81).

### Interprétation scientifique et limites

**Points positifs :**

- Bon accord moyen sur l’échelle ordinale en CV.
- Pipeline simple, reproductible, sans feature engineering avancé ni NLP.

**Points de vigilance (critique) :**

1. **Information redondante avec la cible**  
   La variable **news2_score** (National Early Warning Score 2) est un score de gravité déjà utilisé en clinique et fortement corrélé à la décision de triage. L’inclure comme feature peut faire prédire « la même chose que ce qui a servi à étiqueter » une partie des cas. Le Kappa élevé peut donc refléter en partie cette redondance plutôt qu’une vraie « compréhension » du triage à partir de signaux indépendants. Une analyse d’ablation (sans news2, sans shock_index) serait nécessaire pour quantifier leur contribution.

2. **Pas de validation externe**  
   Les performances sont estimées par cross‑validation sur le même jeu d’entraînement. Il n’y a pas de test sur un autre hôpital, une autre période ou un autre pays. La **généralisation** n’est pas démontrée.

3. **Biais et représentativité**  
   Si les données sont synthétiques ou issues d’un nombre limité de sites, la distribution (démographie, pratiques de triage, mix pathologique) peut ne pas représenter la diversité réelle. Les performances pourraient être optimistes sur ce jeu et se dégrader ailleurs.

4. **Cible = décision humaine**  
   La variable **triage_acuity** est la décision du trieur (étiquette). Le modèle apprend à **reproduire** cette décision, pas à prédire un critère objectif (ex. décès, admission en réa, détérioration). Donc on évalue la **fidélité à la pratique étiquetée**, pas la justesse clinique absolue.

5. **Imputation et fuite potentielle**  
   Les valeurs manquantes sont imputées par la médiane (globale ou par colonne). Si les manquants ne sont pas aléatoires (MCAR), l’imputation peut introduire un biais ; une analyse de sensibilité serait souhaitable.

**Conclusion sur les performances :**  
Un Kappa CV de 0,925 est **très bon sur ce jeu de données** pour un baseline structuré, mais il doit être interprété avec prudence : possible sur-estimation liée à news2/shock_index, absence de validation externe, et cible = jugement humain, pas outcome clinique dur. Pour un usage réel, il faudrait des évaluations complémentaires (ablation, validation externe, analyse des erreurs par niveau d’acuité et par sous‑groupes).

---

## Format de sortie

**Fichier généré** : `submission.csv` (dans `data/` en local, ou `/kaggle/working/` sur Kaggle).

**Colonnes :**

- **patient_id** : identifiant du passage (comme dans `test.csv`).  
- **triage_acuity** : prédiction du niveau d’acuité (1 = le plus urgent, 5 = le moins urgent), entier entre 1 et 5.

**Exemple (10 premières lignes) :**

```
     patient_id  triage_acuity
0  TG-FZUFCRZS3              2
1  TG-SSCOXTYI1              3
2  TG-JY74ZR35D              2
3  TG-JDKD5G62X              5
4  TG-J1BSAAXR0              5
5  TG-5YERUAA5X              4
6  TG-IVU18ZJCF              2
7  TG-K5XYMWTCG              2
8  TG-RCAR2TLEI              4
9  TG-YNDUR0KGO              4
```

---

## Comment reproduire

**En local (depuis le dossier du projet, ex. `1/`) :**

```bash
pip install -r requirements.txt
jupyter notebook triagegeist_baseline.ipynb
```

Exécuter toutes les cellules. Les données sont lues depuis `data/` et `submission.csv` est écrit dans le même dossier (ou selon `OUT_DIR`).

**Sur Kaggle :**  
1. Créer un notebook et attacher le dataset de la compétition Triagegeist.  
2. Copier le contenu de `triagegeist_baseline.ipynb` (ou uploader le notebook).  
3. Lancer « Run all ». Le fichier `submission.csv` apparaît dans « Output » (/kaggle/working/).

---

## Structure des fichiers

```
1/
├── README.md                    # Ce fichier (contexte, problème, performances, reproductibilité)
├── requirements.txt             # pandas, numpy, scikit-learn, lightgbm, nbformat, nbconvert
├── triagegeist_baseline.ipynb   # Notebook principal (EDA, préparation, modèle, submission)
└── data/
    ├── train.csv
    ├── test.csv
    ├── chief_complaints.csv
    ├── patient_history.csv
    ├── sample_submission.csv
    └── submission.csv           # Généré après exécution du notebook
```

---

## Limitations et pistes d’amélioration

- **Données** : imputation par médiane ; pas d’usage du texte libre des chief complaints ni des antécédents détaillés dans ce baseline.  
- **Modèle** : un seul modèle (LightGBM), pas d’ensemble ni de réglage fin systématique.  
- **Évaluation** : Kappa en CV uniquement ; pas de validation temporelle ou par site ; pas d’ablation sur news2/shock_index.  

**Pistes** : intégrer `chief_complaint_raw` (NLP), fusionner `patient_history`, tester une perte ordinale, analyser la contribution de news2/shock_index, et documenter les erreurs par niveau d’acuité et par sous‑groupes pour le writeup et la soumission finale.

---

## Soumission compétition Kaggle

Pour une soumission valide :

1. **Notebook** : public, exécutable de bout en bout sans erreur.  
2. **Writeup** : onglet « Writeups », ≤ 2000 mots, sections demandées (problème clinique, méthodologie, résultats, limitations, reproductibilité), image de couverture 560×280 px, notebook et lien projet public attachés.

Ce README fournit le contexte (nom du projet, fondation), la formulation du problème, la démarche, une **analyse critique et scientifique des performances**, le format de sortie et les instructions de reproductibilité ; il peut servir de base pour le writeup Kaggle.
