# One-to-One SkewGRAM : Optimisation Combinatoire par Embeddings de Nœuds et Décodage Guidé

**Institut Supérieur du Numérique (SupNum), Nouakchott, Mauritanie**  
**Équipe :** Bechir Mady, Abderahmane Abderrahmane, El Moustapha Mohamed El Moustapha, Mohamedou Yahya Cheikh Mohamed Vall  
**Superviseurs :** Dr. Mohamed Lemine Ahmed Sidi, Dr. Hafedh Mohamed Babou  

---

## 📌 À propos de ce projet

Ce dépôt contient le code et les données liés à notre article de recherche, intitulé **"One-to-one SkewGRAM: Applying NLP Embeddings to Biological Graph Optimization"** (titre initial : *"One-to-one SkewGRAM: from biology to NLP"*), accepté à la conférence **I2COMSAPP'26** (Track 11: Other Relevant Topics in AI — Optimisation Combinatoire). La version finale (camera-ready) est dans `paper/en/` (version française à jour dans `paper/fr/`) ; la réponse aux reviewers est un document séparé, `paper/en/response.tex`.

Le projet s'attaque à un problème d'optimisation combinatoire complexe issu de la **bio-informatique**, connu sous le nom de **problème One-to-One SkewGRAM**. Ce problème vise à identifier des signatures métaboliques et génomiques conservées entre différentes espèces, en cherchant le **plus long chemin cohérent** à travers deux réseaux biologiques simultanément.

---

## 🧬 Le Problème (D,G)-consistant

L'objectif du problème One-to-One SkewGRAM est de trouver le plus long chemin qui respecte à la fois un **ordre causal** et une **cohérence topologique** à travers deux réseaux biologiques :

1. **Un graphe orienté acyclique $D = (V, A)$** : modélise des processus biologiques **dirigés** (voies métaboliques, réactions enzymatiques).
2. **Un graphe non-orienté $G = (V, E)$** : modélise des **associations fonctionnelles** (interactions protéine-protéine, proximité génomique).

Un chemin $P = (v_1, v_2, \dots, v_k)$ dans $D$ est dit **(D,G)-consistant** si :
- ✅ C'est un chemin valide dans le graphe orienté $D$ (ordre causal respecté).
- ✅ Le sous-graphe induit par les sommets de $P$ dans $G$ est **connexe** (tous les nœuds se "touchent" fonctionnellement).

**Trouver le plus long tel chemin est un problème NP-difficile.** Les solveurs exacts (ILP) sont trop lents pour les grands graphes biologiques réels.

---

## 🚀 Notre Contribution : One-to-One Skew-GRAM

Les travaux antérieurs (GNN, ILP2) montrent des limites sévères :
- **ILP2** (solveur exact) : trouve la solution optimale mais prend **~30 secondes** par instance de 100 nœuds.
- **GNN (DL2)** : rapide mais trouve des chemins de seulement **1.2 nœuds** en moyenne.

Notre approche, **One-to-One Skew-GRAM**, transpose le principe du Skip-Gram (Mikolov et al., 2013), via le paradigme DeepWalk/node2vec, à ce problème d'optimisation combinatoire sur deux graphes couplés. C'est la **seule** méthode utilisée : les embeddings sont entraînés uniquement sur $D$, et la contrainte de $G$ n'intervient qu'au décodage.

Le pipeline complet :

1. **Marches aléatoires sur $D$ + sous-échantillonnage des hubs :** génération de marches suivant les arcs sortants de $D$, avec réduction du poids des nœuds de haut degré (formule word2vec).
2. **Entraînement des embeddings Skew-GRAM :** modèle Negative Sampling avec **négatifs structurels** (tirés parmi les non-successeurs dans $D$, plutôt qu'une distribution unigramme classique).
3. **Décodage guidé conscient du DAG (Étape A) :** un décodeur naïf (similarité d'embeddings seule) s'enlise dans les culs-de-sac de $D$. Nous ajoutons donc une **anticipation exacte par programmation dynamique** (plus long chemin restant dans le DAG, $O(V+E)$), combinée à la similarité d'embeddings et à un bonus de connexité à $G$.
4. **Extraction exacte du sous-chemin (D,G)-consistant (Étape B) :** la connexité dans $G$ ne peut pas être vérifiée incrémentalement (constat empirique sur les solutions ILP2) — on extrait donc, par fenêtre glissante + Union-Find incrémental, le plus long sous-intervalle contigu strictement connexe dans $G$.

---

## 📊 Résultats Expérimentaux (version finale de l'article)

**Instances synthétiques Erdős–Rényi** (5 graines par instance, 42 à 46). Seules les instances avec une solution ILP2 sont comparées : 298 sur 300 à 100 nœuds, 70 sur 75 à 200 nœuds.

| | 100 nœuds | 200 nœuds |
|---|---|---|
| ILP2 : longueur moyenne / temps moyen | 9.93 nœuds / 29.1 s | 9.64 nœuds / 251.8 s |
| **Skew-GRAM** : longueur moyenne (± écart-type entre graines) | **9.47 ± 0.02** | **7.89 ± 0.12** |
| Exécutions qui atteignent la longueur ILP2 | 71.3 % | 30.0 % |
| Exécutions à au plus 1 nœud d'ILP2 | 90.3 % | 53.1 % |
| Temps moyen Skew-GRAM | 6.9 s | 11.3 s |
| Accélération médiane vs ILP2 | **4.0×** | **21.2×** |

- ⚠️ **Les solutions ILP2 à 200 nœuds ne sont pas toutes optimales** : sur `200_68`, Skew-GRAM trouve un chemin (D,G)-consistant de 10 sommets (vérifié indépendamment) alors que le fichier ILP2 en indique 7. Les longueurs ILP2 sont donc traitées comme des références, et « atteindre ILP2 » signifie « longueur ≥ ILP2 ».
- 🔍 **Ablation** : sans l'anticipation DAG, la longueur moyenne tombe de 9.49 à 7.29 nœuds à 100 nœuds (−23 %), et la longueur ILP2 n'est plus atteinte que sur 47 instances au lieu de 216 (graine 42).
- 🧬 **Réseaux biologiques réels** : sur 25 voies métaboliques KEGG (7 bactéries), Skew-GRAM trouve l'optimum exact (calculé par énumération) sur les 25 voies avec les 5 graines. Ces voies sont petites (au plus 38 sommets) : l'énumération exacte y est plus rapide que Skew-GRAM.
- 📈 **Passage à l'échelle** : une exécution prend en moyenne 49 s à 500 nœuds et 143 s à 1000 nœuds. Le décodage représente la plus grande part du temps, et la qualité augmente encore à 30 000 marches.
- Les temps ILP2 viennent des fichiers `solution.txt` (mesurés sur une autre machine) : les accélérations sont indicatives. Ceux de Skew-GRAM ont été mesurés sur un portable partagé (4 exécutions en parallèle, 1 thread chacune).
- GNN DL2 (travaux antérieurs) : ~1.2 nœuds en moyenne, rapporté sur des instances de 50 nœuds (non relancé ici).

---

## ⚙️ Reproductibilité

Le code complet de One-to-One Skew-GRAM et de l'évaluation est disponible dans le notebook Jupyter :
- `one_to_one_skewgram.ipynb`

Chaque instance est traitée avec la même graine (`SEED = 42`), ré-initialisée au début de l'instance : les résultats sont reproductibles et ne dépendent pas de l'ordre de traitement. Le notebook évalue **toutes** les instances complètes de `data/raw/` et vérifie que chaque chemin retourné est bien (D,G)-consistant.

### Comment exécuter :
1. Installez les dépendances : `pip install -r requirements.txt`.
2. Ouvrez `one_to_one_skewgram.ipynb` dans Jupyter ou VS Code.
3. Cliquez sur **"Run All"** (Exécuter tout).
4. Les résultats sont sauvegardés dans `results/summary.csv` et `results/per_instance_results.csv`, et les figures (en anglais) dans `figures/`. Durée : environ 40 minutes pour les 375 instances.

### Expériences de l'article final
Les expériences répétées (5 graines, ablation, passage à l'échelle, KEGG) sont lancées en parallèle par `scripts/run_experiments.py`, qui réutilise directement les fonctions du notebook ; les figures et tableaux de `paper/en/` sont produits à partir des CSV par `scripts/make_paper_assets.py` :

```bash
python scripts/run_experiments.py seeds      # ~1 h sur 4 cœurs
python scripts/run_experiments.py ablation
python scripts/generate_large_instances.py   # instances 1000 nœuds
python scripts/run_experiments.py scaling
python scripts/run_experiments.py bio
python scripts/make_paper_assets.py             # figures et tableaux anglais (paper/en/)
python scripts/make_paper_assets.py --lang fr   # version française (paper/fr/)
cd paper/en && tectonic main.tex             # ou pdflatex
```

> Les scripts de `scripts/` et le notebook utilisent des chemins relatifs à la racine du dépôt : lancez-les **depuis la racine** (ex. `python scripts/download_instances.py --target 300`).

### Organisation du dépôt
```
One-to-One-SkewGRAM/
├── README.md
├── requirements.txt             # Dépendances Python
├── one_to_one_skewgram.ipynb    # Notebook principal (algorithme + évaluation)
├── data/
│   ├── README.md                # Provenance et format du dataset
│   ├── raw/<n>_<k>/             # Instances synthétiques : graphD.txt, graphG.txt (+ solution.txt ILP2)
│   └── Biological_data/         # Voies métaboliques KEGG (50 organismes)
├── scripts/
│   ├── run_experiments.py       # Expériences parallèles (graines, ablation, échelle, KEGG)
│   ├── make_paper_assets.py     # Figures et tableaux de l'article à partir des CSV
│   ├── generate_large_instances.py  # Instances Erdős–Rényi de 1000 nœuds
│   ├── audit_biological_data.py # Audit et sélection des voies KEGG
│   ├── download_instances.py    # Téléchargement des instances depuis Google Drive
│   ├── fetch_instances.py
│   └── smart_download.py
├── results/                     # Résultats bruts (.csv) du notebook et de run_experiments.py
├── figures/                     # Figures générées par le notebook
│   ├── diagrams/                # Schémas statiques (problème, pipeline, exemple de DAG)
│   └── archive/                 # Anciennes versions des figures
├── paper/
│   ├── en/                      # Article LaTeX final (anglais) ; réponse aux reviewers séparée (response.tex)
│   └── fr/                      # Article LaTeX (français)
└── docs/
    └── references/              # Articles de référence (travaux antérieurs)
```

---
*Ce projet est réalisé dans le cadre de nos études à l'Institut Supérieur du Numérique (SupNum) et de notre publication pour la conférence I2COMSAPP'26, Université de Nouakchott, Mauritanie.*
