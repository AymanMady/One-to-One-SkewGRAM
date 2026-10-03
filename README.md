# One-to-One SkewGRAM : Optimisation Combinatoire par Embeddings de Nœuds et Décodage Guidé

**Institut Supérieur du Numérique (SupNum), Nouakchott, Mauritanie**  
**Équipe :** Bechir Mady, Abderahmane Abderrahmane, El Moustapha Mohamed El Moustapha, Mohamedou Yahya Cheikh Mohamed Vall  
**Superviseurs :** Dr. Mohamed Lemine Ahmed Sidi, Dr. Hafedh Mohamed Babou  

---

## 📌 À propos de ce projet

Ce dépôt contient le code et les données liés à notre article de recherche, intitulé **"One-to-one SkewGRAM: from biology to NLP"**, soumis à la conférence **I2COMSAPP'26** (Track 11: Other Relevant Topics in AI — Optimisation Combinatoire).

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
- **ILP2** (solveur exact) : trouve la solution optimale mais prend **28 secondes** par instance de 100 nœuds.
- **GNN (DL2)** : rapide mais trouve des chemins de seulement **1.2 nœuds** en moyenne.

Notre approche, **One-to-One Skew-GRAM**, transpose le principe du Skip-Gram (Mikolov et al., 2013), via le paradigme DeepWalk/node2vec, à ce problème d'optimisation combinatoire sur deux graphes couplés. C'est la **seule** méthode utilisée : les embeddings sont entraînés uniquement sur $D$, et la contrainte de $G$ n'intervient qu'au décodage.

Le pipeline complet :

1. **Marches aléatoires sur $D$ + sous-échantillonnage des hubs :** génération de marches suivant les arcs sortants de $D$, avec réduction du poids des nœuds de haut degré (formule word2vec).
2. **Entraînement des embeddings Skew-GRAM :** modèle Negative Sampling avec **négatifs structurels** (tirés parmi les non-successeurs dans $D$, plutôt qu'une distribution unigramme classique).
3. **Décodage guidé conscient du DAG (Étape A) :** un décodeur naïf (similarité d'embeddings seule) s'enlise dans les culs-de-sac de $D$. Nous ajoutons donc une **anticipation exacte par programmation dynamique** (plus long chemin restant dans le DAG, $O(V+E)$), combinée à la similarité d'embeddings et à un bonus de connexité à $G$.
4. **Extraction exacte du sous-chemin (D,G)-consistant (Étape B) :** la connexité dans $G$ ne peut pas être vérifiée incrémentalement (constat empirique sur les solutions ILP2) — on extrait donc, par fenêtre glissante + Union-Find incrémental, le plus long sous-intervalle contigu strictement connexe dans $G$.

---

## 📊 Résultats Expérimentaux (sur 38 instances, 100 nœuds, dont 36 avec optimum ILP2 connu)

| Méthode | Longueur moy. | Temps moy. | vs ILP2 (Vitesse) |
|---|---|---|---|
| GNN DL2 (état de l'art) | ~1.2 nœuds | — | — |
| **ILP2** (solveur exact, référence) | **9.97 nœuds** | **28 secondes** | référence |
| **🥇 One-to-One Skew-GRAM (notre méthode)** | **7.13 nœuds** | **2.7 secondes** | **~10x plus rapide** |

- ✅ **Qualité solide pour une heuristique :** ~7.1 nœuds en moyenne (écart moyen de **-30%** vs l'optimum ILP2), égale ou dépasse ILP2 sur **13 des 36** instances avec optimum connu — très largement au-dessus de l'état de l'art GNN (1.2 nœuds).
- ✅ **Accélération significative :** ~2.7 secondes par instance en moyenne, soit **~10x plus rapide** qu'ILP2.
- 🔍 **Diagnostic clé :** l'anticipation DAG (étape A) est le facteur dominant de la qualité — sans elle (décodage par similarité d'embeddings seule), la longueur moyenne retombe à 5.87 nœuds. Les instances les plus faibles restent sensibles à la variance de l'échantillonnage stochastique (graine fixe) plutôt qu'à une limite structurelle de la méthode.

---

## ⚙️ Reproductibilité

Le code complet de l'algorithme G-First Guidé et de l'évaluation est disponible dans le notebook Jupyter :
- `one_to_one_skewgram.ipynb`

### Comment exécuter :
1. Installez les dépendances : `pip install -r requirements.txt`.
2. Ouvrez `one_to_one_skewgram.ipynb` dans Jupyter ou VS Code.
3. Cliquez sur **"Run All"** (Exécuter tout).
4. Les résultats sont sauvegardés dans `results/summary.csv` et `results/per_instance_results.csv`.

> Les scripts de `scripts/` et le notebook utilisent des chemins relatifs à la racine du dépôt : lancez-les **depuis la racine** (ex. `python scripts/download_instances.py --target 300`).

### Organisation du dépôt
```
One-to-One-SkewGRAM/
├── README.md
├── requirements.txt             # Dépendances Python
├── one_to_one_skewgram.ipynb    # Notebook principal (algorithme + évaluation)
├── data/
│   ├── README.md                # Provenance et format du dataset
│   └── raw/100_<k>/             # Instances : graphD.txt, graphG.txt, solution.txt
├── scripts/                     # Téléchargement des instances depuis Google Drive
│   ├── download_instances.py
│   ├── fetch_instances.py
│   └── smart_download.py
├── results/                     # Résultats bruts (.csv) générés par le notebook
├── figures/                     # Figures générées par le notebook
│   ├── diagrams/                # Schémas statiques (problème, pipeline, exemple de DAG)
│   └── archive/                 # Anciennes versions des figures
├── paper/
│   ├── en/                      # Article LaTeX (anglais) pour I2COMSAPP'26
│   └── fr/                      # Article LaTeX (français)
└── docs/
    └── references/              # Articles de référence (travaux antérieurs)
```

---
*Ce projet est réalisé dans le cadre de nos études à l'Institut Supérieur du Numérique (SupNum) et de notre publication pour la conférence I2COMSAPP'26, Université de Nouakchott, Mauritanie.*
