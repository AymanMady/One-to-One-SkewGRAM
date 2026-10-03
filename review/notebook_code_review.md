# Code review — `one_to_one_skewgram.ipynb`

**Date :** 2026-10-03
**Fichier revu :** [../one_to_one_skewgram.ipynb](../one_to_one_skewgram.ipynb) (27 cellules, 16 de code)
**Objectifs :**

1. Corriger ce qui fausse les résultats.
2. Supprimer ce qui ne sert à rien.
3. Simplifier le code pour qu'il soit clair et facile à lire.
4. Produire toutes les figures en **anglais**.
5. Tester sur **toutes** les instances.

> Les constats ci-dessous ont été vérifiés en exécutant les fonctions du notebook dans un script séparé (voir l'[annexe](#annexe--comment-les-constats-ont-été-vérifiés)).
>
> **Statut (2026-10-03) : review appliquée.** Le notebook a été réécrit selon la structure du §7 puis ré-exécuté sur les 200 instances ; les CSV, les figures, le README, `data/README.md`, `requirements.txt` et les deux articles ont été mis à jour. Les chiffres cités dans cette review (195 instances, 9.6 nœuds, etc.) décrivent l'**ancienne** version.

---

## 0. Résumé des actions

| # | Priorité | Où | Problème | Action |
|---|---|---|---|---|
| 1 | 🔴 Critique | `train_skewgram`, `decode_skewgram_path` | Le résultat d'une instance dépend des instances traitées avant elle (non reproductible) | Ré-initialiser les graines **au début de chaque instance** (`set_seed`) |
| 2 | 🔴 Critique | §1, filtre `VALID_INSTANCES` | 5 instances sur 200 sont exclues alors qu'elles sont valides | Supprimer le filtre « G connexe » → 200/200 instances |
| 3 | 🟠 Important | §6, boucle d'évaluation | Les chemins trouvés ne sont **jamais vérifiés** sur les 200 instances | Ajouter `assert is_DG_consistent(path, D, G)` |
| 4 | 🟠 Important | §6, résumé | Les moyennes ILP2 et Skew-GRAM sont calculées sur des ensembles différents ; « reaches_or_beats » mélange deux cas | Calculer sur le même sous-ensemble ; séparer `=` et `>` |
| 5 | 🟠 Important | Figures 02, 04, 05, 06, 08 | Titres et axes en français | Tout traduire (tableau en §4) |
| 6 | 🟡 Moyen | Tout le notebook | Code dupliqué, code mort, cellule gensim inutile | Voir §2 et §3 |
| 7 | 🟡 Moyen | `decode_skewgram_path` | Le chemin retourné contient des `np.int64` | Convertir en `int` |
| 8 | 🟡 Moyen | Markdown §5 + article | Complexité annoncée O(L² α(L)), mais le code est en O(L³ α(L)) | Corriger le texte (le code est assez rapide) |
| 9 | ⚪ Faible | Markdown §9 « Discussion » | Chiffres périmés (« fraction de seconde », « 10× ») | Mettre à jour après la nouvelle exécution |

---

## 1. Problèmes qui faussent les résultats

### 1.1 🔴 Résultats non reproductibles : ils dépendent de l'ordre d'exécution

Le notebook fixe les graines **une seule fois** (cellule Setup). Ensuite, trois sources d'aléa utilisent l'état **global** des générateurs aléatoires, qui évolue d'une instance à l'autre :

| Endroit | Appel | Générateur utilisé |
|---|---|---|
| `SkewGram.__init__` | `nn.init.uniform_(...)` | `torch` global |
| `train_skewgram` | `np.random.shuffle(idx_all)` | `numpy` global |
| `decode_skewgram_path` | `np.random.choice(valid_succs, p=probs)` | `numpy` global |

Conséquence : le résultat de l'instance *k* dépend de toutes les instances traitées avant elle, de la démo exécutée avant la boucle, et de la valeur de `N_INSTANCES_TO_RUN`.

**Vérifié :** les mêmes 8 instances ont été lancées dans deux ordres différents :

| Instance | Valeur dans le CSV actuel | Ordre 1 | Ordre 2 |
|---|---|---|---|
| 100_104 | 8 | 8 | **9** |
| 100_105 | 7 | **8** | **8** |
| 100_123 | 10 | **9** | **11** |
| 100_128 | 10 | 10 | **13** |
| 100_130 | 8 | **12** | **12** |

Les embeddings appris changent aussi d'un ordre à l'autre (la somme de `in_emb` passe par exemple de 112.0 à −213.4 pour 100_10).

**Correction** (vérifiée : avec elle, les résultats sont identiques quel que soit l'ordre) :

```python
def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
```

`set_seed` est appelée au début de `run_pipeline` (voir §3.3), donc **une fois par instance**.

> ⚠️ Après cette correction, tous les chiffres changent (`summary.csv`, tableaux de l'article, README). Il faut tout ré-exécuter et mettre à jour ces trois endroits.

### 1.2 🔴 Toutes les instances ne sont pas évaluées

Il y a deux raisons :

**a) Le filtre `VALID_INSTANCES` exclut 5 instances valides.**
Il exige que `G` soit connexe en entier. Or le problème ne demande que la connexité du **sous-graphe induit par le chemin**, pas celle de `G` tout entier. Les 5 instances exclues ont seulement 1 ou 2 sommets isolés dans `G`, et ILP2 a trouvé une solution pour chacune :

| Instance | Composantes de G | Tailles | Solution ILP2 |
|---|---|---|---|
| 100_126 | 2 | 1 + 99 | longueur 7 |
| 100_44 | 3 | 1 + 1 + 98 | longueur 6 |
| 100_51 | 2 | 1 + 99 | longueur 8 |
| 100_63 | 2 | 1 + 99 | longueur 8 |
| 100_96 | 2 | 1 + 99 | longueur 10 |

L'algorithme fonctionne tel quel sur un `G` non connexe : rien dans le décodeur ni dans l'extraction ne suppose que `G` est connexe.

**b) `N_INSTANCES_TO_RUN = 200` + tri alphabétique.**
`sorted(os.listdir(...))` trie par ordre alphabétique (`100_1, 100_10, 100_100, …`). Si tu ajoutes des instances, `[:200]` prendra un sous-ensemble arbitraire. Pour tester sur toutes les instances, supprime ce paramètre.

**Action :**
- Supprimer `N_INSTANCES_TO_RUN` et le paramètre `limit`.
- Supprimer le filtre `VALID_INSTANCES` et le remplacer par des `assert` (§3.2).
- Trier numériquement : `sorted(..., key=lambda s: int(s.split("_")[1]))`.
- Les 3 instances où ILP2 n'a pas de solution (`100_1`, `100_2`, `100_563`) restent dans les résultats. Elles sont seulement exclues du calcul de l'écart et du speedup (déjà le cas via `found`).

### 1.3 🟠 Les chemins de l'évaluation ne sont jamais vérifiés

La vérification « chemin valide dans D + connexe dans G » n'est faite que pour la démo (instance 100_14). Pour un article, il faut prouver que **chaque** chemin rapporté est valide. Il suffit d'une ligne dans la boucle :

```python
assert is_DG_consistent(path, D, G), inst
```

### 1.4 🟠 Le résumé compare des sous-ensembles différents

```python
"mean_ilp2_length":     found["ilp2_length"].mean(),          # 192 instances
"mean_skewgram_length": df_results["skewgram_length"].mean(),  # 195 instances
```

Les deux moyennes ne portent pas sur les mêmes instances (l'écart est faible ici, 9.615 contre 9.615, mais le calcul est faux en principe). Deux autres remarques :

- **`n_reaches_or_beats_ilp2`** : ILP2 est exact, donc « beats » devrait toujours valoir 0 (c'est le cas : 0 sur 192). Si ce nombre devient un jour > 0, c'est un **bug** à signaler, pas une victoire. Il faut compter `=` et `>` séparément.
- **`mean_speedup_vs_ilp2`** : une moyenne de ratios est tirée vers le haut par les valeurs extrêmes. Sur les résultats actuels : moyenne = 7.52×, **médiane = 6.65×**, ratio des temps moyens = 7.23×. Il vaut mieux rapporter la médiane.

### 1.5 🟡 Des `np.int64` dans les chemins

Sortie actuelle de la démo :
```
[23, np.int64(86), np.int64(38), np.int64(37), ...]
```
Cela vient de `succs = np.array(...)` combiné avec `np.random.choice`. Correction dans le décodeur :

```python
u = int(np.random.choice(valid_succs, p=probs))
```

### 1.6 🟡 La complexité annoncée ne correspond pas au code

Le markdown (§5) et l'article ([paper/en/main.tex:344-350](../paper/en/main.tex#L344-L350)) annoncent **O(L² α(L))**. Mais à chaque nouveau sommet `v`, le code parcourt toute la fenêtre `path[i:j]`, ce qui donne **O(L³ α(L))**.

En pratique ce n'est pas un problème : L ≤ 24 (plus long chemin de D sur les 200 instances) et l'extraction prend environ 0.2 s pour 4 576 appels. **Il faut corriger le texte, pas le code.** Une version « plus simple » avec `nx.is_connected` a été testée : elle donne exactement le même résultat, mais elle est **15× plus lente** (3.35 s contre 0.21 s). On garde donc Union-Find.

---

## 2. Ce qui peut être supprimé

| Élément | Où | Pourquoi |
|---|---|---|
| `from collections import defaultdict` | Setup | Jamais utilisé |
| `BEAM_WIDTH = 64` | Setup | Jamais utilisé (le décodeur ne fait pas de beam search) et absent de l'article |
| `download_drive_instances_info()` | §1 | Affiche seulement des URLs codées en dur ; cette information est déjà dans `data/README.md` |
| `N_INSTANCES_TO_RUN` + paramètre `limit` | §1 | On veut toujours toutes les instances (§1.2) |
| `load_undirected` | §1 | Copie de `load_directed` → une seule fonction `load_graph` (§3.1) |
| Tableau `df_checks` + `print(df_checks.head(10))` | §1 | À remplacer par 3 `assert` (§3.2) |
| `# rng.shuffle(idx_all.tolist())` | `train_skewgram` | Code commenté (mort) |
| Paramètre `D` de `longest_DG_consistent_subpath` | §5 | Jamais utilisé dans la fonction |
| Fonction interne `union()` | §5 | Une seule ligne, utilisée une fois → à mettre directement dans la boucle |
| Variable `history` | §6, boucle | Jamais utilisée |
| **Section 8 « Sanity check gensim »** (cellule entière) | §8 | Elle ne compare rien : elle entraîne un modèle gensim, affiche des voisins, puis dit « cohérence confirmée » sans aucune comparaison. Dans la dernière exécution, gensim n'était même pas installé dans le kernel. Si tu la supprimes, retire aussi `gensim` de `requirements.txt` |
| **Figure 06 (PCA / t-SNE)** + imports `PCA`, `TSNE` | §7 | Pas utilisée dans l'article, et ce sont des nuages de points sans couleur qui ne montrent rien. Si tu veux la garder, colore les points par `lf(v)` (la longueur du plus long chemin restant) pour qu'elle ait un sens |
| Cellule séparée de vérification de la démo | §5 | Remplacée par `is_DG_consistent` (§3.2) |

**Facultatif** (ces figures ne sont pas dans l'article ; tu peux les garder pour une présentation) :
- Figure 02 (histogrammes du sous-échantillonnage)
- Figure 04 (courbe de perte) : peu coûteuse, et elle montre que l'entraînement converge. Je la garderais.

---

## 3. Simplifications

### 3.1 Un seul chargeur de graphe

`load_directed` et `load_undirected` sont identiques à une ligne près. Proposition (vérifiée : mêmes nœuds, dans le même ordre, et mêmes arêtes sur les 200 instances) :

```python
def load_graph(path, directed):
    G = nx.DiGraph() if directed else nx.Graph()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("N:"):
                G.add_nodes_from(range(1, int(line[2:]) + 1))
            elif line and not line.startswith("P:"):
                a, b = map(int, line.split("-"))
                G.add_edge(a, b)
    return G
```

### 3.2 Une seule fonction de vérification, réutilisée partout

La même logique (chemin dans D + connexe dans G) est écrite deux fois aujourd'hui (§1 pour ILP2, §5 pour la démo), et elle manque dans l'évaluation.

```python
def is_DG_consistent(path, D, G):
    """Vrai si `path` est un chemin de D dont les sommets induisent un sous-graphe connexe dans G."""
    in_D = all(D.has_edge(a, b) for a, b in zip(path, path[1:]))
    return bool(path) and in_D and nx.is_connected(G.subgraph(path))
```

Le chargement + contrôle des données devient alors :

```python
INSTANCES = list_instances()
GRAPHS, SOLUTIONS = {}, {}
for inst in INSTANCES:
    d = os.path.join(DATA_DIR, inst)
    D = load_graph(os.path.join(d, "graphD.txt"), directed=True)
    G = load_graph(os.path.join(d, "graphG.txt"), directed=False)
    sol = load_solution(os.path.join(d, "solution.txt"))
    assert nx.is_directed_acyclic_graph(D), inst
    assert set(D) == set(G), inst
    assert sol["path"] is None or is_DG_consistent(sol["path"], D, G), inst
    GRAPHS[inst], SOLUTIONS[inst] = (D, G), sol

print(f"{len(INSTANCES)} instances chargées.")
```

### 3.3 Une seule fonction `run_pipeline`, utilisée par la démo ET l'évaluation

Aujourd'hui, le pipeline est écrit deux fois :
- dans les cellules de démo (§2 → §5), avec `n_walks=10000` ;
- dans la boucle d'évaluation (§6), avec `n_walks=30000`.

Conséquence : **le chemin de longueur 12 de la figure 08 (dans l'article) n'est pas celui obtenu par l'évaluation** pour 100_14. Une seule fonction règle ce problème et celui du §1.1 :

```python
def run_pipeline(D, G, seed=SEED):
    """Phases 1 → 4 sur une instance. Retourne (chemin, modèle, historique de la perte)."""
    set_seed(seed)  # chaque instance repart du même état → reproductible
    walks = generate_walks(D, rng=random.Random(seed))
    freq = node_frequencies(walks, D.number_of_nodes())
    walks = subsample_hubs(walks, freq, rng=random.Random(seed))
    model, history = train_skewgram(D, build_pairs(walks), seed=seed)
    path = decode_skewgram_path(D, G, model, seed=seed)
    return path, model, history
```

La démo devient :

```python
D_demo, G_demo = GRAPHS[REPRESENTATIVE]
path_demo, model_demo, history_demo = run_pipeline(D_demo, G_demo)
```

Et l'évaluation :

```python
rows = []
for inst in INSTANCES:
    D, G = GRAPHS[inst]
    sol = SOLUTIONS[inst]
    t0 = time.time()
    path, _, _ = run_pipeline(D, G)
    elapsed = time.time() - t0
    assert is_DG_consistent(path, D, G), inst
    rows.append({
        "instance": inst,
        "ilp2_status": sol["status"],
        "ilp2_length": sol["length"],
        "ilp2_time_s": sol["time"],
        "skewgram_length": len(path),
        "skewgram_time_s": elapsed,
        "skewgram_path": path,  # utile pour re-vérifier ou tracer plus tard
    })
    print(f"{inst}: ILP2={sol['length']} | Skew-GRAM={len(path)} | {elapsed:.2f}s")

df_results = pd.DataFrame(rows)
df_results["gap_pct_vs_ilp2"] = 100 * (df_results["skewgram_length"] - df_results["ilp2_length"]) / df_results["ilp2_length"]
df_results["speedup_vs_ilp2"] = df_results["ilp2_time_s"] / df_results["skewgram_time_s"]
df_results.to_csv(os.path.join(RESULTS_DIR, "per_instance_results.csv"), index=False)
```

Pour les instances sans solution ILP2, `ilp2_length` vaut `NaN`, donc l'écart vaut aussi `NaN` automatiquement. Plus besoin des deux `if`.

### 3.4 Résumé corrigé

```python
found = df_results[df_results["ilp2_status"] == "found"]
summary = {
    "n_instances": len(df_results),
    "n_ilp2_found": len(found),
    "mean_ilp2_length": found["ilp2_length"].mean(),
    "mean_skewgram_length": found["skewgram_length"].mean(),  # même sous-ensemble qu'ILP2
    "mean_gap_pct_vs_ilp2": found["gap_pct_vs_ilp2"].mean(),
    "n_matches_ilp2": int((found["skewgram_length"] == found["ilp2_length"]).sum()),
    "n_above_ilp2": int((found["skewgram_length"] > found["ilp2_length"]).sum()),  # doit valoir 0
    "mean_skewgram_time_s": found["skewgram_time_s"].mean(),
    "mean_ilp2_time_s": found["ilp2_time_s"].mean(),
    "median_speedup_vs_ilp2": found["speedup_vs_ilp2"].median(),
}
```

### 3.5 Phase 4 nettoyée (même algorithme, même résultat)

Vérifiée : sortie identique à l'original sur 4 000 chemins aléatoires.

```python
def longest_connected_subpath(path, G):
    """Phase 4 : plus long sous-chemin contigu de `path` dont les sommets sont connexes dans G."""
    best_i, best_j = 0, 0
    for i in range(len(path)):
        parent = {}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        n_components = 0
        for j in range(i, len(path)):
            v = path[j]
            parent[v] = v
            n_components += 1
            for u in path[i:j]:
                if G.has_edge(u, v) and find(u) != find(v):
                    parent[find(u)] = find(v)
                    n_components -= 1
            if n_components == 1 and j - i > best_j - best_i:
                best_i, best_j = i, j
    return path[best_i:best_j + 1]
```

Changements par rapport à l'original :
- le paramètre `D`, inutile, est retiré ;
- `union()` est intégrée dans la boucle ;
- le cas `L <= 1` est géré naturellement ;
- le nom de la fonction dit ce qu'elle fait vraiment : elle vérifie uniquement la connexité dans G.

### 3.6 Petites simplifications du décodeur

- Remplacer `raw_path = path` (un alias inutile) par `path`.
- Convertir le sommet choisi en `int` (§1.5).
- **À noter (ne pas changer, car c'est la formule de l'article) :** dans `S(u,v) = lf/τ_lf + β·sim/τ + log(γ)·I`, seul le rapport **β/τ = 0.6** compte. τ et β sont redondants, ce qui est à mentionner si un relecteur demande une étude de sensibilité.

---

## 4. Figures en anglais

### 4.1 Traductions à appliquer

| Figure | Élément | Actuel (FR) | Proposé (EN) |
|---|---|---|---|
| 02 | titre gauche | `Fréquence des nœuds ({inst}) — avant sous-échantillonnage` | `Node frequency ({inst}) — before subsampling` |
| 02 | titre droite | `… — après sous-échantillonnage` | `… — after subsampling` |
| 02 | axe x | `nombre d'apparitions dans les marches` | `occurrences in walks` |
| 02 | axe y | `nombre de nœuds` | `number of nodes` |
| 04 | titre | `Courbe de perte — One-to-One Skew-GRAM ({inst})` | `Training loss — One-to-One Skew-GRAM ({inst})` |
| 04 | axe x | `époque` | `epoch` |
| 04 | axe y | `perte moyenne (Negative Sampling)` | `mean negative-sampling loss` |
| 05 | légende | `ILP2 (référence)` **et** `ILP2 (fondation)` (incohérent) | `ILP2 (exact)` (les deux) |
| 05 | titre gauche | `Longueur du chemin par instance` | `Path length per instance` |
| 05 | axe x gauche | `index d'instance` | `instance index` |
| 05 | axe y gauche | `longueur du chemin (nombre de nœuds)` | `path length (nodes)` |
| 05 | titre droite | `Compromis Temps vs Qualité` | `Runtime vs. solution quality` |
| 05 | axe x droite | `temps d'exécution (s)` | `runtime (s, log scale)` |
| 05 | axe y droite | `longueur du chemin` | `path length (nodes)` |
| 06 | titres | `Projection PCA/t-SNE des embeddings ({inst})` | `PCA / t-SNE projection of node embeddings ({inst})` (si conservée) |
| 08 | titre gauche | `Chemin (D,G)-consistant dans D — longueur {n}` | `(D,G)-consistent path in D (length {n})` |
| 08 | titre droite | `Sous-graphe induit dans G par le chemin (D,G)-consistant — longueur {n}` | `Subgraph of G induced by the path (length {n})` |

Pense aussi à traduire les `print(...)` (par exemple `temps=` → `time=`) si le notebook doit être partagé en anglais.

### 4.2 Autres corrections sur les figures

- **Résolution :** `figure.dpi = 100` donne des PNG flous dans l'article. Utilise `fig.savefig(..., dpi=300, bbox_inches="tight")`.
- **Numérotation :** les fichiers s'appellent `02, 04, 05, 06, 08` (avec des trous). Renumérote-les dans l'ordre d'apparition (`01_…`, `02_…`, `03_…`). Les diagrammes `09, 10, 11` peuvent garder leur numéro.
- **Figure 05, panneau gauche (recommandé) :** une courbe sur 200 instances triées par ordre alphabétique (`100_1, 100_10, 100_100…`) est illisible, et l'axe x n'a aucun sens. Deux alternatives plus claires :
  - un nuage de points `ILP2 length` (x) contre `Skew-GRAM length` (y) avec la diagonale `y = x` : les points sur la diagonale correspondent aux instances où l'optimum est atteint ;
  - ou un histogramme de `gap_pct_vs_ilp2`.

  ⚠️ Cette figure est dans l'article ([paper/en/main.tex:511](../paper/en/main.tex#L511)) : si tu changes son type, il faut adapter la légende et le texte.
- **Copie vers l'article :** les figures 05 et 08 sont copiées à la main dans `paper/en/` **et** `paper/fr/`. Une fois traduites, copie-les seulement dans `paper/en/` ; garde les versions françaises dans `paper/fr/`, sinon l'article français aura des figures en anglais.

---

## 5. Tester sur toutes les instances : checklist

1. Supprimer `N_INSTANCES_TO_RUN` / `limit` et trier numériquement (§1.2b).
2. Supprimer le filtre `VALID_INSTANCES` (§1.2a) : on passe de 195 à **200 / 200** instances.
3. Ajouter `set_seed` dans `run_pipeline` (§1.1).
4. Ajouter `assert is_DG_consistent(...)` dans la boucle (§1.3).
5. Exécuter tout le notebook (« Run All »). Durée estimée : 200 × ~4.1 s ≈ **14 min**.
6. Mettre à jour `results/*.csv`, les tableaux de l'article et le README.

Pour aller au-delà des 200 instances (le dossier Drive en contient ~890) : lance `python scripts/download_instances.py --target N` depuis la racine. Grâce au point 1, le notebook les prendra toutes automatiquement.

---

## 6. Textes markdown à mettre à jour

| Section | Problème |
|---|---|
| Titre / Résumé | Dit que `G` est un « graphe non-orienté **connexe** » : faux pour 5 instances, et ce n'est pas nécessaire. |
| §1 | Mentionne `N_INSTANCES_TO_RUN` (à supprimer). |
| §5 | Complexité O(L² α(L)) → O(L³ α(L)) (§1.6). |
| §8 | Section gensim à supprimer. |
| §9 Discussion | « s'exécute en une **fraction de seconde** » : faux, c'est ~4.1 s par instance. « ~10× plus rapide » : c'est ~7× (médiane 6.65×). « similarité **cosinus** » : faux, c'est un produit scalaire `v_in·v_out` (non normalisé). |
| §6 | Préciser que les temps ILP2 viennent des fichiers `solution.txt` (mesurés sur une autre machine) : le speedup est **indicatif**. |

---

## 7. Structure proposée du notebook simplifié

```
0. Titre + résumé                                   (markdown)
1. Setup        : imports, SEED, chemins, hyperparamètres, set_seed()
2. Données      : load_graph, load_solution, list_instances, is_DG_consistent
                  → chargement de TOUTES les instances + 3 assert
3. Phase 1      : generate_walks, node_frequencies, subsample_hubs, build_pairs
4. Phase 2      : SkewGram, structural_negative_table, sample_structural_negatives, train_skewgram
5. Phases 3-4   : compute_dag_lookahead, longest_connected_subpath, decode_skewgram_path
6. run_pipeline(D, G)
7. Démo (100_14): run_pipeline → print + Fig. 1 (training loss) + Fig. 2 (path in D / subgraph in G)
8. Évaluation   : toutes les instances → per_instance_results.csv
9. Résumé       : summary.csv + Fig. 3 (ILP2 vs Skew-GRAM)
10. Conclusion
```

On passe d'environ 16 à environ 11 cellules de code. Il n'y a plus de pipeline dupliqué, plus de code mort, et les imports `defaultdict`, `PCA`, `TSNE` et `gensim` disparaissent.

---

## 8. Questions de méthode (ne pas changer sans mesurer)

Ce ne sont pas des bugs, mais un relecteur pourrait poser ces questions :

- **Le sous-échantillonnage des hubs supprime 73 % du corpus.** Sur 100_14, il reste 2 991 tokens sur 11 137, avec une probabilité de garder un nœud entre 0.15 et 0.83. Le seuil `t = 1e-3` vient de word2vec, où le vocabulaire compte des millions de mots. Avec 100 nœuds, chaque nœud a une fréquence > 1e-3, donc **tous** les nœuds sont sous-échantillonnés, pas seulement les hubs. Une ablation avec et sans sous-échantillonnage serait utile.
- **Les marches sont très courtes** (3.2 nœuds en moyenne sur 100_14, alors que `WALK_LENGTH = 30`) parce que D est un DAG peu dense. `WALK_LENGTH = 30` n'a presque aucun effet.
- **`build_pairs` est symétrique** : la fenêtre prend aussi les nœuds *précédents*, donc les paires ne gardent pas la direction des arcs. C'est conforme au Skip-Gram classique, mais le nom « **Skew**-GRAM » laisse penser que la direction est prise en compte. À expliquer dans l'article, ou à tester avec une fenêtre uniquement vers l'avant.
- **Temps ILP2** : ces temps n'ont pas été mesurés sur la même machine (voir §6).

---

## 9. Hors notebook (incohérences repérées en passant)

- **[README.md](../README.md)** : annonce 38 instances, 7.13 nœuds et 2.7 s, alors que les résultats actuels sont 195 instances, 9.6 nœuds et 4.1 s.
- **[data/README.md](../data/README.md)** : annonce 39 instances téléchargées, alors que `data/raw/` en contient 200.

---

## Annexe : comment les constats ont été vérifiés

Les fonctions du notebook ont été extraites telles quelles dans un script temporaire (hors du dépôt), puis :

| Vérification | Résultat |
|---|---|
| Même 8 instances, 2 ordres d'exécution | Longueurs différentes sur 5/8 instances → §1.1 |
| Même test avec `set_seed` au début de chaque instance | Résultats identiques dans les deux ordres ✅ |
| Analyse de `G` sur les 5 instances exclues | 1 ou 2 sommets isolés, solution ILP2 valide → §1.2 |
| `load_graph` contre `load_directed` / `load_undirected` sur 200 instances | Identiques (nœuds, ordre, arêtes) ✅ |
| `longest_connected_subpath` (§3.5) contre l'original sur 4 000 chemins | Identiques ✅ |
| Version `nx.is_connected` contre Union-Find sur 4 576 appels réels | Identiques, mais 3.35 s contre 0.21 s → on garde Union-Find |
| Profilage de 100_14 | Marches 0.02 s, entraînement 1.5 s, décodage 3.6 s |
| Analyse de `results/per_instance_results.csv` | 0 instance au-dessus d'ILP2, 142 égales, 50 en dessous ; speedup médian 6.65× |
