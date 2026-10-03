# Provenance et reproductibilité du dataset

## Source

Dossier Google Drive fourni pour le projet : `erdos_renyi`
https://drive.google.com/drive/folders/18meW_x2uaSTFxhaZagfzLYe5p-RwvOHE

Le dossier contient environ **890 instances** nommées `100_<k>`, chacune avec trois fichiers :

- `graphD.txt` — un graphe **orienté** Erdős–Rényi G(100, p) (arêtes `a-b` = arc a → b) ; c'est un DAG (vérifié par le notebook sur les 200/200 instances téléchargées)
- `graphG.txt` — un second graphe sur les mêmes 100 nœuds, à interpréter comme **non-orienté** (les lignes `a-b` sont symétrisées ; `G` n'a pas d'orientation par définition du problème) ; connexe dans 195/200 instances (les 5 autres — `100_44`, `100_51`, `100_63`, `100_96`, `100_126` — ont seulement 1 ou 2 sommets isolés ; elles restent valides, car le problème n'exige que la connexité du sous-graphe induit par le chemin)
- `solution.txt` — résultat d'un solveur exact ILP2 pour le plus long chemin **(D,G)-consistant** : un chemin dans `graphD` dont l'ensemble de sommets induit un sous-graphe connexe dans `graphG` (voir `REPORT.md` §1-2 pour les définitions formelles et la correction méthodologique)

Voir `REPORT.md` (§2) pour le détail de l'analyse ayant permis d'identifier cette structure et de vérifier l'orientation des arêtes.

## Échantillon effectivement téléchargé dans `data/raw/`

**200 instances** :

- **`100_1` à `100_170`** (170 instances consécutives) ;
- **30 instances** issues du premier échantillon, collecté en plusieurs vagues (tirage aléatoire avec la graine `SEED=42` puis sessions séquentielles) : `100_190`, `100_193`, `100_197`, `100_201`, `100_207`, `100_227`, `100_228`, `100_251`, `100_254`, `100_269`, `100_304`, `100_320`, `100_324`, `100_332`, `100_341`, `100_352`, `100_388`, `100_397`, `100_402`, `100_411`, `100_413`, `100_429`, `100_437`, `100_457`, `100_471`, `100_478`, `100_489`, `100_491`, `100_527`, `100_563`.

ILP2 n'a pas de solution (`"No solution found"`) pour 3 d'entre elles : `100_1`, `100_2` et `100_563`.

## Pourquoi seulement 200 instances sur ~890 ?

Le téléchargement programmatique via `gdown` (API publique Google Drive) est soumis à un **quota anti-abus par fichier/IP** (« Cannot retrieve the public link of the file [...] have had many accesses »). Une première tentative de téléchargement **parallèle** (12 workers simultanés) a déclenché ce quota après ~100 fichiers, bloquant ensuite l'accès à **tout** fichier du dossier — y compris des fichiers jamais sollicités auparavant. Le quota s'est ensuite levé et redéclenché à plusieurs reprises au cours de sessions de téléchargement **séquentielles** : chaque vague permettait de récupérer 15 à 25 instances supplémentaires avant de retomber en blocage, sans qu'on puisse prédire précisément la durée de chaque cycle de blocage/déblocage.

**Leçon retenue : ne pas paralléliser les téléchargements `gdown` sur un même dossier Drive public**, et s'attendre à devoir répéter plusieurs vagues de collecte séquentielle (espacées de ~0.7-1 s entre fichiers) séparées par des pauses, pour un dataset de cette taille.

## Comment étendre le dataset

Une fois le quota Google Drive levé (attendre quelques heures, puis tester avec la commande ci-dessous), relancer la collecte :

```bash
# 1. Tester si le quota est levé (doit réussir sans message "many accesses")
gdown 1zpZqmu14CH_Hw72cpWb3THdaDF_Uk2Ne -O /tmp/test.txt

# 2. Reconstruire la liste complète des fichiers du dossier (mapping instance -> IDs Drive)
gdown --folder "https://drive.google.com/drive/folders/18meW_x2uaSTFxhaZagfzLYe5p-RwvOHE" -O data/raw --dry-run
# (interrompre dès que la liste est suffisante ; NE PAS lancer le téléchargement réel en parallèle)

# 3. Télécharger séquentiellement, avec une pause entre chaque fichier, en réutilisant
#    le même schéma d'échantillonnage aléatoire (SEED=42) que celui utilisé dans le notebook
#    pour sélectionner de nouvelles instances jamais tirées.
```

Le notebook `one_to_one_skewgram.ipynb` (fonction `list_instances()`, §1) **détecte automatiquement** toutes les instances complètes présentes dans `data/raw/` — il suffit de ré-exécuter le notebook après avoir ajouté de nouveaux dossiers `100_<k>/{graphD.txt,graphG.txt,solution.txt}` pour obtenir des statistiques et un tableau comparatif portant sur un échantillon plus large, sans modifier une seule ligne de code.

## Format des fichiers

`graphD.txt` / `graphG.txt` :
```
N:100        <- nombre de nœuds
P:0.08       <- probabilité d'arête du modèle Erdős–Rényi (absent de graphG.txt)
1-9          <- arc orienté 1 → 9
1-30
...
```

`solution.txt` (cas résolu) :
```
Result of ILP2 for 100_121:
Path length: 6
Path : [60, 4, 93, 69, 77, 43]
Execution time: 19.5942 seconds
```

`solution.txt` (cas non résolu dans le temps imparti) :
```
Result of ILP2 for 100_1:
No solution found.
Execution time: 0.4958 seconds
```
