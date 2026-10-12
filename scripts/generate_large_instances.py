"""
generate_large_instances.py
===========================
Génère des instances Erdős–Rényi de grande taille (par défaut 5 instances de 1000 nœuds)
pour l'étude de passage à l'échelle, avec le même procédé que le dataset fourni :

- D : ordre topologique aléatoire des nœuds, puis chaque arc « plus tôt → plus tard »
  est ajouté avec la probabilité p (D est donc un DAG par construction) ;
- G : graphe non orienté G(n, p) indépendant, sur les mêmes nœuds.

Ces instances n'ont pas de solution ILP2 (pas de fichier solution.txt) : elles servent
uniquement à mesurer le temps de calcul et la longueur des chemins trouvés.

Usage (depuis la racine du dépôt) :
    python scripts/generate_large_instances.py
    python scripts/generate_large_instances.py --n 1000 --p 0.05 --count 5 --seed 42
"""

import os
import random
import argparse

DATA_DIR = "data/raw"


def generate_instance(n, p, rng):
    order = list(range(1, n + 1))
    rng.shuffle(order)
    arcs = [(order[i], order[j]) for i in range(n) for j in range(i + 1, n) if rng.random() < p]
    edges = [(a, b) for a in range(1, n + 1) for b in range(a + 1, n + 1) if rng.random() < p]
    return sorted(arcs), edges


def write_graph(path, n, edges, p=None):
    with open(path, "w") as f:
        f.write(f"N:{n}\n")
        if p is not None:
            f.write(f"P:{p}\n")
        for a, b in edges:
            f.write(f"{a}-{b}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=1000, help="nombre de nœuds")
    parser.add_argument("--p", type=float, default=0.05, help="probabilité d'arête (0.05 comme les instances 200 et 500)")
    parser.add_argument("--count", type=int, default=5, help="nombre d'instances")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    for k in range(1, args.count + 1):
        inst_dir = os.path.join(DATA_DIR, f"{args.n}_{k}")
        os.makedirs(inst_dir, exist_ok=True)
        arcs, edges = generate_instance(args.n, args.p, rng)
        write_graph(os.path.join(inst_dir, "graphD.txt"), args.n, arcs, p=args.p)
        write_graph(os.path.join(inst_dir, "graphG.txt"), args.n, edges)
        print(f"{inst_dir} : |A| = {len(arcs)}, |E| = {len(edges)}")


if __name__ == "__main__":
    main()
