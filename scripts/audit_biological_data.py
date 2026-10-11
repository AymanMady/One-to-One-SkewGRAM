"""
scripts/audit_biological_data.py
================================
Audit exhaustif des instances réelles du dossier Instances/Biological_data/
Analyse :
- Nombre de sommets (|V|) et d'arcs (|A|, |E|)
- Détection des cycles (DAG pur vs 2-cycles réversibles vs cycles complexes)
- Connexité de G
- Sélection d'un panel de référence de 25-30 instances biologiques pour l'évaluation
"""

import os
import re
import networkx as nx
import pandas as pd
from collections import Counter

BIO_DIR = "data/Biological_data" if os.path.isdir("data/Biological_data") else "Instances/Biological_data"
OUTPUT_CSV = "results/biological_data_audit.csv"
SELECTED_CSV = "results/biological_benchmark_selection.csv"


def load_biological_instance(path_dir):
    """Charge graphD.txt et graphG.txt avec remapping propre des identifiants."""
    path_d = os.path.join(path_dir, "graphD.txt")
    path_g = os.path.join(path_dir, "graphG.txt")
    if not (os.path.isfile(path_d) and os.path.isfile(path_g)):
        return None, None, None

    # Lecture des arêtes brutes
    edges_d = []
    with open(path_d, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("N:") and not line.startswith("P:"):
                parts = line.split("-")
                if len(parts) == 2:
                    edges_d.append((int(parts[0]), int(parts[1])))

    edges_g = []
    with open(path_g, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("N:") and not line.startswith("P:"):
                parts = line.split("-")
                if len(parts) == 2:
                    edges_g.append((int(parts[0]), int(parts[1])))

    # Ensemble de tous les nœuds
    nodes = set()
    for u, v in edges_d:
        nodes.add(u)
        nodes.add(v)
    for u, v in edges_g:
        nodes.add(u)
        nodes.add(v)

    # Si fichier nodes.txt existe, l'inclure également
    nodes_file = os.path.join(path_dir, "nodes.txt")
    if os.path.isfile(nodes_file):
        with open(nodes_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.isdigit():
                    nodes.add(int(line))

    # Mapping bijectif vers [1 .. N]
    node_map = {orig_id: i + 1 for i, orig_id in enumerate(sorted(nodes))}

    # Construction des graphes networkx
    D = nx.DiGraph()
    G = nx.Graph()
    D.add_nodes_from(node_map.values())
    G.add_nodes_from(node_map.values())

    for u, v in edges_d:
        D.add_edge(node_map[u], node_map[v])

    for u, v in edges_g:
        G.add_edge(node_map[u], node_map[v])

    return D, G, node_map


def analyze_cycles(D):
    """Analyse fine des cycles dans D : DAG pur, 2-cycles (réversibilité), cycles longs."""
    if nx.is_directed_acyclic_graph(D):
        return {
            "is_dag": True,
            "n_2cycles": 0,
            "can_be_dag_without_2cycles": True,
            "cycle_type": "Acyclic (Pure DAG)"
        }

    # Recherche des 2-cycles (réactions biochimiques réversibles u <-> v)
    reciprocal_edges = []
    for u, v in D.edges():
        if u < v and D.has_edge(v, u):
            reciprocal_edges.append((u, v))

    n_2cycles = len(reciprocal_edges)

    # Test si la suppression d'un sens pour chaque 2-cycle suffit à rendre D acyclique
    D_no_recip = D.copy()
    for u, v in reciprocal_edges:
        # On supprime temporairement l'arc v -> u
        D_no_recip.remove_edge(v, u)

    can_be_dag = nx.is_directed_acyclic_graph(D_no_recip)

    if n_2cycles > 0 and can_be_dag:
        cycle_type = "Reversible reactions only (2-cycles)"
    elif n_2cycles > 0 and not can_be_dag:
        cycle_type = "Mixed (2-cycles + complex cycles)"
    else:
        cycle_type = "Complex cycles only (no 2-cycles)"

    return {
        "is_dag": False,
        "n_2cycles": n_2cycles,
        "can_be_dag_without_2cycles": can_be_dag,
        "cycle_type": cycle_type
    }


def run_audit():
    print("=" * 70)
    print("🔬 AUDIT EXHAUSTIF DES INSTANCES BIOLOGIQUES (KEGG / PPI)")
    print("=" * 70)

    if not os.path.isdir(BIO_DIR):
        print(f"Erreur : dossier introuvable {BIO_DIR}")
        return

    organisms = sorted([d for d in os.listdir(BIO_DIR) if os.path.isdir(os.path.join(BIO_DIR, d))])
    print(f"Organismes détectés : {len(organisms)}")

    records = []

    for org in organisms:
        org_dir = os.path.join(BIO_DIR, org)
        pathways = sorted([p for p in os.listdir(org_dir) if os.path.isdir(os.path.join(org_dir, p))])

        for pw in pathways:
            pw_dir = os.path.join(org_dir, pw)
            D, G, node_map = load_biological_instance(pw_dir)
            if D is None or G is None:
                continue

            n_nodes = D.number_of_nodes()
            n_edges_d = D.number_of_edges()
            n_edges_g = G.number_of_edges()

            if n_nodes == 0 or n_edges_d == 0:
                continue

            # Analyse cycles
            cycle_info = analyze_cycles(D)

            # Connexité de G
            is_g_conn = nx.is_connected(G)
            n_cc_g = nx.number_connected_components(G)

            records.append({
                "organism": org,
                "pathway": pw,
                "rel_path": f"{org}/{pw}",
                "n_nodes": n_nodes,
                "n_edges_D": n_edges_d,
                "n_edges_G": n_edges_g,
                "is_dag": cycle_info["is_dag"],
                "n_2cycles": cycle_info["n_2cycles"],
                "can_be_dag_without_2cycles": cycle_info["can_be_dag_without_2cycles"],
                "cycle_type": cycle_info["cycle_type"],
                "is_G_connected": is_g_conn,
                "n_connected_components_G": n_cc_g
            })

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Audit terminé sur {len(df)} instances réelles.")
    print(f"📁 Résultats sauvegardés dans : {OUTPUT_CSV}\n")

    # Statistiques globales
    print("-" * 50)
    print("📊 STATISTIQUES GLOBALES SUR LES VOIES BIOLOGIQUES :")
    print("-" * 50)
    print(f"Total instances analysées : {len(df)}")
    print(f"Nœuds min / médian / max  : {df['n_nodes'].min()} / {df['n_nodes'].median():.0f} / {df['n_nodes'].max()}")
    print("\nRépartition par type de structure dans D :")
    for ctype, count in Counter(df["cycle_type"]).items():
        pct = 100 * count / len(df)
        print(f"  - {ctype:<40} : {count:>4} ({pct:5.1f}%)")

    n_pure_dag = df["is_dag"].sum()
    n_can_dag = (df["is_dag"] | df["can_be_dag_without_2cycles"]).sum()
    print(f"\nVoies 100% DAG strictes (sans retouche)      : {n_pure_dag} ({100*n_pure_dag/len(df):.1f}%)")
    print(f"Voies DAG après orientation 2-cycles         : {n_can_dag} ({100*n_can_dag/len(df):.1f}%)")

    # Sélection d'un panel d'évaluation équilibré (25 instances de référence)
    # Sélection des organismes phares : eco (E. coli), bsu (B. subtilis), sau (S. aureus), aae, pae
    top_orgs = ["eco", "bsu", "sau", "pae", "aae", "syn", "mtv"]
    df_top = df[df["organism"].isin(top_orgs)]

    # Sélectionner des instances de taille variée (15 à 70 nœuds), avec priorité aux DAGs purs et 2-cycles résolus
    df_pure = df_top[df_top["is_dag"] & (df_top["n_nodes"] >= 10) & (df_top["n_nodes"] <= 80)].copy()
    df_pure_selected = df_pure.groupby("organism").head(3)

    # Compléter avec quelques voies métaboliques classiques avec 2-cycles résolus
    df_2c = df_top[~df_top["is_dag"] & df_top["can_be_dag_without_2cycles"] & (df_top["n_nodes"] >= 15) & (df_top["n_nodes"] <= 80)].copy()
    df_2c_selected = df_2c.groupby("organism").head(2)

    df_benchmark = pd.concat([df_pure_selected, df_2c_selected]).drop_duplicates(subset=["rel_path"]).head(25)
    df_benchmark.to_csv(SELECTED_CSV, index=False)

    print("\n" + "=" * 70)
    print(f"🎯 SÉLECTION D'UN PANEL DE RÉFÉRENCE BIOLOGIQUE ({len(df_benchmark)} instances) :")
    print("=" * 70)
    for _, row in df_benchmark.iterrows():
        print(f"  • {row['rel_path']:<35} | {row['n_nodes']:>2} nœuds | {row['cycle_type']:<35}")
    print(f"\nFichier du benchmark de référence créé : {SELECTED_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    run_audit()
