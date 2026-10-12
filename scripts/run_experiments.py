"""
run_experiments.py
==================
Expériences complémentaires de la version finale de l'article (réponse aux reviewers d'I2COMSAPP'26).

Le code de l'algorithme n'est pas dupliqué : les cellules de définitions du notebook
`one_to_one_skewgram.ipynb` (imports, hyperparamètres, fonctions, classes) sont exécutées
telles quelles, puis appelées ici en parallèle (un processus par cœur, PyTorch mono-thread).

Sous-commandes (à lancer depuis la racine du dépôt) :
    python scripts/run_experiments.py seeds      # 5 graines × instances 100/200 nœuds (avec solution.txt)
    python scripts/run_experiments.py ablation   # sans lookahead DAG (graine 42)
    python scripts/run_experiments.py scaling    # instances 500 et 1000 nœuds (5 graines, sans ILP2)
    python scripts/run_experiments.py bio        # 25 voies KEGG : optimum exact + Skew-GRAM (5 graines)

Options : --workers N (défaut 4), --seeds 42 43 44 45 46.
Sorties dans results/ : multiseed_results.csv, multiseed_traces.csv, ablation_results.csv,
scaling_results.csv, scaling_traces.csv, biological_multiseed.csv.
Une exécution interrompue reprend là où elle s'est arrêtée (les couples (instance, graine) déjà
présents dans le CSV de sortie sont ignorés).
"""

import os
import ast
import json
import time
import argparse
import multiprocessing as mp

os.environ.setdefault("MPLBACKEND", "Agg")

NOTEBOOK = "one_to_one_skewgram.ipynb"
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
SAVE_EVERY = 20


def load_notebook_definitions(path=NOTEBOOK):
    """Exécute les cellules du notebook qui ne font que définir (imports, constantes, fonctions, classes)."""
    nb = json.load(open(path, encoding="utf-8"))
    ns = {"__name__": "skewgram_notebook"}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        tree = ast.parse(src)
        defines = any(isinstance(n, (ast.FunctionDef, ast.ClassDef)) for n in tree.body)
        if defines or "SEED = 42" in src:
            assert not any(isinstance(n, (ast.For, ast.While, ast.If, ast.With)) for n in tree.body), src[:80]
            exec(compile(src, "<notebook>", "exec"), ns)
    return ns


NS = load_notebook_definitions()
import pandas as pd  # noqa: E402  (après le notebook, qui fixe le backend matplotlib)
import networkx as nx  # noqa: E402


def init_worker():
    import torch
    torch.set_num_threads(1)


# ─── Chargement des instances ──────────────────────────────────────────────────

def load_synthetic(inst):
    d = os.path.join(NS["DATA_DIR"], inst)
    D = NS["load_graph"](os.path.join(d, "graphD.txt"), directed=True)
    G = NS["load_graph"](os.path.join(d, "graphG.txt"), directed=False)
    assert nx.is_directed_acyclic_graph(D) and set(D) == set(G), inst
    sol_path = os.path.join(d, "solution.txt")
    sol = NS["load_solution"](sol_path) if os.path.isfile(sol_path) else {"status": "none", "length": None, "time": None}
    return D, G, sol


def load_bio(rel_path):
    d = os.path.join(NS["BIO_DATA_DIR"], rel_path)
    D, G, _ = NS["load_biological_instance"](os.path.join(d, "graphD.txt"), os.path.join(d, "graphG.txt"))
    assert nx.is_directed_acyclic_graph(D), rel_path
    return D, G


# ─── Tâches exécutées dans les processus ───────────────────────────────────────

def run_synthetic(job):
    inst, seed, decode_kwargs = job
    D, G, sol = load_synthetic(inst)
    timings, trace = {}, []
    t0 = time.time()
    path, _, _ = NS["run_pipeline"](D, G, seed=seed, timings=timings, trace=trace, **decode_kwargs)
    elapsed = time.time() - t0
    assert NS["is_DG_consistent"](path, D, G), (inst, seed)
    row = {
        "instance": inst,
        "n_nodes": D.number_of_nodes(),
        "n_arcs_D": D.number_of_edges(),
        "n_edges_G": G.number_of_edges(),
        "longest_path_D": nx.dag_longest_path_length(D) + 1,
        "seed": seed,
        "ilp2_status": sol["status"],
        "ilp2_length": sol["length"],
        "ilp2_time_s": sol["time"],
        "skewgram_length": len(path),
        "skewgram_time_s": elapsed,
        "time_walks_s": timings["walks"],
        "time_training_s": timings["training"],
        "time_decoding_s": timings["decoding"],
        "skewgram_path": path,
    }
    trace_rows = [{"instance": inst, "seed": seed, "walks": w, "best_length": b, "decoding_elapsed_s": t}
                  for w, b, t in trace]
    return row, trace_rows


def run_bio(job):
    (rel_path, organism, pathway, cycle_type), seed = job
    D, G = load_bio(rel_path)
    t0 = time.time()
    exact = NS["exact_longest_dg_path"](D, G)
    exact_time = time.time() - t0
    assert NS["is_DG_consistent"](exact, D, G), rel_path
    t0 = time.time()
    path, _, _ = NS["run_pipeline"](D, G, seed=seed)
    elapsed = time.time() - t0
    assert NS["is_DG_consistent"](path, D, G), (rel_path, seed)
    row = {
        "organism": organism, "pathway": pathway, "rel_path": rel_path, "cycle_type": cycle_type,
        "n_nodes": D.number_of_nodes(), "n_arcs_D": D.number_of_edges(), "n_edges_G": G.number_of_edges(),
        "longest_path_D": nx.dag_longest_path_length(D) + 1,
        "exact_length": len(exact), "exact_time_s": exact_time, "exact_path": exact,
        "seed": seed, "skewgram_length": len(path), "skewgram_time_s": elapsed, "skewgram_path": path,
    }
    return row, []


# ─── Exécution parallèle avec reprise ──────────────────────────────────────────

def run_jobs(jobs, worker, out_csv, key, trace_csv=None, workers=4):
    done_rows, done_traces = [], []
    if os.path.isfile(out_csv):
        done_rows = pd.read_csv(out_csv).to_dict("records")
        if trace_csv and os.path.isfile(trace_csv):
            done_traces = pd.read_csv(trace_csv).to_dict("records")
    done = {key(r) for r in done_rows}
    todo = [j for j in jobs if key(j) not in done]
    print(f"{len(jobs)} tâches, {len(jobs) - len(todo)} déjà faites, {len(todo)} à lancer ({workers} processus)", flush=True)

    rows, traces = list(done_rows), list(done_traces)

    def save():
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        if trace_csv:
            pd.DataFrame(traces).to_csv(trace_csv, index=False)

    t_start = time.time()
    with mp.Pool(workers, initializer=init_worker) as pool:
        for k, (row, trace_rows) in enumerate(pool.imap_unordered(worker, todo), 1):
            rows.append(row)
            traces.extend(trace_rows)
            print(f"[{k}/{len(todo)}] {key(row)} longueur={row['skewgram_length']} "
                  f"({row['skewgram_time_s']:.1f}s, écoulé {time.time() - t_start:.0f}s)", flush=True)
            if k % SAVE_EVERY == 0:
                save()
    save()
    print(f"Terminé : {out_csv}", flush=True)


def synthetic_key(x):
    # x est soit une tâche (inst, seed, kwargs), soit une ligne de résultat
    return (x["instance"], int(x["seed"])) if isinstance(x, dict) else (x[0], x[1])


def by_size_desc(insts):
    """Les plus grosses instances d'abord, pour mieux répartir la charge entre processus."""
    return sorted(insts, key=lambda s: (-int(s.split("_")[0]), int(s.split("_")[1])))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("experiment", choices=["seeds", "ablation", "scaling", "bio"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    args = parser.parse_args()
    results = NS["RESULTS_DIR"]

    if args.experiment in ("seeds", "ablation"):
        insts = by_size_desc(NS["list_instances"]())
        if args.experiment == "seeds":
            jobs = [(i, s, {}) for s in args.seeds for i in insts]
            run_jobs(jobs, run_synthetic, os.path.join(results, "multiseed_results.csv"), synthetic_key,
                     os.path.join(results, "multiseed_traces.csv"), args.workers)
        else:
            jobs = [(i, NS["SEED"], {"use_lookahead": False}) for i in insts]
            run_jobs(jobs, run_synthetic, os.path.join(results, "ablation_results.csv"), synthetic_key,
                     workers=args.workers)

    elif args.experiment == "scaling":
        data_dir = NS["DATA_DIR"]
        insts = by_size_desc([d for d in os.listdir(data_dir)
                              if d.split("_")[0] in ("500", "1000")
                              and os.path.isfile(os.path.join(data_dir, d, "graphD.txt"))])
        jobs = [(i, s, {}) for s in args.seeds for i in insts]
        run_jobs(jobs, run_synthetic, os.path.join(results, "scaling_results.csv"), synthetic_key,
                 os.path.join(results, "scaling_traces.csv"), args.workers)

    elif args.experiment == "bio":
        bench = pd.read_csv(NS["BIO_BENCHMARK_FILE"])
        specs = list(bench[["rel_path", "organism", "pathway", "cycle_type"]].itertuples(index=False, name=None))
        jobs = [(spec, s) for s in args.seeds for spec in specs]
        key = lambda x: (x["rel_path"], int(x["seed"])) if isinstance(x, dict) else (x[0][0], x[1])
        run_jobs(jobs, run_bio, os.path.join(results, "biological_multiseed.csv"), key, workers=args.workers)


if __name__ == "__main__":
    main()
