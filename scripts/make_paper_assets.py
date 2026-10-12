"""
make_paper_assets.py
====================
Figures et tableaux de l'article, en anglais (paper/en/) ou en français (paper/fr/). Les figures sont
dessinées à la largeur de texte LNCS (12,2 cm = 4,8 in) : incluses à l'échelle 1, les polices gardent
leur taille (7-8 pt). Les tableaux LaTeX (paper/<langue>/tables/*.tex) et les chiffres cités dans le
texte sont calculés directement à partir des CSV de results/ (produits par le notebook et
scripts/run_experiments.py).

    python scripts/make_paper_assets.py                  # tout, en anglais
    python scripts/make_paper_assets.py --lang fr        # tout, en français
    python scripts/make_paper_assets.py problem main     # certaines pièces (voir ASSETS en bas du fichier)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import networkx as nx
import numpy as np
import pandas as pd

LANG = "en"  # remplacé par --lang
OUT_DIR = "paper/en"
TABLE_DIR = os.path.join(OUT_DIR, "tables")
RESULTS_DIR = "results"

# Libellés des figures et tableaux : clé -> (anglais, français)
TEXT = {
    "d_title": ("$D$ (directed, acyclic): causal order", "$D$ (orienté, acyclique) : ordre causal"),
    "g_title": ("$G$ (undirected): functional association", "$G$ (non orienté) : association fonctionnelle"),
    "d_note": ("5→7 extends the\npath in $D$, but 7\nhas no $G$-neighbour\nin $\\{1,\\ldots,5\\}$",
               "5→7 prolonge le\nchemin dans $D$, mais\n7 n'a aucun voisin\ndans $G$ parmi $\\{1,\\ldots,5\\}$"),
    "g_note": ("1 and 2 are not adjacent in $G$,\nyet $G[\\{1,\\ldots,5\\}]$ is connected",
               "1 et 2 ne sont pas adjacents dans $G$,\nmais $G[\\{1,\\ldots,5\\}]$ est connexe"),
    "box1": (("1. Random walks", "follow the arcs of $D$\nwalk = sentence\nhub sub-sampling"),
             ("1. Marches", "aléatoires sur $D$\nmarche = phrase\nhubs sous-pondérés")),
    "box2": (("2. Skip-Gram", "structural negatives\n(non-successors in $D$)\n→ $\\mathbf{V}_{in}, \\mathbf{V}_{out}$"),
             ("2. Skip-Gram", "négatifs structurels\n(non-successeurs de $D$)\n→ $\\mathbf{V}_{in}, \\mathbf{V}_{out}$")),
    "box3": (("3. Guided decoding", "lookahead lf$(v)$\n+ embedding similarity\n+ $G$-connectivity bonus"),
             ("3. Décodage guidé", "anticipation lf$(v)$\n+ similarité apprise\n+ bonus de connexité $G$")),
    "box4": (("4. Exact extraction", "longest sub-path\nconnected in $G$\n(Union-Find)"),
             ("4. Extraction", "plus long sous-chemin\nconnexe dans $G$\n(exacte, Union-Find)")),
    "encoder": ("Encoder: learned on $D$ only", "Encodeur : appris sur $D$ seulement"),
    "decoder": ("Decoder: uses $D$, embeddings and $G$", "Décodeur : utilise $D$, les embeddings et $G$"),
    "missing": ("vertices missing vs. ILP2 solution", "sommets manquants par rapport à ILP2"),
    "share_runs": ("share of runs (%)", "part des exécutions (%)"),
    "gap_title": ("(a) Gap to ILP2", "(a) Écart à ILP2"),
    "equal_time": ("equal time", "temps égal"),
    "ilp2_time": ("ILP2 time (s)", "temps ILP2 (s)"),
    "sg_time": ("Skew-GRAM time (s)", "temps Skew-GRAM (s)"),
    "runtime_title": ("(b) Runtime per instance", "(b) Temps par instance"),
    "walks_axis": ("decoding walks $W$", "marches de décodage $W$"),
    "rel_len": ("relative length (%)", "longueur relative (%)"),
    "rel_title": ("(a) Length relative to $W$ = 30k", "(a) Longueur relative à $W$ = 30k"),
    "reach_axis": ("runs reaching ILP2 (%)", "part des exécutions (%)"),
    "reach_title": ("(b) ILP2 length reached vs. budget", "(b) Longueur ILP2 atteinte selon $W$"),
    # tableau principal
    "t_inst": ("Instances with an ILP2 solution", "Instances avec une solution ILP2"),
    "t_mean_len": (r"\quad Mean length (vertices)", r"\quad Longueur moyenne (sommets)"),
    "t_mean_time": (r"\quad Mean time (s)", r"\quad Temps moyen (s)"),
    "t_gap": (r"\quad Mean gap to ILP2 (\%)", r"\quad Écart moyen à ILP2 (\%)"),
    "t_reach": (r"\quad Runs reaching the ILP2 length (\%)", r"\quad Exécutions atteignant ILP2 (\%)"),
    "t_within": (r"\quad Runs within one vertex of ILP2 (\%)", r"\quad Exécutions à 1 sommet au plus (\%)"),
    "t_speedup": ("Median speedup over ILP2", "Accélération médiane / ILP2"),
    # tableau par instance
    "t_length": ("Length (vertices)", "Longueur (sommets)"),
    "t_gapcol": ("Gap", "Écart"),
    "t_time": ("Time (s)", "Temps (s)"),
    # tableau KEGG
    "t_pathways": ("Pathways", "Voies"),
    "t_acyclic": ("Acyclic", "Acycliques"),
    "t_2cycles": ("With 2-cycles", "Avec 2-cycles"),
    "t_all": ("All", "Toutes"),
    "t_hits": ("Hits", "Atteint"),
    # tableau passage à l'échelle
    "t_lenshort": ("Length", "Longueur"),
    "t_vertices": ("(vertices)", "(sommets)"),
    "t_time_run": ("Time per run (s)", "Temps par exécution (s)"),
    "t_walks": ("Walks", "Marches"),
    "t_training": ("Training", "Entraîn."),
    "t_decoding": ("Decoding", "Décodage"),
}


def t(key):
    en, fr = TEXT[key]
    return fr if LANG == "fr" else en


def num(x, d):
    """Nombre formaté pour une cellule de texte (virgule décimale en français)."""
    s = f"{x:.{d}f}"
    return s.replace(".", "{,}") if LANG == "fr" else s

# Palette (validée avec le skill dataviz) : Skew-GRAM / ILP2, puis rampe ordinale des tailles
SKEW_COLOR = "#2a78d6"
ILP2_COLOR = "#eb6834"
SIZE_COLORS = {100: "#86b6ef", 200: "#3987e5", 500: "#1c5cab", 1000: "#0d366b"}
TEXT_WIDTH = 4.8  # largeur de texte LNCS, en pouces
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#b4b2ab"
CRITICAL = "#d03b3b"

plt.rcParams.update({
    "font.size": 7.5,
    "axes.titlesize": 7.5,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.8,
    "axes.edgecolor": INK_2,
    "axes.labelcolor": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
})


def save(fig, name):
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("écrit :", path)


# ─── Figure 1 : exemple jouet ──────────────────────────────────────────────────

TOY_D = [(1, 2), (2, 3), (3, 4), (4, 5), (1, 3), (2, 5), (3, 6), (6, 7), (5, 7)]
TOY_G = [(1, 3), (2, 3), (3, 5), (4, 5), (5, 6), (6, 7)]
TOY_OPT = [1, 2, 3, 4, 5]
NODE_SIZE = 170


def check_toy():
    """Vérifie les affirmations de la légende : optimum unique, plus long chemin de D non consistant."""
    D, G = nx.DiGraph(TOY_D), nx.Graph(TOY_G)
    paths = [p for s in D for t in D for p in nx.all_simple_paths(D, s, t)] + [[v] for v in D]
    consistent = [p for p in paths if nx.is_connected(G.subgraph(p))]
    best = max(len(p) for p in consistent)
    assert [p for p in consistent if len(p) == best] == [TOY_OPT]
    assert nx.dag_longest_path(D) == TOY_OPT + [7]
    assert not G.has_edge(1, 2)  # le préfixe (1, 2) est déconnecté dans G


def fig_problem():
    check_toy()
    pos = {1: (0.0, 1.0), 2: (1.0, 1.55), 3: (2.0, 1.0), 4: (3.0, 1.55), 5: (4.0, 1.0),
           6: (2.0, 0.0), 7: (4.0, 0.0)}
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.75))
    opt_set = set(TOY_OPT)
    opt_arcs = list(zip(TOY_OPT, TOY_OPT[1:]))

    def draw_nodes(ax, graph):
        others = [v for v in graph if v not in opt_set]
        nx.draw_networkx_nodes(graph, pos, nodelist=TOY_OPT, ax=ax, node_color=SKEW_COLOR,
                               edgecolors="white", linewidths=1.0, node_size=NODE_SIZE)
        nx.draw_networkx_nodes(graph, pos, nodelist=others, ax=ax, node_color="#f0efec",
                               edgecolors=INK_2, linewidths=0.6, node_size=NODE_SIZE)
        nx.draw_networkx_labels(graph, pos, ax=ax, font_size=6.8,
                                font_color={v: ("white" if v in opt_set else INK) for v in graph})

    # D : orienté
    D = nx.DiGraph(TOY_D)
    ax = axes[0]
    rest = [e for e in TOY_D if e not in opt_arcs and e != (5, 7)]
    common = dict(ax=ax, arrows=True, arrowstyle="-|>", node_size=NODE_SIZE, connectionstyle="arc3,rad=0.12")
    nx.draw_networkx_edges(D, pos, edgelist=rest, edge_color=MUTED, width=0.8, arrowsize=6, **common)
    nx.draw_networkx_edges(D, pos, edgelist=[(5, 7)], edge_color=CRITICAL, width=1.0, style="dashed",
                           arrowsize=6, **common)
    nx.draw_networkx_edges(D, pos, edgelist=opt_arcs, edge_color=SKEW_COLOR, width=1.6, arrowsize=7, **common)
    draw_nodes(ax, D)
    ax.text(4.3, 0.5, t("d_note"), fontsize=6.2, color=INK_2, va="center", ha="left")
    ax.set_title(t("d_title"), color=INK)

    # G : non orienté
    G = nx.Graph(TOY_G)
    ax = axes[1]
    induced = [(u, v) for u, v in TOY_G if u in opt_set and v in opt_set]
    rest = [e for e in TOY_G if e not in induced]
    nx.draw_networkx_edges(G, pos, edgelist=rest, ax=ax, edge_color=MUTED, width=0.8)
    nx.draw_networkx_edges(G, pos, edgelist=induced, ax=ax, edge_color=SKEW_COLOR, width=1.6)
    draw_nodes(ax, G)
    ax.text(0.45, 1.85, t("g_note"), fontsize=6.2, color=INK_2, ha="center", va="bottom")
    ax.set_title(t("g_title"), color=INK)

    for ax in axes:
        ax.set_xlim(-0.35, 5.75)
        ax.set_ylim(-0.3, 2.45)
        ax.axis("off")
    fig.tight_layout(w_pad=0.5)
    save(fig, "fig1_problem.pdf")


# ─── Figure 2 : pipeline (encodeur / décodeur) ─────────────────────────────────

def fig_pipeline():
    fig = plt.figure(figsize=(TEXT_WIDTH, 1.42))
    ax = fig.add_axes([0, 0, 1, 1])  # sans les marges par défaut : les boîtes occupent toute la largeur
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 31)
    ax.axis("off")

    enc_fill, enc_edge = "#e8f1fc", SKEW_COLOR
    dec_fill, dec_edge = "#e5f6ee", "#14875e"
    boxes = [(x, *t(key), fill, edge) for x, key, fill, edge in (
        (0.5, "box1", enc_fill, enc_edge), (25.5, "box2", enc_fill, enc_edge),
        (51.5, "box3", dec_fill, dec_edge), (76.5, "box4", dec_fill, dec_edge))]
    w, h, y = 23.0, 20.5, 0.5
    for x, title, body, fill, edge in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.0",
                                    facecolor=fill, edgecolor=edge, linewidth=0.9))
        ax.text(x + w / 2, y + h - 3.4, title, ha="center", va="center", fontsize=7.0, weight="bold", color=INK)
        ax.text(x + w / 2, y + (h - 5.8) / 2, body, ha="center", va="center", fontsize=6.3, color=INK_2,
                linespacing=1.4)
    for x in (23.5, 49.0, 74.5):
        ax.add_patch(FancyArrowPatch((x + 0.1, y + h / 2), (x + 2.0, y + h / 2), arrowstyle="-|>",
                                     mutation_scale=6, color=INK_2, linewidth=0.9))

    # Accolades : encodeur (D seulement) / décodeur (D et G)
    for x0, x1, label, color in ((0.5, 48.5, t("encoder"), enc_edge), (51.5, 99.5, t("decoder"), dec_edge)):
        yb = y + h + 2.6
        ax.plot([x0, x0, x1, x1], [yb - 1.0, yb, yb, yb - 1.0], color=color, linewidth=0.9)
        ax.text((x0 + x1) / 2, yb + 1.0, label, ha="center", va="bottom", fontsize=7.0, color=INK)
    save(fig, "fig2_pipeline.pdf")


# ─── Résultats : chargement et agrégats ────────────────────────────────────────

def load_runs(name):
    df = pd.read_csv(os.path.join(RESULTS_DIR, name))
    df["n"] = df["n_nodes"]
    df["missing"] = df["ilp2_length"] - df["skewgram_length"]
    df["gap_pct"] = 100 * (df["skewgram_length"] - df["ilp2_length"]) / df["ilp2_length"]
    return df


def ilp2_summary(runs):
    """Une ligne par (taille, graine) puis agrégats : moyenne et écart-type entre graines."""
    found = runs[runs["ilp2_status"] == "found"]
    out = {}
    for n, g in found.groupby("n"):
        per_seed = g.groupby("seed").agg(
            length=("skewgram_length", "mean"), gap=("gap_pct", "mean"),
            # « atteint » = longueur >= ILP2 : certaines solutions ILP2 à 200 nœuds ne sont pas optimales
            opt=("missing", lambda m: int((m <= 0).sum())), within1=("missing", lambda m: int((m <= 1).sum())),
            time=("skewgram_time_s", "mean"))
        by_inst = g.groupby("instance")
        inst = by_inst.agg(ilp2_length=("ilp2_length", "first"), ilp2_time=("ilp2_time_s", "first"),
                           sg_mean=("skewgram_length", "mean"), sg_sd=("skewgram_length", "std"),
                           sg_best=("skewgram_length", "max"), sg_time=("skewgram_time_s", "mean"),
                           sg_time_sd=("skewgram_time_s", "std"), gap=("gap_pct", "mean"),
                           n_distinct=("skewgram_length", "nunique"))
        speedup = inst["ilp2_time"] / inst["sg_time"]
        out[n] = {
            "n_found": len(inst), "n_total": runs[runs["n"] == n]["instance"].nunique(),
            "n_seeds": g["seed"].nunique(),
            "ilp2_length": inst["ilp2_length"].mean(), "ilp2_time": inst["ilp2_time"].mean(),
            "ilp2_time_median": inst["ilp2_time"].median(),
            "sg_length": per_seed["length"].mean(), "sg_length_sd": per_seed["length"].std(),
            "gap": per_seed["gap"].mean(), "gap_sd": per_seed["gap"].std(),
            "opt": per_seed["opt"].mean(), "opt_sd": per_seed["opt"].std(),
            "within1": per_seed["within1"].mean(), "within1_sd": per_seed["within1"].std(),
            "sg_time": g["skewgram_time_s"].mean(), "sg_time_sd": g["skewgram_time_s"].std(),
            "sg_time_inst_sd": inst["sg_time_sd"].mean(),
            "inst_length_sd": inst["sg_sd"].mean(), "share_constant": (inst["n_distinct"] == 1).mean(),
            "best_of_seeds_length": inst["sg_best"].mean(),
            "best_of_seeds_opt": int((inst["sg_best"] >= inst["ilp2_length"]).sum()),
            "speedup_median": speedup.median(), "n_faster": int((speedup > 1).sum()),
            "n_above_ilp2": int((g["missing"] < 0).sum()),
            "instances_above_ilp2": sorted(g.loc[g["missing"] < 0, "instance"].unique()),
            "instances": inst,
        }
    return out


def write_table(name, body):
    os.makedirs(TABLE_DIR, exist_ok=True)
    path = os.path.join(TABLE_DIR, name)
    with open(path, "w") as f:
        f.write("% Généré par scripts/make_paper_assets.py — ne pas modifier à la main.\n" + body)
    print("écrit :", path)


def print_numbers(title, d):
    print(f"\n== {title} ==")
    for k, v in d.items():
        if not isinstance(v, (pd.DataFrame, pd.Series)):
            print(f"  {k:<24} {v:.4f}" if isinstance(v, float) else f"  {k:<24} {v}")


# ─── Tableau principal (comparaison à ILP2) ────────────────────────────────────

def table_main():
    S = ilp2_summary(load_runs("multiseed_results.csv"))
    sizes = sorted(S)
    for n in sizes:
        print_numbers(f"{n} sommets", S[n])

    def pm(mean, sd, d):
        return f"{mean:.{d}f} & {sd:.{d}f}"

    def single(text):
        return f"\\multicolumn{{2}}{{c}}{{{text}}}"

    def row(label, f):
        return f"{label} & " + " & ".join(f(S[n]) for n in sizes) + r" \\" + "\n"

    share = lambda x, k: pm(100 * x[k] / x["n_found"], 100 * x[k + "_sd"] / x["n_found"], 1)
    cols = " ".join(r"S[table-format=-3.2]@{\,$\pm$\,}S[table-format=1.2]" for _ in sizes)
    body = ("\\setlength{\\tabcolsep}{7pt}\n"
            + r"\begin{tabular}{@{}l " + cols + "@{}}\n\\toprule\n"
            + " & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$n={n}$}}" for n in sizes) + r" \\" + "\n\\midrule\n"
            + row(t("t_inst"), lambda x: single(f"{x['n_found']} / {x['n_total']}"))
            + "\\addlinespace\n\\emph{ILP2} \\\\\n"
            + row(t("t_mean_len"), lambda x: single(num(x["ilp2_length"], 2)))
            + row(t("t_mean_time"), lambda x: single(num(x["ilp2_time"], 1)))
            + "\\addlinespace\n\\emph{Skew-GRAM} \\\\\n"
            + row(t("t_mean_len"), lambda x: pm(x["sg_length"], x["sg_length_sd"], 2))
            + row(t("t_gap"), lambda x: pm(x["gap"], x["gap_sd"], 1))
            + row(t("t_reach"), lambda x: share(x, "opt"))
            + row(t("t_within"), lambda x: share(x, "within1"))
            + row(t("t_mean_time"), lambda x: pm(x["sg_time"], x["sg_time_sd"], 1))
            + "\\addlinespace\n"
            + row(t("t_speedup"), lambda x: single(f"{num(x['speedup_median'], 1)}$\\times$"))
            + "\\bottomrule\n\\end{tabular}\n")
    write_table("tab_main.tex", body)
    return S


# ─── Tableau par instance (corrigé : colonnes alignées, vrais signes moins) ────

QUANTILES = [0.0, 0.5, 0.8, 0.95, 1.0]


def representative_instances(inst):
    """Instances aux quantiles 0, 50, 80, 95 et 100 % de l'écart moyen (de la meilleure à la pire)."""
    ranked = inst.assign(num=[int(i.split("_")[1]) for i in inst.index]).sort_values(["gap", "num"], ascending=[False, True])
    idx = sorted({min(len(ranked) - 1, int(round(q * (len(ranked) - 1)))) for q in QUANTILES})
    return ranked.iloc[idx]


def table_instances():
    S = ilp2_summary(load_runs("multiseed_results.csv"))
    lines = []
    for k, n in enumerate(sorted(S)):
        if k:
            lines.append("\\addlinespace\n")
        for name, r in representative_instances(S[n]["instances"]).iterrows():
            lines.append(f"{name.replace('_', chr(92) + '_')} & {int(r.ilp2_length)} & {r.sg_mean:.1f} & {r.sg_sd:.1f} & "
                         f"{r.gap:.1f} & {r.ilp2_time:.1f} & {r.sg_time:.1f} \\\\\n")
    body = ("\\setlength{\\tabcolsep}{7pt}\n"
            r"\begin{tabular}{@{}l S[table-format=2.0] S[table-format=2.1]@{\,$\pm$\,}S[table-format=1.1] "
            r"S[table-format=-2.1] S[table-format=3.1] S[table-format=2.1]@{}}" + "\n\\toprule\n"
            + f" & \\multicolumn{{3}}{{c}}{{{t('t_length')}}} & {{{t('t_gapcol')}}} & \\multicolumn{{2}}{{c}}{{{t('t_time')}}} \\\\\n"
            r"\cmidrule(lr){2-4} \cmidrule(l){6-7}" + "\n"
            r"{Instance} & {ILP2} & \multicolumn{2}{c}{Skew-GRAM} & {(\%)} & {ILP2} & {Skew-GRAM} \\" + "\n\\midrule\n"
            + "".join(lines) + "\\bottomrule\n\\end{tabular}\n")
    write_table("tab_instances.tex", body)


# ─── Figure 3 : écart à l'optimum et temps de calcul ───────────────────────────

def fig_results():
    runs = load_runs("multiseed_results.csv")
    S = ilp2_summary(runs)
    found = runs[runs["ilp2_status"] == "found"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.95), gridspec_kw={"width_ratios": [1.0, 1.0]})

    # (a) répartition des sommets manquants (toutes les exécutions), en % des exécutions
    ax = axes[0]
    sizes = sorted(S)
    max_missing = int(found["missing"].max())
    bins = list(range(0, min(max_missing, 6) + 1))
    width = 0.8 / len(sizes)
    for k, n in enumerate(sizes):
        m = found[found["n"] == n]["missing"].clip(lower=0, upper=bins[-1])
        share = [100 * (m == b).mean() for b in bins]
        x = np.array(bins) + (k - (len(sizes) - 1) / 2) * width
        ax.bar(x, share, width=width - 0.03, color=SIZE_COLORS[n], label=f"$n={n}$", zorder=2)
    ax.set_xticks(bins)
    ax.set_xticklabels([str(b) for b in bins[:-1]] + [f"≥{bins[-1]}"])
    ax.set_xlabel(t("missing"))
    ax.set_ylabel(t("share_runs"))
    ax.grid(True, axis="y", color="#e6e5e0", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(t("gap_title"), loc="left")

    # (b) temps par instance : ILP2 vs Skew-GRAM (moyenne sur les graines), échelle log
    ax = axes[1]
    for n in sizes:
        inst = S[n]["instances"]
        ax.scatter(inst["ilp2_time"], inst["sg_time"], s=9, color=SIZE_COLORS[n], edgecolor="white",
                   linewidth=0.3, label=f"$n={n}$", zorder=3)
    lo, hi = 1.0, 2000.0
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.7, linestyle="--", zorder=2)
    ax.text(250, 400, t("equal_time"), fontsize=6.3, color=INK_2, rotation=37, ha="center", va="center")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel(t("ilp2_time"))
    ax.set_ylabel(t("sg_time"))
    ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
    ax.legend(frameon=False, loc="upper left", handletextpad=0.2)
    ax.set_title(t("runtime_title"), loc="left")
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig3_results.pdf")


# ─── Réseaux biologiques réels (KEGG) ──────────────────────────────────────────

def table_bio():
    b = pd.read_csv(os.path.join(RESULTS_DIR, "biological_multiseed.csv"))
    b["group"] = np.where(b["cycle_type"].str.startswith("Acyclic"), "acyclic", "reversible")
    groups = [("acyclic", b[b["group"] == "acyclic"]), ("reversible", b[b["group"] == "reversible"]), ("all", b)]
    labels = {"acyclic": t("t_acyclic"), "reversible": t("t_2cycles"), "all": t("t_all")}
    lines = []
    for label, g in groups:
        inst = g.groupby("rel_path").agg(n=("n_nodes", "first"), a=("n_arcs_D", "first"), e=("n_edges_G", "first"),
                                         lp=("longest_path_D", "first"), opt=("exact_length", "first"),
                                         t_exact=("exact_time_s", "first"))
        per_seed = (g.assign(hit=g["skewgram_length"] == g["exact_length"])
                    .groupby("seed").agg(sg=("skewgram_length", "mean"), hit=("hit", "sum")))
        if label == "all":
            lines.append("\\midrule\n")
        lines.append(f"{labels[label]} ({len(inst)}) & {inst.n.min()}--{inst.n.max()} & {inst.a.mean():.1f} & {inst.e.mean():.1f} & "
                     f"{inst.lp.mean():.1f} & {inst.opt.mean():.2f} & {per_seed.sg.mean():.2f} & "
                     f"{per_seed.hit.mean():.0f}/{len(inst)} \\\\\n")
        if label == "all":
            print_numbers("KEGG", {"n": len(inst), "opt_mean": inst.opt.mean(), "sg_mean": per_seed.sg.mean(),
                                   "hit_min": per_seed.hit.min(), "trivial_opt1": int((inst.opt == 1).sum()),
                                   "exact_time_max_ms": 1000 * inst.t_exact.max(), "sg_time": g.skewgram_time_s.mean(),
                                   "sg_time_sd": g.skewgram_time_s.std(), "n_organisms": g.organism.nunique(),
                                   "V_median": inst.n.median()})
    body = ("\\setlength{\\tabcolsep}{5pt}\n"
            r"\begin{tabular}{@{}l c S[table-format=2.1] S[table-format=1.1] S[table-format=2.1] "
            r"S[table-format=1.2] S[table-format=1.2] c@{}}" + "\n\\toprule\n"
            + f"{t('t_pathways')} & $|V|$ & {{$|A|$}} & {{$|E|$}} & {{LP$(D)$}} & {{Optimum}} & {{Skew-GRAM}} & {t('t_hits')} \\\\\n\\midrule\n"
            + "".join(lines) + "\\bottomrule\n\\end{tabular}\n")
    write_table("tab_bio.tex", body)


# ─── Passage à l'échelle : temps par phase et budget de marches ────────────────

def load_all_sizes():
    runs = load_runs("multiseed_results.csv")
    path = os.path.join(RESULTS_DIR, "scaling_results.csv")
    if os.path.isfile(path):
        runs = pd.concat([runs, load_runs("scaling_results.csv")], ignore_index=True)
    return runs


def table_scaling():
    runs = load_all_sizes()
    lines = []
    for n, g in runs.groupby("n"):
        inst = g.groupby("instance").agg(a=("n_arcs_D", "first"), lp=("longest_path_D", "first"))
        per_seed = g.groupby("seed")["skewgram_length"].mean()
        lines.append(f"{n} & {len(inst)} & {inst.a.mean():.0f} & {inst.lp.mean():.1f} & "
                     f"{per_seed.mean():.2f} & {per_seed.std():.2f} & "
                     f"{g.time_walks_s.mean():.2f} & {g.time_training_s.mean():.1f} & {g.time_decoding_s.mean():.1f} & "
                     f"{g.skewgram_time_s.mean():.1f} & {g.skewgram_time_s.std():.1f} \\\\\n")
        print_numbers(f"échelle n={n}", {"inst": len(inst), "arcs": inst.a.mean(), "lp": inst.lp.mean(),
                                         "len": per_seed.mean(), "len_sd": per_seed.std(),
                                         "walks": g.time_walks_s.mean(), "train": g.time_training_s.mean(),
                                         "decode": g.time_decoding_s.mean(), "total": g.skewgram_time_s.mean(),
                                         "decode_share": g.time_decoding_s.sum() / g.skewgram_time_s.sum()})
    body = (f"\\setlength{{\\tabcolsep}}{{{'3.5' if LANG == 'fr' else '4'}pt}}\n"
            r"\begin{tabular}{@{}S[table-format=4.0] S[table-format=3.0] S[table-format=5.0] S[table-format=2.1] "
            r"S[table-format=2.2]@{\,$\pm$\,}S[table-format=1.2] S[table-format=1.2] S[table-format=2.1] S[table-format=2.1] "
            r"S[table-format=3.1]@{\,$\pm$\,}S[table-format=2.1]@{}}" + "\n\\toprule\n"
            + f" & & & & \\multicolumn{{2}}{{c}}{{{t('t_lenshort')}}} & \\multicolumn{{5}}{{c}}{{{t('t_time_run')}}} \\\\\n"
            r"\cmidrule(lr){5-6} \cmidrule(l){7-11}" + "\n"
            + f"{{$n$}} & {{Inst.}} & {{$|A|$}} & {{LP$(D)$}} & \\multicolumn{{2}}{{c}}{{{t('t_vertices')}}} & {{{t('t_walks')}}} & "
            f"{{{t('t_training')}}} & {{{t('t_decoding')}}} & \\multicolumn{{2}}{{c}}{{Total}} \\\\\n\\midrule\n"
            + "".join(lines) + "\\bottomrule\n\\end{tabular}\n")
    write_table("tab_scaling.tex", body)


def load_traces():
    tr = pd.read_csv(os.path.join(RESULTS_DIR, "multiseed_traces.csv"))
    path = os.path.join(RESULTS_DIR, "scaling_traces.csv")
    if os.path.isfile(path):
        tr = pd.concat([tr, pd.read_csv(path)], ignore_index=True)
    tr["n"] = tr["instance"].str.split("_").str[0].astype(int)
    final = tr.groupby(["instance", "seed"])["best_length"].transform("max")
    tr["relative"] = tr["best_length"] / final
    return tr


def fig_budget():
    tr = load_traces()
    runs = load_runs("multiseed_results.csv")
    opt = runs.groupby("instance").agg(ilp2_length=("ilp2_length", "first"), status=("ilp2_status", "first"))
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH, 1.85))

    ax = axes[0]
    for n, g in tr.groupby("n"):
        curve = g.groupby("walks")["relative"].mean()
        ax.plot(curve.index, 100 * curve.values, color=SIZE_COLORS[n], linewidth=1.5, label=f"$n={n}$")
        print_numbers(f"budget n={n}", {f"rel@{k}": 100 * curve.get(k, np.nan) for k in (500, 1000, 3000, 5000, 10000)})
    ax.set_xscale("log")
    ax.set_xlabel(t("walks_axis"))
    ax.set_ylabel(t("rel_len"))
    ax.set_ylim(35, 103)
    ax.legend(frameon=False, loc="lower right")
    ax.set_title(t("rel_title"), loc="left")

    ax = axes[1]
    tro = tr.join(opt, on="instance")
    tro = tro[tro["status"] == "found"]
    for n, g in tro.groupby("n"):
        curve = g.assign(hit=g["best_length"] >= g["ilp2_length"]).groupby("walks")["hit"].mean()
        ax.plot(curve.index, 100 * curve.values, color=SIZE_COLORS[n], linewidth=1.5, label=f"$n={n}$")
        print_numbers(f"budget opt n={n}", {f"opt@{k}": 100 * curve.get(k, np.nan) for k in (500, 1000, 3000, 5000, 10000, 30000)})
    ax.set_xscale("log")
    ax.set_xlabel(t("walks_axis"))
    ax.set_ylabel(t("reach_axis"))
    ax.legend(frameon=False, loc="upper left")
    ax.set_title(t("reach_title"), loc="left")
    for ax in axes:
        ax.grid(True, color="#e6e5e0", linewidth=0.6, zorder=0)
        ax.set_xticks([500, 1000, 3000, 10000, 30000])
        ax.set_xticklabels(["500", "1k", "3k", "10k", "30k"])
        ax.minorticks_off()
    fig.tight_layout(w_pad=1.2)
    save(fig, "fig4_budget.pdf")


# ─── Ablation du lookahead ─────────────────────────────────────────────────────

def ablation_numbers():
    full = load_runs("multiseed_results.csv")
    full = full[full["seed"] == 42].set_index("instance")
    abl = load_runs("ablation_results.csv").set_index("instance")
    j = abl.join(full[["skewgram_length"]], rsuffix="_full")
    j = j[j["ilp2_status"] == "found"]
    for n, g in j.groupby("n"):
        print_numbers(f"ablation n={n}", {
            "inst": len(g), "full": g.skewgram_length_full.mean(), "no_lookahead": g.skewgram_length.mean(),
            "drop_pct": 100 * (g.skewgram_length.mean() / g.skewgram_length_full.mean() - 1),
            "shorter": int((g.skewgram_length < g.skewgram_length_full).sum()),
            "longer": int((g.skewgram_length > g.skewgram_length_full).sum()),
            "opt_full": int((g.skewgram_length_full >= g.ilp2_length).sum()),
            "opt_no_lookahead": int((g.skewgram_length >= g.ilp2_length).sum()),
            "time_full": full.loc[g.index, "skewgram_time_s"].mean(), "time_no_lookahead": g.skewgram_time_s.mean()})


ASSETS = {"problem": fig_problem, "pipeline": fig_pipeline, "main": table_main, "instances": table_instances,
          "results": fig_results, "bio": table_bio, "scaling": table_scaling, "budget": fig_budget,
          "ablation": ablation_numbers}

if __name__ == "__main__":
    args = sys.argv[1:]
    if "--lang" in args:
        i = args.index("--lang")
        LANG = args[i + 1]
        assert LANG in ("en", "fr"), LANG
        del args[i:i + 2]
        OUT_DIR = f"paper/{LANG}"
        TABLE_DIR = os.path.join(OUT_DIR, "tables")
    for name in args or list(ASSETS):
        ASSETS[name]()
