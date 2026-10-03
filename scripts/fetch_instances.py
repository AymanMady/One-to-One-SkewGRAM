"""
fetch_instances.py
==================
Télécharge les instances 100-nœuds manquantes vers data/raw,
fichier par fichier (graphD.txt, graphG.txt, solution.txt),
en utilisant les IDs Google Drive individuels extraits des logs gdown.

- Évite le quota Drive : télécharge fichier par fichier (pas dossier entier)
- Reprend automatiquement : skip les instances déjà valides
- S'arrête dès que TARGET instances sont atteintes
- Pause PAUSE_S secondes entre chaque instance

Usage :
    python fetch_instances.py              # cible 200 instances
    python fetch_instances.py --target 300
"""

import os
import sys
import time
import shutil
import argparse

import gdown

# ─── Paramètres ────────────────────────────────────────────────────────────────
TARGET   = 200
DATA_DIR = "data/raw"
TMP_DIR  = "data/_tmp_inst"
IDS_FILE = "/tmp/file_ids_100.tsv"   # généré par le script d'extraction
PAUSE_S  = 1.2   # pause entre chaque instance (évite quota)
# ───────────────────────────────────────────────────────────────────────────────


def count_valid(data_dir=DATA_DIR):
    valid = []
    if not os.path.exists(data_dir):
        return valid
    for name in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, name)
        if os.path.isdir(d) and name.startswith("100_"):
            files = [os.path.join(d, f) for f in ("graphD.txt", "graphG.txt", "solution.txt")]
            if all(os.path.exists(fp) and os.path.getsize(fp) > 0 for fp in files):
                valid.append(name)
    return valid


def load_ids(ids_file=IDS_FILE):
    entries = []
    with open(ids_file) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) == 4:
                name, d_id, g_id, s_id = parts
                entries.append((name, d_id, g_id, s_id))
    # Sort numerically by instance number
    entries.sort(key=lambda x: int(x[0].split("_")[1]))
    return entries


def download_file(file_id, dest_path, quiet=True):
    """Télécharge un fichier individuel par son ID Drive."""
    url = f"https://drive.google.com/uc?id={file_id}"
    try:
        result = gdown.download(url=url, output=dest_path, quiet=quiet)
        if result and os.path.exists(dest_path) and os.path.getsize(dest_path) > 0:
            return True
        return False
    except Exception:
        return False


def download_instance(name, d_id, g_id, s_id, tmp_dir=TMP_DIR, data_dir=DATA_DIR):
    """Télécharge les 3 fichiers d'une instance, fichier par fichier."""
    dst = os.path.join(data_dir, name)

    # Skip si déjà valide
    if os.path.exists(dst):
        files = [os.path.join(dst, fn) for fn in ("graphD.txt", "graphG.txt", "solution.txt")]
        if all(os.path.exists(fp) and os.path.getsize(fp) > 0 for fp in files):
            return "skip"

    inst_tmp = os.path.join(tmp_dir, name)
    os.makedirs(inst_tmp, exist_ok=True)

    file_map = [
        (d_id, "graphD.txt"),
        (g_id, "graphG.txt"),
        (s_id, "solution.txt"),
    ]

    for fid, fname in file_map:
        dest = os.path.join(inst_tmp, fname)
        ok = download_file(fid, dest, quiet=True)
        if not ok:
            # Quota ou erreur réseau
            shutil.rmtree(inst_tmp, ignore_errors=True)
            return "quota"

    # Vérifier que les 3 fichiers sont bien présents et non-vides
    required = [os.path.join(inst_tmp, fn) for fn in ("graphD.txt", "graphG.txt", "solution.txt")]
    if all(os.path.exists(fp) and os.path.getsize(fp) > 0 for fp in required):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(inst_tmp, dst)
        return "ok"
    else:
        shutil.rmtree(inst_tmp, ignore_errors=True)
        return "incomplete"


def main():
    parser = argparse.ArgumentParser(
        description="Téléchargeur d'instances 100-nœuds par fichier individuel")
    parser.add_argument("--target", type=int, default=TARGET,
                        help=f"Nombre d'instances cibles (défaut: {TARGET})")
    args = parser.parse_args()
    target = args.target

    print("=" * 62)
    print(" 📥  Fetch Instances — téléchargement fichier par fichier")
    print("=" * 62)

    if not os.path.exists(IDS_FILE):
        print(f"\n[ERREUR] Fichier d'IDs introuvable : {IDS_FILE}")
        print("Régénérez-le avec :")
        print("  python3 extract_ids.py")
        sys.exit(1)

    entries = load_ids(IDS_FILE)
    print(f"  IDs disponibles       : {len(entries)} instances")

    existing = set(count_valid(DATA_DIR))
    print(f"  Déjà valides          : {len(existing)}/{target}")

    if len(existing) >= target:
        print(f"\n✅ Objectif déjà atteint ({len(existing)} ≥ {target}).")
        return

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    ok_count = skip_count = error_count = quota_count = 0
    consecutive_quota = 0
    print()

    for name, d_id, g_id, s_id in entries:
        current_valid = count_valid(DATA_DIR)
        if len(current_valid) >= target:
            print(f"\n✅ Objectif atteint : {len(current_valid)} instances valides.")
            break

        result = download_instance(name, d_id, g_id, s_id)

        if result == "skip":
            skip_count += 1
            continue

        elif result == "ok":
            ok_count += 1
            consecutive_quota = 0
            total = len(count_valid(DATA_DIR))
            print(f"  ✓ {name:<12}  [{total:>3}/{target}]")
            time.sleep(PAUSE_S)

        elif result == "quota":
            quota_count += 1
            consecutive_quota += 1
            wait = min(30 * consecutive_quota, 120)
            print(f"  ⚠  Quota Drive sur {name}. Pause {wait}s "
                  f"({consecutive_quota} blocage(s) consécutif(s))...")
            time.sleep(wait)
            # Retenter cette instance
            result2 = download_instance(name, d_id, g_id, s_id)
            if result2 == "ok":
                ok_count += 1
                consecutive_quota = 0
                total = len(count_valid(DATA_DIR))
                print(f"  ✓ {name:<12}  [{total:>3}/{target}] (après retry)")
                time.sleep(PAUSE_S)
            elif consecutive_quota >= 4:
                print(f"\n  ✗ Quota persistant ({consecutive_quota}x). Arrêt.")
                print(f"    Relancez dans quelques minutes : python fetch_instances.py")
                break
            else:
                error_count += 1

        else:  # incomplete
            error_count += 1
            print(f"  ✗ Incomplet : {name}")
            time.sleep(PAUSE_S)

    # Nettoyage cache temporaire
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    final = count_valid(DATA_DIR)
    print(f"\n{'=' * 62}")
    print(f" 📊  Résultat")
    print(f"{'=' * 62}")
    print(f"  Instances valides dans data/raw   : {len(final)}")
    print(f"  Téléchargées cette session        : {ok_count}")
    print(f"  Déjà présentes (skippées)         : {skip_count}")
    print(f"  Erreurs                           : {error_count}")
    print(f"  Blocages quota Drive              : {quota_count}")

    if len(final) >= target:
        print(f"\n  ✅ Objectif {target} atteint !")
        print(f"     Vous pouvez ré-exécuter le notebook directement.")
    else:
        remaining = target - len(final)
        print(f"\n  ⚠️   Encore {remaining} instances manquantes (quota Drive).")
        print(f"      Relancez dans quelques minutes :")
        print(f"      python fetch_instances.py --target {target}")


if __name__ == "__main__":
    main()
