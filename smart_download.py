"""
smart_download.py
=================
Télécharge directement les instances 100-nœuds manquantes vers data/raw
en utilisant les IDs Google Drive extraits des logs gdown.

- Reprend automatiquement là où il s'arrête
- Télécharge chaque dossier directement par son ID Drive
- S'arrête quand l'objectif (TARGET) est atteint
- Pause PAUSE_S secondes entre chaque instance

Usage :
    python smart_download.py              # cible 200 instances
    python smart_download.py --target 300
"""

import os
import sys
import time
import shutil
import argparse

import gdown

# ─── Paramètres ────────────────────────────────────────────────────────────────
TARGET      = 200
DATA_DIR    = "data/raw"
TMP_DIR     = "data/_tmp_inst"
PAUSE_S     = 1.0   # pause entre chaque dossier (évite quota Drive)
IDS_FILE    = "/tmp/folder_ids_100.txt"   # généré par le grep précédent
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
    """Charge la liste des (folder_id, num) depuis le fichier généré."""
    entries = []
    with open(ids_file) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) == 2:
                fid, num = parts
                entries.append((fid, f"100_{num}"))
    return entries


def download_instance(folder_id, name, tmp_dir=TMP_DIR, data_dir=DATA_DIR):
    """Télécharge un seul dossier d'instance par son ID Drive."""
    dst = os.path.join(data_dir, name)
    if os.path.exists(dst):
        files = [os.path.join(dst, f) for f in ("graphD.txt", "graphG.txt", "solution.txt")]
        if all(os.path.exists(fp) and os.path.getsize(fp) > 0 for fp in files):
            return "skip"

    inst_tmp = os.path.join(tmp_dir, name)
    os.makedirs(inst_tmp, exist_ok=True)

    url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(
            url=url,
            output=inst_tmp,
            quiet=True,
            resume=True,
        )
    except Exception as e:
        err = str(e)
        if "quota" in err.lower() or "many accesses" in err.lower():
            return "quota"
        # Autre erreur — continuer
        return "error"

    # Vérifier les fichiers téléchargés
    required = [os.path.join(inst_tmp, f) for f in ("graphD.txt", "graphG.txt", "solution.txt")]
    if all(os.path.exists(fp) and os.path.getsize(fp) > 0 for fp in required):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.move(inst_tmp, dst)
        return "ok"
    else:
        shutil.rmtree(inst_tmp, ignore_errors=True)
        return "incomplete"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=TARGET)
    args = parser.parse_args()
    target = args.target

    print("=" * 60)
    print(" 📥 Téléchargement ciblé — instances 100-nœuds uniquement")
    print("=" * 60)

    if not os.path.exists(IDS_FILE):
        print(f"[ERREUR] Fichier d'IDs introuvable : {IDS_FILE}")
        print("Lancez d'abord :")
        print('  grep "Retrieving folder .* 100_" logs/task-353.log | sed \'s/Retrieving folder (.*) 100_(.*)/\\1 \\2/\' > /tmp/folder_ids_100.txt')
        sys.exit(1)

    entries = load_ids(IDS_FILE)
    print(f"  IDs Drive disponibles : {len(entries)}")

    existing = set(count_valid(DATA_DIR))
    print(f"  Instances déjà valides : {len(existing)}/{target}")

    if len(existing) >= target:
        print(f"\n✅ Objectif déjà atteint ({len(existing)} ≥ {target}). Rien à faire.")
        return

    os.makedirs(TMP_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    ok = skipped = errors = quota_hits = 0
    consecutive_quota = 0

    for folder_id, name in entries:
        current = count_valid(DATA_DIR)
        if len(current) >= target:
            print(f"\n✅ Objectif atteint : {len(current)} instances valides dans data/raw.")
            break

        result = download_instance(folder_id, name)

        if result == "skip":
            skipped += 1
            continue
        elif result == "ok":
            ok += 1
            consecutive_quota = 0
            total = len(count_valid(DATA_DIR))
            print(f"  ✓ {name}  [{total}/{target}]")
            time.sleep(PAUSE_S)
        elif result == "quota":
            quota_hits += 1
            consecutive_quota += 1
            print(f"  ⚠ Quota Drive atteint après {ok} téléchargements. Pause 30s...")
            time.sleep(30)
            if consecutive_quota >= 3:
                print("  ✗ Quota persistant (3 fois consécutives). Arrêt temporaire.")
                break
        elif result == "incomplete":
            errors += 1
            print(f"  ✗ Incomplet : {name}")
            time.sleep(PAUSE_S)
        else:
            errors += 1
            print(f"  ✗ Erreur : {name}")
            time.sleep(PAUSE_S)

    # Nettoyage
    shutil.rmtree(TMP_DIR, ignore_errors=True)

    final = count_valid(DATA_DIR)
    print(f"\n{'=' * 60}")
    print(f" 📊 Résultat")
    print(f"{'=' * 60}")
    print(f"  Instances valides dans data/raw  : {len(final)}")
    print(f"  Téléchargées cette session       : {ok}")
    print(f"  Déjà présentes (skip)            : {skipped}")
    print(f"  Erreurs / incomplètes            : {errors}")
    print(f"  Blocages quota Drive             : {quota_hits}")

    if len(final) >= target:
        print(f"\n  ✅ Objectif {target} atteint ! Ré-exécutez le notebook directement.")
    else:
        remaining = target - len(final)
        print(f"\n  ⚠️  Encore {remaining} instances manquantes (quota Drive).")
        print(f"     Relancez dans quelques minutes : python smart_download.py")


if __name__ == "__main__":
    main()
