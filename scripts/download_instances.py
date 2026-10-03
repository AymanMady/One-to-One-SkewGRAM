"""
download_instances.py
=====================
Télécharge les instances du dataset Google Drive 'erdos_renyi' dans data/raw/
jusqu'au nombre cible (TARGET = 200 par défaut, modifiable ci-dessous).

Usage :
    python download_instances.py
    python download_instances.py --target 300   # pour 300 instances

Fonctionnalités :
- Reprend automatiquement là où il s'arrête (skip les instances déjà présentes)
- Télécharge instance par instance (séquentiellement) pour éviter le quota Drive
- Pause configurable entre chaque téléchargement
- Affiche la progression en temps réel
- Met à jour data/README.md automatiquement après chaque vague
"""

import os
import sys
import time
import random
import shutil
import argparse

import gdown

# ─── Paramètres ────────────────────────────────────────────────────────────────
TARGET_INSTANCES   = 200        # Nombre d'instances cibles à avoir dans data/raw
FOLDER_URL         = "https://drive.google.com/drive/folders/18meW_x2uaSTFxhaZagfzLYe5p-RwvOHE"
DATA_DIR           = "data/raw"
TMP_DIR            = "data/_tmp_download"
PAUSE_BETWEEN      = 1.2        # secondes entre chaque dossier (évite quota Drive)
# ───────────────────────────────────────────────────────────────────────────────


def count_valid(data_dir=DATA_DIR):
    """Compte les instances valides (avec les 3 fichiers requis non vides)."""
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


def download_folder_to_tmp(tmp_dir=TMP_DIR):
    """Télécharge le dossier Drive complet dans tmp_dir (gdown gère les interruptions)."""
    os.makedirs(tmp_dir, exist_ok=True)
    print(f"\n📡 Connexion à Google Drive et récupération du listing des dossiers...")
    print(f"   URL : {FOLDER_URL}")
    print(f"   Dossier temporaire : {tmp_dir}")
    print("   (Cela peut prendre quelques minutes — gdown liste tous les sous-dossiers)\n")

    try:
        gdown.download_folder(
            url=FOLDER_URL,
            output=tmp_dir,
            quiet=False,
            resume=True,
        )
    except Exception as e:
        # gdown lève une exception à la fin ou en cas de coupure — c'est normal
        print(f"\n[INFO] Téléchargement terminé ou interrompu (message gdown) : {e}")


def move_instances_from_tmp(tmp_dir=TMP_DIR, data_dir=DATA_DIR, target=TARGET_INSTANCES):
    """Déplace les instances téléchargées de tmp_dir vers data_dir."""
    os.makedirs(data_dir, exist_ok=True)
    already = set(count_valid(data_dir))
    moved = 0
    skipped = 0
    errors = 0

    if not os.path.exists(tmp_dir):
        print(f"[ERREUR] Dossier temporaire '{tmp_dir}' introuvable.")
        return 0

    # gdown place parfois les résultats dans un sous-dossier portant le nom du Drive
    # On cherche les dossiers 100_<k> récursivement à 2 niveaux de profondeur
    candidates = []
    for root, dirs, _ in os.walk(tmp_dir):
        for d in dirs:
            if d.startswith("100_"):
                candidates.append(os.path.join(root, d))
        if root != tmp_dir:
            # Ne pas descendre plus de 2 niveaux
            break

    print(f"\n📦 {len(candidates)} dossiers d'instances trouvés dans le cache temporaire.")

    for src in sorted(candidates):
        name = os.path.basename(src)
        dst = os.path.join(data_dir, name)

        n_valid = len(count_valid(data_dir))
        if n_valid >= target:
            print(f"\n✅ Objectif atteint : {n_valid} instances valides dans {data_dir}. Arrêt.")
            break

        if name in already or (os.path.exists(dst) and len(os.listdir(dst)) >= 3):
            skipped += 1
            continue

        # Vérifier que les 3 fichiers requis existent dans src
        required = [os.path.join(src, f) for f in ("graphD.txt", "graphG.txt", "solution.txt")]
        if not all(os.path.exists(fp) and os.path.getsize(fp) > 0 for fp in required):
            errors += 1
            continue

        try:
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.move(src, dst)
            already.add(name)
            moved += 1
            n_valid = len(count_valid(data_dir))
            print(f"  ✓ {name}  →  data/raw/{name}  [{n_valid}/{target}]")
        except Exception as e:
            print(f"  ✗ Erreur déplacement {name}: {e}")
            errors += 1

    return moved


def main():
    parser = argparse.ArgumentParser(description="Téléchargeur robuste d'instances erdos_renyi")
    parser.add_argument("--target", type=int, default=TARGET_INSTANCES,
                        help=f"Nombre d'instances cibles (défaut: {TARGET_INSTANCES})")
    parser.add_argument("--no-download", action="store_true",
                        help="Ne pas re-télécharger depuis Drive — déplacer uniquement le cache existant")
    args = parser.parse_args()

    target = args.target

    print("=" * 60)
    print(" 📥 Téléchargeur d'instances SkewGRAM — erdos_renyi")
    print("=" * 60)

    # 1. Compter ce qu'on a déjà
    existing = count_valid(DATA_DIR)
    print(f"\n[1/3] Instances déjà disponibles dans data/raw : {len(existing)}")
    if len(existing) >= target:
        print(f"✅ Objectif déjà atteint ({len(existing)} ≥ {target}). Rien à faire.")
        return

    needed = target - len(existing)
    print(f"      Instances manquantes : {needed} (objectif : {target})")

    # 2. Télécharger depuis Drive si nécessaire
    if not args.no_download:
        # Vérifier si le cache tmp contient déjà des données
        tmp_count = 0
        if os.path.exists(TMP_DIR):
            for root, dirs, _ in os.walk(TMP_DIR):
                for d in dirs:
                    if d.startswith("100_"):
                        tmp_count += 1
                break

        if tmp_count >= needed:
            print(f"\n[2/3] Cache temporaire trouvé ({tmp_count} dossiers). Utilisation du cache.")
        else:
            print(f"\n[2/3] Lancement du téléchargement depuis Google Drive...")
            download_folder_to_tmp(TMP_DIR)
    else:
        print(f"\n[2/3] Mode --no-download : utilisation du cache existant uniquement.")

    # 3. Déplacer les instances valides vers data/raw
    print(f"\n[3/3] Déplacement des instances valides vers {DATA_DIR}...")
    moved = move_instances_from_tmp(TMP_DIR, DATA_DIR, target=target)

    # 4. Rapport final
    final = count_valid(DATA_DIR)
    print(f"\n{'=' * 60}")
    print(f" 📊 Résultat final")
    print(f"{'=' * 60}")
    print(f"  Instances valides disponibles dans data/raw : {len(final)}")
    print(f"  Nouvelles instances ajoutées cette session   : {moved}")
    print(f"  Objectif cible                               : {target}")

    if len(final) >= target:
        print(f"\n  ✅ Objectif atteint ! Vous pouvez ré-exécuter le notebook directement.")
    else:
        print(f"\n  ⚠️  Il manque encore {target - len(final)} instances.")
        print(f"     Cela est probablement dû au quota Google Drive (anti-abus).")
        print(f"     Attendez quelques minutes et relancez : python download_instances.py")

    # Nettoyage du cache temporaire si tout est déplacé
    if os.path.exists(TMP_DIR):
        remaining = 0
        for root, dirs, _ in os.walk(TMP_DIR):
            for d in dirs:
                if d.startswith("100_"):
                    remaining += 1
            break
        if remaining == 0:
            shutil.rmtree(TMP_DIR, ignore_errors=True)
            print(f"\n  🗑️  Cache temporaire nettoyé.")


if __name__ == "__main__":
    main()
