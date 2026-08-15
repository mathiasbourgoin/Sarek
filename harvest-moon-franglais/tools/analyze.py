#!/usr/bin/env python3
"""Analyse le corpus extrait: volume, contraintes d'affichage, codes de controle,
et testabilite d'un classement chronologique."""

import collections
import re
import sys

# codes de controle documentes dans FOMT-DOC/TextBoxes.txt
CTRL = {0x05: "attendre A", 0x0A: "line feed", 0x0C: "vider la boite", 0x0D: "retour chariot"}


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                continue
            sid, idx, text = parts
            rows.append((int(sid), int(idx), text.encode().decode("unicode_escape")))
    return rows


def display_lines(text):
    """Decoupe une chaine en lignes telles qu'elles s'affichent."""
    # \x0c vide la boite, \r\n passe a la ligne: les deux terminent une ligne
    flat = text.replace("\x0c", "\r\n").replace("\x05", "")
    # \xff suivi d'un selecteur = substitution (nom du joueur, etc.)
    flat = re.sub(r"\xff.", "@@@@@@@@", flat)  # un nom fait ~8 caracteres
    for chunk in re.split(r"\r\n|\r|\n", flat):
        yield chunk


def main(argv):
    rows = load(argv[1] if len(argv) > 1 else "strings_en.tsv")
    texts = [t for _, _, t in rows]

    print("=" * 62)
    print("VOLUME")
    print("=" * 62)
    uniq = set(texts)
    words = sum(len(re.findall(r"[A-Za-z']+", t)) for t in texts)
    uniq_words = sum(len(re.findall(r"[A-Za-z']+", t)) for t in uniq)
    print(f"  chaines totales        : {len(rows)}")
    print(f"  chaines uniques        : {len(uniq)}  ({100*len(uniq)//len(rows)}%)")
    print(f"  mots (total)           : {words}")
    print(f"  mots (dedupliques)     : {uniq_words}")
    print(f"  scripts contenant du texte : {len(set(s for s, _, _ in rows))}")

    print()
    print("=" * 62)
    print("CONTRAINTE D'AFFICHAGE")
    print("=" * 62)
    widths = collections.Counter()
    for t in texts:
        for line in display_lines(t):
            widths[len(line)] += 1
    over = sorted(k for k in widths if k > 28)
    print(f"  largeur max observee   : {max(widths)} caracteres")
    print(f"  lignes > 28 caracteres : {sum(widths[k] for k in over)}")
    print(f"  repartition 24-32      : {[(k, widths[k]) for k in range(24, 33)]}")

    print()
    print("=" * 62)
    print("CODES DE CONTROLE")
    print("=" * 62)
    subs = collections.Counter()
    for t in texts:
        for m in re.finditer(r"\xff(.)", t):
            subs[m.group(1)] += 1
    for code, name in CTRL.items():
        n = sum(t.count(chr(code)) for t in texts)
        print(f"  \\x{code:02x} {name:18s}: {n}")
    print(f"  substitutions \\xff+X   : {dict(sorted(subs.items(), key=lambda kv: -kv[1]))}")

    print()
    print("=" * 62)
    print("DATABILITE DU CORPUS (test de classement chronologique)")
    print("=" * 62)
    markers = {
        "saison": r"\b(Spring|Summer|Fall|Winter)\b",
        "festival/fete": r"\b(Festival|Contest|Eve|Thanksgiving|Ceremony)\b",
        "mariage/coeurs": r"\b(marry|married|marriage|wedding|Blue Feather|love|date)\b",
        "enfant/bebe": r"\b(baby|pregnan|child)\w*",
        "mine/outil": r"\b(Mine|Hammer|Sickle|Hoe|Axe|Ore|Mystrile)\b",
        "tutoriel/debut": r"\b(welcome|first|how to|remember to|don't forget)\b",
    }
    scripts_with = collections.defaultdict(set)
    for sid, _, t in rows:
        for name, pat in markers.items():
            if re.search(pat, t, re.I):
                scripts_with[name].add(sid)
    total_scripts = len(set(s for s, _, _ in rows))
    covered = set()
    for name, pat in markers.items():
        s = scripts_with[name]
        covered |= s
        print(f"  {name:16s}: {len(s):4d} scripts")
    print(f"  --> au moins un marqueur : {len(covered)}/{total_scripts} scripts "
          f"({100*len(covered)//total_scripts}%)")

    print()
    print("=" * 62)
    print("REPETITIONS (travail economise par deduplication)")
    print("=" * 62)
    dup = collections.Counter(texts)
    top = [(n, t) for t, n in dup.items() if n > 1]
    top.sort(reverse=True)
    print(f"  chaines apparaissant >1 fois : {len(top)}")
    print(f"  occurrences economisees      : {sum(n - 1 for n, _ in top)}")
    for n, t in top[:6]:
        print(f"    x{n:3d}  {t[:52]!r}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
