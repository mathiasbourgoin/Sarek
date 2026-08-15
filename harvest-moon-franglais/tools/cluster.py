#!/usr/bin/env python3
"""Teste si la table de scripts est structuree: les scripts d'un meme personnage
sont-ils contigus en ID ? Si oui, dater le corpus devient trivial."""

import collections
import re
import sys

CAST = [
    "Ann", "Popuri", "Karen", "Elli", "Mary", "Cliff", "Gray", "Kai", "Rick",
    "Trent", "Zack", "Won", "Duke", "Manna", "Basil", "Anna", "Doug", "Jeff",
    "Sasha", "Barley", "May", "Lillia", "Ellen", "Stu", "Saibara", "Carter",
    "Harris", "Thomas", "Gotz", "Kano", "Van", "Aja", "Cain", "Louis", "Greg",
]


def load(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            p = line.rstrip("\n").split("\t")
            if len(p) == 3:
                rows.append((int(p[0]), int(p[1]), p[2].encode().decode("unicode_escape")))
    return rows


def runs(ids):
    """Regroupe une liste triee d'ids en plages contigues (tolerance de 2)."""
    out = []
    for i in sorted(ids):
        if out and i - out[-1][1] <= 3:
            out[-1][1] = i
        else:
            out.append([i, i])
    return [(a, b) for a, b in out]


def main(argv):
    rows = load(argv[1] if len(argv) > 1 else "strings_en.tsv")

    by_script = collections.defaultdict(list)
    for sid, _, t in rows:
        by_script[sid].append(t)

    hits = collections.defaultdict(set)
    for sid, texts in by_script.items():
        blob = " ".join(texts)
        for name in CAST:
            if re.search(rf"\b{name}\b", blob):
                hits[name].add(sid)

    print("=" * 70)
    print("REGROUPEMENT PAR PERSONNAGE (scripts citant le nom)")
    print("=" * 70)
    print(f"{'personnage':12s} {'scripts':>8s}  {'plages':>7s}  {'concentration':>13s}  plages principales")
    print("-" * 70)

    total_covered = set()
    concentrations = []

    for name in CAST:
        ids = hits[name]
        if len(ids) < 4:
            continue
        total_covered |= ids
        rs = runs(ids)
        # concentration = part des scripts situes dans les 3 plus grosses plages
        rs_sorted = sorted(rs, key=lambda r: -(r[1] - r[0] + 1))
        big = rs_sorted[:3]
        in_big = sum(1 for i in ids if any(a <= i <= b for a, b in big))
        conc = in_big / len(ids)
        concentrations.append(conc)
        shown = ", ".join(f"{a}-{b}" if a != b else str(a) for a, b in big)
        print(f"{name:12s} {len(ids):8d}  {len(rs):7d}  {conc:12.0%}   {shown}")

    print("-" * 70)
    print(f"scripts avec au moins un nom du cast : {len(total_covered)}/{len(by_script)} "
          f"({100*len(total_covered)//len(by_script)}%)")
    if concentrations:
        avg = sum(concentrations) / len(concentrations)
        print(f"concentration moyenne dans les 3 plus grosses plages : {avg:.0%}")

    # test complementaire: densite de texte par tranche d'ID
    print()
    print("=" * 70)
    print("DENSITE DE TEXTE PAR TRANCHE DE 100 IDS")
    print("=" * 70)
    for base in range(0, 1400, 100):
        ids = [s for s in by_script if base <= s < base + 100]
        n = sum(len(by_script[s]) for s in ids)
        chars = sum(len(t) for s in ids for t in by_script[s])
        bar = "#" * (chars // 1500)
        print(f"  {base:4d}-{base+99:4d}: {len(ids):3d} scripts, {n:5d} chaines, {chars:7d} car. {bar}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
