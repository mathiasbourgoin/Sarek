#!/usr/bin/env python3
"""Extrait toutes les chaines des scripts d'evenement de FoMT/MFoMT.

Sortie: TSV (script_id, string_index, texte brut echappe).
Format documente dans StanHash/mary (goodies/script_dump.py) et
StanHash/fomt (src/script_engine.cc).
"""

import sys
import json

TABLE_ADDR = {"HARVESTMOGBA": (0x080F89D4, 1328), "HM MFOM USA\0": (0x081014BC, 1415)}


def u32(b, off=0):
    return int.from_bytes(b[off : off + 4], "little")


def read_chunks(rom, addr):
    """Retourne {nom_chunk: donnees} pour le conteneur RIFF a l'adresse donnee."""
    off = addr & 0x01FFFFFF

    if rom[off : off + 4] != b"RIFF" or rom[off + 8 : off + 12] != b"SCR ":
        return None

    size = u32(rom, off + 4)
    chunks = {}
    pos = 12

    while pos < size:
        name = rom[off + pos : off + pos + 4].decode("latin-1")
        clen = u32(rom, off + pos + 4)
        chunks[name] = rom[off + pos + 8 : off + pos + 8 + clen]
        pos += clen + 8

    return chunks


def decode_str_chunk(data):
    """Le chunk STR est: count, count*offsets, puis le pool de chaines nul-terminees."""
    count = u32(data)
    pool = data[4 + count * 4 :]
    out = []

    for i in range(count):
        start = u32(data, 4 + i * 4)
        end = pool.index(b"\0", start)
        out.append(pool[start:end])

    return out


def main(argv):
    if len(argv) < 2:
        return f"Usage: {argv[0]} ROM [--json]"

    rom = open(argv[1], "rb").read()
    title = rom[0xA0:0xAC].decode("latin-1")

    if title not in TABLE_ADDR:
        return f"ROM inconnue (titre: {title!r})"

    table_addr, script_count = TABLE_ADDR[title]
    table_off = table_addr & 0x01FFFFFF

    rows = []
    missing = []

    for sid in range(1, script_count + 1):
        addr = u32(rom, table_off + sid * 4)
        chunks = read_chunks(rom, addr) if addr else None

        if chunks is None:
            missing.append(sid)
            continue

        if "STR " not in chunks:
            continue

        for idx, raw in enumerate(decode_str_chunk(chunks["STR "])):
            rows.append((sid, idx, raw))

    if "--json" in argv:
        json.dump(
            [
                {"script": s, "index": i, "text": t.decode("latin-1")}
                for s, i, t in rows
            ],
            sys.stdout,
            ensure_ascii=False,
            indent=1,
        )
    else:
        print("script_id\tstring_index\ttext")
        for sid, idx, raw in rows:
            # on echappe pour garder une ligne TSV par chaine
            text = raw.decode("latin-1").encode("unicode_escape").decode("ascii")
            print(f"{sid}\t{idx}\t{text}")

    print(
        f"# {len(rows)} chaines dans {script_count} scripts "
        f"({len(missing)} scripts illisibles/vides)",
        file=sys.stderr,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
