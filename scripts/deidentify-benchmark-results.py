#!/usr/bin/env python3
# SPDX-License-Identifier: CeCILL-B
# Copyright (c) 2012-2026 Mathias Bourgoin
"""Strip machine-identifying fields out of benchmark result payloads.

WHY THIS EXISTS (backlog-168): benchmark results carried `system.hostname`,
`system.kernel` and `system.memory_gb`. Three personal machines were named in
263 committed files AND in the published gh-pages dashboard payload. Hostname
is an identity; an exact kernel version additionally discloses patch level.
Neither is needed to read a benchmark.

WHAT IT KEEPS, AND WHY THAT IS THE LINE
  kept:    os, cpu.model, cpu.cores, cpu.threads, devices[].name/framework/memory_gb
  removed: hostname, kernel, host-level memory_gb
  added:   machine -- an opaque grouping label DERIVED FROM THE KEPT HARDWARE
           FACTS, never from the hostname.

The line is identity-and-patch-level, not hardware model. `cpu.model` and the
device names ARE the subject matter of a benchmark -- a dashboard that hides
which GPU produced a number is not a dashboard. If those must go too, the
dashboard's premise has to be rethought; that is a bigger decision than this
script.

Deriving `machine` from hardware rather than from the hostname is deliberate:
a hash of the hostname would be trivially reversible (hostnames are low
entropy, a wordlist breaks them), and a committed hostname->label mapping table
would itself be the leak. So the label is a function of data we are keeping
anyway, and discloses nothing new.

Collision: two machines with the same OS and GPU vendor derive the SAME label
and would be merged by the dedup key. That is real -- see the operator override
in benchmarks/system_info.ml. For scrubbing already-collected data, collisions
are reported rather than silently merged.

Usage:
    deidentify-benchmark-results.py [--check] FILE...

  default   rewrite each FILE in place
  --check   exit 1 if any FILE still carries an identifying field, print them.
            This is the mode the CI gate uses; it writes nothing.

Exit: 0 clean/rewritten - 1 identifiers found (--check) - 2 usage or bad input.
"""

import json
import sys

# Fields removed outright. `hostname` is an identity; `kernel` is a patch level;
# host `memory_gb` is a weak fingerprint that no chart reads.
STRIPPED = ("hostname", "kernel", "memory_gb")


VENDOR_TOKENS = (
    ("nvidia", "nvidia"), ("geforce", "nvidia"), ("rtx", "nvidia"),
    ("radeon", "amd"), ("gfx", "amd"), ("amd", "amd"),
    ("arc", "intel"), ("intel", "intel"),
    ("apple", "apple"), ("m1", "apple"), ("m2", "apple"),
    ("m3", "apple"), ("m4", "apple"),
)

# Highest first. Used INSTEAD of device order, see gpu_vendor.
VENDOR_PRIORITY = ("nvidia", "amd", "intel", "apple")

# CPU by construction.
CPU_BACKENDS = ("native", "interpreter")

# Backends on which a CPU can appear as a device, so a name resembling the CPU
# model means "this is the CPU". On CUDA/Metal/Vulkan/HIP a device is a GPU by
# construction and must NOT be excluded by name -- on a unified-memory SoC the
# GPU's name IS the CPU's name. An unrecognised/absent backend is treated as
# possibly-CPU, which is the conservative reading for older payloads.
GPU_ONLY_BACKENDS = ("cuda", "metal", "vulkan", "hip")


def _normalize(text):
    """Lowercase, collapse whitespace runs, trim."""
    return " ".join((text or "").lower().split())


def _is_cpu_device(dev, cpu_model):
    backend = _normalize(dev.get("framework") or dev.get("backend") or "")
    if backend in CPU_BACKENDS:
        return True
    if backend in GPU_ONLY_BACKENDS:
        return False
    name = _normalize(dev.get("name"))
    cpu = _normalize(cpu_model)
    # "unknown" is the producer's CPU-probe FAILURE value, not a CPU name.
    if not cpu or cpu == "unknown":
        return False
    return cpu in name or name in cpu


def gpu_vendor(system):
    """Vendor of the machine's GPU, from names we are keeping anyway.

    Mirrors System_info.gpu_vendor_of in benchmarks/system_info.ml, and must
    keep mirroring it: this function RELABELS payloads, so if the two derive
    different labels the scrubber rewrites a correctly-labelled file and
    desynchronizes it from its own filename (which carries the producer's
    label). Verified against the published payload, which is unchanged by this.

    Order-independent: the vendor is chosen from the whole set of surviving
    matches by VENDOR_PRIORITY, never by device enumeration order, so a
    multi-GPU payload cannot relabel just because devices were listed
    differently.
    """
    cpu_model = (system.get("cpu") or {}).get("model") or ""
    vendors = set()
    for dev in system.get("devices") or []:
        if _is_cpu_device(dev, cpu_model):
            continue
        name = _normalize(dev.get("name"))
        for token, vendor in VENDOR_TOKENS:
            if token in name:
                vendors.add(vendor)
                break
    for vendor in VENDOR_PRIORITY:
        if vendor in vendors:
            return vendor
    return "unknown"


def machine_label(system):
    """Opaque, stable grouping label. Never a function of the hostname."""
    os_name = (system.get("os") or "unknown").strip().lower() or "unknown"
    return f"{os_name}-{gpu_vendor(system)}"


# A payload may legally carry the derived label plus the bounded disambiguating
# suffix an operator set through SAREK_BENCH_MACHINE (see
# scripts/machine-label-shape.sh for the shape and the bound). Without this, a
# suffixed payload looked "wrong" to the relabeller: it was rewritten
# linux-amd-b -> linux-amd, which strips the only thing telling two
# same-hardware machines apart AND desynchronizes the payload from its own
# filename, which still carries the producer's suffixed label. Same defect class
# as the darwin-apple relabelling, one field over.
#
# This is the same rule as Machine_label.is_derived_variant, which is what the
# producer accepts as an override -- deliberately the same rule, since the
# desynchronization above is what any disagreement between them produces.
SUFFIX_MAX_LEN = 8


def keeps_label(existing, derived):
    """True if `existing` is `derived`, or `derived` plus a legal suffix."""
    if not isinstance(existing, str):
        return False
    if existing == derived:
        return True
    if not existing.startswith(derived + "-"):
        return False
    suffix = existing[len(derived) + 1:]
    return (
        1 <= len(suffix) <= SUFFIX_MAX_LEN
        and all(("0" <= c <= "9") or ("a" <= c <= "z") for c in suffix)
    )


def systems_of(doc):
    """Every `system` block in a payload, whichever shape the file uses."""
    if isinstance(doc, dict) and isinstance(doc.get("results"), list):
        # Dashboard aggregate: {"results": [ {benchmark, system, results}, ... ]}
        for entry in doc["results"]:
            if isinstance(entry, dict) and isinstance(entry.get("system"), dict):
                yield entry["system"]
    if isinstance(doc, dict) and isinstance(doc.get("system"), dict):
        # Single run: {"benchmark": ..., "system": ..., "results": [...]}
        yield doc["system"]


def scrub(doc):
    """Rewrite in place. Returns (n_scrubbed, {label: {removed values}})."""
    count = 0
    collisions = {}
    for system in systems_of(doc):
        present = [f for f in STRIPPED if f in system]
        derived = machine_label(system)
        # An operator-set suffix is kept, not relabelled away: it is the only
        # thing separating two same-hardware machines, and the filename carries
        # it too.
        label = system["machine"] if keeps_label(system.get("machine"), derived) else derived
        # Record what each label absorbed, so a merge cannot pass unnoticed.
        collisions.setdefault(label, set()).add(system.get("hostname", "<none>"))
        if not present and system.get("machine") == label:
            continue
        for field in STRIPPED:
            system.pop(field, None)
        system["machine"] = label
        count += 1
    return count, collisions


def offenders(doc):
    """Identifying fields still present. Empty list means clean."""
    found = []
    for system in systems_of(doc):
        for field in STRIPPED:
            if field in system:
                found.append(field)
    return sorted(set(found))


def main(argv):
    check = "--check" in argv
    paths = [a for a in argv if not a.startswith("--")]
    unknown = [a for a in argv if a.startswith("--") and a != "--check"]
    if unknown or not paths:
        print(__doc__.strip().split("Usage:")[1], file=sys.stderr)
        return 2

    dirty = False
    for path in paths:
        try:
            with open(path, encoding="utf-8") as handle:
                doc = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            # Fail closed: an unreadable payload is not a clean payload.
            print(f"::error::{path}: cannot read as JSON: {exc}", file=sys.stderr)
            return 2

        if check:
            found = offenders(doc)
            if found:
                dirty = True
                print(f"::error::{path} carries {', '.join(found)}")
            continue

        count, collisions = scrub(doc)
        # Only write when something was actually stripped or relabelled. The
        # write is a re-serialization (indent=2, sort_keys=True), so doing it
        # unconditionally rewrote every already-clean file it was pointed at
        # and produced pure-formatting diffs unrelated to de-identification --
        # noise that makes a real scrub harder to see in review. Nothing
        # depends on this tool normalizing formatting: the CI gate uses
        # --check, which reads fields and ignores layout.
        if count:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(doc, handle, indent=2, sort_keys=True)
                handle.write("\n")
        merged = {k: v for k, v in collisions.items() if len(v) > 1}
        print(f"{path}: scrubbed {count} system block(s) -> "
              f"{len(collisions)} machine label(s)"
              f"{'' if count else ' (already clean, not rewritten)'}")
        for label, hosts in sorted(merged.items()):
            print(f"  NOTE: {label} merges {len(hosts)} distinct sources "
                  f"-- their runs are now indistinguishable")

    if check and dirty:
        print("\nRun scripts/deidentify-benchmark-results.py on the file(s) above.",
              file=sys.stderr)
        return 1
    if check:
        print(f"OK -- {len(paths)} payload(s) carry no identifying field")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
