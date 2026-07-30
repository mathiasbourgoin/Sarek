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


def gpu_vendor(system):
    """Vendor of the first non-CPU device, from names we are keeping anyway."""
    cpu_model = ((system.get("cpu") or {}).get("model") or "").lower()
    for dev in system.get("devices") or []:
        name = (dev.get("name") or "").lower()
        # A CPU listed as an OpenCL device is not the GPU we want to name.
        if name and name == cpu_model:
            continue
        for token, vendor in (
            ("nvidia", "nvidia"), ("geforce", "nvidia"), ("rtx", "nvidia"),
            ("radeon", "amd"), ("amd", "amd"), ("gfx", "amd"),
            ("arc", "intel"), ("intel", "intel"),
            ("apple", "apple"), ("m1", "apple"), ("m2", "apple"),
            ("m3", "apple"), ("m4", "apple"),
        ):
            if token in name:
                return vendor
    return "unknown"


def machine_label(system):
    """Opaque, stable grouping label. Never a function of the hostname."""
    os_name = (system.get("os") or "unknown").strip().lower() or "unknown"
    return f"{os_name}-{gpu_vendor(system)}"


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
        label = machine_label(system)
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
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(doc, handle, indent=2, sort_keys=True)
            handle.write("\n")
        merged = {k: v for k, v in collisions.items() if len(v) > 1}
        print(f"{path}: scrubbed {count} system block(s) -> "
              f"{len(collisions)} machine label(s)")
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
