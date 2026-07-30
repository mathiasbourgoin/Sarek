#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# THE machine-label shape (backlog-168 / backlog-168b). Sourced, never run.
#
# One rule, one place. The label is the token that replaced the hostname as the
# dedup key and as the leading component of every result FILENAME, so three
# consumers have to agree on what a legal label looks like:
#
#   1. scripts/check-no-machine-identifiers.sh -- refuses to let a tracked
#      filename be committed unless its leading token has this shape;
#   2. benchmarks/machine_label.ml -- refuses a SAREK_BENCH_MACHINE override
#      that does not have this shape, so an operator cannot produce files the
#      gate will later refuse (that contradiction is what backlog-168b fixed);
#   3. scripts/deidentify-benchmark-results.py -- must not strip a legal
#      suffix off a payload it relabels.
#
# Two independently written regexes for one rule is how drift starts, so:
# the bash consumers SOURCE this file, and the OCaml one restates the pattern
# verbatim in MACHINE_LABEL_SHAPE_DOC next to its matcher. The two are not
# merely commented as "must agree" -- scripts/check-no-machine-identifiers.test.sh
# runs BOTH implementations over one shared case table and fails if they
# disagree on any case.
#
# THE SHAPE: <os>-<vendor>[-<suffix>]
#
#   os, vendor   enumerated, not free-form. This is what keeps a bare hostname
#                out: `drangleic` has no <os>-<vendor> prefix and no widening
#                of the suffix can give it one. The enumeration is also why
#                this stays an allowlist rather than a blocklist of the three
#                known hostnames -- a blocklist passes the fourth machine,
#                which is how this class survives.
#   suffix       OPTIONAL, 1-8 chars of [a-z0-9]. It exists because two boxes
#                with the same OS and GPU vendor derive an IDENTICAL label and
#                their runs merge in the dedup key; the operator override is
#                there to tell them apart, and before this suffix existed the
#                override produced filenames the gate refused to commit -- it
#                worked everywhere except for the purpose it exists for.
#
# Why 8, and why exactly one suffix segment: 8 characters is enough for every
# disambiguator a fleet actually needs (`b`, `2`, `lab2`, `office`, `rack12`),
# while a hostname is typically longer -- the three that leaked are 9 and 10
# characters. The bound is not the security boundary (the mandatory enumerated
# prefix and the producer's hostname-equality refusal are); it bounds how much
# free-form operator text can ride along in a committed filename, and keeps the
# whole label under ~24 characters so the filenames stay readable. A single
# segment, rather than `(-[a-z0-9]{1,8})*`, so the token cannot grow by
# repetition into the free-form field the bound is there to deny.

# Components kept separate so a consumer can compose them into a different
# context (the gate needs the shape anchored to a path, not to a whole line).
MACHINE_LABEL_OS='(linux|darwin|windows)'
MACHINE_LABEL_VENDOR='(nvidia|amd|intel|apple|unknown)'
MACHINE_LABEL_SUFFIX='(-[a-z0-9]{1,8})?'

# The label shape as a whole-string ERE.
MACHINE_LABEL_SHAPE="^${MACHINE_LABEL_OS}-${MACHINE_LABEL_VENDOR}${MACHINE_LABEL_SUFFIX}\$"

# The same shape occupying the leading component of a result filename, i.e.
# `.../<label>_<benchmark>_<size>_<timestamp>.json`.
MACHINE_LABEL_PATH_SHAPE="/${MACHINE_LABEL_OS}-${MACHINE_LABEL_VENDOR}${MACHINE_LABEL_SUFFIX}_"
