#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Check that all source files have up-to-date SPDX license headers.
#
# Delegates to add-license-headers.sh --check, which reports what it would
# change without writing to any file. This script itself never mutates the
# working tree and its result does not depend on whether the tree was
# already dirty before it ran (no "git diff" against pre-existing changes).

#
# Exit codes:
#   0  every covered file has an up-to-date header
#   1  at least one covered file needs a header change
#   2  the coverage declaration itself is broken (a declared root directory
#      is missing, or the candidate set came out empty). This is NOT a
#      header problem and must not be "fixed" by running the fixer — it
#      means the gate would otherwise have passed without inspecting
#      anything. See the coverage-scope block in add-license-headers.sh.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

echo "Checking license headers..."
echo ""

# --check makes add-license-headers.sh a pure read: it copies each candidate
# file's would-be output to a temp file and diffs against the original
# in-place, never touching the real file. Exit code 0 means every header is
# already up-to-date; exit code 1 means at least one file needs a change.
# `set -e` must not abort on the checker's own non-zero exit: we need to
# read the code to tell a header failure (1) from a broken coverage
# declaration (2).
set +e
OUTPUT=$("$SCRIPT_DIR/add-license-headers.sh" --check 2>&1)
STATUS=$?
set -e

case "$STATUS" in
    0)
        echo -e "${GREEN}✓ All license headers are up-to-date!${NC}"
        exit 0
        ;;
    1)
        echo -e "${RED}✗ Some files need license header updates:${NC}"
        echo ""
        echo "$OUTPUT"
        echo ""
        echo "To fix, run: ./scripts/add-license-headers.sh"
        echo "Then review and commit the changes."
        exit 1
        ;;
    *)
        # Exit 2, or anything unexpected. Running the fixer will not help and
        # may make it worse: the tree no longer matches what the script claims
        # to cover, so "all headers up-to-date" would have been a lie.
        echo -e "${RED}✗ License-header coverage is broken (exit $STATUS):${NC}"
        echo ""
        echo "$OUTPUT"
        echo ""
        echo "This is a coverage failure, not a header failure. Do NOT run the"
        echo "fixer to silence it — fix the roots/exemptions declared in"
        echo "scripts/add-license-headers.sh so the gate inspects the tree again."
        exit "$STATUS"
        ;;
esac
