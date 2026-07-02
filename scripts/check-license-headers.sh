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
if OUTPUT=$("$SCRIPT_DIR/add-license-headers.sh" --check 2>&1); then
    echo -e "${GREEN}✓ All license headers are up-to-date!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some files need license header updates:${NC}"
    echo ""
    echo "$OUTPUT"
    echo ""
    echo "To fix, run: ./scripts/add-license-headers.sh"
    echo "Then review and commit the changes."
    exit 1
fi
