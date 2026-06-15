#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Check that all source files have up-to-date SPDX license headers
# Simply runs add-license-headers.sh and checks if it would make changes

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

# Run the add script in a dry-run manner by checking git status after
"$SCRIPT_DIR/add-license-headers.sh" > /dev/null 2>&1

# Check if there are any changes
if git diff --quiet; then
    echo -e "${GREEN}✓ All license headers are up-to-date!${NC}"
    exit 0
else
    echo -e "${RED}✗ Some files need license header updates:${NC}"
    echo ""
    git diff --stat
    echo ""
    echo "Files with changes:"
    git diff --name-only
    echo ""
    echo "To fix, run: ./scripts/add-license-headers.sh"
    echo "Then review and commit the changes."
    exit 1
fi
