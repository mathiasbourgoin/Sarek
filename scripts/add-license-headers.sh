#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Automatically add or update SPDX license headers in source files
# Uses git history to determine copyright years
# Uses a canonical maintainer identity (MAINTAINER) for contributor attribution

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

# Default license
LICENSE="CECILL-B"

# Canonical copyright holder for this project. Headers use this single identity
# rather than the git commit author, so the header set stays stable regardless of
# which machine/identity authored a change (avoids accruing per-committer lines).
MAINTAINER="Mathias Bourgoin <mathias.bourgoin@gmail.com>"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
UPDATED_COUNT=0
SKIPPED_COUNT=0

# Get copyright info from git
get_copyright_years() {
    local file="$1"
    
    # Get first commit year
    local first_year=$(git log --follow --format=%aI --reverse "$file" 2>/dev/null | head -1 | cut -d- -f1)
    
    # Get last commit year
    local last_year=$(git log --follow --format=%aI -1 "$file" 2>/dev/null | cut -d- -f1)
    
    # If git info not available, use current year
    if [ -z "$first_year" ]; then
        first_year=$(date +%Y)
        last_year=$(date +%Y)
    fi
    
    # Format year range
    if [ "$first_year" = "$last_year" ]; then
        echo "$first_year"
    else
        echo "$first_year-$last_year"
    fi
}

# Primary contributor. Fixed to the canonical project maintainer so headers carry
# one stable identity rather than a line per git committer.
get_primary_contributor() {
    echo "$MAINTAINER"
}

# Add or update header in OCaml file
add_ocaml_header() {
    local file="$1"
    local last_commit_year=$(git log --format=%aI -1 "$file" 2>/dev/null | cut -d- -f1)
    [ -z "$last_commit_year" ] && last_commit_year=$(date +%Y)
    local contributor=$(get_primary_contributor "$file")
    local contributor_email=$(echo "$contributor" | grep -o '<[^>]*>' | tr -d '<>')
    
    # Check if header already exists
    if head -10 "$file" | grep -q "SPDX-License-Identifier"; then
        # Header exists - check if we need to update it.
        # Match the closing delimiter line "(****...****)" (literal '(', one or
        # more '*', literal ')').
        local header_end_line=$(head -20 "$file" | grep -nE '^\(\*+\)$' | tail -1 | cut -d: -f1)
        # Fall back to a safe window if no delimiter is found, so the head/sed
        # calls below never receive an empty or invalid line number.
        [ -z "$header_end_line" ] && header_end_line=10
        
        # Check if this contributor already has a copyright line
        if head -"$header_end_line" "$file" 2>/dev/null | grep "SPDX-FileCopyrightText:" | grep -q "$contributor_email"; then
            # Contributor exists - check if year needs updating
            local contributor_line=$(head -"$header_end_line" "$file" | grep "SPDX-FileCopyrightText:" | grep "$contributor_email")
            local existing_years=$(echo "$contributor_line" | grep -oP '\d{4}(-\d{4})?')
            local first_year=$(echo "$existing_years" | cut -d- -f1)
            
            if [[ "$existing_years" =~ - ]]; then
                # Has year range - check if last year matches
                local end_year=$(echo "$existing_years" | cut -d- -f2)
                if [ "$last_commit_year" != "$end_year" ]; then
                    # Update year range
                    sed -i "s/\($contributor_email.*\)$existing_years/\1$first_year-$last_commit_year/" "$file"
                    echo -e "${GREEN}UPDATED YEAR${NC}: $file ($first_year-$end_year -> $first_year-$last_commit_year)"
                    UPDATED_COUNT=$((UPDATED_COUNT + 1))
                    return
                fi
            else
                # Single year - check if we need range
                if [ "$last_commit_year" != "$first_year" ]; then
                    sed -i "s/\($contributor_email.*\)$first_year/\1$first_year-$last_commit_year/" "$file"
                    echo -e "${GREEN}UPDATED YEAR${NC}: $file ($first_year -> $first_year-$last_commit_year)"
                    UPDATED_COUNT=$((UPDATED_COUNT + 1))
                    return
                fi
            fi
            
            echo -e "${YELLOW}SKIP${NC}: $file (already up-to-date)"
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        else
            # New contributor - add copyright line before closing delimiter
            local years=$(get_copyright_years "$file")
            local tmpfile=$(mktemp)
            
            head -$((header_end_line - 1)) "$file" > "$tmpfile"
            echo "(* SPDX-FileCopyrightText: $years $contributor *)" >> "$tmpfile"
            tail -n +"$header_end_line" "$file" >> "$tmpfile"
            mv "$tmpfile" "$file"
            
            echo -e "${GREEN}ADDED CONTRIBUTOR${NC}: $file"
            UPDATED_COUNT=$((UPDATED_COUNT + 1))
        fi
        
        return
    fi
    
    # No header - create new one
    local years=$(get_copyright_years "$file")
    local tmpfile=$(mktemp)
    
    cat > "$tmpfile" << 'HEADER_EOF'
(******************************************************************************)
(* SPDX-License-Identifier: CECILL-B                                          *)
HEADER_EOF
    
    echo "(* SPDX-FileCopyrightText: $years $contributor *)" >> "$tmpfile"
    
    cat >> "$tmpfile" << 'HEADER_EOF'
(******************************************************************************)

HEADER_EOF
    
    cat "$file" >> "$tmpfile"
    mv "$tmpfile" "$file"
    
    echo -e "${GREEN}ADDED HEADER${NC}: $file"
    UPDATED_COUNT=$((UPDATED_COUNT + 1))
}

# Add header to shell script
add_shell_header() {
    local file="$1"
    local years=$(get_copyright_years "$file")
    local contributor=$(get_primary_contributor "$file")
    
    # Check if header already exists
    if head -5 "$file" | grep -q "SPDX-License-Identifier"; then
        echo -e "${YELLOW}SKIP${NC}: $file (already has header)"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        return
    fi
    
    # Create temporary file
    local tmpfile=$(mktemp)
    
    # Check if file starts with shebang
    local first_line=$(head -1 "$file")
    if [[ "$first_line" =~ ^#! ]]; then
        # Preserve shebang
        echo "$first_line" > "$tmpfile"
        echo "# SPDX-License-Identifier: $LICENSE" >> "$tmpfile"
        echo "# SPDX-FileCopyrightText: $years $contributor" >> "$tmpfile"
        tail -n +2 "$file" >> "$tmpfile"
    else
        # No shebang
        echo "# SPDX-License-Identifier: $LICENSE" > "$tmpfile"
        echo "# SPDX-FileCopyrightText: $years $contributor" >> "$tmpfile"
        echo "" >> "$tmpfile"
        cat "$file" >> "$tmpfile"
    fi
    
    # Replace original file
    mv "$tmpfile" "$file"
    
    echo -e "${GREEN}UPDATED${NC}: $file"
    UPDATED_COUNT=$((UPDATED_COUNT + 1))
}

# Add header to dune file
add_dune_header() {
    local file="$1"
    local years=$(get_copyright_years "$file")
    local contributor=$(get_primary_contributor "$file")
    
    # Check if header already exists
    if head -5 "$file" | grep -q "SPDX-License-Identifier"; then
        echo -e "${YELLOW}SKIP${NC}: $file (already has header)"
        SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        return
    fi
    
    # Create temporary file
    local tmpfile=$(mktemp)
    
    cat > "$tmpfile" << EOF
; SPDX-License-Identifier: $LICENSE
; SPDX-FileCopyrightText: $years $contributor

EOF
    
    # Append original content
    cat "$file" >> "$tmpfile"
    
    # Replace original file
    mv "$tmpfile" "$file"
    
    echo -e "${GREEN}UPDATED${NC}: $file"
    UPDATED_COUNT=$((UPDATED_COUNT + 1))
}

echo "Adding SPDX license headers..."
echo "License: $LICENSE"
echo ""

# Process OCaml files
echo -e "${BLUE}Processing OCaml files...${NC}"
while IFS= read -r -d '' file; do
    add_ocaml_header "$file"
done < <(find sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal spoc \
    -type f \( -name "*.ml" -o -name "*.mli" \) \
    ! -path "*/.*" \
    ! -path "*/_build/*" \
    ! -path "*/_opam/*" \
    ! -path "*/dependencies/*" \
    -print0 2>/dev/null)

# Process shell scripts
echo ""
echo -e "${BLUE}Processing shell scripts...${NC}"
while IFS= read -r -d '' file; do
    add_shell_header "$file"
done < <(find scripts ci \
    -type f -name "*.sh" \
    ! -path "*/.*" \
    -print0 2>/dev/null)

# Process dune files (optional - uncomment if needed)
# echo ""
# echo -e "${BLUE}Processing dune files...${NC}"
# while IFS= read -r -d '' file; do
#     add_dune_header "$file"
# done < <(find sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal spoc \
#     -type f -name "dune" \
#     ! -path "*/.*" \
#     ! -path "*/_build/*" \
#     ! -path "*/_opam/*" \
#     -print0 2>/dev/null)

# Summary
echo ""
echo "========================================"
echo "License Header Update Summary"
echo "========================================"
echo "Files updated: $UPDATED_COUNT"
echo "Files skipped: $SKIPPED_COUNT"
echo ""

if [ $UPDATED_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Headers added successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review changes: git diff"
    echo "2. Run checker: ./scripts/check-license-headers.sh"
    echo "3. Commit changes: git add -A && git commit -m 'chore: add SPDX license headers'"
fi
