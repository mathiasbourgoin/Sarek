#!/bin/bash
# SPDX-License-Identifier: CECILL-B
# SPDX-FileCopyrightText: 2026 Mathias Bourgoin <mathias.bourgoin@gmail.com>
#
# Automatically add or update SPDX license headers in source files
# Uses git history to determine copyright years
# Uses a canonical maintainer identity (MAINTAINER) for contributor attribution
#
# Pass --check (or --dry-run) to report which files would change without
# touching the working tree. check-license-headers.sh uses this mode so the
# "check" never mutates files. Exits 1 in that mode if any file needs an
# update, 0 otherwise.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_ROOT"

DRY_RUN=false
case "${1:-}" in
    --check|--dry-run)
        DRY_RUN=true
        ;;
esac

# Default license
LICENSE="CECILL-B"

# Canonical copyright holder for this project. Headers use this single identity
# rather than the git commit author, so the header set stays stable regardless of
# which machine/identity authored a change (avoids accruing per-committer lines).
MAINTAINER="Mathias Bourgoin <mathias.bourgoin@gmail.com>"

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Counters
UPDATED_COUNT=0
SKIPPED_COUNT=0
# Files carrying two copyright lines for the same contributor. Not fixable
# here (this script cannot choose which to keep), so it reports and exits
# non-zero rather than feeding a multi-line value to sed and dying namelessly.
DUPLICATE_COUNT=0

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

# Apply a pending change to $file.
#
# In normal mode, moves $tmpfile onto $file and reports "$label".
# In --check/--dry-run mode, never touches $file: diffs $tmpfile against it
# (case-sensitive, byte-for-byte) and only reports/counts a change if they
# actually differ, then discards $tmpfile.
apply_change() {
    local file="$1"
    local tmpfile="$2"
    local label="$3"

    if $DRY_RUN; then
        if ! diff -q "$file" "$tmpfile" >/dev/null 2>&1; then
            echo -e "${GREEN}${label}${NC}: $file"
            UPDATED_COUNT=$((UPDATED_COUNT + 1))
        else
            SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
        fi
        rm -f "$tmpfile"
    else
        # Write THROUGH the existing file rather than `mv`-ing the temp file
        # onto it. mktemp creates 0600, and `mv` carries that mode across, so
        # the previous `mv` silently stripped the executable bit from every
        # script it stamped -- i.e. running the fixer on scripts/ left the
        # whole tooling directory non-executable and CI failing with
        # "Permission denied". Redirecting into $file preserves its mode,
        # owner and inode.
        cat "$tmpfile" > "$file"
        rm -f "$tmpfile"
        echo -e "${GREEN}${label}${NC}: $file"
        UPDATED_COUNT=$((UPDATED_COUNT + 1))
    fi
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
        
        # Check if this contributor already has a copyright line.
        # NB: match case-insensitively. Email providers (e.g. "Gmail.com" vs
        # "gmail.com") are case-insensitive by spec, and mail clients/editors
        # routinely normalize casing over a file's history. A case-sensitive
        # grep here previously caused a contributor with a differently-cased
        # email to look like a "new" contributor on every run, so the fixer
        # kept appending a duplicate SPDX-FileCopyrightText line for the same
        # person (root cause of the sarek/codegen/Sarek_ir_ptx_stmt.mli
        # header mangling).
        # -F, and matched on the BRACKETED address. Without -F the email is a
        # REGEX, so every "." matches any character: two distinct contributors
        # whose addresses differ only where a dot sits would match each other,
        # which now means a false DUPLICATE below and a spurious exit 2 on a
        # correct header. Any of [ ] * ^ $ \ in an address would be worse than
        # loose -- it would be a grep error. Bracketing pins the match to a whole
        # address rather than a substring of a longer one. -i stays: providers
        # are case-insensitive by spec (see the note below).
        local email_pat="<$contributor_email>"
        if head -"$header_end_line" "$file" 2>/dev/null | grep "SPDX-FileCopyrightText:" | grep -qiF "$email_pat"; then
            # Contributor exists - check if the year needs updating.
            #
            # This grep can match MORE THAN ONE line: a header carrying two
            # copyright lines for the same email is exactly what the duplicate
            # bug noted above produces. A multi-line value then reached a
            # `sed "s/$existing_years..."`, which is not a well-formed s command
            # -- sed died with "unterminated `s' command", the fixer exited 1
            # mid-walk having silently skipped every remaining file, and it never
            # said WHICH file. Refuse explicitly and name it, because a duplicate
            # line is itself the defect to fix (four files in this repo had one),
            # not a state to paper over by taking the first match.
            local matches
            matches=$(head -"$header_end_line" "$file" | grep "SPDX-FileCopyrightText:" | grep -ciF "$email_pat")
            if [ "$matches" -gt 1 ]; then
                echo -e "${RED}DUPLICATE${NC}: $file has $matches SPDX-FileCopyrightText lines for $email_pat; remove the extra one (this script cannot choose between them)" >&2
                DUPLICATE_COUNT=$((DUPLICATE_COUNT + 1))
                return
            fi
            local contributor_line lineno
            contributor_line=$(head -"$header_end_line" "$file" | grep "SPDX-FileCopyrightText:" | grep -iF "$email_pat")
            lineno=$(head -"$header_end_line" "$file" | grep -n "SPDX-FileCopyrightText:" | grep -iF "$email_pat" | cut -d: -f1)
            local existing_years=$(echo "$contributor_line" | grep -oP '\d{4}(-\d{4})?' | head -1)
            local first_year=$(echo "$existing_years" | cut -d- -f1)

            # Decide the new year span first, then rewrite the line by NUMBER.
            #
            # The year is substituted with bash ${var/pat/repl}, which is glob and
            # not regex, and the year is digits plus a dash -- no metacharacters.
            # The old code interpolated the email into a sed PATTERN, which had
            # the same regex defect as the greps above; targeting the line we
            # already located removes regex from this path entirely. Carrying the
            # rest of the line verbatim also preserves the address's existing
            # casing, which the old code needed a \1 backreference to do.
            local new_years="" label=""
            if [[ "$existing_years" =~ - ]]; then
                local end_year=$(echo "$existing_years" | cut -d- -f2)
                if [ "$last_commit_year" != "$end_year" ]; then
                    new_years="$first_year-$last_commit_year"
                    label="UPDATED YEAR ($first_year-$end_year -> $new_years)"
                fi
            else
                if [ "$last_commit_year" != "$first_year" ]; then
                    new_years="$first_year-$last_commit_year"
                    label="UPDATED YEAR ($first_year -> $new_years)"
                fi
            fi

            if [ -n "$new_years" ]; then
                local new_line="${contributor_line/$existing_years/$new_years}"
                local tmpfile=$(mktemp)
                awk -v ln="$lineno" -v repl="$new_line" \
                    'NR==ln {print repl; next} {print}' "$file" > "$tmpfile"
                apply_change "$file" "$tmpfile" "$label"
                return
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
            apply_change "$file" "$tmpfile" "ADDED CONTRIBUTOR"
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
    apply_change "$file" "$tmpfile" "ADDED HEADER"
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

    apply_change "$file" "$tmpfile" "UPDATED"
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

    # Write through, not `mv` -- see apply_change for why mode preservation
    # matters here.
    cat "$tmpfile" > "$file"
    rm -f "$tmpfile"

    echo -e "${GREEN}UPDATED${NC}: $file"
    UPDATED_COUNT=$((UPDATED_COUNT + 1))
}

echo "Adding SPDX license headers..."
echo "License: $LICENSE"
echo ""

# ---------------------------------------------------------------------------
# Coverage scope (#137)
# ---------------------------------------------------------------------------
# What is covered is declared here, once, in named lists — not buried in a
# find expression. Two properties this buys us:
#
#   1. A missing root is LOUD. The find calls used to end in `2>/dev/null`,
#      so renaming or deleting a root directory made find print nothing,
#      the read loop body never ran, and the gate passed having inspected
#      no files at all. Every root is now asserted to exist, and the
#      candidate set is asserted non-empty, before any file is examined.
#
#   2. Exemptions are an explicit, reviewable list (EXEMPT_GLOBS) with a
#      stated reason each, rather than an anonymous `! -path` accumulating
#      in a find invocation nobody reads.
#
# NOT covered, deliberately: scripts/**/*.js and scripts/lib/**/*.js (14
# files at the time of writing). JavaScript needs a `//` header and no
# emitter in this script produces one; adding that is a separate change.
# The omission is recorded here so it is a decision rather than an accident
# of the find expression.

# Roots holding first-party OCaml sources (*.ml, *.mli).
OCAML_ROOTS=(sarek sarek-cuda sarek-opencl sarek-vulkan sarek-metal spoc)

# Roots holding first-party tooling. `*.sh` and `*.py` share the `#` comment
# syntax, so add_shell_header serves both.
SCRIPT_ROOTS=(scripts ci)

# The only sanctioned way to leave a matching file out. Each entry is a
# find -path glob plus the reason it is not ours to stamp.
EXEMPT_GLOBS=(
    '*/dependencies/*'  # vendored third-party sources — not ours to relicense
    '*/_build/*'        # dune build output — generated, never committed
    '*/_opam/*'         # local opam switch — not project source
    '*/.*'              # dotfile dirs (.git, .github metadata, editor state)

    # Review-tool bundle members. scripts/REVIEW-BUNDLE.md: "These files are
    # upstream-owned and generated [...] Do not hand-edit any bundle file or
    # the manifest." Each is pinned by sha256 in review-bundle.manifest.json,
    # so a header here fails review-bundle-verify immediately and is silently
    # reverted by the next roster upgrade anyway. Stamping them once already
    # turned check-review-bundle-tracked.sh red with two SHA MISMATCHes.
    #
    # This list is kept by hand rather than derived from the manifest: the
    # manifest is itself a bundle file, and reading it to decide what to skip
    # would let an upstream change quietly widen our exemptions.
    'scripts/check-scope-diff.sh'
    'scripts/xruntime-exec.sh'
)

# An exemption for a file that no longer exists is an exemption nobody is
# reading. Any entry without a wildcard is an exact path and must resolve.
for glob in "${EXEMPT_GLOBS[@]}"; do
    case "$glob" in
        *'*'*) ;;
        *)
            if [ ! -e "$glob" ]; then
                echo "ERROR: stale exemption in EXEMPT_GLOBS: $glob does not exist." >&2
                echo "       Remove it, or point it at the file's new path." >&2
                exit 2
            fi
            ;;
    esac
done

# Build the shared `! -path GLOB ...` argument vector once.
EXEMPT_ARGS=()
for glob in "${EXEMPT_GLOBS[@]}"; do
    EXEMPT_ARGS+=(! -path "$glob")
done

# Fail loudly if a declared root has moved. Without this the loops below
# silently inspect nothing and the gate reports success.
require_roots() {
    local label="$1"; shift
    local missing=()
    local root
    for root in "$@"; do
        [ -d "$root" ] || missing+=("$root")
    done
    if [ ${#missing[@]} -gt 0 ]; then
        echo "ERROR: $label root(s) not found: ${missing[*]}" >&2
        echo "       Update the *_ROOTS list in scripts/add-license-headers.sh." >&2
        echo "       Refusing to report success on an un-inspected tree." >&2
        exit 2
    fi
}

# A gate that examined zero files is not a passing gate.
require_nonempty() {
    local label="$1"
    local count="$2"
    if [ "$count" -eq 0 ]; then
        echo "ERROR: $label matched 0 files." >&2
        echo "       Either the roots or EXEMPT_GLOBS in" >&2
        echo "       scripts/add-license-headers.sh no longer describe this tree." >&2
        exit 2
    fi
}

# Process OCaml files
echo -e "${BLUE}Processing OCaml files...${NC}"
require_roots "OCaml" "${OCAML_ROOTS[@]}"
OCAML_SEEN=0
while IFS= read -r -d '' file; do
    OCAML_SEEN=$((OCAML_SEEN + 1))
    add_ocaml_header "$file"
done < <(find "${OCAML_ROOTS[@]}" \
    -type f \( -name "*.ml" -o -name "*.mli" \) \
    "${EXEMPT_ARGS[@]}" \
    -print0)
require_nonempty "OCaml sources" "$OCAML_SEEN"

# Process shell and Python tooling
echo ""
echo -e "${BLUE}Processing shell and Python scripts...${NC}"
require_roots "Script" "${SCRIPT_ROOTS[@]}"
SCRIPT_SEEN=0
while IFS= read -r -d '' file; do
    SCRIPT_SEEN=$((SCRIPT_SEEN + 1))
    add_shell_header "$file"
done < <(find "${SCRIPT_ROOTS[@]}" \
    -type f \( -name "*.sh" -o -name "*.py" \) \
    "${EXEMPT_ARGS[@]}" \
    -print0)
require_nonempty "Shell/Python tooling" "$SCRIPT_SEEN"

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
if $DRY_RUN; then
    echo "Files needing updates: $UPDATED_COUNT"
else
    echo "Files updated: $UPDATED_COUNT"
fi
echo "Files skipped: $SKIPPED_COUNT"
if [ $DUPLICATE_COUNT -gt 0 ]; then
    echo -e "${RED}Files with duplicate copyright lines: $DUPLICATE_COUNT${NC} (listed above; fix by hand)"
fi
echo ""

# A duplicate is a real defect and must not read as success in either mode.
if [ $DUPLICATE_COUNT -gt 0 ]; then
    exit 2
fi

if $DRY_RUN; then
    if [ $UPDATED_COUNT -gt 0 ]; then
        exit 1
    fi
    exit 0
fi

if [ $UPDATED_COUNT -gt 0 ]; then
    echo -e "${GREEN}✓ Headers added successfully!${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Review changes: git diff"
    echo "2. Run checker: ./scripts/check-license-headers.sh"
    echo "3. Commit changes: git add -A && git commit -m 'chore: add SPDX license headers'"
fi
