#!/usr/bin/env node
// check-alcotest-registration.js — a test case that is written but never
// registered runs never and reports nothing.
//
// Backlog #103. Alcotest suites list their cases explicitly:
//
//     let test_foo () = ...
//     let () = Alcotest.run "S" [ ("g", [Alcotest.test_case "foo" `Quick test_foo]) ]
//
// Forget the list entry and the case is dead weight: it compiles, the binary
// is green, the suite reports N passed, and the behavior it was written to
// protect is unguarded. Nothing in the build, the test run, or coverage
// distinguishes that from a case that genuinely passes — which is exactly the
// shape of failure that reads as success.
//
// ── What this checks, and what it used to check ─────────────────────────────
// The first version of this file checked MENTION, not REGISTRATION: it
// collected `^let test_*` definitions and passed any name that appeared
// anywhere else in the same file. It never looked inside the suite list at
// all, so `let _ = ignore test_never_runs` laundered an orphan, two mutually
// referencing orphans covered for each other, and anything defined with `and`,
// or indented inside a `module ... struct`, or in a helpers file that does not
// itself call Alcotest.run, was invisible. Seven such evasions were confirmed
// by execution; every one exited 0.
//
// This version extracts the identifiers actually passed to `Alcotest.test_case`
// (the only registration form used in this repo — `Alcotest.skip` appears only
// inside case bodies, never as a list entry) and compares that against the set
// of defined test_* functions. A definition absent from the registered set is
// an orphan no matter how often it is mentioned elsewhere.
//
// ── Scoping tradeoff (definitions per-file, registrations global) ───────────
// Cases are routinely defined in one file and registered in another
// (helpers.ml + test_main.ml). So the registered set is built GLOBALLY over
// every scanned file, while definitions are reported per-file with their own
// file:line. The cost is that two same-named functions in unrelated test files
// alias: registering `test_roundtrip` in A silences an orphaned
// `test_roundtrip` in B. The alternative — per-file registration sets — was
// rejected because it makes the ordinary multi-file layout unusable and would
// produce a wall of false positives that trains people to ignore this check.
// Aliasing needs a name collision AND an orphan AND both in scanned trees; a
// missed registration is the common case. Definitions are only collected from
// test trees (a `test/` or `tests/` path component, or a `test_*.ml` /
// `*_test.ml` basename) so that ordinary library code cannot trip the check;
// registrations are harvested from every scanned .ml file regardless.
//
// Ambiguity policy: when the lexer meets something it cannot resolve (an
// unterminated string or comment), it does NOT blank the rest of the file —
// the old stripper did, and a single `'"'` char literal or a `"C:\\"` string
// was enough to erase a file's `Alcotest.run` and drop it from the scan
// silently. It restores the raw text from the point of confusion instead,
// which can only over-report. A false positive a human dismisses beats a
// silent pass.
//
// Usage: node scripts/check-alcotest-registration.js [<dir-or-file> ...]
//        (default: the repo, minus _build/_opam/node_modules)
// Exit: 0 all registered; 1 at least one orphan; 2 usage/IO error.
"use strict";

const fs = require("fs");
const path = require("path");

const SKIP_DIRS = new Set(["_build", "_opam", "node_modules", ".git", "_site", "dist", "gh-pages"]);

// `let`/`and`, any indentation, optional `rec`, followed by at least one
// PARAMETER. The parameter requirement is what separates a test case
// (`let test_foo () = ...`) from an ordinary value that happens to be named
// test_something (`let test_cases = [ ... ]`, which occurs in this repo as a
// local binding inside case bodies). A case with no parameters cannot be
// passed to Alcotest.test_case anyway, since test_case needs `unit -> unit`.
const DEF_RE = /(?:^|[^A-Za-z0-9_'.])(?:let|and)\s+(?:rec\s+)?(test_[A-Za-z0-9_']*)\s+(?![=:])/g;

// The two registration forms this repo actually uses. Both are anchors: the
// function being registered is whatever follows, up to the end of that list
// entry.
//   1. `Alcotest.test_case "name" `Quick test_foo` (or a bare `test_case`
//      under `let open Alcotest in`). \b keeps this off `test_cases`.
//   2. the raw Alcotest tuple `("name", `Quick, test_foo)`, which Alcotest.run
//      accepts directly with no test_case constructor. sarek/tests/unit uses
//      this form almost exclusively; a checker that only knew test_case
//      declared every case in those files an orphan.
const REG_ANCHORS = [
  /(?:\bAlcotest\s*\.\s*)?\btest_case\b/g,
  /`\s*(?:Quick|Slow)\b\s*,/g,
];

const IDENT_RE = /\btest_[A-Za-z0-9_']*\b/g;

// A file is a suite if it calls Alcotest.run, or opens Alcotest and calls a
// bare `run "..."` (the `let open Alcotest in run "S" [...]` idiom).
const SUITE_RE = /\bAlcotest\s*\.\s*run\b/;
const OPEN_ALCOTEST_RE = /\bopen\s+Alcotest\b/;
const BARE_RUN_RE = /\brun\s*\n?\s*"/;
// This repo also contains hand-rolled test drivers that never touch Alcotest
// at all — `let () = test_a (); test_b (); print_endline "ok"`. Those cases ARE
// registered, just by a different mechanism, and are none of this check's
// business. A file must mention Alcotest somewhere to be in scope; that still
// covers the helpers.ml half of a split layout, because a helper that asserts
// anything calls Alcotest.check.
const USES_ALCOTEST_RE = /\bAlcotest\b/;

function isTestTreeFile(file) {
  const parts = file.split(path.sep);
  const base = parts[parts.length - 1];
  if (parts.slice(0, -1).some((p) => p === "test" || p === "tests")) return true;
  return /^test_.*\.ml$/.test(base) || /_test\.ml$/.test(base);
}

function walk(target, acc) {
  let stat;
  try {
    stat = fs.statSync(target);
  } catch {
    return acc;
  }
  if (stat.isFile()) {
    if (target.endsWith(".ml")) acc.push(target);
    return acc;
  }
  if (!stat.isDirectory()) return acc;
  // Guarded like the statSync above: an unreadable directory (permissions, or
  // a race with a concurrent build) must not escape as a raw stack trace when
  // the documented contract is a controlled exit 2.
  let entries;
  try {
    entries = fs.readdirSync(target);
  } catch {
    return acc;
  }
  for (const name of entries) {
    if (SKIP_DIRS.has(name)) continue;
    walk(path.join(target, name), acc);
  }
  return acc;
}

// True when src[i] === "'" opens a CHARACTER LITERAL rather than being a type
// variable ('a), a prime in an identifier (x'), or a polymorphic-variant tick.
// Getting this wrong in either direction is a real failure mode: treating 'a
// as a literal eats the following code, and treating '"' as a type variable
// desyncs the string state and blanks the rest of the file.
function charLiteralEnd(src, i) {
  const prev = src[i - 1];
  // x' / 'a' after an identifier char is a prime, never a literal opener.
  if (prev !== undefined && /[A-Za-z0-9_']/.test(prev)) return -1;
  if (src[i + 1] === "\\") {
    // '\n' '\\' '\'' '\123' '\xFF' '\o101'
    const m = /^'\\(?:[\\'"ntbr ]|[0-9]{3}|x[0-9A-Fa-f]{2}|o[0-3][0-7]{2})'/.exec(src.slice(i, i + 8));
    return m ? i + m[0].length - 1 : -1;
  }
  // 'c' — exactly one non-quote, non-backslash char then a closing tick.
  if (src[i + 2] === "'" && src[i + 1] !== undefined && src[i + 1] !== "\\") return i + 2;
  return -1;
}

// Replaces comments, string literals and char literals with spaces, preserving
// every newline so reported line numbers stay exact (the old version blanked
// newlines inside strings and comments, shifting every line after a multi-line
// comment). Nested (* ... *) is handled, and strings inside comments are
// consumed the way OCaml consumes them.
//
// On an unterminated string or comment, the raw source from the opening
// delimiter onward is restored verbatim rather than blanked. That is the
// difference between "I could not lex this, so I will look at it anyway" and
// "I could not lex this, so I will report the file clean".
function stripCommentsAndStrings(src) {
  const out = [];
  let i = 0;
  let openedAt = -1; // start of the construct we are currently inside
  const blank = (from, to) => {
    for (let k = from; k < to; k++) out.push(src[k] === "\n" ? "\n" : " ");
  };

  while (i < src.length) {
    // ── comment (nesting) ────────────────────────────────────────────────
    if (src[i] === "(" && src[i + 1] === "*") {
      openedAt = i;
      let depth = 0;
      let j = i;
      while (j < src.length) {
        if (src[j] === "(" && src[j + 1] === "*") {
          depth += 1;
          j += 2;
        } else if (src[j] === "*" && src[j + 1] === ")") {
          depth -= 1;
          j += 2;
          if (depth === 0) break;
        } else if (src[j] === '"') {
          // A comment may contain a string literal; OCaml lexes it, so a `*)`
          // inside that string does not close the comment.
          const e = scanString(src, j);
          j = e === -1 ? j + 1 : e + 1;
        } else {
          j += 1;
        }
      }
      if (depth !== 0) break; // unterminated: fall through to the raw tail
      blank(i, j);
      i = j;
      openedAt = -1;
      continue;
    }
    // ── quoted string {id| ... |id} ──────────────────────────────────────
    if (src[i] === "{") {
      const m = /^\{([a-z_]*)\|/.exec(src.slice(i, i + 32));
      if (m) {
        openedAt = i;
        const close = "|" + m[1] + "}";
        const end = src.indexOf(close, i + m[0].length);
        if (end === -1) break;
        blank(i, end + close.length);
        i = end + close.length;
        openedAt = -1;
        continue;
      }
    }
    // ── string ───────────────────────────────────────────────────────────
    if (src[i] === '"') {
      openedAt = i;
      const end = scanString(src, i);
      if (end === -1) break;
      blank(i, end + 1);
      i = end + 1;
      openedAt = -1;
      continue;
    }
    // ── char literal ─────────────────────────────────────────────────────
    if (src[i] === "'") {
      const end = charLiteralEnd(src, i);
      if (end !== -1) {
        blank(i, end + 1);
        i = end + 1;
        continue;
      }
    }
    out.push(src[i]);
    i += 1;
  }

  if (i < src.length) {
    // Unterminated construct. Keep the raw tail from where we got confused so
    // the registrations and definitions in it are still seen. Over-reporting
    // is recoverable; a silently dropped file is not.
    const from = openedAt === -1 ? i : openedAt;
    for (let k = out.length; k < from; k++) out.push(src[k]);
    out.push(src.slice(from));
  }
  return out.join("");
}

// Index of the closing quote of the string opening at `start`, or -1.
// A quote is a real close when the run of backslashes immediately before it is
// EVEN. The old check looked only at src[i-1] !== "\\", so `"C:\\"` was read as
// an unterminated string and everything after it was blanked away.
function scanString(src, start) {
  let j = start + 1;
  while (j < src.length) {
    if (src[j] === "\\") {
      j += 2;
      continue;
    }
    if (src[j] === '"') return j;
    j += 1;
  }
  return -1;
}

function lineOf(code, index) {
  let line = 1;
  for (let k = 0; k < index; k++) if (code[k] === "\n") line += 1;
  return line;
}

// ── binding bodies, for reachability ────────────────────────────────────────
// Not every function named test_* is a test CASE. This repo has helpers with
// case-shaped names — `test_kernels ()` returns a kernel list,
// `test_near_zero name reference` is a two-argument assertion used by two real
// cases. Neither can be passed to Alcotest.test_case and neither is a bug.
// So a definition is live when it is registered OR reachable from a
// registration by ordinary calls: an identifier appearing in the BODY of a live
// binding is itself live, to a fixpoint.
//
// The root set is deliberately narrow — the identifiers inside registration
// anchors plus those inside the `Alcotest.run` suite expression, NOT the whole
// `let () = ...` that encloses the run. That is precisely what keeps
// `let _ = ignore test_never_runs` from laundering an orphan: a bare mention at
// top level is not in any live body, so it grants nothing. Two orphans that
// call each other are likewise unreachable from any root and both get reported.
//
// Bodies are delimited by OCaml layout: a binding's body runs until the next
// `let`/`and` at an indentation less than or equal to its own. The tree is
// ocamlformat'd, so this holds; where it does not, the body runs long, which
// over-approximates liveness for that one binding only.
// `module` and `end` are delimiters as well as `let`/`and`: without them the
// body of the binding preceding a `module M = struct ... end` runs straight
// through the module and every case inside it reads as called. Extra
// delimiters can only SHORTEN bodies, i.e. shrink the live set, i.e. report
// more — the safe direction.
const BINDING_RE =
  /(^|\n)([ \t]*)(?:\b(?:let|and)\s+(?:rec\s+)?([A-Za-z_][A-Za-z0-9_']*|\(\s*\)|_)|(module)\b|(end)\b)/g;
const ANY_IDENT_RE = /\b[A-Za-z_][A-Za-z0-9_']*\b/g;

function collectBindings(code) {
  const out = [];
  BINDING_RE.lastIndex = 0;
  let m;
  while ((m = BINDING_RE.exec(code)) !== null) {
    const start = m.index + m[1].length;
    // m[3] is a bound name; m[4]/m[5] are the bare `module`/`end` delimiters,
    // which get no name and so are never expanded as a live body.
    out.push({ name: m[3] || "", indent: m[2].length, start });
    BINDING_RE.lastIndex = m.index + m[0].length;
  }
  // Body end = next binding at an indentation <= this one's.
  for (let i = 0; i < out.length; i++) {
    let end = code.length;
    for (let j = i + 1; j < out.length; j++) {
      if (out[j].indent <= out[i].indent) {
        end = out[j].start;
        break;
      }
    }
    out[i].body = code.slice(out[i].start, end);
  }
  return out;
}

// The `[ ... ]` suite list handed to Alcotest.run. Bounded by the matching
// bracket rather than by "to end of file", so a top-level binding written after
// the run call cannot smuggle itself into the root set.
function collectRunRoots(code, into) {
  const anchors = [/\bAlcotest\s*\.\s*run\b/g];
  if (OPEN_ALCOTEST_RE.test(code)) anchors.push(/(^|[^.\w])run\s*(?:\n\s*)?"/g);
  for (const anchor of anchors) {
    anchor.lastIndex = 0;
    let m;
    while ((m = anchor.exec(code)) !== null) {
      const open = code.indexOf("[", m.index);
      if (open === -1) continue;
      let depth = 0;
      let j = open;
      for (; j < code.length; j++) {
        const c = code[j];
        if (c === "[" || c === "(" || c === "{") depth += 1;
        else if (c === "]" || c === ")" || c === "}") {
          depth -= 1;
          if (depth === 0) break;
        }
      }
      const span = code.slice(open, j + 1);
      ANY_IDENT_RE.lastIndex = 0;
      let id;
      while ((id = ANY_IDENT_RE.exec(span)) !== null) into.add(id[0]);
      anchor.lastIndex = m.index + m[0].length;
    }
  }
  return into;
}

// Every `test_*` name defined as a function, at any indentation, from `let`,
// `let rec` and `and` chains, including inside `module M = struct ... end`.
function collectDefs(code) {
  const defs = new Map(); // name -> first definition line
  DEF_RE.lastIndex = 0;
  let m;
  while ((m = DEF_RE.exec(code)) !== null) {
    const name = m[1];
    if (!defs.has(name)) defs.set(name, lineOf(code, m.index + m[0].indexOf(name)));
    // The leading [^A-Za-z0-9_'.] is consumed by the match; step back one so
    // adjacent bindings cannot be skipped.
    DEF_RE.lastIndex = m.index + m[0].length - 1;
  }
  return defs;
}

// Identifiers actually handed to Alcotest.test_case. The argument region runs
// from just after the `test_case` token to the first depth-0 `;` or closing
// bracket — i.e. the end of that list entry. Everything test_*-shaped inside
// counts, which deliberately covers partial application
// (`test_case n \`Quick (test_unhandled_raises spec)`) and inline lambdas that
// call a case, both of which occur in this repo.
function collectRegistered(code, into) {
  for (const anchor of REG_ANCHORS) {
    anchor.lastIndex = 0;
    let m;
    while ((m = anchor.exec(code)) !== null) {
      const start = m.index + m[0].length;
      let depth = 0;
      let j = start;
      const limit = Math.min(code.length, start + 4000);
      for (; j < limit; j++) {
        const c = code[j];
        if (c === "(" || c === "[" || c === "{") depth += 1;
        else if (c === ")" || c === "]" || c === "}") {
          if (depth === 0) break;
          depth -= 1;
        } else if (c === ";" && depth === 0) break;
      }
      const span = code.slice(start, j);
      IDENT_RE.lastIndex = 0;
      let id;
      while ((id = IDENT_RE.exec(span)) !== null) into.add(id[0]);
      anchor.lastIndex = start;
    }
  }
  return into;
}

// Returns { isSuite, defs, bindings } and folds this file's registration roots
// into `roots`. One read and one strip per file.
function analyze(file, roots) {
  const raw = fs.readFileSync(file, "utf8");
  const code = stripCommentsAndStrings(raw);
  collectRegistered(code, roots);
  collectRunRoots(code, roots);
  const isSuite = SUITE_RE.test(code) || (OPEN_ALCOTEST_RE.test(code) && BARE_RUN_RE.test(code));
  const inScope = isTestTreeFile(file) && USES_ALCOTEST_RE.test(code);
  const defs = inScope ? collectDefs(code) : new Map();
  return { isSuite, defs, bindings: collectBindings(code) };
}

// Scans a set of .ml files as one unit: roots union globally, liveness closes
// over every binding body, then definitions are checked against the closure.
// Exposed for the tests.
function scanFiles(files) {
  const live = new Set(); // registration roots, then their transitive closure
  const perFile = [];
  const bodies = new Map(); // binding name -> concatenated bodies
  let suites = 0;
  for (const f of files) {
    const r = analyze(f, live);
    if (r.isSuite) suites += 1;
    if (r.defs.size) perFile.push({ file: f, defs: r.defs });
    for (const b of r.bindings) {
      if (!b.name || b.name === "_" || b.name.startsWith("(")) continue;
      bodies.set(b.name, (bodies.get(b.name) || "") + "\n" + b.body);
    }
  }

  // Fixpoint: anything named in the body of a live binding is live too.
  const queue = [...live];
  while (queue.length) {
    const name = queue.pop();
    const body = bodies.get(name);
    if (body === undefined) continue;
    bodies.delete(name); // each binding's body is expanded exactly once
    ANY_IDENT_RE.lastIndex = 0;
    let id;
    while ((id = ANY_IDENT_RE.exec(body)) !== null) {
      if (!live.has(id[0])) {
        live.add(id[0]);
        queue.push(id[0]);
      }
    }
  }

  const problems = [];
  for (const { file, defs } of perFile) {
    const orphans = [];
    for (const [name, line] of defs) if (!live.has(name)) orphans.push({ name, line });
    if (orphans.length) problems.push({ file, orphans });
  }
  return { problems, suites, files: files.length, live };
}

function main(argv) {
  const targets = argv.length ? argv : [path.resolve(__dirname, "..")];
  const files = [];
  for (const t of targets) {
    const abs = path.resolve(t);
    // A path that does not exist is an error, not "nothing to check" — that
    // distinction is the difference between a real pass and a vacuous one.
    if (!fs.existsSync(abs)) {
      console.error(`check-alcotest-registration: target does not exist: ${abs}`);
      return 2;
    }
    walk(abs, files);
  }

  let result;
  try {
    result = scanFiles(files);
  } catch (e) {
    console.error(`check-alcotest-registration: cannot read: ${e.message}`);
    return 2;
  }
  const { problems, suites } = result;

  if (problems.length) {
    for (const p of problems) {
      for (const o of p.orphans) {
        console.error(
          `check-alcotest-registration: UNREGISTERED ${p.file}:${o.line}: ${o.name} is defined but never passed to Alcotest.test_case — it never runs.`
        );
      }
    }
    console.error(
      `check-alcotest-registration: ${problems.reduce((n, p) => n + p.orphans.length, 0)} unregistered case(s) across ${problems.length} file(s) (scanned ${files.length} .ml file(s), ${suites} Alcotest suite(s)).`
    );
    console.error("  Add each to its Alcotest.run suite list, or delete it. A case that never runs reports nothing.");
    return 1;
  }

  // Always report what was actually scanned. "OK" over zero suites is a
  // legitimate result for a project with no Alcotest tests, but it must be
  // visibly distinguishable from "OK" over a real suite set.
  console.log(
    `check-alcotest-registration: OK — scanned ${files.length} .ml file(s), ${suites} Alcotest suite(s), every test_* case registered.`
  );
  return 0;
}

if (require.main === module) process.exitCode = main(process.argv.slice(2));

module.exports = {
  analyze,
  scanFiles,
  collectDefs,
  collectRegistered,
  stripCommentsAndStrings,
  isTestTreeFile,
  main,
};
