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
// Detection: in any file that calls Alcotest.run, every top-level
// `let [rec] test_* ` binding must be mentioned somewhere in the file other
// than its own definition. A helper referenced only by another registered test
// counts as used; this deliberately under-reports rather than crying wolf on
// composed helpers.
//
// Usage: node scripts/check-alcotest-registration.js [<dir-or-file> ...]
//        (default: the repo, minus _build/_opam/node_modules)
// Exit: 0 all registered; 1 at least one orphan; 2 usage/IO error.
"use strict";

const fs = require("fs");
const path = require("path");

const SKIP_DIRS = new Set(["_build", "_opam", "node_modules", ".git", "_site", "dist", "gh-pages"]);
const DEF_RE = /^let\s+(?:rec\s+)?(test_[A-Za-z0-9_']*)\b/;

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
  for (const name of fs.readdirSync(target)) {
    if (SKIP_DIRS.has(name)) continue;
    walk(path.join(target, name), acc);
  }
  return acc;
}

// Strips (* ... *) comments, which nest in OCaml, and string literals. A
// commented-out registration must not count as a registration — that is the
// most likely way this check would silently stop working.
function stripCommentsAndStrings(src) {
  let out = "";
  let depth = 0;
  let inString = false;
  for (let i = 0; i < src.length; i++) {
    if (!inString && src[i] === "(" && src[i + 1] === "*") {
      depth += 1;
      i += 1;
      continue;
    }
    if (!inString && depth > 0 && src[i] === "*" && src[i + 1] === ")") {
      depth -= 1;
      i += 1;
      out += " ";
      continue;
    }
    if (depth > 0) continue;
    if (src[i] === '"' && src[i - 1] !== "\\") {
      inString = !inString;
      out += " ";
      continue;
    }
    out += inString ? " " : src[i];
  }
  return out;
}

function analyze(file) {
  const raw = fs.readFileSync(file, "utf8");
  const code = stripCommentsAndStrings(raw);
  if (!/Alcotest\s*\.\s*run/.test(code)) return null;

  const lines = code.split("\n");
  const defs = new Map(); // name -> 1-based definition line
  lines.forEach((line, idx) => {
    const m = DEF_RE.exec(line);
    if (m && !defs.has(m[1])) defs.set(m[1], idx + 1);
  });

  const orphans = [];
  for (const [name, defLine] of defs) {
    const uses = new RegExp(`\\b${name}\\b`, "g");
    let referenced = false;
    lines.forEach((line, idx) => {
      if (idx + 1 === defLine) return;
      uses.lastIndex = 0;
      if (uses.test(line)) referenced = true;
    });
    if (!referenced) orphans.push({ name, line: defLine });
  }
  return orphans.length ? { file, orphans } : null;
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

  const problems = [];
  let suites = 0;
  for (const f of files) {
    let r;
    try {
      r = analyze(f);
    } catch (e) {
      console.error(`check-alcotest-registration: cannot read ${f}: ${e.message}`);
      return 2;
    }
    if (r === null) continue;
    problems.push(r);
  }
  for (const f of files) {
    try {
      if (/Alcotest\s*\.\s*run/.test(stripCommentsAndStrings(fs.readFileSync(f, "utf8")))) suites += 1;
    } catch {
      /* counted above */
    }
  }

  if (problems.length) {
    for (const p of problems) {
      for (const o of p.orphans) {
        console.error(
          `check-alcotest-registration: UNREGISTERED ${p.file}:${o.line}: ${o.name} is defined but never referenced — it never runs.`
        );
      }
    }
    console.error(
      `check-alcotest-registration: ${problems.reduce((n, p) => n + p.orphans.length, 0)} unregistered case(s) across ${problems.length} file(s).`
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

module.exports = { analyze, stripCommentsAndStrings, main };
