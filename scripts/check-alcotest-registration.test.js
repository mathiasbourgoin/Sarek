#!/usr/bin/env node
// Tests for #103 — an Alcotest case that is DEFINED but never REGISTERED runs
// never, reports nothing, and leaves the suite green.
//
// The checker this exercises used to verify MENTION rather than registration:
// it collected `^let test_*` and passed any name appearing anywhere else in the
// same file. Seven evasions were confirmed against it by execution, all exiting
// 0. Each has a failing-shape fixture below, and each is paired with a positive
// control in the same shape where the case IS registered. Without the pair,
// "it fails" and "it always fails" are the same observation — and a checker
// that reports every case as an orphan is just as useless as one that reports
// none, because it gets switched off within a day.
"use strict";

const assert = require("assert");
const fs = require("fs");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const ROOT = path.resolve(__dirname, "..");
const CHECKER = path.join(ROOT, "scripts/check-alcotest-registration.js");
const { stripCommentsAndStrings, collectDefs, collectRegistered } = require(CHECKER);

let pass = 0;
const results = [];
function check(name, fn) {
  try {
    fn();
    pass += 1;
    results.push(`  ok — ${name}`);
  } catch (e) {
    results.push(`  FAIL — ${name}\n      ${e.message}`);
    process.exitCode = 1;
  }
}

// ── fixture plumbing ────────────────────────────────────────────────────────
const tmpRoots = [];

// Builds <tmp>/test/<name>.ml for each entry. The `test` path component is what
// puts the files in the checker's definition scope, matching the real layout.
function fixture(files) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), "alcotest-reg-"));
  tmpRoots.push(dir);
  fs.mkdirSync(path.join(dir, "test"));
  for (const [name, body] of Object.entries(files)) {
    fs.writeFileSync(path.join(dir, "test", name), body);
  }
  return dir;
}

function runCheck(dir) {
  const r = spawnSync("node", [CHECKER, dir], { encoding: "utf8" });
  return { code: r.status, out: r.stdout, err: r.stderr, all: r.stdout + r.stderr };
}

function assertOrphan(dir, name, file) {
  const r = runCheck(dir);
  assert.strictEqual(r.code, 1, `expected exit 1, got ${r.code}\n${r.all}`);
  assert.ok(
    new RegExp(`UNREGISTERED.*${file}:\\d+: ${name}\\b`).test(r.err),
    `expected ${file}:${name} to be named as unregistered, got:\n${r.all}`
  );
  return r;
}

function assertClean(dir, expectSuites) {
  const r = runCheck(dir);
  assert.strictEqual(r.code, 0, `expected exit 0, got ${r.code}\n${r.all}`);
  if (expectSuites !== undefined) {
    assert.ok(
      new RegExp(`${expectSuites} Alcotest suite\\(s\\)`).test(r.out),
      `expected ${expectSuites} suite(s) to be COUNTED — a pass over zero suites is a vacuous pass. Got:\n${r.out}`
    );
  }
  return r;
}

const HELPER = 'let ok () = Alcotest.(check bool) "ok" true true\n';
const runSuite = (suite, entries) =>
  `let () =\n  Alcotest.run\n    "${suite}"\n    [ ("g", [ ${entries} ]) ]\n`;
const entry = (label, fn) => `Alcotest.test_case "${label}" \`Quick ${fn}`;

// ── evasion 1: `and` in a let rec chain ─────────────────────────────────────
// The old definition regex anchored on `^let`, so an `and`-bound case was never
// collected as a definition at all — it could not be reported even in
// principle.
check("E1: an `and`-bound case in a let rec chain is reported", () => {
  const dir = fixture({
    "test_e1.ml":
      HELPER +
      "\nlet rec test_e1_registered () = ok ()\n\nand test_e1_orphan () = ok ()\n\n" +
      runSuite("e1", entry("reg", "test_e1_registered")),
  });
  assertOrphan(dir, "test_e1_orphan", "test_e1.ml");
});

check("E1 control: an `and` chain with EVERY case registered passes", () => {
  const dir = fixture({
    "test_e1ok.ml":
      HELPER +
      "\nlet rec test_e1ok_a () = ok ()\n\nand test_e1ok_b () = ok ()\n\n" +
      runSuite("e1ok", `${entry("a", "test_e1ok_a")}; ${entry("b", "test_e1ok_b")}`),
  });
  assertClean(dir, 1);
});

// ── evasion 2: definition inside `module M = struct ... end` ────────────────
// `^let` required column 0, so every case in a nested module was invisible.
check("E2: a module-nested case is reported", () => {
  const dir = fixture({
    "test_e2.ml":
      HELPER +
      "\nmodule Inner = struct\n  let test_e2_orphan () = ok ()\n\n  let test_e2_registered () = ok ()\nend\n\n" +
      runSuite("e2", entry("reg", "Inner.test_e2_registered")),
  });
  assertOrphan(dir, "test_e2_orphan", "test_e2.ml");
});

check("E2 control: a module-nested case that IS registered passes", () => {
  const dir = fixture({
    "test_e2ok.ml":
      HELPER +
      "\nmodule Inner = struct\n  let test_e2ok_a () = ok ()\n\n  let test_e2ok_b () = ok ()\nend\n\n" +
      runSuite(
        "e2ok",
        `${entry("a", "Inner.test_e2ok_a")}; ${entry("b", "Inner.test_e2ok_b")}`
      ),
  });
  assertClean(dir, 1);
});

// ── evasion 3: two orphans that reference each other ────────────────────────
// Under a mention rule each one "used" the other, so both read as referenced.
// Reachability is rooted at the registrations, and neither is reachable.
check("E3: two mutually referencing orphans are BOTH reported", () => {
  const dir = fixture({
    "test_e3.ml":
      HELPER +
      "\nlet rec test_e3_a () = if false then test_e3_b () else ok ()\n\n" +
      "and test_e3_b () = if false then test_e3_a () else ok ()\n\n" +
      "let test_e3_registered () = ok ()\n\n" +
      runSuite("e3", entry("reg", "test_e3_registered")),
  });
  const r = assertOrphan(dir, "test_e3_a", "test_e3.ml");
  assert.ok(/test_e3_b\b/.test(r.err), `test_e3_b was laundered by the mutual reference:\n${r.all}`);
});

check("E3 control: a mutually recursive pair with both cases registered passes", () => {
  const dir = fixture({
    "test_e3ok.ml":
      HELPER +
      "\nlet rec test_e3ok_a () = if false then test_e3ok_b () else ok ()\n\n" +
      "and test_e3ok_b () = if false then test_e3ok_a () else ok ()\n\n" +
      runSuite("e3ok", `${entry("a", "test_e3ok_a")}; ${entry("b", "test_e3ok_b")}`),
  });
  assertClean(dir, 1);
});

// ── evasion 4: a stray mention launders an orphan ───────────────────────────
// The systemic one. The old checker never looked inside the suite list, so ANY
// occurrence of the name counted — including one written specifically to shut
// the checker up.
check("E4: `let _ = ignore test_never_runs` does not launder an orphan", () => {
  const dir = fixture({
    "test_e4.ml":
      HELPER +
      "\nlet test_e4_never_runs () = ok ()\n\nlet test_e4_registered () = ok ()\n\n" +
      "let _ = ignore test_e4_never_runs\n\n" +
      runSuite("e4", entry("reg", "test_e4_registered")),
  });
  assertOrphan(dir, "test_e4_never_runs", "test_e4.ml");
});

check("E4b: a mention in a plain top-level binding does not launder either", () => {
  const dir = fixture({
    "test_e4b.ml":
      HELPER +
      "\nlet test_e4b_never_runs () = ok ()\n\nlet test_e4b_registered () = ok ()\n\n" +
      "let unused_alias = test_e4b_never_runs\n\n" +
      runSuite("e4b", entry("reg", "test_e4b_registered")),
  });
  assertOrphan(dir, "test_e4b_never_runs", "test_e4b.ml");
});

check("E4 control: a helper actually CALLED by a registered case is not an orphan", () => {
  // The repo has real cases of this — test_near_zero takes two arguments and is
  // invoked from two registered cases. Reporting it would be a false positive,
  // and this is the assertion that keeps the fix from being "report everything".
  const dir = fixture({
    "test_e4ok.ml":
      HELPER +
      "\nlet test_e4ok_helper name reference = ignore name ; ignore reference ; ok ()\n\n" +
      "let test_e4ok_registered () = test_e4ok_helper \"n\" 1 ; ok ()\n\n" +
      runSuite("e4ok", entry("reg", "test_e4ok_registered")),
  });
  assertClean(dir, 1);
});

// ── evasion 5: defined in one file, registered in another ───────────────────
// `analyze` bailed on any file that did not itself call Alcotest.run, so the
// ordinary helpers.ml + test_main.ml layout was entirely unchecked.
check("E5: an orphan in a helpers file with no Alcotest.run is reported", () => {
  const dir = fixture({
    "helpers.ml":
      'let test_e5_registered () = Alcotest.(check bool) "ok" true true\n\n' +
      'let test_e5_orphan () = Alcotest.(check bool) "ok" true true\n',
    "test_main.ml": runSuite("e5", entry("reg", "Helpers.test_e5_registered")),
  });
  assertOrphan(dir, "test_e5_orphan", "helpers.ml");
});

check("E5 control: a case registered from the OTHER file passes", () => {
  const dir = fixture({
    "helpers.ml": 'let test_e5ok_case () = Alcotest.(check bool) "ok" true true\n',
    "test_main.ml": runSuite("e5ok", entry("case", "Helpers.test_e5ok_case")),
  });
  assertClean(dir, 1);
});

// ── evasion 6: the char literal '"' desynced the string lexer ───────────────
// One such literal blanked the rest of the file, taking Alcotest.run with it,
// so the file stopped counting as a suite and was dropped without a word.
check("E6: an orphan after a '\"' char literal is reported", () => {
  const dir = fixture({
    "test_e6.ml":
      HELPER +
      "\nlet quote_char = '\"'\n\nlet test_e6_orphan () = ignore quote_char ; ok ()\n\n" +
      "let test_e6_registered () = ok ()\n\n" +
      runSuite("e6", entry("reg", "test_e6_registered")),
  });
  assertOrphan(dir, "test_e6_orphan", "test_e6.ml");
});

// ── evasion 7: a string ending in a doubled backslash ───────────────────────
// `src[i-1] !== "\\"` read the closing quote of "C:\\" as escaped, so the
// string never closed and the rest of the file was blanked.
check("E7: an orphan after a \"C:\\\\\" string is reported", () => {
  const dir = fixture({
    "test_e7.ml":
      HELPER +
      '\nlet win_root = "C:\\\\"\n\nlet test_e7_orphan () = ignore win_root ; ok ()\n\n' +
      "let test_e7_registered () = ok ()\n\n" +
      runSuite("e7", entry("reg", "test_e7_registered")),
  });
  assertOrphan(dir, "test_e7_orphan", "test_e7.ml");
});

check("E6+E7 control: a file with '\"', '\\\\', 'a' and \"C:\\\\\" still passes when everything is registered", () => {
  // The suite count assertion is the load-bearing half: exit 0 with the file
  // silently dropped from the scan looks identical to exit 0 with the file
  // fully understood. Only the count distinguishes them.
  const dir = fixture({
    "test_e67ok.ml":
      HELPER +
      "\nlet quote_char = '\"'\n" +
      "let back_char = '\\\\'\n" +
      "let plain_char = 'a'\n" +
      "let newline_char = '\\n'\n" +
      'let win_root = "C:\\\\"\n' +
      'let esc_quote = "he said \\"hi\\""\n' +
      "\nlet test_e67ok_a () = ignore quote_char ; ignore back_char ; ok ()\n" +
      "\nlet test_e67ok_b () = ignore plain_char ; ignore newline_char ; ignore win_root ; ignore esc_quote ; ok ()\n\n" +
      runSuite("e67ok", `${entry("a", "test_e67ok_a")}; ${entry("b", "test_e67ok_b")}`),
  });
  assertClean(dir, 1);
});

// ── the raw Alcotest tuple registration form ───────────────────────────────
// sarek/tests/unit registers with `("name", `Quick, test_foo)` and no test_case
// constructor at all. A checker that only knew test_case called every case in
// those files an orphan.
check("the raw (\"name\", `Quick, fn) tuple counts as a registration", () => {
  const dir = fixture({
    "test_tuple.ml":
      HELPER +
      "\nlet test_tuple_a () = ok ()\n\nlet test_tuple_b () = ok ()\n\n" +
      'let tests = [ ("a", `Quick, test_tuple_a); ("b", `Quick, test_tuple_b) ]\n\n' +
      'let () = Alcotest.run "tuple" [ ("g", tests) ]\n',
  });
  assertClean(dir, 1);
});

check("an orphan alongside tuple-registered cases is still reported", () => {
  const dir = fixture({
    "test_tuple2.ml":
      HELPER +
      "\nlet test_tuple2_a () = ok ()\n\nlet test_tuple2_orphan () = ok ()\n\n" +
      'let tests = [ ("a", `Quick, test_tuple2_a) ]\n\n' +
      'let () = Alcotest.run "tuple2" [ ("g", tests) ]\n',
  });
  assertOrphan(dir, "test_tuple2_orphan", "test_tuple2.ml");
});

// ── a commented-out registration must not count ────────────────────────────
check("a registration inside a (* nested (* comment *) *) does not count", () => {
  const dir = fixture({
    "test_cmt.ml":
      HELPER +
      "\nlet test_cmt_orphan () = ok ()\n\nlet test_cmt_registered () = ok ()\n\n" +
      "(* disabled (* for now *) " +
      entry("orphan", "test_cmt_orphan") +
      " *)\n\n" +
      runSuite("cmt", entry("reg", "test_cmt_registered")),
  });
  assertOrphan(dir, "test_cmt_orphan", "test_cmt.ml");
});

// ── non-Alcotest drivers are out of scope ──────────────────────────────────
check("a hand-rolled driver that never mentions Alcotest is left alone", () => {
  // These exist in the repo (spoc/registry/test, spoc/ir/test): the cases are
  // registered, just by `let () = test_a () ; test_b ()`. Flagging them would
  // have produced 500+ false positives.
  const dir = fixture({
    "test_plain.ml":
      "let test_plain_a () = assert true\n\nlet test_plain_b () = assert true\n\n" +
      'let () = test_plain_a () ; test_plain_b () ; print_endline "ok"\n',
  });
  assertClean(dir, 0);
});

// ── vacuity and CLI contract ───────────────────────────────────────────────
check("a pass over zero suites says so, so a vacuous pass is visible", () => {
  const dir = fixture({ "test_empty.ml": "let x = 1\n" });
  const r = assertClean(dir, 0);
  assert.ok(/scanned \d+ \.ml file\(s\)/.test(r.out), `scan counts missing:\n${r.out}`);
});

check("a nonexistent target is exit 2, not a silent clean pass", () => {
  const r = spawnSync("node", [CHECKER, "/nonexistent/definitely-not-here"], { encoding: "utf8" });
  assert.strictEqual(r.status, 2, `expected exit 2, got ${r.status}\n${r.stdout}${r.stderr}`);
  assert.ok(/does not exist/.test(r.stderr));
});

check("an unreadable directory is reported and exits 2, not silently dropped", () => {
  // This previously asserted exit 0 — "does not crash" — which blessed the
  // silent drop: an unreadable directory removes whatever is inside it from
  // the scan, so `chmod 000` over a directory holding an orphan turned
  // "1 unregistered case" into "OK, 0 scanned". Guarding the exception fixed
  // the stack trace and left the vacuous pass. An assertion that encodes the
  // weaker behaviour promotes it to a specification.
  const dir = fixture({ "test_perm.ml": "let x = 1\n" });
  const locked = path.join(dir, "locked");
  fs.mkdirSync(locked);
  fs.writeFileSync(
    path.join(locked, "test_hidden.ml"),
    'let test_orphan () = Alcotest.(check int) "x" 0 1\nlet () = Alcotest.run "S" []\n'
  );
  fs.chmodSync(locked, 0o000);
  try {
    const r = runCheck(dir);
    assert.strictEqual(r.code, 2, `expected exit 2, got ${r.code}\n${r.all}`);
    assert.ok(/UNREADABLE/.test(r.err), `should name the unreadable path:\n${r.err}`);
    assert.ok(!/at Object\./.test(r.err), `raw stack trace leaked:\n${r.err}`);
  } finally {
    fs.chmodSync(locked, 0o755);
  }
});

check("positive control: the same tree readable reports the orphan inside it", () => {
  const dir = fixture({ "test_perm.ml": "let x = 1\n" });
  const locked = path.join(dir, "locked");
  fs.mkdirSync(locked);
  fs.writeFileSync(
    path.join(locked, "test_hidden.ml"),
    'let test_orphan () = Alcotest.(check int) "x" 0 1\nlet () = Alcotest.run "S" []\n'
  );
  const r = runCheck(dir);
  assert.strictEqual(r.code, 1, `expected exit 1, got ${r.code}\n${r.all}`);
  assert.ok(/test_orphan/.test(r.err), r.err);
});

// ── unit: the lexer ────────────────────────────────────────────────────────
check("the stripper preserves line numbers across a multi-line comment", () => {
  const src = "let a = 1\n(* one\n   two\n   three *)\nlet test_x () = ()\n";
  const code = stripCommentsAndStrings(src);
  assert.strictEqual(code.split("\n").length, src.split("\n").length, "line count changed");
  assert.strictEqual(collectDefs(code).get("test_x"), 5);
});

check("nested comments close at the right depth", () => {
  const code = stripCommentsAndStrings("(* a (* b *) c *) let test_y () = ()\n");
  assert.ok(/let\s+test_y/.test(code), `nested comment swallowed the code: ${JSON.stringify(code)}`);
});

check("a string inside a comment cannot close the comment early", () => {
  const code = stripCommentsAndStrings('(* "*)" still inside *) let test_z () = ()\n');
  assert.ok(/let\s+test_z/.test(code), `bad comment/string interaction: ${JSON.stringify(code)}`);
});

check("an even run of backslashes really does close the string", () => {
  const code = stripCommentsAndStrings('let p = "C:\\\\" let test_w () = ()\n');
  assert.ok(/let\s+test_w/.test(code), `"C:\\\\" ate the rest of the file: ${JSON.stringify(code)}`);
});

check("an odd run of backslashes escapes the quote (positive control)", () => {
  // "a\" b " — the middle quote is escaped, so the string runs to the LAST one
  // and `hidden` must not survive as code.
  const code = stripCommentsAndStrings('let p = "a\\" hidden " let test_v () = ()\n');
  assert.ok(!/hidden/.test(code), `escaped quote was treated as a close: ${JSON.stringify(code)}`);
  assert.ok(/let\s+test_v/.test(code));
});

check("char literals do not desync the string state, type variables are not eaten", () => {
  const code = stripCommentsAndStrings(
    "let q = '\"' let b = '\\\\' let n = '\\n' let f : 'a -> 'a = fun x -> x let test_u () = ()\n"
  );
  assert.ok(/let\s+test_u/.test(code), `char/tyvar handling desynced: ${JSON.stringify(code)}`);
});

check("a prime in an identifier is not read as a char literal", () => {
  const code = stripCommentsAndStrings("let x' = 1 and y' = 2 let test_t () = ()\n");
  assert.ok(/let\s+test_t/.test(code), `identifier prime misread: ${JSON.stringify(code)}`);
});

check("an unterminated comment does NOT blank the rest of the file", () => {
  // The old stripper dropped everything after a lexing surprise, which turned a
  // suite file into a non-suite file and removed it from the scan silently.
  const code = stripCommentsAndStrings("(* oops\nlet test_s () = ()\nAlcotest.run\n");
  assert.ok(/Alcotest\.run/.test(code), `lexing surprise blanked the tail: ${JSON.stringify(code)}`);
});

check("an unterminated string does NOT blank the rest of the file", () => {
  const code = stripCommentsAndStrings('let s = "oops\nlet test_r () = ()\nAlcotest.run\n');
  assert.ok(/Alcotest\.run/.test(code), `lexing surprise blanked the tail: ${JSON.stringify(code)}`);
});

check("{|quoted strings|} are stripped", () => {
  const code = stripCommentsAndStrings("let s = {qq|let test_hidden () = ()|qq} let test_q () = ()\n");
  assert.ok(!/test_hidden/.test(code));
  assert.ok(/let\s+test_q/.test(code));
});

// ── unit: definition and registration extraction ───────────────────────────
check("a value named test_* with no parameters is not mistaken for a case", () => {
  // `let test_cases = [ ... ]` is a real local binding in this repo. It cannot
  // be passed to Alcotest.test_case, which needs unit -> unit.
  const defs = collectDefs("let test_cases = [1; 2]\nlet test_real () = ()\n");
  assert.ok(!defs.has("test_cases"), "a parameterless value was collected as a case");
  assert.ok(defs.has("test_real"));
});

check("partial application inside test_case counts as registering the helper", () => {
  // `test_case name `Quick (test_unhandled_raises spec)` — the real shape in
  // sarek/tests/codegen_golden.
  const reg = collectRegistered("test_case name `Quick (test_unhandled_raises spec))", new Set());
  assert.ok(reg.has("test_unhandled_raises"));
});

check("the argument scan stops at the list separator and does not swallow the next entry", () => {
  // If the span ran past the `;` every subsequent case would be marked
  // registered by proximity, which is the mention bug in a new costume.
  const reg = collectRegistered('test_case "a" `Quick test_alpha; let x = test_beta', new Set());
  assert.ok(reg.has("test_alpha"));
  assert.ok(!reg.has("test_beta"), "the scan ran past the entry boundary");
});

for (const dir of tmpRoots) fs.rmSync(dir, { recursive: true, force: true });

// ── the wrapper layout that used to red correct code (finding 5, false positive)
check("a suite bound to a name and passed by identifier is not reported as unregistered", () => {
  const dir = fixture({
    "helpers.ml": "let case name f = Alcotest.test_case name `Quick f\n",
    "test_wrap.ml":
      'let test_alpha () = Alcotest.(check int) "a" 1 1\n' +
      'let suite = [ ("g", [ Helpers.case "alpha" test_alpha ]) ]\n' +
      'let () = Alcotest.run "S" suite\n',
  });
  const r = runCheck(dir);
  assert.strictEqual(r.code, 0, `false positive on a valid wrapper layout:\n${r.all}`);
});

check("an orphan in that same wrapper layout is STILL caught", () => {
  // The fix widened registration roots to the run call's argument region.
  // Widening a root set can only weaken detection, so this pins that it did not.
  const dir = fixture({
    "helpers.ml": "let case name f = Alcotest.test_case name `Quick f\n",
    "test_wrap.ml":
      'let test_alpha () = Alcotest.(check int) "a" 1 1\n' +
      'let test_forgotten () = Alcotest.(check int) "boom" 0 1\n' +
      'let suite = [ ("g", [ Helpers.case "alpha" test_alpha ]) ]\n' +
      'let () = Alcotest.run "S" suite\n',
  });
  const r = runCheck(dir);
  assert.strictEqual(r.code, 1, `orphan missed after the root set was widened:\n${r.all}`);
  assert.ok(/test_forgotten/.test(r.err), r.err);
});

check("a literally empty suite still reports the cases defined above it", () => {
  const dir = fixture({
    "test_empty.ml":
      'let test_dead () = Alcotest.(check int) "never" 0 1\n' +
      'let () = Alcotest.run "S" []\n',
  });
  const r = runCheck(dir);
  assert.strictEqual(r.code, 1, `an empty suite must not read as registered:\n${r.all}`);
  assert.ok(/test_dead/.test(r.err), r.err);
});

check("KNOWN LIMITATION is documented for dead-code registration (this asserts the gap, not the fix)", () => {
  // Recorded deliberately: `if false then [...]` reads as registered because
  // the case is textually inside the suite expression. Closing it needs OCaml
  // evaluation. The assertion exists so the limitation cannot be forgotten,
  // and so that if someone later closes it this test fails loudly and gets
  // updated rather than the limitation silently outliving its comment.
  const dir = fixture({
    "test_dead_branch.ml":
      'let test_dead () = Alcotest.(check int) "never" 0 1\n' +
      'let () = Alcotest.run "S" (if false then [ ("g", [ Alcotest.test_case "d" `Quick test_dead ]) ] else [])\n',
  });
  const r = runCheck(dir);
  assert.strictEqual(r.code, 0, "if this now exits 1, the limitation is closed — update the KNOWN LIMITATIONS block");
  const src = fs.readFileSync(path.resolve(__dirname, "check-alcotest-registration.js"), "utf8");
  assert.ok(/KNOWN LIMITATIONS/.test(src), "the limitation must stay documented in the source");
  assert.ok(/if false then/.test(src), "the reproducer must stay in the source");
});

console.log(results.join("\n"));
console.log(`\ncheck-alcotest-registration.test: ${pass}/${results.length} passed`);
