#!/usr/bin/env node
// Portable, zero-dependency integrity verifier installed with the review-tool bundle.
//
// This deliberately does NOT implement install/upgrade/remove. Those lifecycle operations stay
// in the externally fetched review-bundle-install.sh so a consumer does not take ownership of its
// bootstrapper. This tool only checks the installed manifest and every manifest-owned file.
//
// Usage: node scripts/review-bundle-verify.js [--target <project-root>]

"use strict";

const crypto = require("crypto");
const fs = require("fs");
const path = require("path");

const MANIFEST_REL = "scripts/review-bundle.manifest.json";
const RECOVERY =
  "Fetch review-bundle-install.sh from a trusted roster source and re-run upgrade, then /recruit update.";

function usage() {
  console.error("usage: node scripts/review-bundle-verify.js [--target <project-root>]");
  return 2;
}

function parseArgs(argv) {
  if (argv.length === 0) return { target: path.resolve(__dirname, "..") };
  if (argv.length === 2 && argv[0] === "--target" && argv[1]) {
    return { target: path.resolve(argv[1]) };
  }
  return null;
}

function sha256(file) {
  return crypto.createHash("sha256").update(fs.readFileSync(file)).digest("hex");
}

function isInside(root, candidate) {
  return candidate === root || candidate.startsWith(`${root}${path.sep}`);
}

function verify(target) {
  let root;
  try {
    root = fs.realpathSync(target);
  } catch {
    console.error(`review-bundle-verify: target does not exist or is not readable: ${target}`);
    return 1;
  }

  const manifestPath = path.resolve(root, MANIFEST_REL);
  let manifest;
  try {
    const stat = fs.lstatSync(manifestPath);
    if (!stat.isFile() || stat.isSymbolicLink()) throw new Error("not a regular file");
    manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  } catch (error) {
    console.error(`review-bundle-verify: missing or invalid ${MANIFEST_REL}: ${error.message}. ${RECOVERY}`);
    return 1;
  }

  if (!manifest || !Array.isArray(manifest.files) || manifest.files.length === 0) {
    console.error(`review-bundle-verify: invalid ${MANIFEST_REL}: files must be a non-empty array. ${RECOVERY}`);
    return 1;
  }

  const seen = new Set();
  let problems = 0;
  for (const entry of manifest.files) {
    const rel = entry && entry.path;
    const expected = entry && entry.sha256;
    if (
      typeof rel !== "string" ||
      rel.length === 0 ||
      typeof expected !== "string" ||
      !/^[a-f0-9]{64}$/.test(expected)
    ) {
      console.error("review-bundle-verify: INVALID manifest file entry");
      problems += 1;
      continue;
    }
    if (seen.has(rel)) {
      console.error(`review-bundle-verify: DUPLICATE ${rel}`);
      problems += 1;
      continue;
    }
    seen.add(rel);

    const absolute = path.resolve(root, rel);
    if (!isInside(root, absolute)) {
      console.error(`review-bundle-verify: UNSAFE PATH ${rel}`);
      problems += 1;
      continue;
    }

    try {
      const stat = fs.lstatSync(absolute);
      if (!stat.isFile() || stat.isSymbolicLink()) {
        console.error(`review-bundle-verify: NOT REGULAR FILE ${rel}`);
        problems += 1;
        continue;
      }
      const real = fs.realpathSync(absolute);
      if (!isInside(root, real)) {
        console.error(`review-bundle-verify: UNSAFE PATH ${rel}`);
        problems += 1;
        continue;
      }
      const actual = sha256(absolute);
      if (actual !== expected) {
        console.error(`review-bundle-verify: SHA MISMATCH ${rel}`);
        problems += 1;
      }
    } catch {
      console.error(`review-bundle-verify: MISSING ${rel}`);
      problems += 1;
    }
  }

  if (problems > 0) {
    console.error(`review-bundle-verify: ${problems} problem(s) found. ${RECOVERY}`);
    return 1;
  }
  console.log(
    `review-bundle-verify: OK — ${manifest.files.length} file(s) present and sha-matched (bundle ${manifest.bundle_version || "unknown"}).`
  );
  return 0;
}

function main(argv = process.argv.slice(2)) {
  const args = parseArgs(argv);
  if (!args) return usage();
  return verify(args.target);
}

if (require.main === module) process.exitCode = main();

module.exports = { main, verify };
