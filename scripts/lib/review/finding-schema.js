// scripts/lib/review/finding-schema.js — CommonJS, zero-dep.
//
// Hand-rolled JSON Schema interpreter for the supported subset only:
// `type`, `enum`, `required`, `properties`, `additionalProperties`, `items`
// (plus the metadata keywords `$schema`/`title`/`description`, which are
// ignored). D-4 (specs/review-skill-slimming.md): the repo carries zero
// runtime dependencies, so no ajv. The interpreter MUST fail closed on any
// schema keyword outside the supported subset (throw at compile time) so a
// future schema edit that assumes richer validation (minimum, pattern,
// oneOf, $ref, ...) cannot silently pass through unenforced — schema/vs
// validator drift becomes impossible to introduce quietly.
//
// Usage:
//   const { compile, loadFindingSchema } = require("./lib/review/finding-schema");
//   const validator = compile(require("../../schema/review-finding.schema.json"));
//   const { valid, errors } = validator.validate(candidate);
"use strict";

const path = require("path");

const SUPPORTED_KEYWORDS = new Set([
  "type",
  "enum",
  "required",
  "properties",
  "additionalProperties",
  "items",
  "$schema",
  "title",
  "description",
]);

// Recursively verifies every keyword in the schema (and nested property/item
// schemas) is one this interpreter understands. Throws on the first miss.
function assertSupported(schema, at) {
  if (schema === null || typeof schema !== "object") return;
  for (const key of Object.keys(schema)) {
    if (!SUPPORTED_KEYWORDS.has(key)) {
      throw new Error(
        `finding-schema: unsupported schema keyword "${key}" at ${at || "$"} — the hand-rolled ` +
          "validator (D-4) fails closed instead of silently ignoring it"
      );
    }
  }
  if (schema.properties) {
    for (const [name, sub] of Object.entries(schema.properties)) {
      assertSupported(sub, `${at || "$"}.properties.${name}`);
    }
  }
  if (schema.items) assertSupported(schema.items, `${at || "$"}.items`);
}

// JSON-Schema-flavoured typeof: distinguishes "integer" from "number" and
// treats `null` and arrays as their own types (not "object").
function typeOf(value) {
  if (value === null) return "null";
  if (Array.isArray(value)) return "array";
  if (typeof value === "number") return Number.isInteger(value) ? "integer" : "number";
  return typeof value; // "string" | "boolean" | "object" | "undefined"
}

function matchesType(value, type) {
  const actual = typeOf(value);
  if (type === "number") return actual === "number" || actual === "integer";
  return actual === type;
}

function validateObjectShape(value, schema, at, errors) {
  if (schema.required) {
    for (const key of schema.required) {
      if (!(key in value)) errors.push(`${at}: missing required property "${key}"`);
    }
  }
  if (schema.properties) {
    for (const [key, subSchema] of Object.entries(schema.properties)) {
      if (key in value) validateValue(value[key], subSchema, `${at}.${key}`, errors);
    }
  }
  if (schema.additionalProperties === false) {
    const allowed = new Set(Object.keys(schema.properties || {}));
    for (const key of Object.keys(value)) {
      if (!allowed.has(key)) errors.push(`${at}: unexpected additional property "${key}"`);
    }
  }
}

function validateValue(value, schema, at, errors) {
  if (schema.type) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => matchesType(value, t))) {
      errors.push(`${at}: expected type ${types.join("|")}, got ${typeOf(value)}`);
      return; // type mismatch — nested checks would be meaningless
    }
  }
  if (schema.enum && !schema.enum.includes(value)) {
    errors.push(`${at}: value ${JSON.stringify(value)} not in enum [${schema.enum.join(", ")}]`);
  }
  if (typeOf(value) === "object" && (schema.properties || schema.required || schema.type === "object")) {
    validateObjectShape(value, schema, at, errors);
  }
  if (schema.items && Array.isArray(value)) {
    value.forEach((item, i) => validateValue(item, schema.items, `${at}[${i}]`, errors));
  }
}

// Compiles a schema object into a `{ validate(value) }` validator. Throws at
// compile time (not validate time) if the schema uses an unsupported
// keyword — a drifted schema fails the very first call, not silently.
function compile(schema) {
  assertSupported(schema, "$");
  return {
    validate(value) {
      const errors = [];
      validateValue(value, schema, "$", errors);
      return { valid: errors.length === 0, errors };
    },
  };
}

// Loads and compiles the canonical schema/review-finding.schema.json,
// resolved relative to the repo root (this file lives in scripts/lib/review/).
function loadFindingSchema() {
  const schemaPath = path.resolve(__dirname, "..", "..", "..", "schema", "review-finding.schema.json");
  return compile(require(schemaPath));
}

module.exports = { compile, loadFindingSchema, typeOf, matchesType };
