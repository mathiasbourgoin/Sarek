// Plain-Node unit tests (no browser, no deps) for the HTML-escaping helpers
// in benchmark-viewer.js. Contributor-submitted benchmark JSON flows into
// innerHTML at several sinks in that file; these tests are a regression
// guard against reintroducing stored-XSS there.
//
// Run: node gh-pages/javascripts/test/escaping_test.js

'use strict';

const assert = require('assert');
const path = require('path');

const { escapeHtml, markdownToHtml } = require(
    path.join(__dirname, '..', 'benchmark-viewer.js')
);

let passed = 0;

function test(name, fn) {
    try {
        fn();
        passed++;
        console.log(`ok - ${name}`);
    } catch (err) {
        console.error(`not ok - ${name}`);
        console.error(err);
        process.exitCode = 1;
    }
}

// (a) An <img onerror=...> payload in a description must never survive as a
// real tag once rendered.
test('markdownToHtml neutralizes an <img onerror> XSS payload', () => {
    const rendered = markdownToHtml('<img src=x onerror=alert(1)>');
    assert.ok(!rendered.includes('<img'), `expected no literal <img tag, got: ${rendered}`);
});

// (b) A javascript: link must never render as a clickable/executable href.
test('markdownToHtml drops javascript: links entirely', () => {
    const rendered = markdownToHtml('[click](javascript:alert(1))');
    assert.ok(!rendered.includes('javascript:'), `expected no javascript: substring, got: ${rendered}`);
    assert.ok(!rendered.includes('<a '), `expected the link to be dropped, got: ${rendered}`);
    assert.ok(rendered.includes('click'), `expected the link text to remain, got: ${rendered}`);
});

// (b1) Common javascript: bypass variants (case, embedded whitespace) must be
// dropped too -- the allow-list is an exact ^https?:// match, so anything
// that isn't literally http(s) is rejected regardless of scheme casing or
// stray whitespace an attacker might use to dodge a naive string check.
test('markdownToHtml drops javascript: link bypass variants (case/whitespace)', () => {
    const variants = [
        '[click](JavaScript:alert(1))',
        '[click](java\tscript:alert(1))',
        '[click]( javascript:alert(1))',
    ];
    for (const md of variants) {
        const rendered = markdownToHtml(md);
        assert.ok(!rendered.includes('<a '), `expected link dropped for ${JSON.stringify(md)}, got: ${rendered}`);
    }
});

// (b2) An https:// link is still allowed and rendered as a real anchor.
test('markdownToHtml keeps https:// links as real anchors', () => {
    const rendered = markdownToHtml('[docs](https://example.com/page)');
    assert.ok(rendered.includes('<a href="https://example.com/page"'), `expected an anchor, got: ${rendered}`);
    assert.ok(rendered.includes('>docs<'), `expected link text preserved, got: ${rendered}`);
});

// (c) Bold formatting must still work after the escape-everything rewrite.
test('markdownToHtml still renders **bold** as <strong>', () => {
    const rendered = markdownToHtml('**bold** text');
    assert.ok(rendered.includes('<strong>bold</strong>'), `expected <strong>bold</strong>, got: ${rendered}`);
});

// (c2) Inline code and fenced code blocks still work, and code content stays escaped.
test('markdownToHtml still renders `inline code` as <code>', () => {
    const rendered = markdownToHtml('`x < y`');
    assert.ok(rendered.includes('<code'), `expected a <code> tag, got: ${rendered}`);
    assert.ok(rendered.includes('&lt;'), `expected escaped < inside code, got: ${rendered}`);
});

test('markdownToHtml keeps fenced code block content escaped and literal', () => {
    const rendered = markdownToHtml('```js\nconst x = "<script>alert(1)</script>";\n```');
    assert.ok(!rendered.includes('<script>'), `expected no literal <script> tag, got: ${rendered}`);
    assert.ok(rendered.includes('<pre'), `expected a <pre> block, got: ${rendered}`);
});

// (c3) The fenced-code-block placeholder must not collide with literal author
// prose. Before the fix the placeholder was the plain string " CODEBLOCK0 ",
// so a description that happened to contain that exact substring got the
// unrelated code block's content spliced into it. The placeholder is now
// wrapped in a Private-Use-Area sentinel (U+E000) that escapeHtml() can never
// produce, so it cannot be forged from author-supplied text.
test('markdownToHtml does not let literal "CODEBLOCK0" prose collide with the placeholder', () => {
    const rendered = markdownToHtml('See CODEBLOCK0 for details.\n```js\nconst secret = 1;\n```');
    assert.ok(
        rendered.includes('See CODEBLOCK0 for details.'),
        `expected the literal prose to survive unchanged, got: ${rendered}`
    );
    assert.ok(rendered.includes('<pre'), `expected the real code block to still render, got: ${rendered}`);
    assert.ok(rendered.includes('const secret = 1;'), `expected the code block content to survive, got: ${rendered}`);
});

// (d) Plain text with & < > " ' round-trips as escaped entities.
test('escapeHtml round-trips & < > " \' as entities', () => {
    const escaped = escapeHtml(`& < > " '`);
    assert.strictEqual(escaped, '&amp; &lt; &gt; &quot; &#39;');
});

test('markdownToHtml escapes raw & < > in plain text', () => {
    const rendered = markdownToHtml('Tom & Jerry < 5 > 3');
    assert.ok(rendered.includes('Tom &amp; Jerry &lt; 5 &gt; 3'), `expected escaped entities, got: ${rendered}`);
});

// Images are documented as dropped: `![alt](src)` must not become a live <img>.
test('markdownToHtml drops image syntax instead of rendering <img>', () => {
    const rendered = markdownToHtml('![alt text](http://evil.example/x.png)');
    assert.ok(!rendered.includes('<img'), `expected no <img> tag, got: ${rendered}`);
});

console.log(`\n${passed} test(s) passed.`);
if (process.exitCode) {
    console.error('FAILURES ABOVE');
} else {
    console.log('ALL PASS');
}
