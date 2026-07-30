# gh-pages/javascripts/test/

Plain-Node unit tests for `gh-pages/javascripts/*.js` (no browser, no deps). Unlike
`gh-pages/learn/test/` (Playwright-driven GPU acceptance tests that need a real
WebGPU browser), these run with a bare `node <file>.js` and exit non-zero on
failure.

Run: `node gh-pages/javascripts/test/escaping_test.js`
