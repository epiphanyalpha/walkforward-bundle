# Bundle Tester — the browser app

`bundle-tester.html` is the whole tool: one self-contained file, no build step,
no dependencies, no server. Open it in a browser and it runs.

The engine is a JavaScript port of the `validation` package, executing inside a
Web Worker so the page stays responsive. Nothing is uploaded: a CSV dropped on
the page is read and processed locally.

Live version: see the link in the top-level README.
