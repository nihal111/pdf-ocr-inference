// Reuse the webpage's diagram templates without starting a browser or server.
// The adapter implements only the DOM operations these two renderers need.
const fs = require('node:fs');
const vm = require('node:vm');
const input = JSON.parse(fs.readFileSync(0, 'utf8'));
const source = fs.readFileSync(input.app, 'utf8');
const start = source.indexOf('function summaryPipelineDiagram(');
const end = source.indexOf('function expandableChangesTable(');
if (start < 0 || end < start) throw new Error('Summary diagram templates not found');
function element() {
  const callbacks = {};
  return {
    innerHTML: '', setAttribute() {},
    querySelectorAll(selector) {
      if (selector !== '[data-decoding-case]') throw new Error(selector);
      return ['match', 'different'].map(value => ({
        dataset: {decodingCase: value},
        addEventListener(event, callback) { callbacks[value] = callback; }
      }));
    },
    querySelector() { return {focus() {}}; },
    select(value) { callbacks[value](); }
  };
}
const context = {
  document: {createElement: element},
  esc: value => String(value ?? '').replace(/[&<>"']/g, c =>
    ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c])),
};
vm.createContext(context);
vm.runInContext(source.slice(start, end) +
  '\nglobalThis.templates={summaryPipelineDiagram,speculativeDecodingDiagram,summaryChangeExplainers,changeDiagram};',
  context);
function table(rows) {
  return {
    querySelectorAll(selector) {
      if (selector !== ':scope > tbody > tr') throw new Error(selector);
      return rows.map(row => ({cells: row.map(textContent => ({textContent}))}));
    },
    replaceWith(figure) { this.figure = figure; }
  };
}
const pipeline = table(input.pipeline);
context.templates.summaryPipelineDiagram(pipeline);
const speculation = table(input.speculation);
context.templates.speculativeDecodingDiagram(speculation);
const matched = speculation.figure.innerHTML;
speculation.figure.select('different');
const corrected = speculation.figure.innerHTML;
const changes = Object.fromEntries(Object.entries(context.templates.summaryChangeExplainers)
  .map(([key, value]) => [key, {...value, diagram: value.kind === 'speculation'
    ? matched : context.templates.changeDiagram(value.kind)}]));
process.stdout.write(JSON.stringify({pipeline: pipeline.figure.innerHTML, matched, corrected, changes}));
