// Lightweight UI logic checks. This is not a browser/layout test.
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const path = require('node:path');
const elements = new Map();
function element(selector) {
  if (!elements.has(selector)) elements.set(selector, {textContent:'',innerHTML:'',hidden:false,dataset:{},handlers:{},classList:{toggle(){}},setAttribute(){},addEventListener(name, handler){this.handlers[name]=handler;},querySelectorAll(){return [];}});
  return elements.get(selector);
}
const root = path.resolve(__dirname, '..');
const snapshot = {generated_at:'2026-09-10T12:00:00Z',snapshot_kind:'legacy',leagues:{'premier-league':[{date:'2026-09-15',matchweek:4,home_team:'Arsenal',away_team:'Away',prediction_available:true,home_win:.5,draw:.3,away_win:.2,expected_home_goals:1.7,expected_away_goals:1.0}]},season_outlook:{}};
const context = vm.createContext({console, Date, setTimeout, document:{querySelector:element,querySelectorAll:()=>[]},fetch: async url => ({ok:true,json:async()=>url.includes('predictions') ? snapshot : {}})});
vm.runInContext(fs.readFileSync(path.join(root,'site/app.js'),'utf8'),context);
setImmediate(() => {
  assert.match(element('#fixtures').innerHTML,/class="fixture"/);
  element('#team-search').handlers.input({target:{value:'Arsenal'}});
  assert.match(element('#fixtures').innerHTML,/Arsenal/);
  vm.runInContext("search='not-a-club'; render()",context);
  assert.match(element('#fixtures').innerHTML,/No fixtures found/);
  vm.runInContext("view='table'; render()",context);
  assert.match(element('#outlook').innerHTML,/Season outlook is unavailable/);
  assert.equal(vm.runInContext('ordinal(21)',context),'21st');
  assert.equal(vm.runInContext('ordinal(12)',context),'12th');
  assert.equal(vm.runInContext("esc('<script>')",context),'&lt;script&gt;');
  assert.match(element('#data-notice').textContent,/Archived forecast preview/);
  console.log('Frontend logic checks passed: loading, search, empty states, ordinal formatting, escaping and archive notice. Browser layout is not tested.');
});
