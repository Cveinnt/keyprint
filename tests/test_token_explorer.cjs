const { test } = require('node:test');
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function setup(reducedMotion = false) {
  const nodes = new Map(), timers = new Map();
  let next = 0;
  function node() {
    return { textContent: '', value: 'marked', children: [], attrs: {}, listeners: {},
      classList: { toggle() {} },
      setAttribute(k,v) { this.attrs[k] = v; },
      addEventListener(k,fn) { this.listeners[k] = fn; },
      replaceChildren() { this.children = []; },
      append(n) { this.children.push(n); },
    };
  }
  const get = id => { if (!nodes.has(id)) nodes.set(id,node()); return nodes.get(id); };
  const ctx = vm.createContext({module:{exports:{}},
    setInterval(fn) { const id = ++next; timers.set(id, fn); return id; },
    clearInterval(id) { timers.delete(id); },
  });
  vm.runInContext(fs.readFileSync(require.resolve('../src/keyprint/web/trace.js'),'utf8'),ctx);
  const explorer = ctx.module.exports.createTokenExplorer({getElementById:get,createElement:node},
    {reducedMotion:()=>reducedMotion});
  return {get,explorer,timers};
}
const trace = [
  {index:0, token_id:1, bytes_hex:'f09f', text:'', start:0,end:0,kind:'pending_bytes'},
  {index:1, token_id:2, bytes_hex:'8cb1', text:'🌱', start:0,end:1,kind:'text'},
  {index:2, token_id:3, bytes_hex:'3c623e', text:'<b>', start:1,end:4,kind:'text'},
  {index:3, token_id:4, bytes_hex:null, text:'', start:4,end:4,kind:'control'},
];
const pair = {outputs:{marked:{trace},ordinary:{trace:trace.slice(0,2)}}};

test('renders exact Unicode as text, exposes pending bytes and control tokens',()=>{
  const {get,explorer}=setup(); explorer.show(pair);
  assert.equal(get('choices').hidden,false);
  assert.equal(get('trace-prefix').textContent,'🌱<b>');
  assert.equal(get('trace-tokens').children[2].textContent,'<b>');
  get('trace-tokens').children[0].listeners.click();
  assert.match(get('trace-details').textContent,/bytes f09f/);
  assert.equal(get('trace-prefix').textContent,'(No visible text yet)');
  get('trace-position').value='1'; get('trace-position').listeners.input();
  assert.equal(get('trace-prefix').textContent,'🌱');
  assert.equal(get('trace-tokens').children[1].attrs['aria-pressed'],'true');
});
test('replay stops at end and switching conditions cancels replay',()=>{
  const {get,explorer,timers}=setup(); explorer.show(pair);
  get('trace-play').listeners.click();
  assert.equal(timers.size,1);
  for(let i=0;i<3;i++) [...timers.values()][0]();
  assert.equal(timers.size,0);
  assert.equal(get('trace-prefix').textContent,'🌱<b>');
  get('trace-play').listeners.click();
  get('trace-condition').value='ordinary'; get('trace-condition').listeners.change();
  assert.equal(timers.size,0);
  assert.equal(get('trace-prefix').textContent,'🌱');
});
test('reduced motion steps without a timer; old recordings hide explorer',()=>{
  const {get,explorer,timers}=setup(true); explorer.show(pair);
  get('trace-play').listeners.click();
  assert.equal(timers.size,0);
  assert.equal(get('trace-position').value,'0');
  explorer.show({outputs:{marked:{text:'Old recording'}}});
  assert.equal(get('choices').hidden,true);
  assert.equal(get('trace-tokens').children.length,0);
});
