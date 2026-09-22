const {test}=require('node:test');
const assert=require('node:assert/strict');
const {mountGallery}=require('../src/keyprint/web/gallery.js');

function setup() {
  const nodes=new Map();
  const make=()=>({children:[],attrs:{},listeners:{},textContent:'',
    append(...items){this.children.push(...items)},replaceChildren(){this.children=[]},
    setAttribute(k,v){this.attrs[k]=v},addEventListener(k,v){this.listeners[k]=v}});
  const get=id=>{if(!nodes.has(id))nodes.set(id,make());return nodes.get(id)};
  return {get,doc:{getElementById:get,createElement:make}};
}
const example=(title,text)=>({title,recording:{schema:'keyprint-comparison-v1',prompt:title,
  experiment:{outputs:{ordinary:{text},marked:{text}}}}});

test('switches real recordings, preserves exact data and uses inert text',()=>{
  const {doc,get}=setup(),seen=[];
  const data={schema:'keyprint-gallery-v1',examples:[example('First','A'),example('<script>','🌱'.repeat(111))]};
  data.examples[1].note='Known constraint failure. <b>Keep literal.</b>';
  mountGallery(doc,data,x=>seen.push(x));
  assert.equal(seen[0],data.examples[0].recording);
  assert.equal(get('gallery').hidden,false);
  const buttons=get('gallery-cards').children;
  assert.equal(buttons[1].children[1].textContent,'<script>');
  assert.equal(buttons[1].children[2].textContent,'🌱'.repeat(110)+'…');
  buttons[1].listeners.click();
  assert.equal(seen[1],data.examples[1].recording);
  assert.equal(buttons[0].attrs['aria-pressed'],'false');
  assert.equal(buttons[1].attrs['aria-pressed'],'true');
  assert.equal(get('gallery-note').hidden,false);
  assert.equal(get('gallery-note').textContent,data.examples[1].note);
  buttons[0].listeners.click();
  assert.equal(get('gallery-note').hidden,true);
});
test('malformed recordings fail before replacing the prior gallery',()=>{
  const {doc,get}=setup();get('gallery-cards').children=['old'];
  assert.throws(()=>mountGallery(doc,{schema:'keyprint-gallery-v1',examples:[example('One','ok'),{}]},()=>{}),/format/);
  assert.deepEqual(get('gallery-cards').children,['old']);
});
