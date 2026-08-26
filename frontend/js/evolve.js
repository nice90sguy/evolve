const $ = s => document.querySelector(s);
let S = null;                      // last server snapshot
// ---------- selection: THE singleton ----------
// INVARIANT (user mandate 2026-08-23): there is exactly ZERO or ONE selected
// Image, app-wide, forever - enforced by construction, not convention:
//  - the selection state lives in a closure nothing else can reach;
//  - Sel.apply() is the ONLY code that writes the .focus / .alias classes,
//    and it always clears every instance before applying exactly one;
//  - a new panel/widget joins selection by tagging elements with
//    data-target/data-index (+ data-id on Images) and calling Sel.set() -
//    never by touching classes or state itself.
// Sel is frozen so its methods cannot be replaced; target/index are getters.
const Sel = (() => {
  let cur = {target: 'working', index: 0};
  const get = () => ({target: cur.target, index: cur.index});
  const el = () => cur.target === 'none' ? null
    : cur.target === 'working' ? $('#stagebox')
    : document.querySelector(`[data-target="${cur.target}"][data-index="${cur.index}"]`);
  const id = () => {
    if (!S || cur.target === 'none') return null;
    if (cur.target === 'working') return S.working;
    if (cur.target === 'ref') return S.controls.refs[cur.index];
    if (cur.target === 'ref0') return S.controls.ref0;
    if (cur.target === 'slot') return (S.candidates[activeTab()] || [])[cur.index] ?? null;
    if (cur.target === 'pin') return S.pins[cur.index] ?? null;
    const l = listFor(cur.target); if (l) return l[cur.index] ?? null;
    return null;
  };
  let lastApplied = '';
  function apply() {
    document.querySelectorAll('.focus, .alias').forEach(e => e.classList.remove('focus', 'alias'));
    const e0 = el();
    const key = cur.target + ':' + cur.index + ':' + id();
    if (e0) {
      e0.classList.add('focus');
      if (e0.tagName === 'IMG' && key !== lastApplied) e0.scrollIntoView({inline: 'nearest', block: 'nearest'});
    }
    lastApplied = key;
    const i = id();
    if (i != null) lastSelId = i;
    if (i != null) {    // SELECTED_ALIAS: every other Image showing the same underlying image.
      // Style the enclosing [data-target] box, not the bare img: an img that
      // doesn't fill its container would otherwise show its own outline
      // inside the container's grey border - a double frame (user-reported).
      const seen = new Set();
      document.querySelectorAll(`[data-id="${i}"]`).forEach(x => {
        const box = x.closest('[data-target]') || x;
        if (box === e0 || (e0 && (e0.contains(box) || box.contains(e0))) || seen.has(box)) return;
        seen.add(box);
        box.classList.add('alias');
      });
    }
    if (spaceHeld) peek(cur.target === 'working' ? null : i);
    syncHistory();
  }
  const set = (target, index) => { cur = {target, index: +index || 0}; apply(); };
  const clear = () => { cur = {target: 'none', index: 0}; apply(); };
  return Object.freeze({get, set, clear, id, el, apply,
    get target() { return cur.target; }, get index() { return cur.index; }});
})();
// where imports land when nothing is selected
const selTarget = () => Sel.target === 'none' ? {target: 'working', index: 0} : Sel.get();
let lastSelId = null;
let lastRescanTs = null;         // the last selected image id, app-global (mode switches keep it)
let pollTimer = null;

const api = (path, body) => fetch('/api/' + path, body === undefined ? {} :
  {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)}).then(r => r.json());
// the standard mutate-and-refresh tail (was hand-repeated at every call site)
const act = (path, body) => api(path, body).then(r => { if (r && r.error) notice(r.error); return refresh(); });
// MUI Material Icons (mui.com/material-ui/material-icons, Apache 2.0),
// inlined as path data; fill-based, colored via currentColor
const I = {
  copy: 'M16 1H4c-1.1 0-2 .9-2 2v14h2V3h12V1zm3 4H8c-1.1 0-2 .9-2 2v14c0 1.1.9 2 2 2h11c1.1 0 2-.9 2-2V7c0-1.1-.9-2-2-2zm0 16H8V7h11v14z',
  done: 'M9 16.2 4.8 12l-1.4 1.4L9 19 21 7l-1.4-1.4L9 16.2z',
  close: 'M19 6.41 17.59 5 12 10.59 6.41 5 5 6.41 10.59 12 5 17.59 6.41 19 12 13.41 17.59 19 19 17.59 13.41 12 19 6.41z',
  grid: 'M3 3v8h8V3H3zm6 6H5V5h4v4zm-6 4v8h8v-8H3zm6 6H5v-4h4v4zm4-16v8h8V3h-8zm6 6h-4V5h4v4zm-6 4v8h8v-8h-8zm6 6h-4v-4h4v4z'
};
const icon = (d, sz) => `<svg viewBox="0 0 24 24" width="${sz || 18}" height="${sz || 18}" fill="currentColor" aria-hidden="true"><path d="${d}"/></svg>`;
const imgURL = id => location.origin + '/img/' + id;

// ---------- views: every word is a view ----------
// The word bar lists every word in use (+ 'all' and 'trash'); the view
// carousel shows every image carrying the chosen word. archived is a BIT,
// not a word: archived images are hidden from every view unless the
// 'show archived' toggle is on; 'trash' is the view of exactly those.
let curWord = localStorage.getItem('view:word') || '';
if (curWord === '*' || curWord === '#trash') curWord = '';   // retired pseudo-views
let showArchived = localStorage.getItem('view:archived') === '1';
const isArchived = id => archivedSet.has(id);
let archivedSet = new Set();
const carShow = k => localStorage.getItem('show:' + k) !== '0';
function viewIds() {
  if (!S || !curWord) return [];
  const ids = (S.all_ids || []).filter(id => (S.tags[id] || []).includes(curWord));
  return showArchived ? ids : ids.filter(id => !isArchived(id));
}
function renderWords() {
  archivedSet = new Set(S.archived || []);
  const bar = $('#wordbar');
  // chip counts match what LIGHTING the chip will show (archived per toggle)
  const counts = {};
  (S.all_ids || []).forEach(id => {
    if (!showArchived && isArchived(id)) return;
    (S.tags[id] || []).forEach(w => { counts[w] = (counts[w] || 0) + 1; });
  });
  if (curWord && !(curWord in (S.words || {}))) curWord = '';
  bar.innerHTML = '';
  Object.keys(S.words || {}).filter(w => w !== 'pinned')   // Pinned has its own carousel
    .sort((a, b) => a.localeCompare(b)).forEach(w => {
      const c = document.createElement('span');
      c.className = 'chip' + (w === curWord ? ' on' : '');
      c.innerHTML = `${w} <span class="cnt">${counts[w] || 0}</span>`;
      c.title = `filter the carousel below to images carrying "${w}" (click again to turn off)`;
      c.addEventListener('click', () => {        // slugs TOGGLE; none lit = no carousel
        curWord = curWord === w ? '' : w;
        localStorage.setItem('view:word', curWord);
        render();
      });
      bar.appendChild(c);
    });
  const tg = document.createElement('label');
  tg.className = 'chip tog' + (showArchived ? ' on' : '');
  tg.title = 'show archived images inside the views (browse the trash itself in A mode)';
  tg.innerHTML = `<input type="checkbox" ${showArchived ? 'checked' : ''}> show trashed`;
  tg.querySelector('input').addEventListener('change', e => { showArchived = e.target.checked; localStorage.setItem('view:archived', showArchived ? '1' : '0'); render(); });
  bar.appendChild(tg);
  $('#viewname').textContent = curWord || 'View';
  $('#viewhint').textContent = curWord ? `every image carrying "${curWord}", by number` : '';
}

// ---------- named strips: ONE carousel that works anywhere ----------
// Every thumbnail row is a `.car` strip registered here. listFor() feeds the
// generic focus/keyboard machinery (arrows, Space-peek, Enter), so a new
// strip only has to appear in this table to get the full behavior.
let famData = null, famKey = null; // Genealogy sheets cache (anchored to the WI)
const RW = ['working', 'ref', 'ref0', 'slot', 'pin'];   // the only WRITABLE containers
// The strip registry: name -> getter. FLAT by design - a getter carries its
// own availability check, so there is no ordering to get wrong (the old
// if-chain let a genealogy guard swallow the grid/lora lookups). Contract:
// null = "not a list" (unknown name, or its source isn't loaded); an array
// is a list even when empty.
const LISTS = {
  hist:  () => S.history,
  pin:   () => S.pins,
  all:   () => viewIds(),
  gpar:  () => famData && famData.parents.map(x => x.id),
  gsib:  () => famData && famData.siblings.map(x => x.id),
  gkid:  () => famData && famData.children.map(x => x.id),
  grid:  () => (gridName && gridName !== 'grid') ? listFor(gridName) : null,
  lora:  () => { const x = curLora(); return x ? datasetIds(x) : null; },
  // NOT mode-gated: a selection made in A keeps its identity when the pane
  // is hidden, so its aliases (dashed) still mark every visible twin in E
  abrowse: () => S ? placesIds() : null,
};
function listFor(name) {
  if (!S) return null;
  const get = LISTS[name];
  const l = get ? get() : null;
  return Array.isArray(l) ? l : null;
}
// ---- grid view: any carousel, full workspace ----
let gridOn = false, gridName = null;
function openGrid(name) {
  gridName = name;
  gridOn = true;
  $('#gridview').hidden = false;
  renderGrid();
}
let gridPeekOn = false;
function gridPeek(start) {
  // hold-Space preview: the selection at ACTUAL size (never upscaled,
  // capped to the viewport), centred over its tile when there is room and
  // pushed inside the viewport at edges/corners; live while arrows move it
  if (start === true) gridPeekOn = true;
  if (!gridPeekOn) return;
  const gp = $('#gridpeek');
  const gid = Sel.target === 'grid' ? Sel.id() : null;
  if (gid == null) { gp.hidden = true; return; }
  const im = gp.firstElementChild, url = imgURL(gid);
  if (im.src !== url) {
    gp.hidden = true;
    im.onload = gridPeekPlace;
    im.src = url;
    if (im.complete && im.naturalWidth) gridPeekPlace();
  } else gridPeekPlace();
}
function gridPeekPlace() {
  if (!gridPeekOn) return;
  const gp = $('#gridpeek'), im = gp.firstElementChild;
  if (!im.naturalWidth) return;
  const M = 8, vw = innerWidth, vh = innerHeight;
  const sc = Math.min(1, (vw - 2 * M) / im.naturalWidth, (vh - 2 * M) / im.naturalHeight);
  const w = Math.round(im.naturalWidth * sc), h = Math.round(im.naturalHeight * sc);
  const el = Sel.el();
  const r = el ? el.getBoundingClientRect() : {left: vw / 2, top: vh / 2, width: 0, height: 0};
  const x = Math.max(M, Math.min(vw - w - M, r.left + r.width / 2 - w / 2));
  const y = Math.max(M, Math.min(vh - h - M, r.top + r.height / 2 - h / 2));
  im.style.width = w + 'px'; im.style.height = h + 'px';
  Object.assign(gp.style, {left: x + 'px', top: y + 'px'});
  gp.hidden = false;
}
function gridPeekEnd() {
  gridPeekOn = false;
  const gp = $('#gridpeek');
  if (gp) gp.hidden = true;
}
function closeGrid() {
  gridPeekEnd();
  gridOn = false;
  $('#gridview').hidden = true;
  if (S) render();
}
function renderGrid() {
  if (!gridOn || !S) return;
  const ids = (listFor(gridName) || []).filter(x => x != null);
  const src = document.querySelector('#car-' + gridName + ' summary b');
  $('#gridtitle').textContent = src ? src.textContent : gridName;
  $('#gridcount').textContent = ids.length;
  $('#gridview').classList.toggle('big', localStorage.getItem('size:grid') === '1');
  $('#gridsz').textContent = localStorage.getItem('size:grid') === '1' ? '-' : '+';
  const g = $('#gridbody');
  g.innerHTML = '';
  ids.forEach((id, k) => {
    const im = thumb(id);
    im.dataset.target = 'grid';
    im.dataset.index = k;
    im.addEventListener('dblclick', () => { closeGrid(); act('place', {id, target: 'working'}); });
    g.appendChild(im);
  });
}
function setOpen(car, open) {   // programmatic toggle: not a user preference
  car._forced = open;
  car.open = open;
}
function fillStrip(name, els, empty, key) {
  const car = $('#car-' + name);
  car.querySelector('.n').textContent = els.length;
  // STRICT rule (user, 2026-08-24): zero items = cannot expand. No
  // empty-hint exception - that loophole let sheets open onto nothing.
  // Forcing shut must not clobber the sticky preference, hence setOpen().
  const openable = els.length > 0;
  car.classList.toggle('empty', !openable);
  if (!openable && car.open) setOpen(car, false);
  else if (openable && car.dataset.wasEmpty === '1' && localStorage.getItem('open:' + car.id) !== '0') setOpen(car, true);
  car.dataset.wasEmpty = openable ? '0' : '1';
  const strip = car.querySelector('.strip');
  const fullKey = (key || '') + '|' + car.open + '|' + els.length;
  if (key && strip.dataset.key === fullKey) return;   // unchanged: keep the DOM
  strip.dataset.key = fullKey;
  strip.innerHTML = '';
  if (!car.open) return;
  if (!els.length) { if (empty) strip.innerHTML = '<span class="hint">' + empty + '</span>'; return; }
  els.forEach((el, k) => { el.dataset.target = name; el.dataset.index = k; strip.appendChild(el); });
}

// ---------- rendering ----------
async function refresh() {
  try {
    S = await api('state');
    $('#evolve-dot').className = 'dot ok';
  } catch (err) {
    $('#evolve-dot').className = 'dot bad';
    clearTimeout(pollTimer);
    pollTimer = setTimeout(refresh, 3000);
    return;
  }
  render();
  clearTimeout(pollTimer);
  // 1s while generating (slots fill in as candidates land), 3s idle (cheap;
  // picks up anything another tab or the server did)
  pollTimer = setTimeout(refresh, S.busy ? 1000 : 3000);
}

function thumb(id, cls) {
  const im = document.createElement('img');
  im.src = imgURL(id); im.dataset.id = id; im.draggable = true; im.className = cls || '';
  im.loading = 'lazy';
  im.title = tip(id);
  // recompute on hover: memoized strips keep their DOM across polls, so a
  // baked title goes stale (a moved image kept showing its old path)
  im.addEventListener('mouseenter', () => { im.title = tip(id); });
  im.addEventListener('dragstart', e => dragStart(e, id));
  im.addEventListener('error', () => {   // file gone (emptied / deleted outside)
    const d = document.createElement('div');
    d.className = 'gonebox';
    d.dataset.id = id;
    d.textContent = '#' + id + String.fromCharCode(10) + 'file gone';
    d.title = '#' + id + ' — the record survives in the journal; the file was emptied or deleted';
    if (im.parentNode) im.replaceWith(d);
  }, {once: true});
  return im;
}
function tip(id) {
  const m = S.meta[id]; if (!m) return '#' + id;
  const r = m.recipe;
  return `#${id}  ${m.w}x${m.h}` + (r ? `\n${r.family || 'klein'}  seed ${r.seed}  ${r.lock || 'fiat'}  vary ${r.vary}` +
    (r.lora ? `  lora ${r.lora}@${r.lora_strength}` : '') + `\n${r.prompt}` : `\n${m.source}`) + `\n${m.path}`;
}

function render() {
  if (!S) return;
  renderCarousel('hist', S.history);
  renderCarousel('pin', S.pins);
  renderWords();
  // carousel presence: Pinned everywhere (checkbox), History E-only
  // (checkbox), the tag carousel only while a slug is lit; Family is
  // decided in renderGenealogy (E + checkbox + non-trivial)
  $('#car-pin').hidden = !carShow('pin');
  $('#car-hist').hidden = !(mode === 'evolver' && carShow('hist'));
  $('#car-all').hidden = !curWord;
  renderCarousel('all', viewIds());
  // stage
  const box = $('#stagebox'); box.innerHTML = '';
  if (S.working != null) {
    const im = thumb(S.working); box.appendChild(im);
    $('#recipe').textContent = tip(S.working).replace(/\n/g, '  ·  ');
  } else {
    box.innerHTML = '<div class="hint">working image<br>drop · paste · double-click a candidate</div>';
    $('#recipe').textContent = '';
  }
  if (peekId != null) peek(peekId);     // survive a re-render mid-hold
  renderSlots();
  // controls: never overwrite a widget you are editing (idle polls re-render)
  const c = S.controls;
  if (!$('#controls').contains(document.activeElement) && !$('#bottom').contains(document.activeElement)) {
    $('#prompt').value = c.prompt; $('#negative').value = c.negative || '';
    const fs = $('#family');
    fs.innerHTML = Object.entries(S.families).map(([k, f]) => `<option value="${k}">${f.label}</option>`).join('');
    fs.value = S.families[c.family] ? c.family : 'klein';
    $('#steps').value = c.steps || ''; $('#cfg').value = c.cfg || '';
    fillLoras(c.lora);
    $('#lstr').value = c.lora_strength; $('#lock').value = c.lock;
    $('#vary').value = c.vary; $('#varyv').textContent = c.vary;
    $('#width').value = c.width; $('#height').value = c.height;
    $('#whitebg').checked = c.whitebg;
    $('#fresh').checked = !!c.fresh_model;
    ['create', 'derive', 'camera'].forEach(t => {
      $('#outputs_' + t).value = c['outputs_' + t] || 6;
      $('#seed_' + t).value = c['seed_' + t] || 0;
    });
    camSync(c);
    if (c.tab && c.tab !== activeTab()) setTab(c.tab, false);
  }
  $('#seedused').textContent = S.last_base_seed ? `(last used ${S.last_base_seed})` : '';
  $('#rootname').textContent = S.root_name || '';
  if (S.rescan && S.rescan.ts !== lastRescanTs) {
    lastRescanTs = S.rescan.ts;
    if (S.rescan.missing || S.rescan.moved || S.rescan.imported)
      toast(`rescan: ${S.rescan.moved} moved, ${S.rescan.imported} imported, ${S.rescan.missing} missing`,
            S.rescan.missing ? 'warn' : '');
  }
  const dtg = $('#deftags');
  if (document.activeElement !== dtg) dtg.value = ((S.settings && S.settings.default_tags) || []).join(', ');
  familyUI();
  document.querySelectorAll('.ref[data-target="ref"]').forEach((r, i) => {
    r.innerHTML = '';
    const id = c.refs[i];
    if (id != null) r.appendChild(thumb(id)); else r.innerHTML = '<span class="hint">+</span>';
  });
  const r0 = document.querySelector('.ref[data-target="ref0"]');
  r0.innerHTML = '';
  if (c.ref0 != null) r0.appendChild(thumb(c.ref0)); else r0.innerHTML = '<span class="hint">ref0</span>';
  $('#status').textContent = S.busy ? 'per-step progress is on the evolve.py console' : '';   // #msg is separate: never clobbered by polls
  genUI();
  statusBar();
  renderGenealogy();
  renderLoras();
  renderPlaces();
  trainUI();
  renderGrid();
  Sel.apply();
}

// ---------- A mode: the Places tree + browser ----------
// One image, one place. The tree is real directories under the root; the
// browser shows the open folder; drops onto tree nodes MOVE (place is
// exclusive - unlike tag/dataset drops, which only add a word).
// A anywhere = reveal the selected image: open its folder in the manager,
// scrolled into view. The SELECTION STAYS WHERE IT WAS (its solid blue
// border in E survives the round trip); the revealed tile shows as an
// ALIAS (dashed) because it displays the same underlying image - reveal
// never steals the selection (user rule). Inert when already in A; with
// no selection it just opens the manager where it was.
async function revealInPlaces() {
  if (mode === 'assets') return;
  const id = lastSelId;
  const dir = (id != null && S) ? S.paths[id] : null;
  if (dir && dir !== S.cwd) await api('cwd', {dir});
  setMode('assets');
  await refresh();
  if (id == null) return;
  const t = document.querySelector('#abgrid [data-id="' + id + '"]');
  if (t) t.scrollIntoView({block: 'nearest'});
}
function placesIds() {
  const cwd = S.cwd;
  return (S.all_ids || []).filter(id => S.paths[id] === cwd);
}
let placesKey = '';
function renderPlaces() {
  if (mode !== 'assets' || !S) return;
  const counts = {};
  Object.values(S.paths || {}).forEach(d => { counts[d] = (counts[d] || 0) + 1; });
  const key = JSON.stringify([S.cwd, S.dirs, counts, placesIds(), S.missing]);
  if (key === placesKey) { return; }
  placesKey = key;
  const tb = $('#treebody');
  tb.innerHTML = '';
  (S.dirs || []).forEach(d => {
    const n = document.createElement('div');
    const depth = d === '.trash' ? 0 : d.split('/').length - 1;
    n.className = 'dnode' + (d === S.cwd ? ' on' : '') + (d === '.trash' ? ' trash' : '');
    n.style.paddingLeft = (8 + depth * 14) + 'px';
    n.innerHTML = `<span>${d === '.trash' ? 'trash' : d.split('/').pop()}</span><span class="cnt">${counts[d] || 0}</span>`;
    n.title = d;
    n.addEventListener('click', () => act('cwd', {dir: d}));
    n.addEventListener('dragover', e => { e.preventDefault(); e.stopPropagation(); n.classList.add('over'); });
    n.addEventListener('dragleave', () => n.classList.remove('over'));
    n.addEventListener('drop', async e => {
      e.preventDefault(); e.stopPropagation(); n.classList.remove('over');
      const own = e.dataTransfer.getData('application/x-evolver');
      if (own) { const r = await api('move', {ids: [+own], to: d}); if (r.error) notice(r.error); else flash(`#${own} → ${d}`); refresh(); }
    });
    tb.appendChild(n);
  });
  $('#abtitle').textContent = S.cwd === '.trash' ? 'trash' : S.cwd;
  const ids = placesIds();
  $('#abcount').textContent = ids.length;
  const g = $('#abgrid');
  if (g.contains(document.activeElement)) return;
  g.innerHTML = '';
  ids.forEach((id, k) => {
    if ((S.missing || []).includes(id)) {
      const d = document.createElement('div');
      d.className = 'gonebox';
      d.textContent = '#' + id + ' missing' + String.fromCharCode(10) + '(file not found - rescan when it returns)';
      g.appendChild(d);
      return;
    }
    const im = thumb(id);
    im.dataset.target = 'abrowse';
    im.dataset.index = k;
    im.addEventListener('dblclick', () => act('place', {id, target: 'working'}));
    g.appendChild(im);
  });
}
$('#mkdir').addEventListener('click', () => {
  const name = prompt('new folder under "' + (S ? S.cwd : '') + '" (name only, no slashes):');
  if (!name || !S) return;
  const d = (S.cwd === '.trash' ? 'images' : S.cwd) + '/' + name.trim();
  api('mkdir', {dir: d}).then(r => { if (r.error) notice(r.error); else act('cwd', {dir: r.dir}); });
});
$('#rescan').addEventListener('click', async () => {
  flash('rescanning…');
  const r = await api('rescan', {});
  if (r.error) { notice(r.error); return; }
  toast(`rescan: ${r.moved} moved, ${r.imported} imported, ${r.missing} missing` +
    (r.skipped.length ? `, ${r.skipped.length} skipped` : ''));
  refresh();
});
{ // tree | browser divider in A mode
  const box = $('#page-assets'), bar = $('#asplit'), KEY = 'split:assets';
  const apply = px => { box.style.gridTemplateColumns = `${px}px 6px 1fr`; };
  const saved = +localStorage.getItem(KEY);
  if (saved) requestAnimationFrame(() => apply(saved));
  bar.addEventListener('pointerdown', e => {
    e.preventDefault(); bar.setPointerCapture(e.pointerId); bar.classList.add('drag');
    const move = ev => apply(Math.max(150, Math.min(box.clientWidth - 300, Math.round(ev.clientX - box.getBoundingClientRect().left))));
    const up = () => { bar.classList.remove('drag'); bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', up);
      const w = parseInt(box.style.gridTemplateColumns, 10); if (w) localStorage.setItem(KEY, w); };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => { localStorage.removeItem(KEY); box.style.gridTemplateColumns = ''; });
}

// ---------- LoRA editor: images grouped by a LoRA's dataset word, with
// descriptions; remove the word, train the LoRA ----------
// ---- the task rail: modes (Blender-style: one key each) ----
const MODES = ['evolver', 'loras', 'assets', 'story'];
const MODE_KEYS = {e: 'evolver', l: 'loras', a: 'assets', s: 'story'};
let mode = localStorage.getItem('mode') || 'evolver';
let lastTask = null;          // the last non-A mode of THIS session (Tab's return target)
let curLoraName = localStorage.getItem('lora:last') || null;
function setMode(m) {
  const next = MODES.includes(m) ? m : 'evolver';
  if (mode !== 'assets' && next !== mode) lastTask = mode;
  mode = next;
  localStorage.setItem('mode', mode);
  document.querySelectorAll('#rail .tab[data-mode]').forEach(t =>
    t.classList.toggle('on', t.dataset.mode === mode));
  ['#stage', '#split', '#genpanel'].forEach(sel => { $(sel).style.display = mode === 'evolver' ? '' : 'none'; });
  $('#loras').hidden = mode !== 'loras';
  $('#page-assets').hidden = mode !== 'assets';
  placesKey = '';                          // re-render the pane on entry
  $('#page-story').hidden = mode !== 'story';
  document.querySelectorAll('#carshow .eonly').forEach(l =>
    l.style.display = mode === 'evolver' ? '' : 'none');
  if (S) render();
}
document.querySelectorAll('#rail .tab[data-mode]').forEach(t =>
  t.addEventListener('click', () => setMode(t.dataset.mode)));

function curLora() {
  return (S.loras || []).find(x => x.name === curLoraName) || (S.loras || [])[0] || null;
}
const dsTag = a => 'lora_dataset_' + a.name;
// the dataset = every unarchived image carrying the LoRA's word
function datasetIds(a) {
  const w = dsTag(a);
  return (S.all_ids || []).filter(id => (S.tags[id] || []).includes(w) && !isArchived(id));
}
function renderLoras() {
  if (mode !== 'loras' || !S) return;
  const sel = $('#lorasel');
  const a = curLora();
  curLoraName = a ? a.name : null;
  if (document.activeElement !== sel) {
    sel.innerHTML = (S.loras || []).map(x => `<option>${x.name}</option>`).join('');
    if (a) sel.value = a.name;
  }
  // never rebuild the grid under an in-progress caption edit
  if ($('#lgrid').contains(document.activeElement)) return;
  const g = $('#lgrid');
  const lkey = a ? JSON.stringify([a.name, datasetIds(a), (S.descriptions || {})]) : 'none';
  if (g.dataset.key === lkey) return;
  g.dataset.key = lkey;
  g.innerHTML = '';
  if (!a) { g.innerHTML = '<span class="hint">no LoRAs yet — “+ new” creates one</span>'; return; }
  datasetIds(a).forEach((id, k) => {
    const t = document.createElement('div');
    t.className = 'atile';
    t.dataset.target = 'lora';
    t.dataset.index = k;
    t.tabIndex = 0;
    const im = thumb(id);
    im.addEventListener('dblclick', () => act('place', {id, target: 'working'}));
    const x = document.createElement('button');
    x.className = 'ax'; x.innerHTML = icon(I.close, 13);
    x.title = 'remove the dataset word from this image (the image itself is untouched)';
    x.addEventListener('click', () => act('tag', {ids: [id], remove: [dsTag(a)]}));
    const d = document.createElement('textarea');
    d.className = 'adesc';
    d.value = (S.descriptions || {})[id] || '';
    d.placeholder = 'description of the image ("' + a.name + ', " is prefixed at training time)';
    const mark = () => d.classList.toggle('bad', d.value.trim().startsWith(a.name));
    mark();
    d.title = 'a plain description of the image; red = it already starts with the trigger (double prefix)';
    d.addEventListener('input', mark);
    d.addEventListener('change', () => api('describe', {id, description: d.value}).then(refresh));
    t.append(im, x, d);
    g.appendChild(t);
  });
}
function renderLoraFiles() {
  const box = $('#lorafiles');
  const x = curLora();
  if (!x) { box.innerHTML = ''; return; }
  const newest = {};
  x.files.forEach(f => { newest[f.family] = f.path; });   // last per family wins
  box.innerHTML = x.files.length ? '<b>trained files</b>' : '<span class="hint">no trained files yet</span>';
  [...x.files].reverse().forEach(f => {
    const d = document.createElement('div');
    d.className = 'lf' + (newest[f.family] === f.path ? ' cur' : '');
    d.innerHTML = `<span class="fam">${f.family}</span><span>${f.path.split('/').pop()}</span>`;
    d.title = f.path + (newest[f.family] === f.path ? '  (what the dropdown resolves to)' : '  (older version)');
    box.appendChild(d);
  });
}
function trainUI() {
  if (mode === 'loras' && S) renderLoraFiles();
  const tf = $('#trainfam');
  if (S && !tf.dataset.built) {   // trainable families come from the server
    tf.dataset.built = '1';
    tf.innerHTML = Object.entries(S.families).filter(([, f]) => f.trainable)
      .map(([k, f]) => `<option value="${k}">${f.label}</option>`).join('');
  }
  const t = S && S.train;
  const running = !!(t && t.running);
  $('#maketrain').disabled = running || !!(S && S.busy);
  $('#trainstop').hidden = !running;
  $('#trainstat').textContent = !t ? '' :
    t.running ? `training ${t.name} (${t.family}) · ${Math.floor(t.elapsed / 60)}m${t.elapsed % 60}s · tail -f ${t.log}` :
    t.error ? `${t.name}: ${t.error}` : `${t.name}: done ✓`;
}
$('#maketrain').addEventListener('click', async () => {
  const a = curLora();
  if (!a) { flash('create a LoRA first'); return; }
  const fam = $('#trainfam').value;
  if (!confirm(`Train "${a.name}" (${fam}) on ${datasetIds(a).length} image(s)?` +
      String.fromCharCode(10) + 'This runs on the GPU and blocks generation until done.')) return;
  const r = await api('train', {name: a.name, family: fam, steps: +$('#trainsteps').value || undefined});
  if (r.error) alert(r.error);
  refresh();
});
$('#trainstop').addEventListener('click', async () => {
  if (!confirm('Abort training? Progress in this run is lost.')) return;
  await api('train_abort', {});
  refresh();
});
$('#lorasel').addEventListener('change', () => {
  curLoraName = $('#lorasel').value;
  localStorage.setItem('lora:last', curLoraName);
  renderLoras();
});
$('#lorafolder').addEventListener('click', async () => {
  const a = curLora();
  if (!a) { flash('create a LoRA first'); return; }
  const path = prompt('folder to import into "' + a.name + '" (recursive):');
  if (!path) return;
  flash('importing folder…');
  const r = await api('import_folder', {path: path.trim(), tags: [dsTag(a)]});
  if (r.error) { alert(r.error); return; }
  flash(`${r.added} added, ${r.duplicates} duplicates, ${r.skipped} skipped (of ${r.total})`);
  refresh();
});
$('#loranew').addEventListener('click', () => {
  const name = prompt('new LoRA name (it is the trigger word — letters, digits, - _):');
  if (!name) return;
  api('lora', {op: 'create', name: name.trim()}).then(r => {
    if (r.error) { flash(r.error); return; }
    curLoraName = name.trim();
    localStorage.setItem('lora:last', curLoraName);
    refresh();
  });
});
$('#loradel').addEventListener('click', () => {
  const a = curLora();
  if (!a) return;
  if (!confirm(`forget LoRA "${a.name}"? Its files stay on disk; images and their words are untouched.`)) return;
  act('lora', {op: 'delete', name: a.name});
});
async function walkEntry(en) {
  // recursive directory walk of a dropped Explorer folder. Entries must be
  // grabbed synchronously at drop time (they go inert after an await).
  if (en.isFile) {
    return new Promise(res => en.file(f => res(f.type.startsWith('image/') ? [f] : []),
                                      () => res([])));
  }
  if (en.isDirectory) {
    const rd = en.createReader();
    let out = [], batch;
    do {
      batch = await new Promise(res => rd.readEntries(res, () => res([])));
      for (const c2 of batch) out = out.concat(await walkEntry(c2));
    } while (batch.length);
    return out;
  }
  return [];
}
async function loraAddId(a, id) {
  // put the LoRA's dataset word on the image (its description was seeded
  // from the recipe prompt at birth/import; edit it on the tile)
  await api('tag', {ids: [id], add: [dsTag(a)]});
}
async function loraDrop(dt) {
  const a = curLora();
  if (!a) { flash('create a LoRA first'); return; }
  const entries = [...(dt.items || [])]
    .map(it => it.webkitGetAsEntry && it.webkitGetAsEntry())
    .filter(Boolean);
  if (entries.some(en => en.isDirectory)) {
    flash('reading folder…');
    let files = [];
    for (const en of entries) files = files.concat(await walkEntry(en));
    let added = 0;
    for (const f of files) {
      const r = await fetch('/api/import', {method: 'POST', body: f}).then(x => x.json());
      if (r.error) continue;
      await loraAddId(a, r.id);
      added++;
    }
    flash(`${added} of ${files.length} images added from folder`);
    refresh();
    return;
  }
  const own = dt.getData('application/x-evolver');
  if (own) { await loraAddId(a, +own); refresh(); return; }
  const uri = (dt.getData('text/uri-list') || '').split(String.fromCharCode(10)).map(x => x.trim()).find(x => x && !x.startsWith('#'));
  if (uri && uri.startsWith(location.origin + '/img/')) {
    await loraAddId(a, +uri.split('/').pop());
    refresh(); return;
  }
  const files = [...(dt.files || [])].filter(f => f.type.startsWith('image/'));
  for (const f of files) {   // external file: import, then the word
    const r = await fetch('/api/import', {method: 'POST', body: f}).then(x => x.json());
    if (r.error) { flash(r.error); continue; }
    await loraAddId(a, r.id);
  }
  if (files.length) refresh();
  else if (!own && !uri) flash('drop an image (or drag one from any strip)');
}

// ---------- dialogs (the prune box doubles as a generic confirm) ----------
let dlgGo = null;
function openDialog(title, body, goLabel, onGo) {
  dlgGo = onGo;
  $('#pforce').parentElement.style.display = 'none';
  $('#prunedlg .ptitle').textContent = title;
  $('#prunedlg .pbody').textContent = body;
  $('#pgo').textContent = goLabel;
  $('#pgo').disabled = false;
  $('#prunedlg').hidden = false;
}
let pruneId = null, prunePlan = null;
function pruneText(p) {
  const n = p.archive.length;
  let t = `Branch under #${p.root}: ${p.branch} image${p.branch === 1 ? '' : 's'} (mother-line).` + String.fromCharCode(10);
  t += `Archive ${n}.`;
  if (p.keep.length) {
    t += ` Keep ${p.keep.length}:` + String.fromCharCode(10) +
      p.keep.map(k => `   #${k.id} — ${k.why}`).join(String.fromCharCode(10));
  }
  if (p.unpin.length) t += String.fromCharCode(10) + `Unpin: ${p.unpin.map(i => '#' + i).join(', ')}.`;
  if (p.dataset_removals.length) {
    const by = {};
    p.dataset_removals.forEach(r => { by[r.lora] = (by[r.lora] || 0) + 1; });
    t += String.fromCharCode(10) + 'Leaves LoRA datasets: ' +
      Object.entries(by).map(([a, c]) => `${a} ×${c}`).join(', ') + '.';
  }
  const live = [];
  if (p.live.working) live.push('the working image');
  if (p.live.ref0) live.push('ref0');
  if (p.live.refs) live.push(`${p.live.refs} reference slot(s)`);
  if (live.length) t += String.fromCharCode(10) + 'Clears: ' + live.join(', ') + '.';
  if (p.outside_refs) t += String.fromCharCode(10) + `Also referenced by ${p.outside_refs} image(s) outside the branch (not touched).`;
  if (!n) t += String.fromCharCode(10) + 'Nothing to archive under this plan.';
  return t;
}
async function openPrune(id, plan) {
  pruneId = id; prunePlan = plan; dlgGo = null;
  $('#pforce').parentElement.style.display = '';
  $('#pgo').textContent = 'Prune';
  $('#pforce').checked = false;
  $('#prunedlg .ptitle').textContent = `Prune #${id} and its branch`;
  $('#prunedlg .pbody').textContent = pruneText(plan);
  $('#pgo').disabled = !plan.archive.length;
  $('#prunedlg').hidden = false;
}
function closePrune() { $('#prunedlg').hidden = true; pruneId = null; }
$('#pforce').addEventListener('change', async () => {
  const plan = await api('prune', {id: pruneId, force: $('#pforce').checked});
  if (plan.error) { notice(plan.error); return; }
  prunePlan = plan;
  $('#prunedlg .pbody').textContent = pruneText(plan);
  $('#pgo').disabled = !plan.archive.length;
});
$('#pcancel').addEventListener('click', closePrune);
$('#pgo').addEventListener('click', async () => {
  if (dlgGo) { const go = dlgGo; dlgGo = null; closePrune(); await go(); return; }
  const r = await api('prune', {id: pruneId, force: $('#pforce').checked, apply: true});
  closePrune();
  if (r.error) notice(r.error);
  else flash(`pruned: ${r.archive.length} archived`);
  refresh();
});

// ---------- the tabbed action panel: Create | Derive | Camera | Tween ----------
// One Generate button for all tabs (fixed, bottom-right), dispatching on the
// active tab. The tab IS the mode: Create = fiat (no refs, model dropdown
// live), Derive = Klein refs+IFT, Camera = absolute re-shoot of the WI.
function activeTab() { const b = document.querySelector('.tabb.on'); return b ? b.dataset.tab : 'create'; }
function setTab(t, user) {
  document.querySelectorAll('.tabb').forEach(b => b.classList.toggle('on', b.dataset.tab === t));
  document.querySelectorAll('.tpage').forEach(pg => { pg.style.display = pg.dataset.tab === t ? 'block' : 'none'; });
  $('#shared').style.display = (t === 'create' || t === 'derive') ? '' : 'none';
  if (S) {
    familyUI(); genUI(); renderSlots();   // the grid shows THIS tab's outputs
    if (user) {   // the Output grid previews the ACTIVE tab's outputs count
      const o = $('#outputs_' + t);
      if (o) act('slots', {slots: +o.value || 1});
      saveControls();
      // USER-switch to Derive with an empty ref0: seed it from the WI
      // (w/h follow, like the Space-click gesture). Occupied ref0 is never
      // touched - a pick's restored parent outranks the WI. Del + return
      // refills; changing the WI while ON the tab does not.
      if (t === 'derive' && S.controls.ref0 == null && S.working != null) {
        api('place', {id: S.working, target: 'ref0'}).then(() => {
          const m = S.meta[S.working];
          if (m) { $('#width').value = m.w; $('#height').value = m.h; saveControls(); }
          refresh();
        });
      }
    }
  }
}
document.querySelectorAll('.tabb').forEach(b => b.addEventListener('click', () => setTab(b.dataset.tab, true)));

// ---- camera axes: checkbox reveals the control; unchecked = token omitted ----
function camVal(axis) {
  if (!S || !$('#chk_' + axis).checked) return null;
  if (axis === 'azim') return dialSel;
  const list = axis === 'dist' ? S.pov_dist : S.pov_elev;
  const k = +$('#cam_' + axis).value;
  return (list[k] || list[0])[0];
}
function setAxis(axis, list, key, dflt) {
  const on = key != null;
  $('#chk_' + axis).checked = on;
  $('#body_' + axis).parentElement.classList.toggle('on', on);
  const k = list.findIndex(x => x[0] === key);
  if (k >= 0) $('#cam_' + axis).value = k;
  else if (!on) $('#cam_' + axis).value = dflt;
}
function stopsFill(axis, list) {
  const box = $('#stops_' + axis);
  if (box.dataset.built !== '1') {
    box.dataset.built = '1';
    box.innerHTML = list.map((x, i) => `<span data-i="${i}">${x[1]}</span>`).join('');
    box.addEventListener('click', e => {
      const sp = e.target.closest('span'); if (!sp) return;
      $('#cam_' + axis).value = sp.dataset.i;
      $('#chk_' + axis).checked = true;
      $('#body_' + axis).parentElement.classList.add('on');
      camPaint(); saveControls(); genUI();
    });
  }
  [...box.children].forEach((sp, i) => sp.classList.toggle('on', +$('#cam_' + axis).value === i && $('#chk_' + axis).checked));
}
function camPaint() {
  if (!S) return;
  stopsFill('dist', S.pov_dist);
  stopsFill('elev', S.pov_elev);
  dialSet(dialSel);
}
function camSync(c) {
  if (!S) return;
  setAxis('dist', S.pov_dist, c.pov_dist, 1);
  setAxis('elev', S.pov_elev, c.pov_elev, 1);
  $('#chk_azim').checked = c.pov_azim != null;
  $('#body_azim').parentElement.classList.toggle('on', c.pov_azim != null);
  if (c.pov_azim != null) dialSel = c.pov_azim;
  camPaint();
}
['dist', 'elev', 'azim'].forEach(a => $('#chk_' + a).addEventListener('change', () => {
  $('#body_' + a).parentElement.classList.toggle('on', $('#chk_' + a).checked);
  camPaint(); saveControls(); genUI();
}));
['dist', 'elev'].forEach(a => $('#cam_' + a).addEventListener('input', () => { camPaint(); saveControls(); }));

// ---- the Orbit compass dial: front at 12 o'clock, clockwise ----
const AZ_ORDER = ['front', 'front-right', 'right', 'back-right', 'back', 'back-left', 'left', 'front-left'];
const AZ_SHORT = {front: 'F', 'front-right': 'FR', right: 'R', 'back-right': 'BR',
                  back: 'B', 'back-left': 'BL', left: 'L', 'front-left': 'FL'};
let dialSel = 'front';
function dialBuild() {
  const svg = $('#dial'), C = 66, R = 44, LR = 58;
  let h = `<circle class="ring" cx="${C}" cy="${C}" r="${R}"></circle>` +
          `<line class="needle" x1="${C}" y1="${C}" x2="${C}" y2="${C - R + 10}"></line>`;
  AZ_ORDER.forEach((az, k) => {
    const a = k * Math.PI / 4;
    h += `<circle class="pt" data-az="${az}" cx="${C + R * Math.sin(a)}" cy="${C - R * Math.cos(a)}" r="7"><title>${az} view</title></circle>`;
    h += `<text x="${C + LR * Math.sin(a)}" y="${C - LR * Math.cos(a)}">${AZ_SHORT[az]}</text>`;
  });
  svg.innerHTML = h;
  svg.addEventListener('click', e => {
    const pt = e.target.closest('.pt'); if (!pt) return;
    $('#chk_azim').checked = true;
    $('#body_azim').parentElement.classList.add('on');
    dialSet(pt.dataset.az); saveControls(); genUI();
  });
}
function dialSet(az) {
  dialSel = az;
  const k = AZ_ORDER.indexOf(az), a = k * Math.PI / 4, C = 66, R = 44;
  const n = document.querySelector('#dial .needle');
  if (n) { n.setAttribute('x2', C + (R - 10) * Math.sin(a)); n.setAttribute('y2', C - (R - 10) * Math.cos(a)); }
  document.querySelectorAll('#dial .pt').forEach(c2 => c2.classList.toggle('on', c2.dataset.az === az));
  $('#dialval').textContent = $('#chk_azim').checked ? az + ' view' : '';
}
dialBuild();

// ---- one button, four meanings ----
function genUI() {
  if (!S) return;
  const t = activeTab(), b = $('#gen');
  $('#controls').classList.toggle('busy', !!S.busy);
  if (S.train && S.train.running) {
    b.disabled = true;
    b.classList.remove('stop');
    b.textContent = 'Training…';
    b.title = 'a LoRA is training - one GPU';
    return;
  }
  if (S.busy) {   // the button MORPHS into the abort control
    b.disabled = false;
    b.classList.add('stop');
    b.textContent = `Stop ${S.busy.done}/${S.busy.total}`;
    b.title = 'abort this round: finished candidates stay, the in-flight render is interrupted';
    return;
  }
  b.classList.remove('stop');
  let on = true, label = 'Generate', tip = '';
  if (t === 'derive' && S.controls.ref0 == null) { on = false; tip = 'set ref0 — Derive breeds FROM an image'; }
  else if (t === 'camera') {
    label = 'Re-shoot';
    if (S.working == null) { on = false; tip = 'no working image to re-shoot'; }
    else if (!(camVal('azim') || camVal('elev') || camVal('dist'))) { on = false; tip = 'check at least one axis'; }
  } else if (t === 'tween') { on = false; tip = 'coming soon'; }
  b.disabled = !on;
  b.textContent = label;
  b.title = tip;
}

// ---------- Genealogy sheets: parents / siblings / children of the WI ----------
// v2 (2026-08-23): flat read-only carousels anchored to the Working Image.
// No walking, no decks - dbl-click an ancestor to make IT the WI instead.
async function renderGenealogy() {
  const g = $('#genea');
  if (mode !== 'evolver' || !carShow('fam') || !S || S.working == null) {
    g.hidden = true; famData = null; famKey = null;
    return;
  }
  const key = S.working + ':' + JSON.stringify(S.candidates);   // children grow as a round lands
  if (key !== famKey) {
    const r = await api('family', {id: S.working});
    if (r.error) { g.hidden = true; return; }
    famData = r; famKey = key;
  }
  // auto-hide a trivial family: no parents, no children, no siblings but itself
  g.hidden = !famData.parents.filter(t => !t.gone).length && !famData.children.length
    && famData.siblings.length <= 1 && !famData.parents.length;
  if (g.hidden) return;
  const tile = t => {
    if (t.gone) {
      const d = document.createElement('div');
      d.className = 'gonebox car-gone';
      d.textContent = '#' + t.id + String.fromCharCode(10) + 'gone';
      d.title = '#' + t.id + ' — ancestor whose file is gone (journal remembers its recipe)';
      return d;
    }
    const im = thumb(t.id);
    im.addEventListener('dblclick', () => act('place', {id: t.id, target: 'working'}));
    return im;
  };
  fillStrip('gpar', famData.parents.map(tile), 'fiat — no reference images', 'gp:' + famKey);
  fillStrip('gsib', famData.siblings.map(tile), null, 'gs:' + famKey);
  fillStrip('gkid', famData.children.map(tile), 'no children yet', 'gk:' + famKey);
}
{ // the Genealogy section itself: sticky open/close like any carousel
  const gn = $('#genea'), gk = 'open:genea';
  const saved = localStorage.getItem(gk);
  if (saved != null) gn.open = saved === '1';
  gn.addEventListener('toggle', () => { localStorage.setItem(gk, gn.open ? '1' : '0'); render(); });
}

// ---------- Output slots (the Candidates grid, inside the Generate panel) ----------
function renderSlots() {
  const cand = S.candidates[activeTab()] || [];
  const pending = (S.busy && S.busy.tab === activeTab()) ? S.busy.total - S.busy.done : 0;
  $('#output .n').textContent = S.slots;
  const sheet = $('#sheet');
  const skey = activeTab() + ':' + cand.join(',') + ':' + S.slots + ':' + pending;
  if (sheet.dataset.key === skey) { layoutSlots(); return; }
  sheet.dataset.key = skey;
  sheet.innerHTML = '';
  for (let k = 0; k < S.slots; k++) {
    const id = cand[k];
    const d = document.createElement('div');
    d.className = 'slot drop'; d.dataset.target = 'slot'; d.dataset.index = k; d.tabIndex = 0;
    if (id != null) {
      d.appendChild(thumb(id));
      d.addEventListener('dblclick', () => act('place', {id, target: 'working'}));
    } else if (k < cand.length + pending) {
      d.innerHTML = '<div class="hint busy">generating…</div>';
    } else {
      d.innerHTML = '<div class="hint">empty</div>';
    }
    sheet.appendChild(d);
  }
  layoutSlots();
}

// ---------- Status + Session bar ----------
function statusBar() {
  $('#comfy-dot').className = 'dot ' + (S.comfy_ok ? 'ok' : 'bad');
  const p = $('#prog'), t = $('#progtxt');
  if (S.busy) {
    p.style.visibility = 'visible';
    p.max = S.busy.total; p.value = S.busy.done;
    t.textContent = S.busy.done + '/' + S.busy.total;
  } else { p.style.visibility = 'hidden'; t.textContent = ''; }
}

{ // controls|output divider inside the Generate panel (height-adjustable controls)
  const gp = $('#genpanel'), bar = $('#hsplit'), KEY = 'split:gen';
  const apply = px => { gp.style.gridTemplateRows = px + 'px 6px minmax(80px,1fr)'; if (S) layoutSlots(); };
  const saved = +localStorage.getItem(KEY);
  if (saved) requestAnimationFrame(() => apply(saved));
  bar.addEventListener('pointerdown', e => {
    e.preventDefault(); bar.setPointerCapture(e.pointerId); bar.classList.add('drag');
    const move = ev => apply(Math.max(110, Math.min(gp.clientHeight - 90, Math.round(ev.clientY - gp.getBoundingClientRect().top))));
    const up = () => { bar.classList.remove('drag'); bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', up); const h = parseInt(gp.style.gridTemplateRows, 10); if (h) localStorage.setItem(KEY, h); };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => { localStorage.removeItem(KEY); gp.style.gridTemplateRows = ''; if (S) layoutSlots(); });
}

document.addEventListener('keydown', e => {
  if (typing(e)) return;
  if (e.key === 'Escape') {
    if (!$('#prunedlg').hidden) { closePrune(); return; }
    if (infoHide()) return;
    if (gridOn) { closeGrid(); return; }
    Sel.clear();
  }
});

let sheetCols = 1;
// the two top carousels behave identically: click selects (blue), dbl-click /
// Enter picks, Space peeks, arrows move, thumbs drag, the end buttons scroll while held
function renderCarousel(name, ids) {
  fillStrip(name, ids.map(id => {
    const im = thumb(id);
    im.addEventListener('dblclick', () => act('place', {id, target: 'working'}));
    return im;
  }), null, name + ':' + ids.join(','));
  const strip = $('#car-' + name + ' .strip');
  if (name === 'hist' && S.working !== lastCurWorking) {
    lastCurWorking = S.working;
    const cur = strip.querySelector('[data-id="' + S.working + '"]');
    if (cur && !fullyVisible(cur, strip)) cur.scrollIntoView({inline: 'center', block: 'nearest'});
  }
}
let lastCurWorking = null;
function fullyVisible(el, box) {
  const a = el.getBoundingClientRect(), b = box.getBoundingClientRect();
  return a.left >= b.left && a.right <= b.right;
}
function layoutSlots() {
  // one square cell size chosen so all N slots fit the Output pane
  const grid = $('#sheet');
  const W = grid.clientWidth - 20, H = grid.clientHeight - 4;
  let best = {cols: 1, size: 0};
  for (let cols = 1; cols <= S.slots; cols++) {
    const rows = Math.ceil(S.slots / cols);
    const size = Math.min((W - 8 * (cols - 1)) / cols, (H - 8 * (rows - 1)) / rows);
    if (size > best.size) best = {cols, size};
  }
  sheetCols = best.cols;
  const size = Math.max(64, Math.floor(best.size));
  grid.style.gridTemplateColumns = 'repeat(' + best.cols + ', ' + size + 'px)';
  grid.style.gridAutoRows = size + 'px';
}
window.addEventListener('resize', () => S && render());

// ---------- controls ----------
function readControls() {
  const t = activeTab();
  const sEl = $('#seed_' + t);
  return {prompt: $('#prompt').value, negative: $('#negative').value, family: $('#family').value,
    steps: +$('#steps').value || 0, cfg: +$('#cfg').value || 0,
    ref0: S ? S.controls.ref0 : null,
    refs: S ? S.controls.refs : [null, null, null],
    lora: $('#lora').value, lora_strength: +$('#lstr').value, lock: $('#lock').value,
    vary: +$('#vary').value, whitebg: $('#whitebg').checked,
    width: +$('#width').value || 1024, height: +$('#height').value || 1024,
    fresh_model: $('#fresh').checked,
    tab: t,
    seed: sEl ? (+sEl.value || 0) : 0,           // the ACTIVE tab's seed
    seed_create: +$('#seed_create').value || 0,
    seed_derive: +$('#seed_derive').value || 0,
    seed_camera: +$('#seed_camera').value || 0,
    outputs_create: +$('#outputs_create').value || 6,
    outputs_derive: +$('#outputs_derive').value || 6,
    outputs_camera: +$('#outputs_camera').value || 4,
    pov_azim: camVal('azim'), pov_elev: camVal('elev'), pov_dist: camVal('dist')};
}

// ---- draggable divider between the preview image and the board/sheet ----
// The grid is `<preview> 6px <sheet>`; dragging rewrites the first track and
// localStorage keeps it. layoutSlots() sizes cells from clientWidth, so the
// sheet has to be re-laid out as the drag moves, not just at the end.
(function () {
  const work = $('#work'), bar = $('#split');
  const KEY = 'split:work';
  const apply = px => {
    work.style.gridTemplateColumns = `${px}px 6px minmax(220px,1fr)`;
    if (S) layoutSlots();
  };
  const saved = +localStorage.getItem(KEY);
  if (saved) requestAnimationFrame(() => apply(saved));
  bar.addEventListener('pointerdown', e => {
    e.preventDefault();
    bar.setPointerCapture(e.pointerId);
    bar.classList.add('drag');
    const move = ev => {
      const px = Math.round(ev.clientX - work.getBoundingClientRect().left);
      apply(Math.max(220, Math.min(work.clientWidth - 226, px)));
    };
    const up = () => {
      bar.classList.remove('drag');
      bar.removeEventListener('pointermove', move);
      bar.removeEventListener('pointerup', up);
      const w = parseInt(work.style.gridTemplateColumns, 10);
      if (w) localStorage.setItem(KEY, w);
    };
    bar.addEventListener('pointermove', move);
    bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => { localStorage.removeItem(KEY); work.style.gridTemplateColumns = ''; if (S) layoutSlots(); });
})();
{ // Dataset | Train divider in L mode: dragging sets the Train pane width
  const box = $('#loras'), bar = $('#lsplit'), KEY = 'split:lora';
  const apply = px => { box.style.gridTemplateColumns = `minmax(260px,1fr) 6px ${px}px`; };
  const saved = +localStorage.getItem(KEY);
  if (saved) requestAnimationFrame(() => apply(saved));
  bar.addEventListener('pointerdown', e => {
    e.preventDefault(); bar.setPointerCapture(e.pointerId); bar.classList.add('drag');
    const move = ev => apply(Math.max(230, Math.min(box.clientWidth - 280, Math.round(box.getBoundingClientRect().right - ev.clientX))));
    const up = () => { bar.classList.remove('drag'); bar.removeEventListener('pointermove', move); bar.removeEventListener('pointerup', up);
      const w = parseInt(box.style.gridTemplateColumns.split(' ').pop(), 10); if (w) localStorage.setItem(KEY, w); };
    bar.addEventListener('pointermove', move); bar.addEventListener('pointerup', up);
  });
  bar.addEventListener('dblclick', () => { localStorage.removeItem(KEY); box.style.gridTemplateColumns = ''; });
}
let saveT = null;
function saveControls() { clearTimeout(saveT); saveT = setTimeout(() => api('controls', readControls()), 400); }
['#prompt', '#negative', '#family', '#steps', '#cfg', '#lora', '#lstr', '#lock', '#vary', '#width', '#height', '#whitebg', '#fresh',
 '#seed_create', '#seed_derive', '#seed_camera', '#outputs_create', '#outputs_derive', '#outputs_camera']
  .forEach(s => $(s).addEventListener('input', () => { $('#varyv').textContent = $('#vary').value; saveControls(); }));
// dbl-click the Working Image = copy it to ref0 (works from any tab;
// the same w/h sync as the Space-click gesture)
$('#stagebox').addEventListener('dblclick', async () => {
  if (!S || S.working == null) return;
  await api('place', {id: S.working, target: 'ref0'});
  const m = S.meta[S.working];
  if (m) { $('#width').value = m.w; $('#height').value = m.h; saveControls(); }
  flash(`ref0 ← #${S.working}`);
  refresh();
});

// ONE rule: click on ANY image while Space is held = copy it to ref0
// (shortcut for dragging it into the ref0 slot). Plain clicks only select.
document.addEventListener('click', async e => {
  if (!spaceHeld || !S || !e.target.closest('[data-target]')) return;
  const id = Sel.id();          // mousedown just focused the clicked box
  if (id == null) return;
  await api('place', {id, target: 'ref0'});
  const m = S.meta[id];
  if (m) { $('#width').value = m.w; $('#height').value = m.h; saveControls(); }
  flash(`ref0 ← #${id}`);
  refresh();
});

// hidden setting (settings page later): selecting an image anywhere scrolls
// History to it, if it is there. localStorage sync_history_to_selection=0 off.
let lastHistSync = null;
function syncHistory() {
  // selecting an image outside History mirrors the blue border onto its
  // History twin; the strip scrolls (centered) ONLY when the selection just
  // changed AND the twin is not fully visible - manual scrolls stay put
  if (!S || Sel.target === 'hist') { lastHistSync = Sel.id(); return; }
  const id = Sel.id();
  const k = id == null ? -1 : S.history.indexOf(id);
  if (k < 0) { lastHistSync = id; return; }
  const im = document.querySelector(`img[data-target="hist"][data-index="${k}"]`);
  if (!im) { lastHistSync = id; return; }
  const changed = id !== lastHistSync;
  lastHistSync = id;
  if (localStorage.getItem('sync_history_to_selection') === '0') return;
  if (changed && !fullyVisible(im, im.closest('.strip')))
    im.scrollIntoView({inline: 'center', block: 'nearest'});
}
// family is a fiat-only choice: live while ref0 + ref slots are empty
// (references exist only in the Klein graph; the stage may hold anything)
// the model family in force: Klein on Derive, the dropdown on Create
function activeFamily() {
  const fam = activeTab() === 'derive' ? 'klein' : ($('#family').value || 'klein');
  return S && S.families[fam] ? fam : 'klein';
}
// LoRAs are specific to their model: the dropdown only ever lists the
// active family's (S.loras is {family: [entries]})
function fillLoras(want) {
  const lsel = $('#lora');
  const opts = (S.lora_menu && S.lora_menu[activeFamily()]) || [];
  const cur = want !== undefined ? want : lsel.value;
  lsel.innerHTML = '<option value="">(none)</option>' + opts.map(l => `<option>${l}</option>`).join('');
  lsel.value = opts.includes(cur) ? cur : '';
}
function familyUI() {
  // v3: the TAB is the fiat gate - the dropdown is always live on Create,
  // and Derive is Klein by construction (no dropdown there at all)
  if (!S) return;
  const fam = activeFamily();
  const f = S.families[fam] || S.families.klein;
  if (!$('#controls').contains(document.activeElement)) fillLoras();
  $('#steps').placeholder = f.steps; $('#cfg').placeholder = f.cfg;
  // Klein ignores its -ive prompt: only models that read it show the box
  $('#negative').style.display = (activeTab() === 'create' && fam !== 'klein') ? '' : 'none';
  // white bg is a Flux-only mechanism (prompt prefix): off-screen otherwise.
  // The eventual solution is the user-editable boilerplate-vars table.
  $('#whitebg').parentElement.style.display = fam === 'klein' ? '' : 'none';
  const noLora = !f.lora;                    // the family says whether it takes one
  $('#lora').parentElement.classList.toggle('off', noLora);
  $('#lstr').parentElement.classList.toggle('off', noLora);
}
$('#family').addEventListener('change', () => { $('#steps').value = ''; $('#cfg').value = ''; fillLoras(''); familyUI(); saveControls(); });
['create', 'derive', 'camera'].forEach(t => $('#outputs_' + t).addEventListener('change', () => {
  if (activeTab() === t) act('slots', {slots: +$('#outputs_' + t).value || 1});
}));
$('#gen').addEventListener('click', async () => {
  if (S && S.busy) {          // morphed: Stop
    await api('abort', {});
    refresh();
    return;
  }
  const t = activeTab();
  let r;
  if (t === 'camera') {
    r = await api('pov', {azim: camVal('azim'), elev: camVal('elev'), dist: camVal('dist'),
      outputs: +$('#outputs_camera').value || 4, seed: +$('#seed_camera').value || 0});
  } else {
    r = await api('generate', {...readControls(), op: t,
      outputs: +$('#outputs_' + t).value || 6});
  }
  if (r.error) alert(r.error);
  refresh();
});
[['show_pin', 'pin'], ['show_hist', 'hist'], ['show_fam', 'fam']].forEach(([id, k]) => {
  const cb = $('#' + id);
  cb.checked = carShow(k);
  cb.addEventListener('change', () => { localStorage.setItem('show:' + k, cb.checked ? '1' : '0'); if (S) render(); });
});
$('#deftags').addEventListener('change', () => {
  api('settings', {default_tags: $('#deftags').value.split(',')}).then(refresh);
});
$('#gc').addEventListener('click', async () => {
  const p = await api('empty_trash', {});
  if (p.error) { notice(p.error); return; }
  if (!p.count) { flash('the trash is empty'); return; }
  const mb = (p.bytes / 1048576).toFixed(1);
  let body = `${p.count} file(s), ${mb} MB \u2192 Windows Recycle Bin.`;
  if (p.load_bearing) {
    body += String.fromCharCode(10) + String.fromCharCode(10) +
      `${p.load_bearing} of them are ancestors of ${p.orphaned} live image(s)` +
      ` (e.g. ${p.sample_orphaned.map(i => '#' + i).join(', ')}).` + String.fromCharCode(10) +
      'Those images will show missing-parent placeholders and their recipes can no longer be re-rendered exactly.';
  }
  body += String.fromCharCode(10) + String.fromCharCode(10) +
    'These images leave evolve for good (no placeholders, no tracking). ' +
    'The files go to the Windows Recycle Bin, not oblivion - recover the bytes from there if you change your mind.';
  openDialog('Empty trash', body, 'Empty trash', async () => {
    const r = await api('empty_trash', {apply: true});
    if (r.error) notice(r.error); else flash(`${r.removed} file(s) \u2192 Recycle Bin`);
    refresh();
  });
});
// ---------- the carousel component ----------
// Every .car gets: sticky open/close, single-image scroll steps that
// accelerate to fast scroll after [GS carousel_step] steps (0 = fast
// immediately), and a +/- toggling between exactly two thumb sizes.
// New carousels inherit ALL of it from the markup pattern alone.
const GS = {carousel_step: +(localStorage.getItem('GS:carousel_step') ?? 3)};
document.querySelectorAll('.car .arrow').forEach(btn => {
  const strip = btn.parentElement.querySelector('.strip'), dir = +btn.dataset.dir;
  let t = null, n = 0;
  const w = () => { const im = strip.querySelector('[data-id]'); return (im ? im.getBoundingClientRect().width : 64) + 6; };
  const step = () => strip.scrollBy({left: dir * w(), behavior: 'smooth'});
  const fast = () => { clearInterval(t); t = setInterval(() => { strip.scrollLeft += dir * 14; }, 16); };
  btn.addEventListener('mousedown', () => {
    clearInterval(t); n = 0;
    if (!GS.carousel_step) { fast(); return; }
    step(); n = 1;
    t = setInterval(() => { if (n++ < GS.carousel_step) step(); else fast(); }, 300);
  });
  ['mouseup', 'mouseleave'].forEach(ev => btn.addEventListener(ev, () => clearInterval(t)));
});
document.querySelectorAll('details.car').forEach(c => {
  const k = 'open:' + c.id;
  const saved = localStorage.getItem(k);
  if (saved != null) c.open = saved === '1';
  c.addEventListener('toggle', () => {
    if (c._forced !== undefined && c.open === c._forced) { c._forced = undefined; return; }
    localStorage.setItem(k, c.open ? '1' : '0');
    render();
  });
  const sm = c.querySelector('summary');
  const b = document.createElement('button');
  b.className = 'sz'; b.title = 'toggle thumbnail size';
  const kk = 'size:' + c.id;
  const setSz = () => { const big = localStorage.getItem(kk) === '1'; c.classList.toggle('big', big); b.textContent = big ? '-' : '+'; };
  b.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    localStorage.setItem(kk, localStorage.getItem(kk) === '1' ? '0' : '1');
    setSz(); render();
  });
  const gb = document.createElement('button');
  gb.className = 'sz';
  gb.title = 'show as a full-workspace grid';
  gb.innerHTML = icon(I.grid, 13);
  gb.addEventListener('click', e => {
    e.preventDefault(); e.stopPropagation();
    openGrid(c.id.replace('car-', ''));
  });
  // PROXIMITY: the controls sit right after the title+count cluster, never
  // floated across an ultrawide row (association problem, user-reported)
  const nEl = sm.querySelector('.n');
  if (nEl) nEl.after(b, gb); else sm.append(b, gb);
  setSz();
});
$('#gridclose').innerHTML = icon(I.close, 20);
$('#gridclose').addEventListener('click', closeGrid);
$('#gridsz').addEventListener('click', () => {
  localStorage.setItem('size:grid', localStorage.getItem('size:grid') === '1' ? '0' : '1');
  renderGrid();
});
let navBusy = false;
function prefetchNeighbors(ids) {
  // warm the browser cache for likely scrub targets (immutable /img)
  (ids || []).forEach(id => { if (id != null) { const im = new Image(); im.src = imgURL(id); } });
}
async function navWI(d) {
  // browser back/forward for the WI (its own stack in state.json, pushed by
  // every pick, never by scrubbing). History (the strip) stays the sparse
  // bred-from work log - this is the browse log. One request in flight -
  // key autorepeat must not queue a flood.
  if (navBusy) return;
  navBusy = true;
  try {
    const r = await api('winav', {dir: d});
    if (r.id == null) flash(d < 0 ? 'start of WI history' : 'end of WI history');
    await refresh();
    prefetchNeighbors(r.around);
  } finally { navBusy = false; }
}

// ---------- focus + keyboard ----------
let spaceHeld = false;
function slotPinned(t) {   // board cells never take a replacing drop/paste
  return S && t.target === 'pin' && t.index < S.pins.length;
}
document.addEventListener('mousedown', e => {
  const d = e.target.closest('[data-target]'); if (!d) return;
  Sel.set(d.dataset.target, d.dataset.index);
});
const typing = e => ['INPUT', 'TEXTAREA', 'SELECT'].includes(e.target.tagName);
document.addEventListener('keydown', async e => {
  if (typing(e)) return;
  const id = Sel.id();
  if (gridOn) {
    if (e.key === ' ') {
      e.preventDefault();
      if (!e.repeat) gridPeek(true);          // hold = full-size preview
      return;
    }
    if (!e.key.startsWith('Arrow')) return;   // modal: arrows browse, Esc closes
    queueMicrotask(() => gridPeek());         // preview follows the selection while held
  }
  if (e.key === ' ') {                       // hold = the stage shows the selection; on the
    e.preventDefault();                      // preview image itself: shows its PARENT (provenance)
    if (!e.repeat) { spaceHeld = true; Sel.apply(); }
  } else if (e.key === 'Delete' || e.key === 'Backspace') {
    if (e.shiftKey && id != null) {   // PRUNE: a collectable leaf just goes; anything with impact asks
      e.preventDefault();
      const plan = await api('prune', {id, force: false});
      if (plan.error) { notice(plan.error); return; }
      if (plan.branch === 1 && plan.archive.length === 1) {
        await api('prune', {id, force: false, apply: true});
        refresh();
      } else openPrune(id, plan);
      return;
    }
    if (Sel.target === 'lora') {    // remove the dataset WORD; the image is untouched
      const a = curLora();
      if (a && id != null) { e.preventDefault(); act('tag', {ids: [id], remove: [dsTag(a)]}); }
      return;
    }
    if (id != null && RW.includes(Sel.target)) {   // r/o sheets (history, genealogy) can't be edited
      e.preventDefault();
      await api('clear', {target: Sel.target, index: Sel.index}); refresh();
    }
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'c') {
    if (id != null) { e.preventDefault(); copyImage(id); }
  } else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'x') {
    if (id != null) { e.preventDefault(); await copyImage(id); if (RW.includes(Sel.target)) { await api('clear', {target: Sel.target, index: Sel.index}); refresh(); } }
  } else if (e.key === 'Enter') {
    if (id != null && Sel.target !== 'working') {
      act('place', {id, target: 'working'});
    }
  } else if (e.key.startsWith('Arrow')) {
    const d = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -sheetCols, ArrowDown: sheetCols}[e.key];
    if (Sel.target === 'slot') { e.preventDefault(); Sel.set('slot', Math.max(0, Math.min(S.slots - 1, Sel.index + d))); }
    else if (Sel.target === 'pin') { e.preventDefault(); Sel.set('pin', Math.max(0, Math.min(S.pins.length - 1, Sel.index + Math.sign(d)))); }
    else if (Sel.target === 'working' && Math.abs(d) === 1) { e.preventDefault(); navWI(d); }
    else if (Sel.target === 'ref' && Math.abs(d) === 1) { Sel.set('ref', Math.max(0, Math.min(2, Sel.index + d))); }
    else if (Sel.target === 'grid' && listFor('grid')) {   // a GRID: up/down move by a row
      e.preventDefault();
      const cols = getComputedStyle($('#gridbody')).gridTemplateColumns.split(' ').length || 1;
      const step = {ArrowLeft: -1, ArrowRight: 1, ArrowUp: -cols, ArrowDown: cols}[e.key];
      Sel.set('grid', Math.max(0, Math.min(listFor('grid').length - 1, Sel.index + step)));
    }
    else if (listFor(Sel.target) && Math.abs(d) === 1) { e.preventDefault(); Sel.set(Sel.target, Math.max(0, Math.min(listFor(Sel.target).length - 1, Sel.index + d))); }
  } else if (e.key === 'p' && id != null) {
    act('pin', {id, on: true});   // P pins the selection from ANYWHERE; Del inside Pinned unpins
  } else if (e.key === 'Tab' && !e.ctrlKey && !e.altKey && !gridOn) {
    // Tab = strict A <-> task toggle (in a form field it keeps its native
    // focus-move meaning via the typing() guard above): in a task -> the
    // FULL A action (switch AND reveal the selection's folder - Tab is a
    // synonym, not a lesser A); in A -> the last task of this session, or
    // nothing if there is none yet
    e.preventDefault();
    if (mode === 'assets') { if (lastTask) setMode(lastTask); }
    else revealInPlaces();
  } else if (!e.ctrlKey && !e.metaKey && !e.altKey && !gridOn && MODE_KEYS[e.key.toLowerCase()]) {
    const m = MODE_KEYS[e.key.toLowerCase()];  // Blender-style: E/L/A/S switch modes;
    if (m === 'assets') revealInPlaces();      // A = REVEAL the selection in its folder
    else setMode(m);
  }
});

document.addEventListener('keyup', e => { if (e.key === ' ') { spaceHeld = false; peek(null); gridPeekEnd(); } });
window.addEventListener('blur', () => { spaceHeld = false; peek(null); gridPeekEnd(); });
let peekId = null;
function peek(id) {
  peekId = id;
  let ov = $('#peek');
  if (!ov) { ov = document.createElement('img'); ov.id = 'peek'; $('#stagebox').appendChild(ov); }
  if (id == null) { ov.style.display = 'none'; ov.removeAttribute('src'); return; }
  ov.src = imgURL(id); ov.style.display = 'block';
}

// ---------- Info Window: right-click any Image ----------
const SVG_COPY = icon(I.copy, 16);
const SVG_TICK = icon(I.done, 16);
let infoEl = null, infoX = 0, infoY = 0;
function infoHide() { if (infoEl && !infoEl.hidden) { infoEl.hidden = true; return true; } return false; }
async function showInfo(id, x, y) {
  infoX = x; infoY = y;
  const q = {id};
  const r0 = await api('meta', q);         // always: the gc verdict is computed live
  if (r0.error) { notice(r0.error); return; }
  const m = r0;
  if (!infoEl) {
    infoEl = document.createElement('div');
    infoEl.id = 'infowin';
    document.body.appendChild(infoEl);
    document.addEventListener('mousedown', e => { if (!infoEl.hidden && !infoEl.contains(e.target)) infoHide(); });
  }
  infoEl.innerHTML = '';
  const path = m.path || '';
  const cut = Math.max(path.lastIndexOf('/'), path.lastIndexOf(String.fromCharCode(92)));
  const dir = cut < 0 ? '' : path.slice(0, cut + 1);
  const rows = [['file', path],
    ['size', (m.w && m.h) ? m.w + ' × ' + m.h : null, true],   // true = no copy button
    ['created', m.ts],
    ['refs', (m.parents || []).map(q => dir + q + '.png').join(String.fromCharCode(10)) || null],
    ['prompt', m.recipe && m.recipe.prompt]];
  for (const [k, v, nocopy] of rows) {
    if (v == null || v === '') continue;
    const row = document.createElement('div'); row.className = 'iw-row';
    const kk = document.createElement('span'); kk.className = 'iw-k'; kk.textContent = k;
    const vv = document.createElement('span'); vv.className = 'iw-v'; vv.textContent = v;
    if (nocopy) { row.append(kk, vv); infoEl.appendChild(row); continue; }
    const c = document.createElement('button'); c.className = 'iw-c'; c.innerHTML = SVG_COPY; c.title = 'copy to clipboard';
    c.addEventListener('click', () => {
      navigator.clipboard.writeText(String(v));
      c.innerHTML = SVG_TICK; c.title = 'Copied!'; c.classList.add('done');
      setTimeout(() => { c.innerHTML = SVG_COPY; c.title = 'copy to clipboard'; c.classList.remove('done'); }, 1200);
    });
    row.append(kk, vv, c);
    infoEl.appendChild(row);
  }
  infoEl.appendChild(tagEditor(m));
  infoEl.appendChild(archivedEditor(m));
  infoEl.appendChild(descEditor(m));
  infoEl.hidden = false;
  const r = infoEl.getBoundingClientRect();
  infoEl.style.left = Math.min(x, innerWidth - r.width - 12) + 'px';
  infoEl.style.top = Math.min(y, innerHeight - r.height - 12) + 'px';
}
// ---- tag editor (in the Info Window): words, add/remove, cascade ----
// No word is protected. "apply to descendants" (default on) cascades along
// parent-0 edges; ADD skips archived descendants, REMOVE does not (server
// rule). archived is a bit with its own checkbox, not a word.
let cascadeOn = localStorage.getItem('tag:cascade') !== '0';
function tagEditor(m) {
  const row = document.createElement('div'); row.className = 'iw-row';
  const kk = document.createElement('span'); kk.className = 'iw-k'; kk.textContent = 'tags';
  const box = document.createElement('div'); box.className = 'iw-tags';
  const apply = (add, remove) => api('tag', {ids: [m.id], add, remove, cascade: cascadeOn})
    .then(r => {
      if (r.error) { notice(r.error); return refresh(); }
      flash(`${add.length ? '+' + add.join(',') : ''}${remove.length ? ' -' + remove.join(',') : ''} on ${r.touched.length} image(s)`);
      return refresh();
    })
    .then(() => showInfo(m.id, infoX, infoY));
  (m.tags || []).forEach(w => {
    const c = document.createElement('span'); c.className = 'chip';
    c.innerHTML = `${w}<span class="x" title="remove this word">×</span>`;
    c.querySelector('.x').addEventListener('click', () => apply([], [w]));
    box.appendChild(c);
  });
  const inp = document.createElement('input'); inp.className = 'iw-tagin'; inp.placeholder = 'add word…';
  inp.setAttribute('list', 'wordlist');
  let dl = $('#wordlist');
  if (!dl) { dl = document.createElement('datalist'); dl.id = 'wordlist'; document.body.appendChild(dl); }
  dl.innerHTML = Object.keys(S.words || {}).sort().map(w => `<option value="${w}">`).join('');
  const submit = () => { const v = inp.value; inp.value = ''; if (v.trim()) apply(v.split(','), []); };
  inp.addEventListener('keydown', e => { e.stopPropagation(); if (e.key === 'Enter') submit(); });
  inp.addEventListener('change', submit);
  const casc = document.createElement('label'); casc.className = 'iw-casc';
  casc.innerHTML = `<input type="checkbox" ${cascadeOn ? 'checked' : ''}> apply to descendants`;
  casc.querySelector('input').addEventListener('change', e => { cascadeOn = e.target.checked; localStorage.setItem('tag:cascade', cascadeOn ? '1' : '0'); });
  box.append(inp, casc);
  row.append(kk, box);
  return row;
}
function archivedEditor(m) {
  const row = document.createElement('div'); row.className = 'iw-row';
  const kk = document.createElement('span'); kk.className = 'iw-k'; kk.textContent = 'state';
  const lab = document.createElement('label'); lab.className = 'iw-casc';
  const pinned = (m.tags || []).includes('pinned');
  lab.innerHTML = `<input type="checkbox" ${m.archived ? 'checked' : ''}> archived` +
    (pinned && !m.archived ? ' <span class="hint">(pinned — archiving unpins)</span>' : '');
  lab.querySelector('input').addEventListener('change', async e => {
    const r = await api('archive', {id: m.id, on: e.target.checked, force: true});
    if (r.error) notice(r.error);
    await refresh();
    showInfo(m.id, infoX, infoY);
  });
  const gc = document.createElement('span'); gc.className = 'hint'; gc.style.marginLeft = '10px'; gc.textContent = m.gc || '';
  row.append(kk, lab, gc);
  return row;
}
function descEditor(m) {
  const row = document.createElement('div'); row.className = 'iw-row';
  const kk = document.createElement('span'); kk.className = 'iw-k'; kk.textContent = 'descr.';
  const ta = document.createElement('textarea'); ta.className = 'iw-desc';
  ta.value = m.description || '';
  ta.placeholder = 'a plain description of the image (LoRA triggers are prefixed at training time)';
  ta.addEventListener('keydown', e => e.stopPropagation());
  ta.addEventListener('change', () => api('describe', {id: m.id, description: ta.value}).then(refresh));
  row.append(kk, ta);
  return row;
}
document.addEventListener('contextmenu', e => {
  const d = e.target.closest('[data-id]');
  if (d) { e.preventDefault(); showInfo(+d.dataset.id, e.clientX, e.clientY); }
});

// ---------- clipboard ----------
async function copyImage(id) {
  // PNG bitmap for Paint/Photoshop/browsers + the file path as text for editors
  try {
    const blob = await (await fetch(imgURL(id))).blob();
    const item = {'image/png': blob};
    try { item['text/plain'] = new Blob([S.meta[id].path], {type: 'text/plain'}); } catch (_) {}
    await navigator.clipboard.write([new ClipboardItem(item)]);
    flash(`copied #${id} to clipboard`);
  } catch (err) {
    // some browsers refuse multi-type items: fall back to image only
    try {
      const blob = await (await fetch(imgURL(id))).blob();
      await navigator.clipboard.write([new ClipboardItem({'image/png': blob})]);
      flash(`copied #${id} (image only)`);
    } catch (e2) { flash('copy failed: ' + e2); }
  }
}
document.addEventListener('paste', async e => {
  if (typing(e) && e.target.id !== 'prompt') return;
  const items = [...(e.clipboardData?.items || [])];
  const img = items.find(it => it.type.startsWith('image/'));
  if (img && (Sel.target === 'lgrid' || Sel.target === 'lora')) {
    e.preventDefault();
    const a = curLora();
    if (!a) { flash('create a LoRA first'); return; }
    const r = await fetch('/api/import', {method: 'POST', body: img.getAsFile()}).then(x => x.json());
    if (r.error) { flash(r.error); return; }
    await loraAddId(a, r.id);
    flash(`pasted into "${a.name}"`);
    refresh();
    return;
  }
  if (img) { e.preventDefault(); await importBlob(img.getAsFile(), slotPinned(Sel.get()) ? {target: 'pin', index: 999} : selTarget()); return; }
  const txt = e.clipboardData?.getData('text') || '';
  if (/^https?:\/\/\S+$/.test(txt.trim()) && !typing(e)) { e.preventDefault(); await importURL(txt.trim(), selTarget()); }
});
function flash(msg) {
  const el = $('#msg');
  el.classList.remove('notice');
  el.textContent = msg;
  setTimeout(() => { if (el.textContent === msg && !el.classList.contains('notice')) el.textContent = ''; }, 3000);
}
function toast(msg, kind) {
  // dismissable, stays ~8s: long enough to digest a multi-part message
  const t = document.createElement('div');
  t.className = 'toast' + (kind ? ' ' + kind : '');
  t.textContent = msg;
  t.addEventListener('click', () => t.remove());
  $('#toasts').appendChild(t);
  setTimeout(() => t.remove(), 8000);
}
function notice(msg) {
  // a REFUSAL deserves to be seen: toast + the bar flashes
  toast(msg, 'warn');
  const bar = $('#bottom');
  bar.classList.remove('alert'); void bar.offsetWidth;   // restart the animation
  bar.classList.add('alert');
}

// ---------- drag & drop ----------
function dragStart(e, id) {
  const url = imgURL(id);
  e.dataTransfer.effectAllowed = 'all';
  e.dataTransfer.setData('application/x-evolver', String(id));
  e.dataTransfer.setData('text/uri-list', url);
  e.dataTransfer.setData('text/plain', S.meta[id] ? S.meta[id].path : url);
  e.dataTransfer.setData('DownloadURL', `image/png:${id}.png:${url}`);   // Chromium: drag out as a file
}
// never show the no-entry cursor, never let a missed drop navigate the page
['dragenter', 'dragover'].forEach(t => document.addEventListener(t, e => {
  e.preventDefault();
  let d = e.target.closest('.drop');
  if (d && d.dataset.target === 'pin' && +d.dataset.index < 900) d = d.closest('#car-pin');
  e.dataTransfer.dropEffect = d ? 'copy' : 'none';
  document.querySelectorAll('.drop.over').forEach(x => x !== d && x.classList.remove('over'));
  if (d) d.classList.add('over');
}));
document.addEventListener('dragleave', e => { if (!e.relatedTarget) document.querySelectorAll('.drop.over').forEach(x => x.classList.remove('over')); });
document.addEventListener('drop', async e => {
  e.preventDefault();
  document.querySelectorAll('.drop.over').forEach(x => x.classList.remove('over'));
  if (e.target.closest('#loras')) { await loraDrop(e.dataTransfer); return; }
  let d = e.target.closest('.drop'); if (!d) return;
  if (d.dataset.target === 'pin' && +d.dataset.index < 900) d = d.closest('#car-pin');
  const target = {target: d.dataset.target, index: +(d.dataset.index || 0)};
  if (target.target !== 'pin') Sel.set(target.target, target.index);
  const dt = e.dataTransfer;
  const own = dt.getData('application/x-evolver');
  if (own) { const r = await api('place', {id: +own, target: target.target, index: target.index}); if (r.error) flash(r.error); refresh(); return; }
  const files = [...(dt.files || [])].filter(f => f.type.startsWith('image/') || /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
  if (files.length) {
    // first file to the drop target; extra files flow into following slots
    await importBlob(files[0], target);
    for (let k = 1; k < files.length && target.target === 'slot'; k++) await importBlob(files[k], {target: 'slot', index: target.index + k});
    return;
  }
  const uri = (dt.getData('text/uri-list') || dt.getData('text/plain') || '').split('\n').map(s => s.trim()).find(s => s && !s.startsWith('#'));
  if (uri) {
    if (uri.startsWith('data:')) { await importBlob(await (await fetch(uri)).blob(), target); return; }
    const html = dt.getData('text/html');
    const m = html && html.match(/<img[^>]+src="([^"]+)"/i);
    await importURL(m ? m[1].replace(/&amp;/g, '&') : uri, target);
  }
});
async function importBlob(blob, target) {
  const r = await fetch('/api/import', {method: 'POST', body: blob}).then(r => r.json());
  if (r.error) { flash(r.error); return; }
  await api('place', {id: r.id, target: target.target, index: target.index});
  refresh();
}
async function importURL(url, target) {
  if (url.startsWith('data:')) return importBlob(await (await fetch(url)).blob(), target);
  if (url.startsWith(location.origin + '/img/')) return api('place', {id: +url.split('/').pop(), ...target}).then(refresh);
  flash('fetching ' + url.slice(0, 60) + '…');
  const r = await api('import_url', {url});
  if (r.error) { flash(r.error); return; }
  await api('place', {id: r.id, target: target.target, index: target.index});
  refresh();
}

setMode(mode);
refresh();
