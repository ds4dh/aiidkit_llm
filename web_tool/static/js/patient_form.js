/**
 * AIIDKIT — patient_form.js
 * Handles the structured event entry form and CSV/JSON file upload.
 * Exposes window.AIIDKIT_FORM
 */
'use strict';

window.AIIDKIT_FORM = (() => {

  let _vocabulary = {};
  let _uploadedEvents = [];
  let _rowCount = 0;

  // Example patient configurations are now loaded dynamically from the backend CSV files.

  // ------------------------------------------------------------------
  // INIT
  // ------------------------------------------------------------------

  function initForm(vocabulary) {
    _vocabulary = vocabulary || {};

    // Tab switching
    document.getElementById('tab-btn-form')?.addEventListener('click',   () => switchTab('form'));
    document.getElementById('tab-btn-upload')?.addEventListener('click', () => switchTab('upload'));

    // Add-event button
    document.getElementById('add-event-btn')?.addEventListener('click', () => {
      addEventRow();
      scrollToBottom();
    });

    // File upload
    initFileUpload();

    // Pre-load one starter row
    addEventRow({ entity: 'Medication', attribute: 'Tacrolimus', value: 'High', days_since_tpx: 7 });
  }

  function switchTab(name) {
    ['form', 'upload'].forEach(t => {
      const isActive = t === name;
      document.getElementById(`tab-btn-${t}`)?.classList.toggle('active', isActive);
      document.getElementById(`tab-btn-${t}`)?.setAttribute('aria-selected', String(isActive));
      const panel = document.getElementById(`tab-${t}`);
      if (panel) {
        panel.classList.toggle('active',  isActive);
        panel.classList.toggle('hidden', !isActive);
      }
    });
  }

  // ------------------------------------------------------------------
  // EVENT ROWS
  // ------------------------------------------------------------------

  function addEventRow(defaults = {}) {
    const tbody = document.getElementById('patient-events-tbody');
    if (!tbody) return;

    _rowCount++;
    const id = `r${_rowCount}`;
    const entities = Object.keys(_vocabulary);

    const defaultEntity = defaults.entity || entities[0] || '';
    const defaultAttrs  = defaultEntity ? Object.keys(_vocabulary[defaultEntity] || {}) : [];
    const defaultAttr   = defaults.attribute || defaultAttrs[0] || '';
    const defaultVals   = (defaultEntity && defaultAttr)
      ? (_vocabulary[defaultEntity]?.[defaultAttr] || [])
      : [];

    // --- build selects ---
    const entitySel = _makeSelect(`ent-${id}`, entities, defaultEntity);
    const attrSel   = _makeSelect(`att-${id}`, defaultAttrs, defaultAttr);
    const valSel    = _makeSelect(`val-${id}`, defaultVals,  defaults.value);

    entitySel.addEventListener('change', () => _onEntityChange(entitySel, attrSel, valSel));
    attrSel.addEventListener('change',   () => _onAttrChange(entitySel, attrSel, valSel));

    // --- day input ---
    const dayInput = document.createElement('input');
    dayInput.type = 'number'; dayInput.min = '-9999'; dayInput.max = '9999';
    dayInput.id = `day-${id}`;
    dayInput.value = defaults.days_since_tpx ?? '';
    dayInput.placeholder = '—';

    // --- remove button ---
    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'remove-row-btn';
    removeBtn.title = 'Remove this event';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', () => tr.remove());

    // --- assemble row ---
    const tr = document.createElement('tr');
    [entitySel, attrSel, valSel, dayInput, removeBtn].forEach(el => {
      const td = document.createElement('td'); td.appendChild(el); tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }

  function _onEntityChange(entitySel, attrSel, valSel) {
    const entity  = entitySel.value;
    const attribs = entity ? Object.keys(_vocabulary[entity] || {}) : [];
    _populateSelect(attrSel, attribs);
    const vals = (entity && attrSel.value)
      ? (_vocabulary[entity]?.[attrSel.value] || []) : [];
    _populateSelect(valSel, vals);
  }

  function _onAttrChange(entitySel, attrSel, valSel) {
    const entity = entitySel.value;
    const attr   = attrSel.value;
    const vals   = (entity && attr) ? (_vocabulary[entity]?.[attr] || []) : [];
    _populateSelect(valSel, vals);
  }

  function _makeSelect(id, options, selectedValue) {
    const sel = document.createElement('select');
    sel.id = id;
    _populateSelect(sel, options, selectedValue);
    return sel;
  }

  function _populateSelect(sel, options, selectedValue) {
    const prev = sel.value;
    sel.innerHTML = '';
    options.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt; o.textContent = opt;
      if (opt === (selectedValue ?? prev)) o.selected = true;
      sel.appendChild(o);
    });
    if (!selectedValue && options.includes(prev)) sel.value = prev;
  }

  // ------------------------------------------------------------------
  // COLLECT FORM EVENTS
  // ------------------------------------------------------------------

  function getFormEvents() {
    const tbody = document.getElementById('patient-events-tbody');
    if (!tbody) return [];
    const events = [];
    tbody.querySelectorAll('tr').forEach(tr => {
      const entity  = tr.querySelector('[id^="ent-"]')?.value?.trim();
      const attr    = tr.querySelector('[id^="att-"]')?.value?.trim();
      const value   = tr.querySelector('[id^="val-"]')?.value?.trim();
      const dayStr  = tr.querySelector('[id^="day-"]')?.value;
      const day     = parseFloat(dayStr);
      if (entity && attr && value && !isNaN(day)) {
        events.push({ entity, attribute: attr, value, days_since_tpx: day });
      }
    });
    return events;
  }

  // ------------------------------------------------------------------
  // FILE UPLOAD
  // ------------------------------------------------------------------

  function initFileUpload() {
    const dropZone  = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');

    if (dropZone) {
      dropZone.addEventListener('dragover', e => {
        e.preventDefault(); dropZone.classList.add('drag-over');
      });
      dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
      dropZone.addEventListener('drop', e => {
        e.preventDefault(); dropZone.classList.remove('drag-over');
        const file = e.dataTransfer?.files[0];
        if (file) _handleFile(file);
      });
    }

    fileInput?.addEventListener('change', e => {
      const file = e.target.files?.[0];
      if (file) _handleFile(file);
    });
  }

  function _handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv', 'json'].includes(ext)) {
      alert('Unsupported file format. Please upload a .csv or .json file.');
      return;
    }
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const text = e.target?.result || '';
        const events = ext === 'json' ? parseJSON(text) : parseCSV(text);
        if (!events.length) {
          alert('No valid events found in the uploaded file.');
          return;
        }

        // Reset results first
        if (window.AIIDKIT && typeof window.AIIDKIT.clearPredictionResults === 'function') {
          window.AIIDKIT.clearPredictionResults();
        }

        // Set raw events in main orchestrator state
        if (window.AIIDKIT && typeof window.AIIDKIT.setCurrentPatientRawEvents === 'function') {
          window.AIIDKIT.setCurrentPatientRawEvents(events);
        }

        // Filter events to FUP boundary
        const fupSel = document.getElementById('fup-select');
        const selectedFup = fupSel ? parseInt(fupSel.value, 10) : 90;
        const filteredEvents = events.filter(ev => ev.days_since_tpx <= selectedFup);

        // Load events into manual entry form
        const tbody = document.getElementById('patient-events-tbody');
        if (tbody) {
          tbody.innerHTML = '';
          _rowCount = 0;
          filteredEvents.forEach(ev => addEventRow(ev));
        }

        // Switch active tab to Manual Entry form
        switchTab('form');
        scrollToBottom();

        // Clear file input so it can be uploaded again
        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';

        // Show toast
        if (window.AIIDKIT && typeof window.AIIDKIT.showToast === 'function') {
          window.AIIDKIT.showToast(`${filteredEvents.length} events loaded into manual entry form (filtered to ${selectedFup}-day follow-up).`, 'success');
        }
      } catch (err) {
        alert(`Could not parse file: ${err.message}`);
      }
    };
    reader.readAsText(file);
  }

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) return [];
    const first = lines[0].toLowerCase();
    const hasHeader = first.includes('entity') || first.includes('attribute');
    const data = hasHeader ? lines.slice(1) : lines;
    const events = [];
    for (const line of data) {
      const parts = line.split(',').map(p => p.trim().replace(/^["']|["']$/g, ''));
      if (parts.length < 4) continue;
      const [entity, attribute, value, daysStr] = parts;
      const days = parseFloat(daysStr);
      if (entity && attribute && value && !isNaN(days)) {
        events.push({ entity, attribute, value, days_since_tpx: days });
      }
    }
    return events;
  }

  function parseJSON(text) {
    const data = JSON.parse(text);
    const arr  = Array.isArray(data) ? data : (data.events ?? []);
    return arr
      .filter(ev => ev && ev.entity && ev.attribute && ev.value != null && ev.days_since_tpx != null)
      .map(ev => ({
        entity:         String(ev.entity),
        attribute:      String(ev.attribute),
        value:          String(ev.value),
        days_since_tpx: parseFloat(ev.days_since_tpx),
      }));
  }

  // ------------------------------------------------------------------
  // EXAMPLE PATIENTS
  // ------------------------------------------------------------------

  async function loadExample(type) {
    if (window.AIIDKIT && typeof window.AIIDKIT.showLoading === 'function') {
      window.AIIDKIT.showLoading('Loading example patient data…');
    }
    try {
      const res = await fetch(`/api/examples/${type}`);
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const j = await res.json(); msg = j.error || msg; } catch (_) {}
        throw new Error(msg);
      }
      const events = await res.json();
      
      // Set raw events in main orchestrator state
      if (window.AIIDKIT && typeof window.AIIDKIT.setCurrentPatientRawEvents === 'function') {
        window.AIIDKIT.setCurrentPatientRawEvents(events);
      }

      // Filter events to FUP boundary
      const fupSel = document.getElementById('fup-select');
      const selectedFup = fupSel ? parseInt(fupSel.value, 10) : 90;
      const filteredEvents = events.filter(ev => ev.days_since_tpx <= selectedFup);

      const tbody = document.getElementById('patient-events-tbody');
      if (!tbody) return;
      tbody.innerHTML = '';
      _rowCount = 0;
      filteredEvents.forEach(ev => addEventRow(ev));
      scrollToBottom();
      if (window.AIIDKIT && typeof window.AIIDKIT.showToast === 'function') {
        const labels = { high: 'High', mod: 'Moderate', low: 'Low' };
        window.AIIDKIT.showToast(`${labels[type] || type}-risk example patient loaded (filtered to ${selectedFup}-day follow-up).`, 'success');
      }
    } catch (err) {
      console.error('[AIIDKIT] Failed to load example patient:', err);
      if (window.AIIDKIT && typeof window.AIIDKIT.showToast === 'function') {
        window.AIIDKIT.showToast(`Failed to load example patient: ${err.message}`, 'error');
      }
    } finally {
      if (window.AIIDKIT && typeof window.AIIDKIT.hideLoading === 'function') {
        window.AIIDKIT.hideLoading();
      }
    }
  }

  function clearForm() {
    const tbody = document.getElementById('patient-events-tbody');
    if (tbody) tbody.innerHTML = '';
    _rowCount = 0;
    addEventRow({ entity: 'Medication', attribute: 'Tacrolimus', value: 'High', days_since_tpx: 7 });
  }

  function clearUpload() {
    const fileInput = document.getElementById('file-input');
    if (fileInput) fileInput.value = '';
  }

  function scrollToBottom() {
    setTimeout(() => {
      const wrapper = document.querySelector('.event-table-wrapper');
      if (wrapper) {
        wrapper.scrollTop = wrapper.scrollHeight;
      }
    }, 50);
  }

  function setEvents(events) {
    const tbody = document.getElementById('patient-events-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';
    _rowCount = 0;
    events.forEach(ev => addEventRow(ev));
    scrollToBottom();
  }

  // ------------------------------------------------------------------
  // PUBLIC API
  // ------------------------------------------------------------------
  return { initForm, addEventRow, getFormEvents, loadExample, clearForm, clearUpload, setEvents };
})();
