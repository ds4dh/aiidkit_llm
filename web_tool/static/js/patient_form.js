/**
 * AIIDKIT — patient_form.js
 * Handles the structured event entry form and header-aware CSV/JSON file upload.
 * Exposes window.AIIDKIT_FORM
 */
'use strict';

window.AIIDKIT_FORM = (() => {

  let _vocabulary = {};
  let _rowCount = 0;

  function _notifyError(msg) {
    if (window.AIIDKIT && typeof window.AIIDKIT.showToast === 'function') {
      window.AIIDKIT.showToast(msg, 'error');
    } else {
      console.error('[AIIDKIT Form]', msg);
    }
  }

  // ------------------------------------------------------------------
  // INIT
  // ------------------------------------------------------------------

  function initForm(vocabulary) {
    _vocabulary = vocabulary || {};

    document.getElementById('tab-btn-form')?.addEventListener('click',   () => switchTab('form'));
    document.getElementById('tab-btn-upload')?.addEventListener('click', () => switchTab('upload'));

    document.getElementById('add-event-btn')?.addEventListener('click', () => {
      addEventRow();
      scrollToBottom();
    });

    initFileUpload();
    addEventRow();
  }

  function switchTab(name) {
    ['form', 'upload'].forEach(t => {
      const isActive = t === name;
      document.getElementById(`tab-btn-${t}`)?.classList.toggle('active', isActive);
      document.getElementById(`tab-btn-${t}`)?.setAttribute('aria-selected', String(isActive));
      const panel = document.getElementById(`tab-${t}`);
      if (panel) {
        panel.classList.toggle('active', isActive);
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

    const defaultEntity = defaults.entity ? String(defaults.entity).trim() : '';
    const defaultAttrs  = defaultEntity ? Object.keys(_vocabulary[defaultEntity] || {}) : [];
    const defaultAttr   = defaults.attribute ? String(defaults.attribute).trim() : '';
    const defaultVals   = (defaultEntity && defaultAttr)
      ? (_vocabulary[defaultEntity]?.[defaultAttr] || [])
      : [];
    const defaultValue  = defaults.value != null ? String(defaults.value).trim() : '';

    const entitySel = _makeSelect(`ent-${id}`, entities, defaultEntity, '—');
    const attrSel   = _makeSelect(`att-${id}`, defaultAttrs, defaultAttr, '—');
    const valSel    = _makeSelect(`val-${id}`, defaultVals, defaultValue, '—');

    entitySel.addEventListener('change', () => _onEntityChange(entitySel, attrSel, valSel));
    attrSel.addEventListener('change',   () => _onAttrChange(entitySel, attrSel, valSel));

    const dayInput = document.createElement('input');
    dayInput.type = 'number';
    dayInput.step = 'any';
    dayInput.min = '-9999';
    dayInput.max = '9999';
    dayInput.id = `day-${id}`;
    dayInput.value = defaults.days_since_tpx ?? '';
    dayInput.placeholder = '—';

    const removeBtn = document.createElement('button');
    removeBtn.type = 'button';
    removeBtn.className = 'remove-row-btn';
    removeBtn.title = 'Remove this event';
    removeBtn.textContent = '×';
    removeBtn.addEventListener('click', () => tr.remove());

    const tr = document.createElement('tr');
    [entitySel, attrSel, valSel, dayInput, removeBtn].forEach(el => {
      const td = document.createElement('td');
      td.appendChild(el);
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }

  function _onEntityChange(entitySel, attrSel, valSel) {
    const entity  = entitySel.value;
    const attribs = entity ? Object.keys(_vocabulary[entity] || {}) : [];
    _updatePlaceholderClass(entitySel);
    _populateSelect(attrSel, attribs, '', '—');
    _populateSelect(valSel, [], '', '—');
  }

  function _onAttrChange(entitySel, attrSel, valSel) {
    const entity = entitySel.value;
    const attr   = attrSel.value;
    const vals   = (entity && attr) ? (_vocabulary[entity]?.[attr] || []) : [];
    _updatePlaceholderClass(attrSel);
    _populateSelect(valSel, vals, '', '—');
  }

  function _makeSelect(id, options, selectedValue, placeholder = '—') {
    const sel = document.createElement('select');
    sel.id = id;
    _populateSelect(sel, options, selectedValue, placeholder);
    sel.addEventListener('change', () => _updatePlaceholderClass(sel));
    return sel;
  }

  function _populateSelect(sel, options, selectedValue, placeholder = '—') {
    sel.innerHTML = '';

    if (placeholder) {
      const ph = document.createElement('option');
      ph.value = '';
      ph.textContent = placeholder;
      ph.disabled = true;
      if (!selectedValue) {
        ph.selected = true;
      }
      sel.appendChild(ph);
    }

    const finalOptions = Array.isArray(options) ? [...options] : [];
    if (selectedValue && !finalOptions.includes(selectedValue)) {
      // Preserve out-of-dropdown or raw numeric values from uploaded CSV/JSON files
      finalOptions.push(selectedValue);
    }

    finalOptions.forEach(opt => {
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      if (selectedValue && opt === selectedValue) {
        o.selected = true;
      }
      sel.appendChild(o);
    });

    _updatePlaceholderClass(sel);
  }

  function _updatePlaceholderClass(sel) {
    sel.classList.toggle('is-placeholder', !sel.value);
  }

  // ------------------------------------------------------------------
  // COLLECT FORM EVENTS (Chronologically Sorted)
  // ------------------------------------------------------------------

  function getFormEvents() {
    const tbody = document.getElementById('patient-events-tbody');
    if (!tbody) return [];
    const events = [];
    tbody.querySelectorAll('tr').forEach(tr => {
      const entity = tr.querySelector('[id^="ent-"]')?.value?.trim();
      const attr   = tr.querySelector('[id^="att-"]')?.value?.trim();
      const value  = tr.querySelector('[id^="val-"]')?.value?.trim();
      const dayStr = tr.querySelector('[id^="day-"]')?.value;
      const day    = parseFloat(dayStr);
      if (entity && attr && value && !isNaN(day)) {
        events.push({ entity, attribute: attr, value, days_since_tpx: day });
      }
    });
    events.sort((a, b) => a.days_since_tpx - b.days_since_tpx);
    return events;
  }

  // ------------------------------------------------------------------
  // FILE UPLOAD (Header-Aware RFC-4180 CSV + JSON)
  // ------------------------------------------------------------------

  function initFileUpload() {
    const dropZone  = document.getElementById('file-drop-zone');
    const fileInput = document.getElementById('file-input');

    if (dropZone) {
      dropZone.addEventListener('dragover', e => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
      });
      dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
      dropZone.addEventListener('drop', e => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        const file = e.dataTransfer?.files[0];
        if (file) _handleFile(file);
      });
    }

    fileInput?.addEventListener('change', e => {
      const file = e.target.files?.[0];
      if (file) _handleFile(file);
    });
  }

  function _splitCSVLine(line) {
    const fields = [];
    let cur = '';
    let inQuotes = false;
    for (let i = 0; i < line.length; i++) {
      const ch = line[i];
      if (ch === '"') {
        if (inQuotes && line[i + 1] === '"') {
          cur += '"';
          i++;
        } else {
          inQuotes = !inQuotes;
        }
      } else if (ch === ',' && !inQuotes) {
        fields.push(cur.trim());
        cur = '';
      } else {
        cur += ch;
      }
    }
    fields.push(cur.trim());
    return fields;
  }

  function parseCSV(text) {
    const lines = text.trim().split(/\r?\n/).filter(l => l.trim());
    if (!lines.length) return [];

    const firstParts = _splitCSVLine(lines[0]).map(p => p.toLowerCase().replace(/^['"]|['"]$/g, ''));
    const hasHeader = firstParts.some(h => ['entity', 'attribute', 'days_since_tpx', 'value_binned', 'value', 'time', 'day'].includes(h));

    let idxEntity = 0, idxAttr = 1, idxValue = 2, idxDays = 3;
    if (hasHeader) {
      const findCol = (names, fallback) => {
        for (const n of names) {
          const pos = firstParts.indexOf(n);
          if (pos >= 0) return pos;
        }
        return fallback;
      };
      idxEntity = findCol(['entity'], -1);
      idxAttr   = findCol(['attribute', 'attr'], -1);
      idxValue  = findCol(['value_binned', 'value', 'val'], -1);
      idxDays   = findCol(['days_since_tpx', 'time', 'day', 'days'], -1);
      if (idxEntity < 0 || idxAttr < 0 || idxValue < 0 || idxDays < 0) {
        // Fallback to standard 4-column positional order
        [idxEntity, idxAttr, idxValue, idxDays] = [0, 1, 2, 3];
      }
    }

    const dataLines = hasHeader ? lines.slice(1) : lines;
    const events = [];
    for (const line of dataLines) {
      const parts = _splitCSVLine(line).map(p => p.replace(/^['"]|['"]$/g, '').trim());
      if (parts.length < 4) continue;
      const entity    = parts[idxEntity];
      const attribute = parts[idxAttr];
      const value     = parts[idxValue];
      const days      = parseFloat(parts[idxDays]);
      if (entity && attribute && value !== '' && !isNaN(days)) {
        events.push({ entity, attribute, value, days_since_tpx: days });
      }
    }
    events.sort((a, b) => a.days_since_tpx - b.days_since_tpx);
    return events;
  }

  function parseJSON(text) {
    const data = JSON.parse(text);
    const arr  = Array.isArray(data) ? data : (data.events ?? []);
    const events = arr
      .filter(ev => ev && ev.entity && ev.attribute && (ev.value != null || ev.value_binned != null) && (ev.days_since_tpx != null || ev.time != null))
      .map(ev => ({
        entity:         String(ev.entity).trim(),
        attribute:      String(ev.attribute).trim(),
        value:          String(ev.value ?? ev.value_binned).trim(),
        days_since_tpx: parseFloat(ev.days_since_tpx ?? ev.time),
      }))
      .filter(ev => !isNaN(ev.days_since_tpx));
    events.sort((a, b) => a.days_since_tpx - b.days_since_tpx);
    return events;
  }

  function _handleFile(file) {
    const ext = file.name.split('.').pop().toLowerCase();
    if (!['csv', 'json'].includes(ext)) {
      _notifyError('Unsupported file format. Please upload a .csv or .json file.');
      return;
    }
    const reader = new FileReader();
    reader.onload = e => {
      try {
        const text = e.target?.result || '';
        const events = ext === 'json' ? parseJSON(text) : parseCSV(text);
        if (!events.length) {
          _notifyError('No valid clinical events found in the uploaded file.');
          return;
        }

        window.AIIDKIT?.clearPredictionResults?.();
        window.AIIDKIT?.setCurrentPatientRawEvents?.(events);

        setEvents(events);
        switchTab('form');

        const fileInput = document.getElementById('file-input');
        if (fileInput) fileInput.value = '';

        window.AIIDKIT?.showToast?.(`${events.length} events loaded into manual entry form.`, 'success');
      } catch (err) {
        _notifyError(`Could not parse file: ${err.message}`);
      }
    };
    reader.readAsText(file);
  }

  // ------------------------------------------------------------------
  // EXAMPLE PATIENTS
  // ------------------------------------------------------------------

  async function loadExample(type) {
    window.AIIDKIT?.showLoading?.('Loading example patient data…');
    try {
      const res = await fetch(`/api/examples/${encodeURIComponent(type)}`);
      if (!res.ok) {
        let msg = `HTTP ${res.status}`;
        try { const j = await res.json(); msg = j.error || msg; } catch (_) {}
        throw new Error(msg);
      }
      const events = await res.json();
      events.sort((a, b) => parseFloat(a.days_since_tpx) - parseFloat(b.days_since_tpx));

      window.AIIDKIT?.setCurrentPatientRawEvents?.(events);
      setEvents(events);

      const labels = { high: 'High', mod: 'Moderate', low: 'Low' };
      window.AIIDKIT?.showToast?.(`${labels[type] || type}-risk example patient loaded (${events.length} events).`, 'success');
    } catch (err) {
      console.error('[AIIDKIT] Failed to load example patient:', err);
      _notifyError(`Failed to load example patient: ${err.message}`);
    } finally {
      window.AIIDKIT?.hideLoading?.();
    }
  }

  function clearForm() {
    const tbody = document.getElementById('patient-events-tbody');
    if (tbody) tbody.innerHTML = '';
    _rowCount = 0;
    addEventRow();
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

  return { initForm, addEventRow, getFormEvents, loadExample, clearForm, clearUpload, setEvents };
})();
