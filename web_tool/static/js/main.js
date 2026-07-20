/**
 * AIIDKIT — main.js
 * Application orchestrator: state management, API calls,
 * risk gauge animation, UI updates, and event listeners.
 * Exposes window.AIIDKIT  (call init() on DOMContentLoaded)
 */
'use strict';

window.AIIDKIT = (() => {

  // ================================================================
  // STATE
  // ================================================================
  const state = {
    config:           null,
    cohort:           null,
    prediction:       null,
    selectedHorizon:  30,
    selectedFup:      90,
    chartsInitialized: false,
    currentPatientRawEvents: null,
  };

  // ================================================================
  // API
  // ================================================================

  async function _fetch(url, options) {
    const res = await fetch(url, options);
    if (!res.ok) {
      let msg = `HTTP ${res.status}`;
      try { const j = await res.json(); msg = j.error || msg; } catch (_) {}
      throw new Error(msg);
    }
    return res.json();
  }

  const fetchConfig = () => _fetch('/api/config');
  const fetchCohort = (fup) => _fetch(`/api/cohort?fup=${fup || 90}`);

  function submitPrediction(events, horizon, fup) {
    return _fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ events, horizon, fup }),
    });
  }

  // ================================================================
  // LOADING OVERLAY
  // ================================================================

  const showLoading = (msg = 'Analysing patient data…') => {
    const el = document.querySelector('#loading-overlay .loading-text');
    if (el) el.textContent = msg;
    document.getElementById('loading-overlay')?.classList.remove('hidden');
  };
  const hideLoading = () => document.getElementById('loading-overlay')?.classList.add('hidden');

  // ================================================================
  // TOAST NOTIFICATIONS
  // ================================================================

  function showToast(message, type = 'info', duration = 4200) {
    const container = document.getElementById('toast-container');
    if (!container) return null;
    const icons = {
      success: '\u2713',
      error: '\u2717',
      info: '\u2139',
      loading: '<span class="toast-spinner"></span>'
    };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${icons[type] || '\u2139'}</span>${message}`;
    container.appendChild(toast);
    
    if (duration > 0) {
      setTimeout(() => {
        toast.style.cssText += 'opacity:0;transform:translateX(40px);transition:opacity 0.3s,transform 0.3s;';
        setTimeout(() => toast.remove(), 350);
      }, duration);
    }
    return toast;
  }

  // ================================================================
  // RESULTS AREA
  // ================================================================

  function showResults() {
    document.getElementById('results-area')?.classList.remove('hidden');
    document.getElementById('empty-state')?.classList.add('hidden');

    // Initialize charts once when first shown to ensure correct dimensions
    if (!state.chartsInitialized) {
      const C = window.AIIDKIT_CHARTS;
      if (C && state.cohort) {
        requestAnimationFrame(() => {
          setTimeout(() => {
            const activeCohort = state.cohort.horizons[state.selectedHorizon];
            C.initUMAP(activeCohort);
            C.initHistogram(activeCohort.risk_distribution);
            C.initClusterProfiles(activeCohort.cluster_profiles);
            
            C.initAttributionChart(state.cohort.global_attributions || []);

            // Draw current prediction values if already set
            if (state.prediction) {
              C.updateUMAPWithPatient(state.prediction.umap_x, state.prediction.umap_y);
              C.updateHistogramWithPatient(state.prediction.risk_score);
              C.highlightCluster(state.prediction.cluster);
            }
            state.chartsInitialized = true;
          }, 50);
        });
      }
    } else {
      // Trigger a resize on already initialized charts
      requestAnimationFrame(() => {
        setTimeout(() => {
          window.AIIDKIT_CHARTS?.resizeCharts?.();
        }, 50);
      });
    }
  }

  // ================================================================
  // SVG GAUGE
  // Half-circumference of r=80 circle: pi*80 ~ 251 px.
  // rotate(180 100 110) makes arc sweep left->top->right.
  // ================================================================

  const GAUGE_HALF = 251;

  function updateRiskGauge(riskScore, category, color) {
    const arc   = document.getElementById('gauge-arc');
    const track = document.getElementById('gauge-track');
    const pctEl = document.getElementById('gauge-pct');
    const badge = document.getElementById('risk-badge');

    if (track) {
      track.setAttribute('transform',        'rotate(180 100 110)');
      track.setAttribute('stroke-dasharray', `${GAUGE_HALF} 502`);
      track.setAttribute('stroke-dashoffset','0');
    }
    if (arc) {
      arc.setAttribute('transform',        'rotate(180 100 110)');
      arc.setAttribute('stroke-dashoffset','0');
      arc.setAttribute('stroke',            color);
      requestAnimationFrame(() => {
        arc.setAttribute('stroke-dasharray', `${Math.round(riskScore * GAUGE_HALF)} 502`);
      });
    }

    if (pctEl) {
      const startVal = parseInt(pctEl.textContent) || 0;
      const endVal   = Math.round(riskScore * 100);
      const t0       = performance.now();
      const DUR      = 1400;
      function tick(now) {
        const p    = Math.min((now - t0) / DUR, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        pctEl.textContent = `${Math.round(startVal + (endVal - startVal) * ease)}/100`;
        pctEl.setAttribute('fill', color);
        if (p < 1) requestAnimationFrame(tick);
      }
      requestAnimationFrame(tick);
    }

    if (badge) {
      badge.textContent = category;
      badge.className   = `risk-badge ${_badgeClass(category)}`;
    }
  }

  function _badgeClass(cat) {
    return { 'Low risk': 'badge-low', 'Moderate risk': 'badge-mod', 'High risk': 'badge-high', 'Very high risk': 'badge-vhigh' }[cat] || 'badge-low';
  }

  // ================================================================
  // RISK SCORE CARDS
  // ================================================================

  function updateRiskScoreCards(riskScores, calibratedRisks, horizon) {
    [30, 60, 90].forEach(h => {
      const card = document.getElementById(`horizon-card-${h}`);
      const el = document.getElementById(`risk-${h}d`);
      if (!el) return;
      const score = riskScores?.[`${h}d`];
      const risk = calibratedRisks?.[`${h}d`];
      
      if (score != null) {
        let htmlVal = `<span class="horizon-score-val">${(score * 100).toFixed(1)}/100</span>`;
        if (risk != null) {
          htmlVal += `<span class="horizon-risk-pct">Risk: ${(risk * 100).toFixed(1)}%</span>`;
        }
        el.innerHTML = htmlVal;
      } else {
        el.textContent = '\u2014';
      }
      
      if (card) {
        if (h === horizon) {
          card.classList.add('active-horizon');
        } else {
          card.classList.remove('active-horizon');
        }
      }
    });
  }

  // ================================================================
  // COHORT POSITION
  // ================================================================

  function updateCohortPosition(result) {
    const numEl   = document.getElementById('percentile-number');
    const sufEl   = document.getElementById('percentile-suffix');
    const badgeEl = document.getElementById('cluster-badge');
    const rateEl  = document.getElementById('cluster-event-rate');

    if (numEl)   numEl.textContent = result.percentile ?? '\u2014';
    if (sufEl)   sufEl.textContent = _ordSuffix(result.percentile);
    if (badgeEl) {
      badgeEl.textContent      = result.cluster_name ?? '\u2014';
      badgeEl.style.background = result.cluster_color || '#444';
    }
    if (rateEl) {
      const activeCohort = state.cohort?.horizons?.[state.selectedHorizon];
      const prof = activeCohort?.cluster_profiles?.[String(result.cluster)];
      rateEl.textContent = `Event rate: ${prof ? (prof.event_rate * 100).toFixed(0) + '%' : '\u2014'}`;
    }
    window.AIIDKIT_CHARTS?.highlightCluster?.(result.cluster);
  }

  function _ordSuffix(n) {
    if (!Number.isInteger(n)) return '';
    const s = ['th','st','nd','rd'];
    const v = n % 100;
    return s[(v - 20) % 10] || s[v] || s[0];
  }

  // ================================================================
  // CLINICAL TIMELINE
  // ================================================================

  function renderTimeline(eventsWithScores) {
    const list = document.getElementById('timeline-list');
    if (!list) return;
    if (!eventsWithScores?.length) {
      list.innerHTML = '<li class="timeline-placeholder">No events to display.</li>';
      return;
    }

    function getTimelineTextColor(score) {
      const clamped = Math.max(-0.2, Math.min(0.2, score));
      const t = (clamped + 0.2) / 0.4;
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';

      // Pick anchors based on light/dark theme
      // Dark mode: Blue #4F8EF7 (79, 142, 247) -> Gray #B4BED2 (180, 190, 210) -> Red #FF4757 (255, 71, 87)
      // Light mode: Blue #1A56DB (26, 86, 219) -> Gray #64748B (100, 116, 139) -> Red #B32431 (179, 36, 49)
      const startB = isLight ? [26, 86, 219] : [79, 142, 247];
      const midG   = isLight ? [100, 116, 139] : [180, 190, 210];
      const endR   = isLight ? [179, 36, 49] : [255, 71, 87];

      let r, g, b;
      if (t < 0.5) {
        const u = t * 2;
        r = Math.round(startB[0] + (midG[0] - startB[0]) * u);
        g = Math.round(startB[1] + (midG[1] - startB[1]) * u);
        b = Math.round(startB[2] + (midG[2] - startB[2]) * u);
      } else {
        const u = (t - 0.5) * 2;
        r = Math.round(midG[0] + (endR[0] - midG[0]) * u);
        g = Math.round(midG[1] + (endR[1] - midG[1]) * u);
        b = Math.round(midG[2] + (endR[2] - midG[2]) * u);
      }
      return `rgb(${r}, ${g}, ${b})`;
    }

    const sorted = [...eventsWithScores].sort((a, b) => a.days_since_tpx - b.days_since_tpx);
    list.innerHTML = '';

    const rawFeatures = state.cohort?.cluster_profiles?.['0']?.top_features || [];
    const highRiskFeatures = Array.isArray(rawFeatures)
      ? rawFeatures
      : [
          ...(rawFeatures.static || []),
          ...(rawFeatures.recent || [])
        ];

    sorted.forEach((ev, i) => {
      const sc = ev.score ?? 0;
      const absScore = Math.abs(sc);
      const intensity = Math.min(1.0, absScore / 0.2); // Using 0.2 as a reference max absolute value
      const textC = getTimelineTextColor(sc);

      let barC, bgStyle, badgeStyle, badgeClass, badgeText;
      if (sc > 0) {
        barC = `rgba(255, 71, 87, ${0.35 + 0.65 * intensity})`;
        bgStyle = `background: rgba(255, 71, 87, ${0.02 + 0.08 * intensity})`;
        badgeStyle = `background: rgba(255, 71, 87, ${0.08 + 0.22 * intensity}); color: ${textC}; border: 1px solid rgba(255, 71, 87, ${0.12 + 0.28 * intensity}); font-weight: 700;`;
        badgeText = `Risk Impact: +${sc.toFixed(3)}`;
        badgeClass = 'score-pos';
      } else if (sc < 0) {
        barC = `rgba(79, 142, 247, ${0.35 + 0.65 * intensity})`;
        bgStyle = `background: rgba(79, 142, 247, ${0.02 + 0.08 * intensity})`;
        badgeStyle = `background: rgba(79, 142, 247, ${0.08 + 0.22 * intensity}); color: ${textC}; border: 1px solid rgba(79, 142, 247, ${0.12 + 0.28 * intensity}); font-weight: 700;`;
        badgeText = `Risk Impact: ${sc.toFixed(3)}`;
        badgeClass = 'score-neg';
      } else {
        barC = `rgba(61, 79, 114, 0.35)`;
        bgStyle = `background: transparent`;
        badgeStyle = `background: rgba(255, 255, 255, 0.05); color: ${textC}; border: 1px solid rgba(255, 255, 255, 0.04);`;
        badgeText = `Risk Impact: 0.000`;
        badgeClass = 'score-neu';
      }

      const scBadge = `<span class="timeline-score ${badgeClass}" style="${badgeStyle}">${badgeText}</span>`;

      const matchesCohortRisk = highRiskFeatures.some(f => {
        const featName = `${ev.entity} - ${ev.attribute}`;
        return f.feature === featName && String(f.value) === String(ev.value);
      });

      const cohortRiskBadge = matchesCohortRisk 
        ? `<span class="cohort-risk-badge">⚠️ Cohort Risk Driver</span>` 
        : '';

      const li = document.createElement('li');
      li.className = 'timeline-item';
      li.style.animationDelay = `${i * 35}ms`;
      li.style.cssText += `; ${bgStyle}; padding-left: 8px; padding-right: 8px; border-radius: var(--radius-sm); margin-bottom: 4px;`;
      
      li.innerHTML = `
        <div class="timeline-day">
          <div class="timeline-day-num">${ev.days_since_tpx}</div>
          <div class="timeline-day-unit">day</div>
        </div>
        <div class="timeline-bar" style="background:${barC}"></div>
        <div class="timeline-content horizontal-layout">
          <div class="timeline-details">
            <span class="timeline-entity-tag">${ev.entity === 'Infection' ? 'Previous infection' : ev.entity}</span>
            <span class="timeline-event-name" style="color: ${textC}"><strong>${ev.attribute}</strong>: ${ev.value}</span>
          </div>
          <div class="timeline-badges-wrap">
            ${scBadge}
            ${cohortRiskBadge}
          </div>
        </div>`;
      list.appendChild(li);
    });
  }

  // ================================================================
  // RISK CATEGORY HELPER
  // ================================================================

  function _riskCategory(score) {
    const threshold = state.config?.risk_threshold ?? 0.25;
    if (score < threshold) return { category: 'Low risk',  color: '#00C9A7' };
    return                 { category: 'High risk', color: '#FF4757' };
  }

  // ================================================================
  // ANALYSE HANDLER
  // ================================================================

  function clearPredictionResults() {
    state.prediction = null;
    state.chartsInitialized = false;

    // Hide results area, show empty state
    document.getElementById('results-area')?.classList.add('hidden');
    document.getElementById('empty-state')?.classList.remove('hidden');
  }

  function resetAnalysis() {
    state.currentPatientRawEvents = null;
    clearPredictionResults();

    // Reset sidebar inputs
    window.AIIDKIT_FORM?.clearForm();
    window.AIIDKIT_FORM?.clearUpload();

    // Reset attribution chart to global importances
    if (state.chartsInitialized && window.AIIDKIT_CHARTS && state.cohort) {
      window.AIIDKIT_CHARTS.updateAttributionChart(state.cohort.global_attributions || []);
    }

    showToast('Ready for new analysis.', 'info');
  }

  async function handleAnalyse(events, silent = false) {
    if (!events?.length) { showToast('Please add at least one clinical event.', 'error'); return; }
    
    const loadToast = showToast('Analysing patient sequence…', 'loading', 0);
    
    try {
      const result     = await submitPrediction(events, state.selectedHorizon, state.selectedFup);
      state.prediction = result;

      // Always show results first to ensure containers render
      showResults();

      updateRiskGauge(result.risk_score, result.risk_category, result.risk_color);
      updateRiskScoreCards(result.risk_scores, result.calibrated_risks, state.selectedHorizon);
      updateCohortPosition(result);
      renderTimeline(result.events_with_scores || []);

      const narr = document.getElementById('narrative-text');
      if (narr) narr.textContent = result.narrative || '\u2014';

      // If charts are initialized, update them immediately
      if (state.chartsInitialized) {
        const C = window.AIIDKIT_CHARTS;
        if (C) {
          C.updateUMAPWithPatient(result.umap_x, result.umap_y);
          C.updateHistogramWithPatient(result.risk_score);
          if (state.cohort && typeof C.updateAttributionChart === 'function') {
            C.updateAttributionChart(state.cohort.global_attributions || []);
          }
        }
      }

      if (loadToast) {
        loadToast.style.cssText += 'opacity:0;transform:translateX(40px);transition:opacity 0.3s,transform 0.3s;';
        setTimeout(() => loadToast.remove(), 350);
      }

      showToast('Analysis updated!', 'success');
    } catch (err) {
      if (loadToast) {
        loadToast.remove();
      }
      showToast(`Analysis failed: ${err.message}`, 'error');
      console.error('[AIIDKIT] Prediction error:', err);
    }
  }

  // ================================================================
  // THEME MANAGEMENT
  // ================================================================

  function _initTheme() {
    const toggleBtn = document.getElementById('theme-toggle');
    if (!toggleBtn) return;

    // Determine initial theme
    const savedTheme = localStorage.getItem('theme');
    const systemPref = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    const initialTheme = savedTheme || systemPref || 'light';

    // Apply initial theme
    _setTheme(initialTheme);

    toggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
      _setTheme(nextTheme);
    });
  }

  function _setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      if (theme === 'light') {
        // Moon icon + label (indicates next action or current state, here we show current state)
        toggleBtn.innerHTML = `
          <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
          <span>Light theme</span>
        `;
        toggleBtn.setAttribute('title', 'Switch to dark theme');
      } else {
        // Sun icon + label
        toggleBtn.innerHTML = `
          <svg viewBox="0 0 24 24" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
          <span>Dark theme</span>
        `;
        toggleBtn.setAttribute('title', 'Switch to light theme');
      }
    }

    // Update charts tick and grid colors
    window.AIIDKIT_CHARTS?.updateThemeColors?.();

    // Redraw timeline to update text colors
    if (state.prediction) {
      renderTimeline(state.prediction.events_with_scores || []);
    }
  }

  // ================================================================
  // EVENT LISTENERS
  // ================================================================

  function _setupListeners() {
    [30, 60, 90].forEach(h => {
      const card = document.getElementById(`horizon-card-${h}`);
      card?.addEventListener('click', () => {
        const select = document.getElementById('horizon-select');
        if (select && parseInt(select.value, 10) !== h) {
          select.value = h;
          select.dispatchEvent(new Event('change'));
        }
      });
    });

    document.getElementById('horizon-select')?.addEventListener('change', async e => {
      state.selectedHorizon = parseInt(e.target.value, 10);
      
      updateRiskScoreCards(
        state.prediction?.risk_scores,
        state.prediction?.calibrated_risks,
        state.selectedHorizon
      );

      const C = window.AIIDKIT_CHARTS;
      if (state.chartsInitialized && C && state.cohort) {
        const activeCohort = state.cohort.horizons[state.selectedHorizon];
        if (activeCohort) {
          C.initUMAP(activeCohort); // Re-initialize UMAP completely since coordinates and scales bounds changed
          if (state.prediction) {
            C.updateUMAPWithPatient(state.prediction.umap_x, state.prediction.umap_y);
          }
          C.updateHistogramData(activeCohort.risk_distribution);
          C.updateClusterProfiles(activeCohort.cluster_profiles);
        }
      }

      // Auto-re-run prediction if a patient has been analyzed
      const formEvents = window.AIIDKIT_FORM?.getFormEvents?.() || [];
      if (formEvents.length > 0 && state.prediction) {
        await handleAnalyse(formEvents, true);
      }
    });

    document.getElementById('fup-select')?.addEventListener('change', async e => {
      state.selectedFup = parseInt(e.target.value, 10);
      showLoading('Updating cohort data…');
      try {
        state.cohort = await fetchCohort(state.selectedFup);
        const C = window.AIIDKIT_CHARTS;
        if (state.chartsInitialized && C && state.cohort) {
          const activeCohort = state.cohort.horizons[state.selectedHorizon];
          if (activeCohort) {
            // Re-initialize UMAP completely since coordinates and scales bounds changed
            C.initUMAP(activeCohort);
            // Draw gold star immediately if a patient has been analyzed
            if (state.prediction) {
              C.updateUMAPWithPatient(state.prediction.umap_x, state.prediction.umap_y);
            }
            C.updateHistogramData(activeCohort.risk_distribution);
            C.updateClusterProfiles(activeCohort.cluster_profiles);
            if (typeof C.updateAttributionChart === 'function') {
              C.updateAttributionChart(state.cohort.global_attributions || []);
            }
          }
        }
        
        // Update form events based on FUP
        let updatedEvents = [];
        if (state.currentPatientRawEvents) {
          updatedEvents = state.currentPatientRawEvents.filter(ev => ev.days_since_tpx <= state.selectedFup);
          window.AIIDKIT_FORM?.setEvents?.(updatedEvents);
        } else {
          // If manually entered, trim events that exceed the new FUP
          const formEvents = window.AIIDKIT_FORM?.getFormEvents?.() || [];
          updatedEvents = formEvents.filter(ev => ev.days_since_tpx <= state.selectedFup);
          if (updatedEvents.length !== formEvents.length) {
            window.AIIDKIT_FORM?.setEvents?.(updatedEvents);
          }
        }

        // Auto-re-run prediction if a patient has been analyzed
        if (updatedEvents.length > 0 && state.prediction) {
          await handleAnalyse(updatedEvents);
        } else {
          showToast(`Cohort loaded for ${state.selectedFup}-day follow-up.`, 'success');
        }
      } catch (err) {
        showToast(`Failed to load cohort: ${err.message}`, 'error');
        console.error('[AIIDKIT] FUP cohort load error:', err);
      } finally {
        hideLoading();
      }
    });

    document.getElementById('reset-analysis-btn')?.addEventListener('click', () => {
      resetAnalysis();
    });

    document.getElementById('umap-toggle-cluster')?.addEventListener('click', () =>
      window.AIIDKIT_CHARTS?.toggleUMAPColorMode?.('cluster'));
    document.getElementById('umap-toggle-risk')?.addEventListener('click', () =>
      window.AIIDKIT_CHARTS?.toggleUMAPColorMode?.('risk'));
    document.getElementById('umap-toggle-imminence')?.addEventListener('click', () =>
      window.AIIDKIT_CHARTS?.toggleUMAPColorMode?.('imminence'));

    document.getElementById('analyze-form-btn')?.addEventListener('click', () =>
      handleAnalyse(window.AIIDKIT_FORM?.getFormEvents?.() || []));

    document.querySelectorAll('.example-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const type = btn.dataset.example;
        clearPredictionResults();
        document.getElementById('tab-btn-form')?.click();
        if (window.AIIDKIT_FORM?.loadExample) {
          await window.AIIDKIT_FORM.loadExample(type);
        }
      });
    });
  }

  // ================================================================
  // INIT
  // ================================================================

  async function init() {
    showLoading('Loading…');
    try {
      _initTheme();
      state.config = await fetchConfig();

      const badge = document.getElementById('mock-badge');
      if (badge) {
        if (state.config.using_mock_data) {
          badge.textContent = 'DEMO MODE'; badge.className = 'mode-badge badge-demo';
        } else {
          badge.textContent = 'LIVE MODEL'; badge.className = 'mode-badge badge-live';
        }
      }

      state.selectedFup = state.config.default_fup || 90;

      // Populate follow-up period select dropdown
      const fupSel = document.getElementById('fup-select');
      if (fupSel && state.config.available_fups) {
        fupSel.innerHTML = '';
        state.config.available_fups.forEach(f => {
          const opt = document.createElement('option');
          opt.value = f;
          opt.textContent = `${f} days`;
          if (f === state.selectedFup) opt.selected = true;
          fupSel.appendChild(opt);
        });
      }

      window.AIIDKIT_FORM?.initForm?.(state.config.vocabulary);

      state.cohort = await fetchCohort(state.selectedFup);

      _setupListeners();
      updateRiskScoreCards(null, null, state.selectedHorizon);
      hideLoading();
    } catch (err) {
      hideLoading();
      showToast(`Startup error: ${err.message}`, 'error');
      console.error('[AIIDKIT] Init error:', err);
    }
  }

  return {
    init,
    showLoading,
    hideLoading,
    showToast,
    setCurrentPatientRawEvents(events) { state.currentPatientRawEvents = events; },
    getCurrentPatientRawEvents() { return state.currentPatientRawEvents; },
    clearPredictionResults,
    handleAnalyse,
  };
})();
