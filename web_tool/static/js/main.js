/**
 * AIIDKIT — main.js
 * Application orchestrator: multi-domain specialist routing (Bacteria, Viruses,
 * Fungi, Graft Loss, Mortality), state management, dynamic horizon cards,
 * per-domain UMAP manifolds & trajectories, and UI event listeners.
 * Exposes window.AIIDKIT  (call init() on DOMContentLoaded)
 */
'use strict';

window.AIIDKIT = (() => {

  function _escHtml(val) {
    return String(val ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function _formatHorizonLabel(h, config) {
    const fromCfg = config?.horizon_labels?.[String(h)];
    if (fromCfg) return fromCfg;
    if (h >= 360 && h % 360 === 0) {
      const y = h / 360;
      return y === 1 ? '1 year' : `${y} years`;
    }
    if (h === 180) return '6 months';
    return `${h} days`;
  }

  // ================================================================
  // STATE (Multi-Domain Specialist Aware)
  // ================================================================
  const state = {
    config:                  null,
    cohortsByDomain:         {},   // { [domain]: { [fup]: cohortObj } }
    predictionsByDomain:     {},   // { [domain]: { [fup]: predObj } }
    trajectoriesByDomain:    {},   // { [domain]: trajectoryArray }
    renderedUmapDomain:      null,
    fupSteps:                [],
    maxFup:                  0,
    patientMaxDay:           0,
    showTrajectory:          true,
    bgFetchToken:            0,
    selectedTask:            'infection_bacteria',
    selectedHorizon:         90,
    preferredAcuteHorizon:   90,
    preferredLongHorizon:    360,
    selectedFup:             30,
    chartsInitialized:       false,
    currentPatientRawEvents: null,
  };

  function _getTaskDomain(taskKey = state.selectedTask) {
    return state.config?.task_to_domain?.[taskKey] || 'default';
  }

  function _getDomainLabel(domainKey) {
    const dObj = (state.config?.domains || []).find(d => d.domain === domainKey);
    return dObj?.label || '';
  }

  function _getAvailableHorizons(taskKey = state.selectedTask) {
    const taskSpecific = state.config?.task_horizons?.[taskKey];
    if (Array.isArray(taskSpecific) && taskSpecific.length > 0) {
      return taskSpecific;
    }
    return state.config?.available_horizons?.length
      ? state.config.available_horizons
      : [30, 60, 90];
  }

  function _syncHorizonForTask(taskKey) {
    const allowed = _getAvailableHorizons(taskKey);
    if (allowed.includes(state.selectedHorizon)) return false;
    const isLongTerm = allowed[0] >= 180;
    const pref = isLongTerm ? state.preferredLongHorizon : state.preferredAcuteHorizon;
    state.selectedHorizon = allowed.includes(pref) ? pref : allowed[Math.min(2, allowed.length - 1)];
    return true;
  }

  function _rememberHorizonPreference(h) {
    if (h >= 360) {
      state.preferredLongHorizon = h;
    } else {
      state.preferredAcuteHorizon = h;
    }
  }

  function _getActiveTrajectory() {
    const dom = _getTaskDomain(state.selectedTask);
    return state.trajectoriesByDomain[dom] || [];
  }

  function getActiveTaskCohort() {
    const dom = _getTaskDomain(state.selectedTask);
    const domMap = state.cohortsByDomain[dom] || {};
    const activeCohortObj = domMap[state.selectedFup] || domMap[state.config?.default_fup ?? 30] || Object.values(domMap)[0];
    if (!activeCohortObj) return null;
    const taskCohort = activeCohortObj.tasks?.[state.selectedTask] || activeCohortObj;
    if (activeCohortObj.global_umap_bounds && !taskCohort.global_umap_bounds) {
      taskCohort.global_umap_bounds = activeCohortObj.global_umap_bounds;
    }
    return taskCohort;
  }

  function _buildLightweightStagePred(fup) {
    const dom = _getTaskDomain(state.selectedTask);
    const domPreds = state.predictionsByDomain[dom] || {};
    const basePred = domPreds[state.maxFup] || Object.values(domPreds)[0];
    if (!basePred) return null;

    const baseTask = basePred.tasks?.[state.selectedTask] || basePred;
    const baseHoriz = baseTask.by_horizon?.[String(state.selectedHorizon)] || baseTask;
    const traj = _getActiveTrajectory();
    const trajPt = traj.find(pt => pt.fup === fup);
    if (!trajPt) return baseHoriz;

    const stTask = trajPt.tasks?.[state.selectedTask];
    const hKey = `${state.selectedHorizon}d`;
    const riskScores = stTask?.risk_scores || baseHoriz.risk_scores;
    const calibRisks = stTask?.calibrated_risks || baseHoriz.calibrated_risks;
    const selRisk = riskScores?.[hKey] ?? baseHoriz.risk_score;
    const selCalib = calibRisks?.[hKey] ?? baseHoriz.calibrated_risk;
    const evsUpToFup = (baseHoriz.events_with_scores || []).filter(e => parseFloat(e.days_since_tpx ?? 0) <= fup);

    const taskCohort = getActiveTaskCohort();
    const cohortPts = taskCohort?.horizons?.[state.selectedHorizon]?.patients || [];
    let clusterId = baseHoriz.cluster;
    let clusterName = baseHoriz.cluster_name;
    let clusterColor = baseHoriz.cluster_color;
    let percentile = baseHoriz.percentile;
    if (cohortPts.length > 0) {
      let bestDist = Infinity;
      for (const p of cohortPts) {
        const d2 = (p.umap_x - trajPt.umap_x) ** 2 + (p.umap_y - trajPt.umap_y) ** 2;
        if (d2 < bestDist) {
          bestDist = d2;
          clusterId = p.cluster;
          clusterName = p.cluster_name;
          clusterColor = p.cluster_color;
        }
      }
      const lowerCount = cohortPts.filter(p => p.risk_score < selRisk).length;
      percentile = Math.round((lowerCount / cohortPts.length) * 100);
    }

    return {
      ...baseHoriz,
      fup,
      umap_x: trajPt.umap_x,
      umap_y: trajPt.umap_y,
      risk_scores: riskScores,
      calibrated_risks: calibRisks,
      risk_score: selRisk,
      calibrated_risk: selCalib,
      risk_category: clusterName,
      risk_color: clusterColor,
      cluster: clusterId,
      cluster_name: clusterName,
      cluster_color: clusterColor,
      percentile,
      events_with_scores: evsUpToFup,
    };
  }

  function _getUnifiedTimelineContext() {
    const dom = _getTaskDomain(state.selectedTask);
    const domPreds = state.predictionsByDomain[dom] || {};
    const fullPred = domPreds[state.maxFup] || Object.values(domPreds)[0];
    if (!fullPred) return null;
    const fullTask = fullPred.tasks?.[state.selectedTask] || fullPred;
    const fullHoriz = fullTask.by_horizon?.[String(state.selectedHorizon)] || fullTask;
    const fullEvents = fullHoriz.events_with_scores || [];
    const finalFullScore = fullHoriz.risk_score;

    const traj = _getActiveTrajectory();
    const day0Pt = traj.find(pt => pt.fup === 0) || traj[0];
    const baselineScore = day0Pt?.tasks?.[state.selectedTask]?.risk_scores?.[`${state.selectedHorizon}d`];
    return {
      fullEvents,
      visibleEvents: fullEvents.filter(e => parseFloat(e.days_since_tpx ?? 0) <= state.selectedFup),
      finalFullScore,
      baselineScore,
      cutoffFup: state.selectedFup,
    };
  }

  function getActiveTaskPrediction() {
    const dom = _getTaskDomain(state.selectedTask);
    const domPreds = state.predictionsByDomain[dom] || {};
    if (state.selectedFup === state.maxFup && domPreds[state.maxFup]) {
      const taskPred = domPreds[state.maxFup].tasks?.[state.selectedTask] || domPreds[state.maxFup];
      return taskPred.by_horizon?.[String(state.selectedHorizon)] || taskPred;
    }
    return _buildLightweightStagePred(state.selectedFup);
  }

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

  const fetchConfig = () => _fetch('api/config');

  const fetchCohort = async (fup, domainKey = null) => {
    const targetFup = fup != null ? fup : 30;
    const dom = domainKey || _getTaskDomain(state.selectedTask);
    if (!state.cohortsByDomain[dom]) {
      state.cohortsByDomain[dom] = {};
    }
    if (state.cohortsByDomain[dom][targetFup]) {
      return state.cohortsByDomain[dom][targetFup];
    }
    const c = await _fetch(`api/cohort?fup=${targetFup}&domain=${encodeURIComponent(dom)}`);
    state.cohortsByDomain[dom][targetFup] = c;
    return c;
  };

  function submitPrediction(events, horizon, fup, task, maxFup, background = false) {
    return _fetch('api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        events,
        horizon,
        fup,
        max_fup: maxFup != null ? maxFup : fup,
        task: task || state.selectedTask,
        background,
      }),
    });
  }

  // ================================================================
  // LOADING OVERLAY & TOASTS
  // ================================================================

  const showLoading = (msg = 'Analysing patient data…') => {
    const el = document.querySelector('#loading-overlay .loading-text');
    if (el) el.textContent = msg;
    document.getElementById('loading-overlay')?.classList.remove('hidden');
  };
  const hideLoading = () => document.getElementById('loading-overlay')?.classList.add('hidden');

  function showToast(message, type = 'info', duration = 4200) {
    const container = document.getElementById('toast-container');
    if (!container) return null;
    const icons = {
      success: '\u2713',
      error: '\u2717',
      info: '\u2139',
      loading: '<span class="toast-spinner"></span>',
    };
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<span>${icons[type] || '\u2139'}</span>${_escHtml(message)}`;
    container.appendChild(toast);

    if (duration > 0) {
      setTimeout(() => {
        toast.style.cssText += 'opacity:0;transform:translateY(-15px);transition:opacity 0.3s,transform 0.3s;';
        setTimeout(() => toast.remove(), 350);
      }, duration);
    }
    return toast;
  }

  // ================================================================
  // RESULTS AREA & COLLAPSIBLE LATERAL SIDEBAR
  // ================================================================

  function _refreshChartsAfterLayoutChange() {
    if (!state.chartsInitialized) return;
    const C = window.AIIDKIT_CHARTS;
    const dom = _getTaskDomain(state.selectedTask);
    const taskCohort = getActiveTaskCohort();
    const activeCohort = taskCohort?.horizons?.[state.selectedHorizon];
    const activePred = getActiveTaskPrediction();
    if (!C || !activeCohort) return;

    C.initUMAP(activeCohort, taskCohort.global_umap_bounds);
    state.renderedUmapDomain = dom;
    if (activePred) {
      C.updateUMAPWithPatient(
        activePred.umap_x,
        activePred.umap_y,
        activePred,
        _getActiveTrajectory(),
        state.selectedFup,
        state.showTrajectory,
        (clickedFup) => setSelectedFup(clickedFup)
      );
      const tlCtx = _getUnifiedTimelineContext();
      if (tlCtx && tlCtx.fullEvents.length) {
        C.drawRiskTimelineGraph?.(tlCtx.fullEvents, tlCtx.finalFullScore, state.selectedTask, {
          cutoffFup: tlCtx.cutoffFup,
          baselineScore: tlCtx.baselineScore,
        });
      }
    }
    C.resizeCharts?.();
  }

  function setSidebarCollapsed(collapsed, refitCharts = true) {
    const sidebar = document.getElementById('app-sidebar');
    if (!sidebar) return;
    sidebar.classList.toggle('collapsed', Boolean(collapsed));
    if (refitCharts && state.chartsInitialized) {
      setTimeout(_refreshChartsAfterLayoutChange, 280);
    }
  }

  function showResults() {
    document.getElementById('results-area')?.classList.remove('hidden');
    document.getElementById('empty-state')?.classList.add('hidden');
    document.getElementById('sidebar-collapse-btn')?.classList.remove('hidden');
    document.getElementById('header-new-patient-btn')?.classList.remove('hidden');

    setSidebarCollapsed(true, false);

    if (!state.chartsInitialized) {
      const C = window.AIIDKIT_CHARTS;
      const dom = _getTaskDomain(state.selectedTask);
      const taskCohort = getActiveTaskCohort();
      if (C && taskCohort) {
        requestAnimationFrame(() => {
          setTimeout(() => {
            const activeCohort = taskCohort.horizons[state.selectedHorizon];
            const activePred = getActiveTaskPrediction();
            C.initUMAP(activeCohort, taskCohort.global_umap_bounds);
            state.renderedUmapDomain = dom;
            C.initHistogram(activeCohort.risk_distribution, state.selectedTask);
            C.initClusterProfiles(activeCohort.cluster_profiles);

            if (activePred) {
              C.updateUMAPWithPatient(
                activePred.umap_x,
                activePred.umap_y,
                activePred,
                _getActiveTrajectory(),
                state.selectedFup,
                state.showTrajectory,
                (clickedFup) => setSelectedFup(clickedFup)
              );
              C.updateHistogramWithPatient(activePred.risk_score);
              C.highlightCluster(activePred.cluster);
              const tlCtx = _getUnifiedTimelineContext();
              if (tlCtx && tlCtx.fullEvents.length) {
                C.drawRiskTimelineGraph?.(tlCtx.fullEvents, tlCtx.finalFullScore, state.selectedTask, {
                  cutoffFup: tlCtx.cutoffFup,
                  baselineScore: tlCtx.baselineScore,
                });
              }
            }
            state.chartsInitialized = true;
          }, 280);
        });
      }
    } else {
      requestAnimationFrame(() => {
        setTimeout(_refreshChartsAfterLayoutChange, 280);
      });
    }
  }

  // ================================================================
  // DYNAMIC HORIZON CARDS & RISK BADGE
  // ================================================================

  function _badgeClass(cat) {
    return {
      'Low risk': 'badge-low',
      'Medium risk': 'badge-mod',
      'Moderate risk': 'badge-mod',
      'High risk': 'badge-high',
      'Very high risk': 'badge-vhigh',
    }[cat] || 'badge-low';
  }

  function updateRiskBadge(category) {
    const badge = document.getElementById('risk-badge');
    if (badge) {
      badge.textContent = category || 'Low risk';
      badge.className = `risk-badge ${_badgeClass(category)}`;
    }
  }

  function renderHorizonCards() {
    const container = document.getElementById('risk-horizons-container');
    const metricsRow = document.getElementById('unified-metrics-row');
    if (!container) return;

    const horizons = _getAvailableHorizons(state.selectedTask);
    const isMultiGrid = horizons.length > 3;
    container.classList.toggle('multi-horizon-grid', isMultiGrid);
    metricsRow?.classList.toggle('has-multi-horizons', isMultiGrid);

    container.innerHTML = '';
    horizons.forEach((h, idx) => {
      const item = document.createElement('div');
      item.className = `risk-horizon-item${h === state.selectedHorizon ? ' active-horizon' : ''}`;
      if (isMultiGrid && horizons.length % 2 === 1 && idx === horizons.length - 1) {
        item.classList.add('span-full-col');
      }
      item.id = `horizon-card-${h}`;
      item.dataset.horizon = String(h);
      item.innerHTML = `
        <span class="horizon-label">${_escHtml(_formatHorizonLabel(h, state.config))}</span>
        <span id="risk-${h}d" class="horizon-value">\u2014</span>
      `;
      item.addEventListener('click', () => {
        if (state.selectedHorizon === h) return;
        state.selectedHorizon = h;
        _rememberHorizonPreference(h);
        applyActiveTaskAndHorizon(true);
      });
      container.appendChild(item);
    });
  }

  function updateRiskScoreCards(riskScores, calibratedRisks, horizon) {
    const horizons = _getAvailableHorizons(state.selectedTask);
    if (!document.getElementById(`horizon-card-${horizons[0]}`) ||
        document.querySelectorAll('#risk-horizons-container .risk-horizon-item').length !== horizons.length) {
      renderHorizonCards();
    }
    horizons.forEach(h => {
      const card = document.getElementById(`horizon-card-${h}`);
      const el = document.getElementById(`risk-${h}d`);
      if (!el) return;
      const score = riskScores?.[`${h}d`];
      const risk = calibratedRisks?.[`${h}d`];

      if (score != null) {
        let htmlVal = `<span class="horizon-score-val">${Math.round(score * 100)}/100</span>`;
        if (risk != null) {
          const pctStr = risk < 0.01 ? '<1%' : `${Math.round(risk * 100)}%`;
          htmlVal += `<span class="horizon-risk-pct">Prob: ${pctStr}</span>`;
        }
        el.innerHTML = htmlVal;
      } else {
        el.textContent = '\u2014';
      }

      if (card) {
        card.classList.toggle('active-horizon', h === horizon);
      }
    });
  }

  // ================================================================
  // CLINICAL TIMELINE
  // ================================================================

  function renderTimeline(eventsWithScores) {
    const list = document.getElementById('timeline-list');
    if (!list) return;
    const tlCtx = _getUnifiedTimelineContext();
    const listEvents = tlCtx ? tlCtx.visibleEvents : (eventsWithScores || []);
    if (!listEvents.length) {
      list.innerHTML = '<li class="timeline-placeholder">No events up to this follow-up stage.</li>';
      if (tlCtx && tlCtx.fullEvents.length) {
        window.AIIDKIT_CHARTS?.drawRiskTimelineGraph?.(tlCtx.fullEvents, tlCtx.finalFullScore, state.selectedTask, {
          cutoffFup: tlCtx.cutoffFup,
          baselineScore: tlCtx.baselineScore,
        });
      }
      return;
    }

    function getTimelineTextColor(score) {
      const clamped = Math.max(-0.2, Math.min(0.2, score));
      const t = (clamped + 0.2) / 0.4;
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const startB = isLight ? [26, 86, 219] : [79, 142, 247];
      const midG   = isLight ? [100, 116, 139] : [180, 190, 210];
      const endR   = isLight ? [179, 36, 49] : [255, 71, 87];

      const u = t < 0.5 ? t * 2 : (t - 0.5) * 2;
      const c0 = t < 0.5 ? startB : midG;
      const c1 = t < 0.5 ? midG : endR;
      return `rgb(${Math.round(c0[0] + (c1[0] - c0[0]) * u)}, ${Math.round(c0[1] + (c1[1] - c0[1]) * u)}, ${Math.round(c0[2] + (c1[2] - c0[2]) * u)})`;
    }

    const sorted = [...listEvents].sort((a, b) => a.days_since_tpx - b.days_since_tpx);
    list.innerHTML = '';

    const activeCohort = getActiveTaskCohort()?.horizons?.[state.selectedHorizon];
    const rawFeatures = activeCohort?.cluster_profiles?.['0']?.top_features || [];
    const highRiskFeatures = Array.isArray(rawFeatures)
      ? rawFeatures
      : [...(rawFeatures.static || []), ...(rawFeatures.recent || [])];

    sorted.forEach((ev, i) => {
      const sc = ev.score ?? 0;
      const intensity = Math.min(1.0, Math.abs(sc) / 0.2);
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
        barC = 'rgba(61, 79, 114, 0.35)';
        bgStyle = 'background: transparent';
        badgeStyle = `background: rgba(255, 255, 255, 0.05); color: ${textC}; border: 1px solid rgba(255, 255, 255, 0.04);`;
        badgeText = 'Risk Impact: 0.000';
        badgeClass = 'score-neu';
      }

      const matchesCohortRisk = highRiskFeatures.some(f => {
        const featName = `${ev.entity} - ${ev.attribute}`;
        return f.feature === featName && String(f.value) === String(ev.value);
      });

      const cohortRiskBadge = matchesCohortRisk
        ? '<span class="cohort-risk-badge">Cohort Risk Driver</span>'
        : '';
      const entityLabel = _escHtml(ev.entity === 'Infection' ? 'Previous infection' : ev.entity);

      const li = document.createElement('li');
      li.className = 'timeline-item';
      li.style.animationDelay = `${i * 25}ms`;
      li.style.cssText += `; ${bgStyle}; padding: 10px 14px; border-radius: var(--radius-sm); margin-bottom: 5px; display: flex; align-items: center; gap: 12px;`;

      li.innerHTML = `
        <div class="timeline-day" style="flex-shrink: 0; margin-right: 4px;">
          <div class="timeline-day-num">${_escHtml(ev.days_since_tpx)}</div>
          <div class="timeline-day-unit">day</div>
        </div>
        <div class="timeline-bar" style="background:${barC}; flex-shrink: 0; width: 3.5px; border-radius: 2px; align-self: stretch;"></div>
        <div class="timeline-content-2line" style="flex: 1; display: flex; flex-direction: column; gap: 4px; min-width: 0; padding-right: 8px;">
          <div class="timeline-row-top" style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
            <span class="timeline-entity-tag" style="margin: 0; flex-shrink: 0;">${entityLabel}</span>
            <span class="timeline-value-text" style="font-size: 0.74rem; font-weight: 700; font-variant-numeric: tabular-nums; color: ${textC}; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; margin-left: auto;"><strong>Value:</strong> ${_escHtml(ev.value)}</span>
          </div>
          <div class="timeline-row-bottom" style="display: flex; align-items: center; justify-content: space-between; gap: 10px;">
            <span class="timeline-attribute-name" style="font-size: 0.75rem; color: var(--text); font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">${_escHtml(ev.attribute)}</span>
            <div class="timeline-badges-wrap" style="display: flex; align-items: center; gap: 6px; flex-shrink: 0; margin-left: auto;">
              <span class="timeline-score ${badgeClass}" style="${badgeStyle}">${badgeText}</span>
              ${cohortRiskBadge}
            </div>
          </div>
        </div>`;
      list.appendChild(li);
    });

    if (tlCtx && tlCtx.fullEvents.length) {
      window.AIIDKIT_CHARTS?.drawRiskTimelineGraph?.(tlCtx.fullEvents, tlCtx.finalFullScore, state.selectedTask, {
        cutoffFup: tlCtx.cutoffFup,
        baselineScore: tlCtx.baselineScore,
      });
    } else {
      const activePred = getActiveTaskPrediction();
      window.AIIDKIT_CHARTS?.drawRiskTimelineGraph?.(sorted, activePred?.risk_score, state.selectedTask);
    }
  }

  // ================================================================
  // ANALYSE, SPECIALIST SWITCHING & FUP STAGE MANAGEMENT
  // ================================================================

  function clearPredictionResults() {
    state.bgFetchToken += 1;
    state.predictionsByDomain = {};
    state.trajectoriesByDomain = {};
    state.renderedUmapDomain = null;
    state.fupSteps = [];
    state.chartsInitialized = false;

    document.getElementById('results-area')?.classList.add('hidden');
    document.getElementById('empty-state')?.classList.remove('hidden');
    document.getElementById('sidebar-collapse-btn')?.classList.add('hidden');
    document.getElementById('header-new-patient-btn')?.classList.add('hidden');
    setSidebarCollapsed(false, false);
  }

  function resetAnalysis() {
    state.currentPatientRawEvents = null;
    clearPredictionResults();
    window.AIIDKIT_FORM?.clearForm();
    window.AIIDKIT_FORM?.clearUpload();
    showToast('Ready for new analysis.', 'info');
  }

  function calculateAutoFup(events) {
    if (!events || !events.length) return { fup: 0, maxDay: 0, fupSteps: [0] };
    const validDays = events
      .map(ev => parseFloat(ev.days_since_tpx))
      .filter(d => !isNaN(d) && d >= 0);
    const maxDay = validDays.length ? Math.max(...validDays) : 0;
    const avail = (state.config?.available_fups || [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360])
      .slice()
      .sort((a, b) => a - b);

    let bestFup = avail.find(f => f >= maxDay);
    if (bestFup === undefined) {
      bestFup = avail[avail.length - 1] ?? 0;
    }
    const fupSteps = avail.filter(f => f <= bestFup);
    if (!fupSteps.length) fupSteps.push(0);
    return { fup: bestFup, maxDay: Math.round(maxDay), fupSteps };
  }

  function _updateFupLabelsAndBadges() {
    const fup = state.selectedFup;
    const dom = _getTaskDomain(state.selectedTask);
    const domLabel = _getDomainLabel(dom);
    const domPreds = state.predictionsByDomain[dom] || {};
    const stageText = fup === 0 ? 'Day 0 (Baseline)' : `Day ${fup}`;
    const shortStageText = `Day ${fup}`;

    const badge1 = document.getElementById('fup-slider-val-badge');
    if (badge1) badge1.textContent = stageText;

    const badge2 = document.getElementById('assessment-fup-badge');
    if (badge2) badge2.textContent = shortStageText;

    const specBadge = document.getElementById('assessment-specialist-badge');
    if (specBadge) {
      if (domLabel && state.config?.multi_domain_mode) {
        specBadge.textContent = domLabel;
        specBadge.classList.remove('hidden');
      } else {
        specBadge.classList.add('hidden');
      }
    }

    const panelTitle = document.getElementById('assessment-panel-title');
    if (panelTitle) {
      panelTitle.textContent = state.selectedTask.startsWith('infection_')
        ? 'Infection risk assessment'
        : 'Clinical outcome risk assessment';
    }

    const taskBadge = document.getElementById('assessment-task-badge');
    if (taskBadge) {
      const taskObj = (state.config?.available_tasks || []).find(t => t.key === state.selectedTask);
      taskBadge.textContent = taskObj?.label || state.selectedTask;
    }

    const umapCaption = document.getElementById('umap-cohort-caption');
    if (umapCaption) {
      const prefix = domLabel ? `${domLabel} manifold` : 'Reference cohort';
      if (fup === state.maxFup) {
        umapCaption.textContent = `${prefix} at Day ${fup} post-transplant (matched to patient timeline: Day ${state.patientMaxDay}) — proximity reflects clinical similarity.`;
      } else {
        umapCaption.textContent = `Historical stage view: ${prefix} & patient state at Day ${fup} post-transplant (full patient timeline reaches Day ${state.patientMaxDay}).`;
      }
    }

    const slider = document.getElementById('fup-slider');
    if (slider && state.fupSteps.length > 0) {
      const idx = state.fupSteps.indexOf(fup);
      if (idx >= 0 && parseInt(slider.value, 10) !== idx) {
        slider.value = String(idx);
      }
    }

    document.querySelectorAll('.fup-tick-btn').forEach(btn => {
      const btnFup = parseInt(btn.dataset.fup, 10);
      btn.classList.toggle('active', btnFup === fup);
      btn.classList.toggle('cached', Boolean(domPreds[state.maxFup] && state.cohortsByDomain[dom]?.[btnFup]));
    });
  }

  function _updateFupBackgroundStatus() {
    const statusEl = document.getElementById('fup-bg-status');
    if (!statusEl) return;
    const dom = _getTaskDomain(state.selectedTask);
    const domPreds = state.predictionsByDomain[dom] || {};
    const domCohorts = state.cohortsByDomain[dom] || {};
    const total = state.fupSteps.length;
    const hasMainPred = Boolean(domPreds[state.maxFup]);
    const readyCount = hasMainPred ? state.fupSteps.filter(f => Boolean(domCohorts[f])).length : 0;

    if (total <= 1) {
      statusEl.className = 'fup-bg-status ready';
      statusEl.innerHTML = '\u2713 Baseline stage';
    } else if (readyCount < total) {
      statusEl.className = 'fup-bg-status computing';
      statusEl.innerHTML = `<span class="toast-spinner"></span>History ${readyCount}/${total}\u2026`;
    } else {
      statusEl.className = 'fup-bg-status ready';
      statusEl.innerHTML = `\u2713 All ${total} stages ready`;
    }

    document.querySelectorAll('.fup-tick-btn').forEach(btn => {
      const btnFup = parseInt(btn.dataset.fup, 10);
      btn.classList.toggle('cached', Boolean(hasMainPred && domCohorts[btnFup]));
    });
  }

  function renderFupSliderBar() {
    const slider = document.getElementById('fup-slider');
    const ticksWrap = document.getElementById('fup-slider-ticks');
    const trajBtn = document.getElementById('fup-trajectory-toggle');
    const dom = _getTaskDomain(state.selectedTask);
    const domPreds = state.predictionsByDomain[dom] || {};
    const steps = state.fupSteps.length ? state.fupSteps : [state.selectedFup];

    if (slider) {
      slider.min = '0';
      slider.max = String(Math.max(0, steps.length - 1));
      slider.step = '1';
      slider.value = String(Math.max(0, steps.indexOf(state.selectedFup)));
      slider.disabled = steps.length <= 1;
    }

    if (trajBtn) {
      trajBtn.classList.toggle('active', state.showTrajectory);
      trajBtn.style.display = steps.length > 1 ? 'inline-flex' : 'none';
    }

    if (ticksWrap) {
      ticksWrap.innerHTML = '';
      steps.forEach((f) => {
        const btn = document.createElement('button');
        btn.type = 'button';
        btn.className = `fup-tick-btn${f === state.selectedFup ? ' active' : ''}${domPreds[f] ? ' cached' : ''}`;
        btn.dataset.fup = String(f);
        btn.textContent = f === 0 ? '0d (Base)' : `${f}d`;
        btn.title = `Switch to Day ${f} post-transplant stage`;
        btn.addEventListener('click', () => setSelectedFup(f));
        ticksWrap.appendChild(btn);
      });
    }

    _updateFupLabelsAndBadges();
    _updateFupBackgroundStatus();
  }

  async function _precomputeEarlierFupsInBackground(events, domainKey, taskKey, horizon, priorFups, maxFup, token) {
    // Only pre-fetch lightweight reference cohort JSONs per FUP; patient trajectory & unified IG
    // attributions across all FUP stages are already resolved by the single maxFup prediction!
    for (const f of priorFups) {
      if (state.bgFetchToken !== token) return;
      try {
        await fetchCohort(f, domainKey);
        if (_getTaskDomain(state.selectedTask) === domainKey) {
          _updateFupBackgroundStatus();
        }
      } catch (err) {
        console.warn(`[AIIDKIT] Background cohort fetch for ${domainKey}@${f} failed:`, err);
      }
    }
  }

  async function _ensureDomainAndTaskPredictionIfNeeded() {
    const events = state.currentPatientRawEvents || window.AIIDKIT_FORM?.getFormEvents?.() || [];
    if (!events.length || !state.chartsInitialized) return;

    const dom = _getTaskDomain(state.selectedTask);
    const currentToken = state.bgFetchToken;
    if (!state.predictionsByDomain[dom]) state.predictionsByDomain[dom] = {};

    const maxFupPred = state.predictionsByDomain[dom][state.maxFup];
    if (!maxFupPred) {
      const domLabel = _getDomainLabel(dom) || dom;
      const loadToast = showToast(`Scoring patient on ${domLabel}…`, 'loading', 0);
      try {
        await fetchCohort(state.selectedFup, dom);
        const res = await submitPrediction(
          events, state.selectedHorizon, state.maxFup, state.selectedTask, state.maxFup, false
        );
        if (state.bgFetchToken !== currentToken) {
          loadToast?.remove();
          return;
        }
        state.predictionsByDomain[dom][state.maxFup] = res;
        if (Array.isArray(res.trajectory)) {
          state.trajectoriesByDomain[dom] = res.trajectory;
        }
        loadToast?.remove();
        _updateFupBackgroundStatus();
        applyActiveTaskAndHorizon(false);

        const priorFups = state.fupSteps.filter(f => f !== state.selectedFup).sort((a, b) => b - a);
        if (priorFups.length > 0) {
          _precomputeEarlierFupsInBackground(
            events, dom, state.selectedTask, state.selectedHorizon, priorFups, state.maxFup, currentToken
          );
        }
      } catch (err) {
        loadToast?.remove();
        showToast(`Specialist prediction failed: ${err.message}`, 'error');
      }
      return;
    }

    const tPred = maxFupPred.tasks?.[state.selectedTask];
    if (tPred && tPred.ig_computed === false) {
      try {
        const updated = await submitPrediction(
          events, state.selectedHorizon, state.maxFup, state.selectedTask, state.maxFup, true
        );
        if (state.bgFetchToken !== currentToken) return;
        state.predictionsByDomain[dom][state.maxFup] = updated;
        applyActiveTaskAndHorizon(true);
      } catch (err) {
        console.warn('[AIIDKIT] Lazy IG fetch error:', err);
      }
    }
  }

  async function setSelectedFup(newFup) {
    const targetFup = parseInt(newFup, 10);
    if (isNaN(targetFup) || !state.fupSteps.includes(targetFup)) return;
    state.selectedFup = targetFup;
    _updateFupLabelsAndBadges();

    const dom = _getTaskDomain(state.selectedTask);
    try {
      await fetchCohort(targetFup, dom);
    } catch (err) {
      console.warn('[AIIDKIT] Could not fetch cohort for FUP', targetFup, err);
    }

    applyActiveTaskAndHorizon(true);
    _ensureDomainAndTaskPredictionIfNeeded();
  }

  function applyActiveTaskAndHorizon(smoothUmap = true) {
    const dom = _getTaskDomain(state.selectedTask);
    const taskCohort = getActiveTaskCohort();
    const activeCohort = taskCohort?.horizons?.[state.selectedHorizon];
    const activePred = getActiveTaskPrediction();
    const C = window.AIIDKIT_CHARTS;

    _updateFupLabelsAndBadges();

    if (activePred) {
      updateRiskBadge(activePred.risk_category);
      updateRiskScoreCards(activePred.risk_scores, activePred.calibrated_risks, state.selectedHorizon);
      C?.highlightCluster?.(activePred.cluster);
      renderTimeline(activePred.events_with_scores || []);
      const narr = document.getElementById('narrative-text');
      if (narr) narr.textContent = activePred.narrative || '\u2014';
    } else {
      updateRiskScoreCards(null, null, state.selectedHorizon);
    }

    if (state.chartsInitialized && C && activeCohort) {
      const sameManifold = smoothUmap && state.renderedUmapDomain === dom;
      if (sameManifold && typeof C.updateUMAPCohort === 'function') {
        C.updateUMAPCohort(activeCohort, taskCohort.global_umap_bounds);
      } else {
        C.initUMAP(activeCohort, taskCohort.global_umap_bounds);
        state.renderedUmapDomain = dom;
      }
      if (activePred) {
        C.updateUMAPWithPatient(
          activePred.umap_x,
          activePred.umap_y,
          activePred,
          _getActiveTrajectory(),
          state.selectedFup,
          state.showTrajectory,
          (clickedFup) => setSelectedFup(clickedFup)
        );
      }
      C.updateHistogramData(activeCohort.risk_distribution, state.selectedTask);
      if (activePred) {
        C.updateHistogramWithPatient(activePred.risk_score);
      }
      C.updateClusterProfiles(activeCohort.cluster_profiles);
      if (activePred) {
        C.highlightCluster(activePred.cluster);
      }
    }
  }

  async function setSelectedTask(newTask) {
    if (!newTask || state.selectedTask === newTask) return;
    const prevDom = _getTaskDomain(state.selectedTask);
    const nextDom = _getTaskDomain(newTask);
    state.selectedTask = newTask;

    const sel1 = document.getElementById('task-select');
    if (sel1 && sel1.value !== newTask) sel1.value = newTask;

    const horizonChanged = _syncHorizonForTask(newTask);
    if (horizonChanged || prevDom !== nextDom) {
      renderHorizonCards();
    }

    try {
      await fetchCohort(state.selectedFup, nextDom);
    } catch (err) {
      console.warn('[AIIDKIT] Cohort load failed when switching task:', err);
    }

    _updateFupBackgroundStatus();
    applyActiveTaskAndHorizon(prevDom === nextDom);
    await _ensureDomainAndTaskPredictionIfNeeded();
  }

  async function handleAnalyse(events) {
    if (!events?.length) {
      showToast('Please add at least one clinical event.', 'error');
      return;
    }

    state.currentPatientRawEvents = events;
    const { fup: autoFup, maxDay, fupSteps } = calculateAutoFup(events);
    const activeDom = _getTaskDomain(state.selectedTask);
    _syncHorizonForTask(state.selectedTask);

    state.bgFetchToken += 1;
    const currentToken = state.bgFetchToken;

    state.maxFup = autoFup;
    state.patientMaxDay = maxDay;
    state.fupSteps = fupSteps;
    state.selectedFup = autoFup;
    state.predictionsByDomain = { [activeDom]: {} };
    state.trajectoriesByDomain = {};

    const loadToast = showToast('Analysing patient sequence…', 'loading', 0);

    try {
      await fetchCohort(autoFup, activeDom);

      const result = await submitPrediction(events, state.selectedHorizon, autoFup, state.selectedTask, autoFup, false);
      if (state.bgFetchToken !== currentToken) return;

      state.predictionsByDomain[activeDom][autoFup] = result;
      state.trajectoriesByDomain[activeDom] = Array.isArray(result.trajectory) ? result.trajectory : [];

      renderHorizonCards();
      renderFupSliderBar();
      showResults();
      applyActiveTaskAndHorizon(false);

      if (loadToast) {
        loadToast.style.cssText += 'opacity:0;transform:translateX(40px);transition:opacity 0.3s,transform 0.3s;';
        setTimeout(() => loadToast.remove(), 350);
      }

      showToast('Analysis updated!', 'success');
      window.scrollTo({ top: 0, behavior: 'smooth' });

      const priorFups = fupSteps.filter(f => f < autoFup).sort((a, b) => b - a);
      if (priorFups.length > 0) {
        _precomputeEarlierFupsInBackground(
          events, activeDom, state.selectedTask, state.selectedHorizon, priorFups, autoFup, currentToken
        );
      }
    } catch (err) {
      loadToast?.remove();
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

    const savedTheme = localStorage.getItem('theme');
    const systemPref = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
    _setTheme(savedTheme || systemPref || 'light');

    toggleBtn.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme');
      _setTheme(currentTheme === 'light' ? 'dark' : 'light');
    });
  }

  function _setTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    localStorage.setItem('theme', theme);

    const toggleBtn = document.getElementById('theme-toggle');
    if (toggleBtn) {
      if (theme === 'light') {
        toggleBtn.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"></path></svg>
          <span>Dark theme</span>
        `;
        toggleBtn.setAttribute('title', 'Switch to dark theme');
      } else {
        toggleBtn.innerHTML = `
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="5"></circle><line x1="12" y1="1" x2="12" y2="3"></line><line x1="12" y1="21" x2="12" y2="23"></line><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"></line><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"></line><line x1="1" y1="12" x2="3" y2="12"></line><line x1="21" y1="12" x2="23" y2="12"></line><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"></line><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"></line></svg>
          <span>Light theme</span>
        `;
        toggleBtn.setAttribute('title', 'Switch to light theme');
      }
    }

    window.AIIDKIT_CHARTS?.updateThemeColors?.();
    const activePred = getActiveTaskPrediction();
    if (activePred) {
      renderTimeline(activePred.events_with_scores || []);
    }
  }

  // ================================================================
  // EVENT LISTENERS
  // ================================================================

  function _setupListeners() {
    document.getElementById('task-select')?.addEventListener('change', (e) => {
      setSelectedTask(e.target.value);
    });

    const fupSlider = document.getElementById('fup-slider');
    fupSlider?.addEventListener('input', (e) => {
      const idx = parseInt(e.target.value, 10);
      const targetFup = state.fupSteps[idx];
      if (targetFup !== undefined && targetFup !== state.selectedFup) {
        setSelectedFup(targetFup);
      }
    });

    const fupTrajToggle = document.getElementById('fup-trajectory-toggle');
    fupTrajToggle?.addEventListener('click', () => {
      state.showTrajectory = !state.showTrajectory;
      fupTrajToggle.classList.toggle('active', state.showTrajectory);
      const activePred = getActiveTaskPrediction();
      if (activePred && window.AIIDKIT_CHARTS) {
        window.AIIDKIT_CHARTS.updateUMAPWithPatient(
          activePred.umap_x,
          activePred.umap_y,
          activePred,
          _getActiveTrajectory(),
          state.selectedFup,
          state.showTrajectory,
          (clickedFup) => setSelectedFup(clickedFup)
        );
      }
    });

    document.getElementById('header-brand')?.addEventListener('click', resetAnalysis);
    document.getElementById('header-new-patient-btn')?.addEventListener('click', resetAnalysis);
    document.getElementById('reset-analysis-btn')?.addEventListener('click', resetAnalysis);

    document.getElementById('sidebar-collapsed-rail')?.addEventListener('click', () => setSidebarCollapsed(false, true));
    document.getElementById('sidebar-collapse-btn')?.addEventListener('click', () => setSidebarCollapsed(true, true));

    document.getElementById('umap-toggle-cluster')?.addEventListener('click', () =>
      window.AIIDKIT_CHARTS?.toggleUMAPColorMode?.('cluster'));
    document.getElementById('umap-toggle-risk')?.addEventListener('click', () =>
      window.AIIDKIT_CHARTS?.toggleUMAPColorMode?.('risk'));
    document.getElementById('umap-toggle-imminence')?.addEventListener('click', () =>
      window.AIIDKIT_CHARTS?.toggleUMAPColorMode?.('imminence'));

    const btnList   = document.getElementById('timeline-toggle-list');
    const btnGraph  = document.getElementById('timeline-toggle-graph');
    const wrapList  = document.getElementById('timeline-list-wrapper');
    const wrapGraph = document.getElementById('timeline-graph-outer');

    btnList?.addEventListener('click', () => {
      btnList.classList.add('active');
      btnGraph?.classList.remove('active');
      wrapList?.classList.remove('hidden');
      wrapGraph?.classList.add('hidden');
    });

    btnGraph?.addEventListener('click', () => {
      btnGraph.classList.add('active');
      btnList?.classList.remove('active');
      wrapGraph?.classList.remove('hidden');
      wrapList?.classList.add('hidden');
      const tlCtx = _getUnifiedTimelineContext();
      if (tlCtx && tlCtx.fullEvents.length) {
        window.AIIDKIT_CHARTS?.drawRiskTimelineGraph?.(
          tlCtx.fullEvents,
          tlCtx.finalFullScore,
          state.selectedTask,
          { cutoffFup: tlCtx.cutoffFup, baselineScore: tlCtx.baselineScore }
        );
      }
    });

    document.getElementById('analyze-form-btn')?.addEventListener('click', () =>
      handleAnalyse(window.AIIDKIT_FORM?.getFormEvents?.() || []));

    document.querySelectorAll('.example-btn').forEach(btn => {
      btn.addEventListener('click', async () => {
        const type = btn.dataset.example;
        clearPredictionResults();
        document.getElementById('tab-btn-form')?.click();
        await window.AIIDKIT_FORM?.loadExample?.(type);
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
      state.selectedTask = state.config.task || 'infection_bacteria';
      state.selectedHorizon = state.config.default_horizon || _getAvailableHorizons(state.selectedTask)[0] || 90;
      _syncHorizonForTask(state.selectedTask);

      if (Array.isArray(state.config.available_tasks) && state.config.available_tasks.length > 0) {
        const sel = document.getElementById('task-select');
        if (sel) {
          const hasGroups = state.config.available_tasks.some(t => t.group);
          if (hasGroups) {
            const groups = [];
            const groupMap = new Map();
            state.config.available_tasks.forEach(t => {
              const g = t.group || 'Other';
              if (!groupMap.has(g)) {
                groupMap.set(g, []);
                groups.push(g);
              }
              groupMap.get(g).push(t);
            });
            sel.innerHTML = groups
              .map(g => `<optgroup label="${_escHtml(g)}">${groupMap.get(g).map(t => `<option value="${_escHtml(t.key)}">${_escHtml(t.label)}</option>`).join('')}</optgroup>`)
              .join('');
          } else {
            sel.innerHTML = state.config.available_tasks
              .map(t => `<option value="${_escHtml(t.key)}">${_escHtml(t.label)}</option>`)
              .join('');
          }
          sel.value = state.selectedTask;
        }
      }

      renderHorizonCards();
      state.selectedFup = state.config.default_fup || 30;
      window.AIIDKIT_FORM?.initForm?.(state.config.vocabulary);
      await fetchCohort(state.selectedFup, _getTaskDomain(state.selectedTask));

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
