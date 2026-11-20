/**
 * AIIDKIT — charts.js
 * All data visualisations: D3 UMAP scatter, Chart.js KM curves,
 * attribution bar chart, risk histogram, and cluster profiles table.
 * Exposes window.AIIDKIT_CHARTS
 */
'use strict';

window.AIIDKIT_CHARTS = (() => {

  // ---- Fluid font & formatting helpers ------------------------------
  function _escHtml(val) {
    return String(val ?? '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function _shortHorizonLabel(h) {
    const n = parseInt(h, 10);
    if (n >= 360 && n % 360 === 0) return `${n / 360}y`;
    if (n === 180) return '6m';
    return `${n}d`;
  }

  function _scoreTitle(taskKey) {
    return String(taskKey || 'infection_bacteria').startsWith('infection_')
      ? 'Infection Score'
      : 'Risk Score';
  }

  // Returns the current computed root font size in px (tracks CSS clamp).
  function _rootFontSize() {
    return parseFloat(getComputedStyle(document.documentElement).fontSize) || 14;
  }
  // Convert a rem fraction to pixels at the current root size.
  function _scaledPx(remFraction) {
    return Math.round(_rootFontSize() * remFraction);
  }

  // Chart.js global theme — base size tracks CSS fluid root
  Chart.defaults.color = '#7C8DB5';
  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.size = _scaledPx(0.78);

  // UMAP dot radii — set at initUMAP() from canvas size so they stay
  // visually proportional regardless of screen width/DPI.
  let _umapDotR = 4.5;   // normal cohort dots
  let _umapEventR = 6.5;  // event-positive dots (slightly larger)
  let _umapStarOuter = 13; // new-patient star outer radius
  let _umapStarInner = 5.5; // new-patient star inner radius

  // Module-level chart instances (preserved for update calls)
  let _umapG = null, _umapScaleX = null, _umapScaleY = null;
  let _umapCurScaleX = null, _umapCurScaleY = null;
  let _umapPatientG = null;
  let _umapColorMode = 'cluster';
  let _umapData = [];
  let _histChart = null;

  function _getThemeColors() {
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    return {
      grid: isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.04)',
      textMuted: isLight ? '#64748B' : '#7C8DB5',
      text: isLight ? '#0F172A' : '#E4E8F0',
    };
  }

  // ================================================================
  // UMAP SCATTER  (D3.js v7)
  // ================================================================

  let _umapViewBoxW = 420;
  let _lastGlobalBounds = null;
  let _lastPatientData = null;
  let _lastPatientCoords = null;
  let _lastTrajectory = null;
  let _lastSelectedFup = null;
  let _showTrajectory = true;
  let _onSelectFupCb = null;

  function _computeDotFill(d, mode) {
    if (mode === 'cluster') return d.cluster_color || '#888';
    if (mode === 'risk') {
      const t = d.risk_score;
      if (t < 0.5) {
        const u = t * 2;
        return `rgb(${Math.round(79 + 176 * u)},${Math.round(142 + 23 * u)},${Math.round(247 - 245 * u)})`;
      } else {
        const u = (t - 0.5) * 2;
        return `rgb(255,${Math.round(165 - 94 * u)},${Math.round(2 + 85 * u)})`;
      }
    }
    if (mode === 'imminence') {
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const startColor = isLight ? [200, 200, 200] : [120, 130, 150];
      const endColor = [255, 71, 87];
      const imminence = d.event_imminence || 0.0;
      const r = Math.round(startColor[0] + (endColor[0] - startColor[0]) * imminence);
      const g = Math.round(startColor[1] + (endColor[1] - startColor[1]) * imminence);
      const b = Math.round(startColor[2] + (endColor[2] - startColor[2]) * imminence);
      return `rgb(${r},${g},${b})`;
    }
    return d.cluster_color || '#888';
  }

  function _bindUMAPDotEvents(selection, container, tip) {
    selection
      .style('cursor', 'pointer')
      .on('mouseover', function (ev, d) {
        d3.select(this).attr('opacity', 1).attr('r', d.true_label === 1 ? _umapEventR * 1.25 : _umapDotR * 1.45);
        const bx = container.getBoundingClientRect();
        const tipW = 195;
        const ex = ev.clientX - bx.left;
        const ey = ev.clientY - bx.top;

        let summaryLine = '';
        if (d.days_to_event !== undefined && d.days_to_event !== null) {
          summaryLine = `Next event: ${d.days_to_event} days`;
          if (d.true_label === 1) {
            summaryLine += `<br><span style="font-weight: 500; font-size: 0.65rem; opacity: 0.85;">(within prediction horizon)</span>`;
          }
        } else {
          summaryLine = 'No event observed';
        }

        let featuresHtml = '';
        if (d.summary && d.summary !== 'Stable course' && d.summary !== 'Standard profile') {
          featuresHtml = `<div class="tooltip-features" style="font-size:0.65rem;color:var(--text-muted);margin-top:2px;">Features: ${_escHtml(d.summary)}</div>`;
        }

        tip.classed('visible', true)
          .style('left', `${ex + tipW > bx.width ? ex - tipW - 8 : ex + 10}px`)
          .style('top', `${ey - 12}px`)
          .html(`
            <div class="tooltip-id">${_escHtml(d.id)}</div>
            <div class="tooltip-risk">Score: <strong style="color:${_escHtml(d.cluster_color)}">${Math.round(d.risk_score * 100)}/100</strong></div>
            <div class="tooltip-risk" style="font-size: 0.65rem; color: var(--text-muted);">Prob: ${d.calibrated_risk != null ? (d.calibrated_risk * 100).toFixed(1) + '%' : (d.risk_score * 100).toFixed(1) + '%'}</div>
            <div class="tooltip-cluster" style="color:${_escHtml(d.cluster_color)}">${_escHtml(d.cluster_name)}</div>
            <div class="tooltip-summary" style="font-weight: 600;">${summaryLine}</div>
            ${featuresHtml}
          `);
      })
      .on('mouseout', function (ev, d) {
        d3.select(this).attr('opacity', d.true_label === 1 ? 1.0 : 0.65).attr('r', d.true_label === 1 ? _umapEventR : _umapDotR);
        tip.classed('visible', false);
      });
  }

  function initUMAP(cohortData, globalBounds) {
    const container = document.getElementById('umap-svg-container');
    if (!container) return;
    container.innerHTML = '';
    _umapData = cohortData.patients || [];
    if (globalBounds) _lastGlobalBounds = globalBounds;

    const W = container.clientWidth || 420;
    const H = container.clientHeight || 260;
    _umapViewBoxW = W;
    const _legW = Math.round(_rootFontSize() * 7.8);
    const m = { top: 14, right: _legW + 14, bottom: 14, left: 14 };
    const iW = W - m.left - m.right;
    const iH = H - m.top - m.bottom;

    _umapDotR = Math.max(3.5, Math.min(7, iH * 0.012));
    _umapEventR = Math.max(5.0, Math.min(10, iH * 0.016));
    _umapStarOuter = Math.max(11, Math.min(18, iH * 0.034));
    _umapStarInner = _umapStarOuter * 0.42;

    const bounds = _lastGlobalBounds || cohortData.global_umap_bounds;
    const xs = _umapData.map(d => d.umap_x);
    const ys = _umapData.map(d => d.umap_y);
    const minX = bounds?.min_x != null ? Math.min(bounds.min_x, d3.min(xs) ?? bounds.min_x) : d3.min(xs);
    const maxX = bounds?.max_x != null ? Math.max(bounds.max_x, d3.max(xs) ?? bounds.max_x) : d3.max(xs);
    const minY = bounds?.min_y != null ? Math.min(bounds.min_y, d3.min(ys) ?? bounds.min_y) : d3.min(ys);
    const maxY = bounds?.max_y != null ? Math.max(bounds.max_y, d3.max(ys) ?? bounds.max_y) : d3.max(ys);
    const px = ((maxX - minX) || 1) * 0.09;
    const py = ((maxY - minY) || 1) * 0.09;

    _umapScaleX = d3.scaleLinear().domain([minX - px, maxX + px]).range([0, iW]);
    _umapScaleY = d3.scaleLinear().domain([minY - py, maxY + py]).range([iH, 0]);
    _umapCurScaleX = _umapScaleX;
    _umapCurScaleY = _umapScaleY;

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('width', '100%').attr('height', '100%');

    svg.append('rect').attr('width', W).attr('height', H).attr('fill', 'var(--umap-bg)');

    const defs = svg.append('defs');
    defs.append('clipPath').attr('id', 'umap-clip')
      .append('rect').attr('x', -15).attr('y', -15).attr('width', iW + 30).attr('height', iH + 30);

    // Arrowhead marker for patient longitudinal trajectory segments
    defs.append('marker')
      .attr('id', 'umap-traj-arrow')
      .attr('viewBox', '0 -4 8 8')
      .attr('refX', 7)
      .attr('refY', 0)
      .attr('markerWidth', 5.5)
      .attr('markerHeight', 5.5)
      .attr('orient', 'auto')
      .append('path')
      .attr('d', 'M0,-3.2L7,0L0,3.2Z')
      .attr('fill', '#F59E0B');

    _umapG = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    const ptG = _umapG.append('g').attr('clip-path', 'url(#umap-clip)').attr('class', 'pts');

    // Semantic zoom: rescales X/Y coordinates without scaling dot radii, stars, or text
    const zoom = d3.zoom().scaleExtent([0.4, 20]).on('zoom', ev => {
      _umapCurScaleX = ev.transform.rescaleX(_umapScaleX);
      _umapCurScaleY = ev.transform.rescaleY(_umapScaleY);
      ptG.selectAll('circle.umap-pt')
        .attr('cx', d => _umapCurScaleX(d.umap_x))
        .attr('cy', d => _umapCurScaleY(d.umap_y));
      if (_umapPatientG) {
        _umapPatientG.selectAll('line.umap-traj-seg')
          .attr('x1', d => _umapCurScaleX(d.p1.umap_x))
          .attr('y1', d => _umapCurScaleY(d.p1.umap_y))
          .attr('x2', d => _umapCurScaleX(d.p2.umap_x))
          .attr('y2', d => _umapCurScaleY(d.p2.umap_y));
        _umapPatientG.selectAll('g.umap-traj-milestone')
          .attr('transform', d => `translate(${_umapCurScaleX(d.umap_x)},${_umapCurScaleY(d.umap_y)})`);
        _umapPatientG.selectAll('g.umap-active-star')
          .attr('transform', d => `translate(${_umapCurScaleX(d.umapX)},${_umapCurScaleY(d.umapY)})`);
        _umapPatientG.selectAll('circle.umap-pulse-circle')
          .attr('cx', d => _umapCurScaleX(d.umapX))
          .attr('cy', d => _umapCurScaleY(d.umapY));
      }
    });
    svg.call(zoom);

    const tip = d3.select(container).append('div').attr('class', 'umap-tooltip');
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';

    const circles = ptG.selectAll('circle.umap-pt')
      .data([..._umapData].sort((a, b) => (a.true_label || 0) - (b.true_label || 0)), d => d.id)
      .join('circle')
      .attr('class', 'umap-pt')
      .attr('cx', d => _umapCurScaleX(d.umap_x))
      .attr('cy', d => _umapCurScaleY(d.umap_y))
      .attr('r', d => d.true_label === 1 ? _umapEventR : _umapDotR)
      .attr('fill', d => _computeDotFill(d, _umapColorMode))
      .attr('opacity', d => d.true_label === 1 ? 1.0 : 0.65)
      .attr('stroke', d => d.true_label === 1 ? (isLight ? '#475569' : '#ffffff') : 'none')
      .attr('stroke-width', d => d.true_label === 1 ? 1.8 : 0);

    _bindUMAPDotEvents(circles, container, tip);

    _umapPatientG = _umapG.append('g').attr('clip-path', 'url(#umap-clip)');
    if (_lastPatientCoords) {
      updateUMAPWithPatient();
    }
    _drawUMAPLegend(_umapColorMode);
  }

  function updateUMAPCohort(cohortData, globalBounds) {
    if (!cohortData) return;
    if (globalBounds) _lastGlobalBounds = globalBounds;
    if (!_umapG || !_umapScaleX || !_umapScaleY) {
      initUMAP(cohortData, globalBounds);
      return;
    }
    const sx = _umapCurScaleX || _umapScaleX;
    const sy = _umapCurScaleY || _umapScaleY;
    _umapData = cohortData.patients || [];
    const container = document.getElementById('umap-svg-container');
    if (!container) return;
    let tip = d3.select(container).select('.umap-tooltip');
    if (tip.empty()) {
      tip = d3.select(container).append('div').attr('class', 'umap-tooltip');
    }

    const ptG = _umapG.select('g.pts');
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const strokeCol = isLight ? '#475569' : '#ffffff';

    const joined = ptG.selectAll('circle.umap-pt')
      .data([..._umapData].sort((a, b) => (a.true_label || 0) - (b.true_label || 0)), d => d.id)
      .join(
        enter => enter.append('circle')
          .attr('class', 'umap-pt')
          .attr('cx', d => sx(d.umap_x))
          .attr('cy', d => sy(d.umap_y))
          .attr('r', 0)
          .attr('fill', d => _computeDotFill(d, _umapColorMode))
          .attr('opacity', 0),
        update => update,
        exit => exit.transition().duration(250).attr('r', 0).attr('opacity', 0).remove()
      );

    _bindUMAPDotEvents(joined, container, tip);

    joined.transition().duration(380).ease(d3.easeCubicOut)
      .attr('cx', d => sx(d.umap_x))
      .attr('cy', d => sy(d.umap_y))
      .attr('r', d => d.true_label === 1 ? _umapEventR : _umapDotR)
      .attr('fill', d => _computeDotFill(d, _umapColorMode))
      .attr('opacity', d => d.true_label === 1 ? 1.0 : 0.65)
      .attr('stroke', d => d.true_label === 1 ? strokeCol : 'none')
      .attr('stroke-width', d => d.true_label === 1 ? 1.8 : 0);

    _drawUMAPLegend(_umapColorMode);
  }

  function _drawUMAPLegend(mode) {
    const container = document.getElementById('umap-svg-container');
    if (!container) return;
    const svg = d3.select(container).select('svg');
    if (svg.empty()) return;

    svg.select('.umap-legend-group').remove();

    const svgNode = svg.node();
    let viewBoxW = _umapViewBoxW || 420;
    if (svgNode && svgNode.viewBox && svgNode.viewBox.baseVal && svgNode.viewBox.baseVal.width > 0) {
      viewBoxW = svgNode.viewBox.baseVal.width;
    }

    const legW = Math.round(_rootFontSize() * 7.8);

    const legDotR = _umapDotR;
    const legStarOuter = legDotR * 1.3;
    const legStarInner = legStarOuter * 0.42;
    const legFontPx = _scaledPx(0.68);

    const rowH = Math.max(legStarOuter * 2 + 6, _scaledPx(1.55));
    const cy = rowH / 2;
    const symX = legStarOuter + 3;
    const textX = symX + legStarOuter + 7;

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const strokeColor = isLight ? '#475569' : '#ffffff';

    let colorItems = [];
    if (mode === 'cluster') {
      const hasMedium = _umapData && _umapData.some(d => d.cluster === 2 || d.cluster_name === 'Medium risk');
      colorItems = [
        { label: 'High risk', color: '#FF4757' },
      ];
      if (hasMedium) {
        colorItems.push({ label: 'Medium risk', color: '#FFA502' });
      }
      colorItems.push({ label: 'Low risk', color: '#00C9A7' });
    } else if (mode === 'risk') {
      colorItems = [
        { label: 'High risk', color: '#FF4757' },
        { label: 'Low risk', color: '#4F8EF7' },
      ];
    } else if (mode === 'imminence') {
      colorItems = [
        { label: 'Close', color: '#FF4757' },
        { label: 'Distant/No', color: '#7C8DB5' },
      ];
    }

    const hasTraj = _showTrajectory && Array.isArray(_lastTrajectory) && _lastTrajectory.length > 1;
    const numRows = colorItems.length + 2 + (hasTraj ? 1 : 0);
    const legHeight = numRows * rowH + 8;

    const lg = svg.append('g')
      .attr('class', 'umap-legend-group')
      .attr('transform', `translate(${viewBoxW - legW - 10}, 18)`);

    lg.append('rect')
      .attr('x', -4).attr('y', -4)
      .attr('width', legW).attr('height', legHeight)
      .attr('rx', 6)
      .attr('fill', 'var(--legend-bg)').attr('stroke', 'var(--legend-border)');

    function _legText(parent, label) {
      parent.append('text')
        .attr('x', textX).attr('y', cy)
        .attr('dominant-baseline', 'central')
        .attr('fill', 'var(--legend-text)')
        .attr('font-size', `${legFontPx}px`)
        .attr('font-family', "'Inter',sans-serif")
        .text(label);
    }

    colorItems.forEach((e, i) => {
      const g = lg.append('g').attr('transform', `translate(0,${i * rowH})`);
      g.append('circle')
        .attr('cx', symX).attr('cy', cy).attr('r', legDotR)
        .attr('fill', e.color).attr('opacity', 0.85);
      _legText(g, e.label);
    });

    const evRow = lg.append('g').attr('transform', `translate(0,${colorItems.length * rowH})`);
    evRow.append('circle')
      .attr('cx', symX).attr('cy', cy).attr('r', legDotR)
      .attr('fill', '#7C8DB5').attr('opacity', 0.65)
      .attr('stroke', strokeColor).attr('stroke-width', 1.8);
    _legText(evRow, 'Event observed');

    const stRow = lg.append('g').attr('transform', `translate(0,${(colorItems.length + 1) * rowH})`);
    stRow.append('path')
      .attr('d', _starPath(symX, cy, legStarOuter, legStarInner, 5))
      .attr('fill', '#FFD700').attr('stroke', 'none');
    _legText(stRow, 'Active stage');

    if (hasTraj) {
      const trRow = lg.append('g').attr('transform', `translate(0,${(colorItems.length + 2) * rowH})`);
      trRow.append('line')
        .attr('x1', symX - legStarOuter).attr('x2', symX + legStarOuter)
        .attr('y1', cy).attr('y2', cy)
        .attr('stroke', '#F59E0B').attr('stroke-width', 2.2);
      trRow.append('circle')
        .attr('cx', symX).attr('cy', cy).attr('r', Math.max(2.5, legDotR * 0.68))
        .attr('fill', '#FFD700').attr('stroke', strokeColor).attr('stroke-width', 1.2);
      _legText(trRow, 'Patient course');
    }
  }

  function updateUMAPWithPatient(umapX, umapY, patientData, trajectory, selectedFup, showTrajectory, onSelectFup) {
    if (umapX != null && umapY != null) {
      _lastPatientCoords = { umapX, umapY };
    }
    if (patientData) {
      _lastPatientData = patientData;
    }
    if (Array.isArray(trajectory)) {
      _lastTrajectory = trajectory;
    }
    if (selectedFup != null) {
      _lastSelectedFup = selectedFup;
    }
    if (typeof showTrajectory === 'boolean') {
      _showTrajectory = showTrajectory;
    }
    if (typeof onSelectFup === 'function') {
      _onSelectFupCb = onSelectFup;
    }

    const activeCoords = _lastPatientCoords || (patientData ? { umapX: patientData.umap_x, umapY: patientData.umap_y } : null);
    const activeData = patientData || _lastPatientData;
    const activeTraj = _lastTrajectory || activeData?.trajectory || [];
    const activeFup = _lastSelectedFup != null ? _lastSelectedFup : (activeData?.fup ?? 0);

    if (!activeCoords || !_umapPatientG || !_umapScaleX || !_umapScaleY) return;

    const sx = _umapCurScaleX || _umapScaleX;
    const sy = _umapCurScaleY || _umapScaleY;
    const px = sx(activeCoords.umapX);
    const py = sy(activeCoords.umapY);
    _umapPatientG.selectAll('*').remove();
    _umapPatientG.raise();

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const starStroke = isLight ? '#1E293B' : '#ffffff';
    const haloStroke = isLight ? 'rgba(255,255,255,0.88)' : 'rgba(15,17,21,0.88)';
    const pathColor = isLight ? '#D97706' : '#FBBF24';

    const container = document.getElementById('umap-svg-container');
    if (!container) return;
    let tip = d3.select(container).select('.umap-tooltip');
    if (tip.empty()) {
      tip = d3.select(container).append('div').attr('class', 'umap-tooltip');
    }

    // 1. Draw longitudinal patient trajectory across FUP landmarks if enabled
    if (_showTrajectory && activeTraj.length > 1) {
      const trajG = _umapPatientG.append('g').attr('class', 'umap-trajectory-layer');

      // Draw segment-by-segment lines (solid up to activeFup, dashed after activeFup)
      for (let i = 0; i < activeTraj.length - 1; i++) {
        const p1 = activeTraj[i];
        const p2 = activeTraj[i + 1];
        const x1 = sx(p1.umap_x);
        const y1 = sy(p1.umap_y);
        const x2 = sx(p2.umap_x);
        const y2 = sy(p2.umap_y);
        const isPast = p2.fup <= activeFup;

        // Subtle high-contrast halo under segment
        trajG.append('line')
          .datum({ p1, p2 })
          .attr('class', 'umap-traj-seg')
          .attr('x1', x1).attr('y1', y1)
          .attr('x2', x2).attr('y2', y2)
          .attr('stroke', haloStroke)
          .attr('stroke-width', isPast ? 5.2 : 3.8)
          .attr('stroke-linecap', 'round')
          .attr('opacity', isPast ? 0.75 : 0.45);

        trajG.append('line')
          .datum({ p1, p2 })
          .attr('class', 'umap-traj-seg')
          .attr('x1', x1).attr('y1', y1)
          .attr('x2', x2).attr('y2', y2)
          .attr('stroke', pathColor)
          .attr('stroke-width', isPast ? 2.6 : 1.9)
          .attr('stroke-dasharray', isPast ? 'none' : '4,4')
          .attr('stroke-linecap', 'round')
          .attr('opacity', isPast ? 0.95 : 0.65)
          .attr('marker-end', isPast ? 'url(#umap-traj-arrow)' : null);
      }

      // Draw clickable milestone nodes at each non-active FUP landmark
      activeTraj.forEach(pt => {
        if (pt.fup === activeFup) return;
        const mx = sx(pt.umap_x);
        const my = sy(pt.umap_y);
        const isPast = pt.fup < activeFup;
        const nodeR = Math.max(4.8, _umapDotR * 1.15);

        const mGroup = trajG.append('g')
          .datum(pt)
          .attr('class', 'umap-traj-milestone')
          .attr('transform', `translate(${mx},${my})`)
          .style('cursor', 'pointer')
          .attr('pointer-events', 'all');

        // Hit target
        mGroup.append('circle')
          .attr('cx', 0).attr('cy', 0).attr('r', 15)
          .attr('fill', 'transparent');

        const mCircle = mGroup.append('circle')
          .attr('cx', 0).attr('cy', 0)
          .attr('r', nodeR)
          .attr('fill', isPast ? '#FFD700' : (isLight ? '#FFFFFF' : '#1E222B'))
          .attr('stroke', isPast ? starStroke : pathColor)
          .attr('stroke-width', 2)
          .attr('opacity', isPast ? 0.95 : 0.78);

        // Compact day label next to milestone node
        const lblText = `${pt.fup}d`;
        mGroup.append('text')
          .attr('x', 0)
          .attr('y', -nodeR - 5)
          .attr('text-anchor', 'middle')
          .attr('font-family', "'Inter', sans-serif")
          .attr('font-size', `${_scaledPx(0.60)}px`)
          .attr('font-weight', '700')
          .attr('fill', isLight ? '#111827' : '#F3F4F6')
          .attr('stroke', haloStroke)
          .attr('stroke-width', 3)
          .attr('paint-order', 'stroke')
          .text(lblText);

        mGroup
          .on('mouseenter mouseover', function (ev) {
            mCircle.attr('r', nodeR * 1.3).attr('stroke-width', 2.8);
            const bx = container.getBoundingClientRect();
            const tipW = 215;
            const ex = ev.clientX - bx.left;
            const ey = ev.clientY - bx.top;
            const pData = activeData || {};
            const activeTask = pData.task || 'infection_bacteria';
            const activeHorizon = pData.selected_horizon || 90;
            const hKey = `${activeHorizon}d`;
            const stTask = pt.tasks?.[activeTask];
            const stScore = stTask?.risk_scores?.[hKey] != null ? Math.round(stTask.risk_scores[hKey] * 100) : '—';
            const stCalib = stTask?.calibrated_risks?.[hKey] != null ? (stTask.calibrated_risks[hKey] * 100).toFixed(1) + '%' : '—';

            tip.classed('visible', true)
              .style('left', `${ex + tipW > bx.width ? ex - tipW - 10 : ex + 12}px`)
              .style('top', `${ey - 15}px`)
              .html(`
                <div class="tooltip-id" style="color:#F59E0B; font-weight:700; font-size:0.76rem;">● Patient at Day ${pt.fup}</div>
                <div class="tooltip-risk" style="font-size:0.72rem;">${_scoreTitle(activeTask)}: <strong>${stScore}/100</strong> <span style="font-size:0.65rem; opacity:0.8;">(${_shortHorizonLabel(activeHorizon)})</span></div>
                <div class="tooltip-risk" style="font-size:0.66rem; color:var(--text-muted);">Calibrated prob: ${stCalib} · ${pt.n_events ?? '—'} events</div>
                <div class="tooltip-summary" style="font-weight:600; margin-top:3px; font-size:0.66rem; color:var(--accent);">Click to inspect Day ${pt.fup} stage</div>
              `);
          })
          .on('mouseleave mouseout', function () {
            mCircle.attr('r', nodeR).attr('stroke-width', 2);
            tip.classed('visible', false);
          })
          .on('click', function (ev) {
            ev.stopPropagation();
            tip.classed('visible', false);
            if (typeof _onSelectFupCb === 'function') {
              _onSelectFupCb(pt.fup);
            }
          });
      });
    }

    // 2. Draw pulsing halo and golden star at the active FUP stage
    const pulseR = _umapStarOuter * 1.08;
    const pulseEnd = _umapStarOuter * 2.5;

    _umapPatientG.append('circle')
      .datum(activeCoords)
      .attr('class', 'umap-pulse-circle')
      .attr('cx', px).attr('cy', py).attr('r', pulseR)
      .attr('fill', 'none').attr('stroke', '#FFD700').attr('stroke-width', 2)
      .attr('opacity', 0.8)
      .transition().duration(1000).ease(d3.easeExpOut)
      .attr('r', pulseEnd).attr('opacity', 0);

    const starGroup = _umapPatientG.append('g')
      .datum(activeCoords)
      .attr('class', 'umap-active-star')
      .attr('transform', `translate(${px},${py})`)
      .style('cursor', 'pointer')
      .attr('pointer-events', 'all');

    starGroup.append('circle')
      .attr('cx', 0)
      .attr('cy', 0)
      .attr('r', 22)
      .attr('fill', 'transparent')
      .attr('pointer-events', 'all');

    const starNode = starGroup.append('path')
      .attr('d', _starPath(0, 0, _umapStarOuter * 1.4, _umapStarInner * 1.4, 5))
      .attr('fill', '#FFD700')
      .attr('stroke', starStroke)
      .attr('stroke-width', 2)
      .attr('opacity', 0)
      .attr('pointer-events', 'none');

    starNode.transition().duration(350).attr('opacity', 1);

    if (activeTraj.length > 1) {
      starGroup.append('text')
        .attr('x', 0)
        .attr('y', -_umapStarOuter * 1.4 - 5)
        .attr('text-anchor', 'middle')
        .attr('font-family', "'Plus Jakarta Sans', 'Inter', sans-serif")
        .attr('font-size', `${_scaledPx(0.65)}px`)
        .attr('font-weight', '800')
        .attr('fill', isLight ? '#B45309' : '#FFD700')
        .attr('stroke', haloStroke)
        .attr('stroke-width', 3.5)
        .attr('paint-order', 'stroke')
        .attr('pointer-events', 'none')
        .text(`Day ${activeFup}`);
    }

    starGroup
      .on('mouseenter mouseover', function (ev) {
        starNode.attr('stroke-width', 3.5);
        const bx = container.getBoundingClientRect();
        const tipW = 235;
        const ex = ev.clientX - bx.left;
        const ey = ev.clientY - bx.top;

        const pData = activeData || {};
        const activeTask = pData.task || 'infection_bacteria';
        const activeHorizon = pData.selected_horizon || 90;
        const horizonKey = `${activeHorizon}d`;
        const scoreVal = pData.risk_scores?.[horizonKey] ?? pData.risk_score ?? 0;
        const scoreInt = Math.round(scoreVal * 100);
        const catName = _escHtml(pData.risk_category || pData.cluster_name || 'Active Patient');
        const catColor = _escHtml(pData.risk_color || pData.cluster_color || (catName === 'Medium risk' ? '#FFA502' : catName === 'Low risk' ? '#00C9A7' : '#FF4757'));
        const horizonLabel = _shortHorizonLabel(activeHorizon);

        let nextEv = 'Stable course (no imminent event)';
        if (pData.days_to_event != null) {
          nextEv = `Next event: ~${pData.days_to_event} days`;
        }

        let summaryHtml = '';
        if (pData.events_with_scores && pData.events_with_scores.length > 0) {
          const topEvs = pData.events_with_scores
            .slice(0, 3)
            .map(e => `${_escHtml(e.entity || '')}: ${_escHtml(e.attribute || '')} (${_escHtml(e.value || '')})`)
            .join('<br>');
          summaryHtml = `<div class="tooltip-features" style="font-size:0.65rem;color:var(--text-muted);margin-top:4px;line-height:1.3;"><strong>Key events (up to Day ${activeFup}):</strong><br>${topEvs}</div>`;
        }

        tip.classed('visible', true)
          .style('left', `${ex + tipW > bx.width ? ex - tipW - 10 : ex + 12}px`)
          .style('top', `${ey - 15}px`)
          .html(`
            <div class="tooltip-id" style="color:#FFD700; font-weight:700; font-size:0.78rem;">★ Active Patient (Day ${activeFup})</div>
            <div class="tooltip-risk" style="font-size:0.74rem;">${_scoreTitle(activeTask)}: <strong style="color:${catColor}">${scoreInt}/100</strong> <span style="font-size:0.68rem; opacity:0.8;">(${horizonLabel})</span></div>
            <div class="tooltip-cluster" style="color:${catColor}; font-weight:600; font-size:0.72rem;">Subgroup: ${catName}</div>
            <div class="tooltip-summary" style="font-weight:600; margin-top:2px; font-size:0.70rem;">${nextEv}</div>
            ${summaryHtml}
          `);
      })
      .on('mouseleave mouseout', function () {
        starNode.attr('stroke-width', 2);
        tip.classed('visible', false);
      });

    _drawUMAPLegend(_umapColorMode);
  }

  function _starPath(cx, cy, outerR, innerR, pts) {
    let path = '';
    const step = Math.PI / pts;
    for (let i = 0; i < 2 * pts; i++) {
      const r = i % 2 === 0 ? outerR : innerR;
      const a = i * step - Math.PI / 2;
      path += `${i === 0 ? 'M' : 'L'}${(cx + r * Math.cos(a)).toFixed(2)},${(cy + r * Math.sin(a)).toFixed(2)} `;
    }
    return path + 'Z';
  }

  function toggleUMAPColorMode(mode) {
    _umapColorMode = mode;
    document.getElementById('umap-toggle-cluster')?.classList.toggle('active', mode === 'cluster');
    document.getElementById('umap-toggle-risk')?.classList.toggle('active', mode === 'risk');
    document.getElementById('umap-toggle-imminence')?.classList.toggle('active', mode === 'imminence');

    _drawUMAPLegend(mode);
    if (!_umapG || !_umapData.length) return;

    _umapG.selectAll('circle.umap-pt')
      .transition().duration(350)
      .attr('fill', d => _computeDotFill(d, mode));
  }

  // ================================================================
  // RISK HISTOGRAM  (Chart.js v4)
  // ================================================================

  function initHistogram(distribution, taskKey = 'infection_bacteria') {
    const canvas = document.getElementById('hist-canvas');
    if (!canvas) return;
    if (_histChart) { _histChart.destroy(); _histChart = null; }

    const { binLabels, counts } = _binValues(distribution.values, 10);
    const barColors = binLabels.map((_, i) => {
      const t = i / (binLabels.length - 1);
      return `rgba(${Math.round(79 + 176 * t)},${Math.round(142 - 71 * t)},${Math.round(247 - 160 * t)},0.78)`;
    });

    const histLinePlugin = {
      id: 'histLine',
      afterDatasetsDraw(chart) {
        const opt = chart.options.plugins.histLine;
        if (opt.riskScore == null) return;
        const { ctx, scales } = chart;
        const xPx = scales.x.left + opt.riskScore * (scales.x.right - scales.x.left);
        const yTop = scales.y.top;
        const yBot = scales.y.bottom;

        // Retrieve active patient badge colors dynamically from CSS variables
        const style = getComputedStyle(document.documentElement);
        const textCol = style.getPropertyValue('--active-patient-text').trim() || '#FFD700';
        const bgCol = style.getPropertyValue('--active-patient-bg').trim() || 'rgba(255, 215, 0, 0.12)';
        const borderCol = style.getPropertyValue('--active-patient-border').trim() || 'rgba(255, 215, 0, 0.35)';

        ctx.save();

        // 1. Draw the vertical dashed line
        ctx.beginPath();
        ctx.moveTo(xPx, yTop);
        ctx.lineTo(xPx, yBot);
        ctx.strokeStyle = textCol;
        ctx.lineWidth = 2;
        ctx.setLineDash([4, 3]);
        ctx.stroke();
        ctx.setLineDash([]);

        // 2. Setup text styling for badge — scale with root font size
        const badgeFontPx = _scaledPx(0.62);
        ctx.font = `700 ${badgeFontPx}px Inter, sans-serif`;
        ctx.textBaseline = 'alphabetic';
        const text = 'Active Patient';
        const textWidth = ctx.measureText(text).width;

        // 3. Compute badge dimensions and position
        const rectW = textWidth + 14;
        const rectH = 16;
        const rectY = yTop - 22;
        let rectX = xPx - rectW / 2;

        // Clamp to prevent horizontal clipping
        if (rectX < scales.x.left) {
          rectX = scales.x.left;
        } else if (rectX + rectW > scales.x.right) {
          rectX = scales.x.right - rectW;
        }

        // 4. Draw subtle rounded Swiss badge container
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(rectX, rectY, rectW, rectH, 4);
        } else {
          ctx.rect(rectX, rectY, rectW, rectH);
        }
        ctx.fillStyle = bgCol;
        ctx.fill();
        ctx.lineWidth = 1;
        ctx.strokeStyle = borderCol;
        ctx.stroke();

        // 5. Draw text inside badge
        ctx.fillStyle = textCol;
        ctx.textAlign = 'center';
        ctx.fillText(text, rectX + rectW / 2, rectY + 11);

        // 6. Draw pointer triangle (pointing from bottom of badge to the dashed line)
        ctx.beginPath();
        ctx.moveTo(xPx - 4, yTop - 6);
        ctx.lineTo(xPx + 4, yTop - 6);
        ctx.lineTo(xPx, yTop - 1);
        ctx.closePath();
        ctx.fillStyle = borderCol;
        ctx.fill();

        ctx.restore();
      },
    };

    const tc = _getThemeColors();
    _histChart = new Chart(canvas, {
      type: 'bar',
      data: { labels: binLabels, datasets: [{ label: 'Patients', data: counts, backgroundColor: barColors, borderRadius: 4 }] },
      plugins: [histLinePlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
        layout: { padding: { top: 25 } },
        plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(6,13,31,0.92)', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, callbacks: { label: ctx => ` ${ctx.parsed.y} patients` } }, histLine: { riskScore: null } },
        scales: {
          x: { title: { display: true, text: _scoreTitle(taskKey).toLowerCase(), color: tc.textMuted, font: { size: _scaledPx(0.71) } }, grid: { display: false }, ticks: { color: tc.textMuted, font: { size: _scaledPx(0.71) }, maxTicksLimit: 6 } },
          y: { title: { display: true, text: 'N patients', color: tc.textMuted, font: { size: _scaledPx(0.71) } }, grid: { color: tc.grid }, ticks: { color: tc.textMuted, font: { size: _scaledPx(0.71) } } },
        },
      },
    });
  }

  function updateHistogramWithPatient(riskScore) {
    if (!_histChart) return;
    _histChart.options.plugins.histLine = { riskScore: riskScore };
    _histChart.update('none');
  }

  function updateHistogramData(distribution, taskKey = 'infection_bacteria') {
    if (!_histChart) return;
    const { binLabels, counts } = _binValues(distribution.values, 10);
    _histChart.data.labels = binLabels;
    _histChart.data.datasets[0].data = counts;
    _histChart.options.scales.x.title.text = _scoreTitle(taskKey).toLowerCase();
    _histChart.update();
  }

  function _binValues(values, nBins) {
    const counts = new Array(nBins).fill(0);
    const step = 1 / nBins;
    for (const v of values) {
      if (typeof v !== 'number') continue;
      const idx = Math.min(Math.floor(v / step), nBins - 1);
      counts[idx]++;
    }
    return {
      binLabels: Array.from({ length: nBins }, (_, i) => `${Math.round(i * step * 100)}-${Math.round((i + 1) * step * 100)}`),
      counts,
    };
  }

  // ================================================================
  // CLUSTER PROFILES TABLE
  // ================================================================

  function initClusterProfiles(profiles) {
    const container = document.getElementById('cluster-profiles-container');
    if (!container) return;
    container.innerHTML = '';

    if (!profiles) {
      container.innerHTML = '<div style="text-align:center;color:var(--text-muted);padding:20px;">No cluster profiles available</div>';
      return;
    }

    const formatFeat = feat => String(feat ?? '').replace(/^Infection\b/, 'Previous infection');

    const renderTraitCard = (f) => {
      const fullFeat = formatFeat(f.feature);
      const parts = fullFeat.split(' - ');
      const entity = _escHtml(parts[0] || 'Feature');
      const attr = _escHtml(parts.slice(1).join(' - ') || fullFeat);
      const val = _escHtml(f.value);
      const isPos = f.score >= 0;
      const cardClass = isPos ? 'pos' : 'neg';

      return `<div class="profile-trait-card ${cardClass}" title="${entity} &ndash; ${attr}: ${val}">
        <div class="trait-header">
          <span class="trait-entity" title="${entity}">${entity}</span>
          <span class="trait-val" title="${val}">${val}</span>
        </div>
        <div class="trait-attr" title="${attr}">${attr}</div>
      </div>`;
    };

    try {
      Object.entries(profiles)
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .forEach(([id, prof]) => {
          let staticTags = '';
          let recentTags = '';

          if (Array.isArray(prof.top_features)) {
            // Backward compatibility for old flat array format
            staticTags = prof.top_features.map(renderTraitCard).join('') || '<span style="color:#3D4F72">—</span>';
            recentTags = '<span style="color:#3D4F72">—</span>';
          } else if (prof.top_features && typeof prof.top_features === 'object') {
            // New dictionary format with static and recent subsets
            staticTags = (prof.top_features.static || []).map(renderTraitCard).join('') || '<span style="color:#3D4F72">—</span>';
            recentTags = (prof.top_features.recent || []).map(renderTraitCard).join('') || '<span style="color:#3D4F72">—</span>';
          } else {
            staticTags = '<span style="color:#3D4F72">—</span>';
            recentTags = '<span style="color:#3D4F72">—</span>';
          }

          const card = document.createElement('div');
          card.className = 'cluster-profile-card';
          card.dataset.clusterId = id;
          card.style.setProperty('--card-color', prof.color);
          const evRate = prof.event_rate != null ? `${(prof.event_rate * 100).toFixed(0)}%` : '—';

          card.innerHTML = `
            <div class="cluster-profile-header">
              <div class="cluster-name-cell">
                <span class="cluster-dot" style="background:${prof.color}"></span>
                <span class="cluster-profile-name">${prof.name}</span>
                <span class="cluster-n-separator">&ndash;</span>
                <span class="cluster-n-label">N</span>
                <span class="cluster-n-value">${prof.n_patients ?? 0}</span>
                <span class="active-patient-badge">Active Patient</span>
              </div>
              <div class="cluster-metric-item">
                <span class="metric-label">Event Rate</span>
                <span class="event-rate-pill" style="background: ${prof.color}1f; color: ${prof.color}; border: 1px solid ${prof.color}33;">${evRate}</span>
              </div>
            </div>
            <div class="cluster-profile-drivers">
              <div class="drivers-column">
                <div class="column-title">Baseline</div>
                <div class="tags-container">${staticTags}</div>
              </div>
              <div class="drivers-column">
                <div class="column-title">Recent</div>
                <div class="tags-container">${recentTags}</div>
              </div>
            </div>
          `;
          container.appendChild(card);
        });
    } catch (err) {
      console.error('[AIIDKIT] Error rendering cluster profiles:', err);
      container.innerHTML = '<div style="text-align:center;color:var(--risk-high);padding:20px;">Error loading cluster profiles</div>';
    }
  }

  function highlightCluster(clusterId) {
    document.querySelectorAll('.cluster-profile-card').forEach(card => {
      card.classList.toggle('active-cluster', card.dataset.clusterId === String(clusterId));
    });
  }

  function resizeCharts() {
    if (_histChart) _histChart.resize();
  }

  function updateThemeColors() {
    const tc = _getThemeColors();
    Chart.defaults.color = tc.textMuted;
    Chart.defaults.font.size = _scaledPx(0.78);

    if (_histChart) {
      const axFontPx = _scaledPx(0.71);
      _histChart.options.scales.x.ticks.color = tc.textMuted;
      _histChart.options.scales.x.ticks.font = { size: axFontPx };
      _histChart.options.scales.x.title.color = tc.textMuted;
      _histChart.options.scales.x.title.font = { size: axFontPx };
      _histChart.options.scales.y.ticks.color = tc.textMuted;
      _histChart.options.scales.y.ticks.font = { size: axFontPx };
      _histChart.options.scales.y.title.color = tc.textMuted;
      _histChart.options.scales.y.title.font = { size: axFontPx };
      _histChart.options.scales.y.grid.color = tc.grid;
      _histChart.update();
    }

    // Update UMAP stroke colors on theme change
    if (_umapG) {
      const isLight = document.documentElement.getAttribute('data-theme') === 'light';
      const strokeColor = isLight ? '#475569' : '#ffffff';

      _umapG.selectAll('circle.umap-pt')
        .attr('stroke', d => d.true_label === 1 ? strokeColor : 'none');

      if (_umapPatientG) {
        _umapPatientG.selectAll('path')
          .attr('stroke', strokeColor);
      }

      _drawUMAPLegend(_umapColorMode);
    }
  }

  // ================================================================
  // RISK TIMELINE GRAPH (MODE 2)
  // ================================================================

  function drawRiskTimelineGraph(eventsWithScores, finalScore, taskKey = 'infection_bacteria') {
    const container = document.getElementById('timeline-graph-svg');
    if (!container) return;

    // Clear SVG container
    d3.select(container).selectAll('*').remove();

    // Track active pinned node for toggle-off re-clicking
    let activePinnedData = null;

    // Append compact interactive popover card to document.body (always centered at left:50%)
    d3.select('body').selectAll('.timeline-graph-tooltip').remove();
    const tip = d3.select('body').append('div')
      .attr('class', 'umap-tooltip timeline-graph-tooltip')
      .style('position', 'fixed')
      .style('z-index', '999999')
      .style('pointer-events', 'auto')
      .style('left', '50%')
      .style('transform', 'translateX(-50%)')
      .style('min-width', '320px')
      .style('max-width', '450px')
      .style('width', 'max-content')
      .style('max-height', '220px')
      .style('overflow-y', 'auto')
      .style('transition', 'all 0.22s cubic-bezier(0.16, 1, 0.3, 1)')
      .style('box-shadow', '0 12px 32px rgba(0,0,0,0.3)');

    if (!eventsWithScores || eventsWithScores.length === 0) {
      d3.select(container).append('div')
        .style('padding', '40px 16px')
        .style('text-align', 'center')
        .style('color', 'var(--text-muted)')
        .style('font-style', 'italic')
        .style('font-size', '0.78rem')
        .text('Submit a patient to view the cumulative risk timeline graph.');
      return;
    }

    // Target Final Risk Score (e.g. 51 for 51/100) shown in Risk Assessment panel
    const endScore = Math.round((finalScore || 0.5) * 100);

    const STATIC_ENTITIES = new Set([
      'Patient info', 'Donor info', 'Mismatch info', 'Transplant info',
      'Serology info', 'Patient serology', 'Immunology', 'History', 'Patient', 'Donor',
    ]);
    const isStaticEvent = ev => {
      const d = ev.days_since_tpx ?? 0;
      return d <= 0 || STATIC_ENTITIES.has(ev.entity);
    };

    const postBaselineEvents = eventsWithScores.filter(ev => !isStaticEvent(ev));

    // Sum of raw attribution deltas
    let totalRawDeltaSum = 0;
    postBaselineEvents.forEach(ev => { totalRawDeltaSum += (ev.score || 0) * 100; });

    // Determine realistic baseline risk R0 (0-100 scale)
    let baseRisk;
    const rawBaseline = Math.round(endScore - totalRawDeltaSum);

    if (rawBaseline >= 10 && rawBaseline <= 85) {
      baseRisk = rawBaseline;
    } else {
      baseRisk = endScore >= 50
        ? Math.max(15, Math.min(45, Math.round(endScore * 0.4)))
        : Math.max(10, Math.min(35, Math.round(endScore * 0.6 + 5)));
    }

    // Required total delta sum across post-baseline timeline to hit endScore
    const reqTotalDelta = endScore - baseRisk;

    // Normalization / Scaling factor kScale
    const kScale = Math.abs(totalRawDeltaSum) > 1e-4
      ? (reqTotalDelta / totalRawDeltaSum)
      : 1.0;

    // Attach normalized/scaled scores to all events
    const eventsWithScaledScores = eventsWithScores.map(ev => {
      const isPost = !isStaticEvent(ev);
      const rawSc = (ev.score || 0) * 100;
      const scaledSc = isPost ? rawSc * kScale : rawSc;
      return {
        ...ev,
        scaled_score: scaledSc / 100,
        scaled_delta: scaledSc
      };
    });

    // Group events chronologically by days_since_tpx
    const daysMap = new Map();
    eventsWithScaledScores.forEach(ev => {
      const d = isStaticEvent(ev) ? 0 : ev.days_since_tpx;
      if (!daysMap.has(d)) daysMap.set(d, []);
      daysMap.get(d).push(ev);
    });

    const sortedDays = Array.from(daysMap.keys()).sort((a, b) => a - b);
    const postDays = sortedDays.filter(d => d > 0);

    // Construct timeline points with guaranteed convergence to endScore
    const timelineData = [];
    let currentRisk = baseRisk;

    const minDay = sortedDays[0];
    const startDay = minDay <= 0 ? minDay - 6 : -6;

    // Baseline point (day 0)
    timelineData.push({
      day: startDay,
      riskBefore: baseRisk,
      riskAfter: baseRisk,
      delta: 0,
      events: daysMap.get(0) || [],
      isBaseline: true
    });

    let runningAccumulator = baseRisk;
    postDays.forEach((day, dayIdx) => {
      const evs = daysMap.get(day);
      let dayDelta = 0;
      evs.forEach(e => { dayDelta += e.scaled_delta; });

      const riskBefore = currentRisk;
      let riskAfter;
      if (dayIdx === postDays.length - 1) {
        // Guarantee EXACT convergence to endScore on final timeline point!
        riskAfter = endScore;
      } else {
        runningAccumulator += dayDelta;
        riskAfter = Math.max(1, Math.min(99, Math.round(runningAccumulator)));
      }
      currentRisk = riskAfter;

      let mainEv = evs[0];
      let maxAbs = Math.abs(mainEv.scaled_delta || 0);
      evs.forEach(e => {
        if (Math.abs(e.scaled_delta || 0) > maxAbs) {
          maxAbs = Math.abs(e.scaled_delta || 0);
          mainEv = e;
        }
      });

      timelineData.push({
        day: day,
        riskBefore: Math.round(riskBefore),
        riskAfter: Math.round(riskAfter),
        delta: dayDelta,
        events: evs,
        mainEvent: mainEv,
        isBaseline: false
      });
    });

    // Proportional Linear X-Scale based on Calendar Days — Twice as dense (8px per day)
    const maxDay = sortedDays[sortedDays.length - 1];
    const daySpan = Math.max(30, maxDay - startDay);
    const containerW = container.clientWidth || 550;
    const containerH = 340;
    const pxPerDay = 8;
    const calculatedW = Math.max(containerW, Math.round(daySpan * pxPerDay + 120));

    const m = { top: 56, right: 60, bottom: 48, left: 24 };
    const iW = calculatedW - m.left - m.right;
    const iH = containerH - m.top - m.bottom;

    const svg = d3.select(container)
      .append('svg')
      .attr('width', calculatedW)
      .attr('height', containerH);

    const g = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);

    // Range starts at 14px so Baseline dot sits clear of the Y-axis sticky white gutter bar
    const xScale = d3.scaleLinear()
      .domain([startDay, maxDay])
      .range([14, iW]);

    const yScale = d3.scaleLinear()
      .domain([0, 100])
      .range([iH, 0]);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const gridColor = isLight ? 'rgba(0,0,0,0.06)' : 'rgba(255,255,255,0.05)';
    const textColor = isLight ? '#475569' : '#94A3B8';

    // Horizontal Y Gridlines (spanning full chart width iW)
    const yTicks = [0, 25, 50, 75, 100];
    yTicks.forEach(tickVal => {
      const yPos = yScale(tickVal);
      g.append('line')
        .attr('x1', 0).attr('x2', iW)
        .attr('y1', yPos).attr('y2', yPos)
        .attr('stroke', gridColor)
        .attr('stroke-dasharray', tickVal === 50 ? '4,4' : 'none')
        .attr('stroke-width', tickVal === 50 ? 1.5 : 1);
    });

    // Sticky Y-Axis Group (follows horizontal scroll right at the left border frame)
    const yAxisG = svg.append('g')
      .attr('class', 'y-axis-group')
      .attr('transform', `translate(${m.left},${m.top})`);

    const bgGutterFill = isLight ? '#F8FAFC' : '#12151B';
    yAxisG.append('rect')
      .attr('x', -m.left - 60)
      .attr('y', -m.top)
      .attr('width', m.left + 68)
      .attr('height', containerH)
      .attr('fill', bgGutterFill);

    yTicks.forEach(tickVal => {
      const yPos = yScale(tickVal);
      yAxisG.append('text')
        .attr('x', -4).attr('y', yPos)
        .attr('dominant-baseline', 'central')
        .attr('text-anchor', 'end')
        .attr('fill', textColor)
        .attr('font-size', '0.66rem')
        .attr('font-weight', tickVal === 50 ? '700' : '400')
        .text(`${tickVal}`);
    });

    // Attach horizontal scroll listener to keep y-axis pinned to left frame
    const wrapper = document.getElementById('timeline-graph-wrapper');
    if (wrapper && !wrapper.dataset.hasScrollListener) {
      wrapper.dataset.hasScrollListener = 'true';
      wrapper.addEventListener('scroll', () => {
        const sl = wrapper.scrollLeft;
        d3.select(container).select('.y-axis-group')
          .attr('transform', `translate(${m.left + sl}, ${m.top})`);
      });
    }

    // 50 Risk Threshold Line (Dashed reference line)
    g.append('line')
      .attr('x1', 0).attr('x2', iW)
      .attr('y1', yScale(50)).attr('y2', yScale(50))
      .attr('stroke', '#FF4757')
      .attr('stroke-dasharray', '3,3')
      .attr('stroke-width', 1);

    // ----------------------------------------------------------------
    // Vertical Faint Grid Lines & Regular 7-Day X Axis Ticks
    // ----------------------------------------------------------------
    const gGridVertical = g.append('g').attr('class', 'timeline-grid-vertical');

    const weeklyTicks = [startDay]; // Always include Baseline
    for (let day = 0; day <= maxDay + 7; day += 7) {
      if (day >= startDay) {
        weeklyTicks.push(day);
      }
    }

    let lastTickX = -999;
    weeklyTicks.forEach(tickDay => {
      const xPos = xScale(tickDay);
      if (xPos >= 0 && xPos <= iW) {
        // Draw faint vertical grid line spanning full plot height from y=0 to y=iH
        gGridVertical.append('line')
          .attr('x1', xPos).attr('x2', xPos)
          .attr('y1', 0).attr('y2', iH)
          .attr('stroke', gridColor)
          .attr('stroke-width', 1)
          .attr('stroke-dasharray', '2,2')
          .attr('opacity', 0.6);

        if (xPos - lastTickX >= 32 || tickDay === startDay || tickDay === 0) {
          lastTickX = xPos;
          g.append('line')
            .attr('x1', xPos).attr('x2', xPos)
            .attr('y1', iH).attr('y2', iH + 6)
            .attr('stroke', gridColor);

          const textG = g.append('text')
            .attr('x', xPos).attr('y', iH + 18)
            .attr('text-anchor', 'middle')
            .attr('fill', textColor)
            .attr('font-size', '0.66rem')
            .attr('font-weight', '600');

          if (tickDay === startDay) {
            textG.text('Baseline');
          } else {
            textG.text(`Day ${tickDay > 0 ? '+' : ''}${tickDay}`);
          }
        }
      }
    });

    // Build Staircase Path Points
    const stepPoints = [];
    timelineData.forEach((d, i) => {
      if (i === 0) {
        stepPoints.push({ x: xScale(d.day), y: yScale(d.riskBefore) });
      } else {
        stepPoints.push({ x: xScale(d.day), y: yScale(d.riskBefore) });
        stepPoints.push({ x: xScale(d.day), y: yScale(d.riskAfter) });
      }
    });

    // Dedicated SVG layer groups for proper z-ordering:
    // Area -> Staircase Line -> Stems -> Callout Badges -> Interactive Dots (on top!)
    const gArea = g.append('g').attr('class', 'timeline-area-layer');
    const gLines = g.append('g').attr('class', 'timeline-line-layer');
    const gCalloutStems = g.append('g').attr('class', 'timeline-stems-layer');
    const gCalloutBadges = g.append('g').attr('class', 'timeline-badges-layer');
    const gNodes = g.append('g').attr('class', 'timeline-nodes-layer');

    // Gradient Area Fill under Staircase
    const areaDef = svg.append('defs').append('linearGradient')
      .attr('id', 'risk-timeline-grad')
      .attr('x1', '0').attr('y1', '0')
      .attr('x2', '0').attr('y2', '1');

    areaDef.append('stop').attr('offset', '0%').attr('stop-color', '#FF4757').attr('stop-opacity', '0.22');
    areaDef.append('stop').attr('offset', '100%').attr('stop-color', '#2D6BE4').attr('stop-opacity', '0.02');

    const areaPath = d3.area()
      .x(pt => pt.x)
      .y0(iH)
      .y1(pt => pt.y);

    gArea.append('path')
      .datum(stepPoints)
      .attr('fill', 'url(#risk-timeline-grad)')
      .attr('d', areaPath);

    // Staircase Line Path
    const linePath = d3.line()
      .x(pt => pt.x)
      .y(pt => pt.y);

    gLines.append('path')
      .datum(stepPoints)
      .attr('fill', 'none')
      .attr('stroke', '#2D6BE4')
      .attr('stroke-width', 2.8)
      .attr('stroke-linejoin', 'miter')
      .attr('d', linePath);

    // Collect all node dots for dot-clearing checks
    const allDots = timelineData.map(d => ({
      cx: xScale(d.day),
      cy: yScale(d.riskBefore),
      r: d.isBaseline ? 5.0 : 6.5
    }));

    // ----------------------------------------------------------------
    // Collect & Resolve Callout Annotations (User Strategy + 2D Anti-Collision)
    // ----------------------------------------------------------------
    const calloutCandidates = [];

    timelineData.forEach((d, idx) => {
      const cx = xScale(d.day);
      const cy = yScale(d.riskBefore);
      const absDelta = Math.abs(d.delta);
      const dotR = d.isBaseline ? 5.0 : 6.5;

      if (!d.isBaseline && absDelta >= 2.5 && d.mainEvent) {
        const sign = d.delta > 0 ? '+' : '';
        const entityLabel = d.mainEvent.entity === 'Infection' ? 'Previous infection' : d.mainEvent.entity;
        const calloutText = `${sign}${d.delta.toFixed(1)}%: ${entityLabel}`;

        let calloutTextColor, calloutBorder, calloutBg;
        if (d.delta > 0) {
          calloutTextColor = isLight ? '#B32431' : '#FF8A95';
          calloutBorder = isLight ? '#EF4444' : '#FF4757';
          calloutBg = isLight ? '#FFF1F2' : 'rgba(255, 71, 87, 0.25)';
        } else {
          calloutTextColor = isLight ? '#1D4ED8' : '#93C5FD';
          calloutBorder = isLight ? '#3B82F6' : '#5B9BF8';
          calloutBg = isLight ? '#EFF6FF' : 'rgba(37, 99, 235, 0.25)';
        }

        // Measure text dimensions
        const dummyG = gCalloutBadges.append('g').style('visibility', 'hidden');
        const dummyText = dummyG.append('text')
          .attr('font-size', '0.64rem')
          .attr('font-weight', '700')
          .text(calloutText);
        const bbox = dummyText.node().getBBox();
        dummyG.remove();

        const padX = 6;
        const padY = 4;
        const boxW = bbox.width + padX * 2;
        const boxH = bbox.height + padY * 2;

        const isRed = d.delta > 0;
        let targetSide = isRed ? 'below' : 'above';

        if (targetSide === 'below' && cy > iH - 35) {
          targetSide = 'above';
        } else if (targetSide === 'above' && cy < 25) {
          targetSide = 'below';
        }

        const clampedX = Math.max(boxW / 2 + 4, Math.min(iW - boxW / 2 - 4, cx));
        // Generous initial vertical gap (longer dashed connecting lines)
        const idealY = targetSide === 'above' ? (cy - 58) : (cy + 58);

        calloutCandidates.push({
          idx,
          d,
          cx,
          cy,
          dotR,
          calloutText,
          calloutTextColor,
          calloutBorder,
          calloutBg,
          boxW,
          boxH,
          x: clampedX,
          y: idealY,
          targetSide
        });
      }
    });

    calloutCandidates.sort((a, b) => a.cx - b.cx || a.cy - b.cy);

    // Initial separation pass: push overlapping boxes further vertically (support long stems!)
    for (let i = 0; i < calloutCandidates.length; i++) {
      for (let j = i + 1; j < calloutCandidates.length; j++) {
        const c1 = calloutCandidates[i];
        const c2 = calloutCandidates[j];
        const dx = Math.abs(c1.x - c2.x);
        const reqX = (c1.boxW + c2.boxW) / 2 + 6;

        if (dx < reqX) {
          const dy = Math.abs(c1.y - c2.y);
          const reqY = (c1.boxH + c2.boxH) / 2 + 6;
          if (dy < reqY) {
            const pushDist = reqY - dy;
            if (c2.targetSide === 'above') {
              c2.y -= pushDist;
            } else if (c2.targetSide === 'below') {
              c2.y += pushDist;
            } else {
              if (c2.y >= c1.y) c2.y += pushDist; else c2.y -= pushDist;
            }
          }
        }
      }
    }

    // Dot-clearing pass: ensure NO speech bubble box overlaps ANY dot on the chart!
    calloutCandidates.forEach(c => {
      allDots.forEach(dot => {
        const dx = Math.abs(c.x - dot.cx);
        const dy = Math.abs(c.y - dot.cy);
        const clearMarginX = c.boxW / 2 + dot.r + 10;
        const clearMarginY = c.boxH / 2 + dot.r + 14;

        if (dx < clearMarginX && dy < clearMarginY) {
          // Push box vertically clear of this dot!
          if (c.targetSide === 'above') {
            c.y = Math.min(c.y, dot.cy - clearMarginY - 8);
          } else {
            c.y = Math.max(c.y, dot.cy + clearMarginY + 8);
          }
        }
      });
    });

    // Multi-pass relaxation solver to finalize 2D spacing
    for (let iter = 0; iter < 12; iter++) {
      let moved = false;
      for (let i = 0; i < calloutCandidates.length; i++) {
        for (let j = i + 1; j < calloutCandidates.length; j++) {
          const c1 = calloutCandidates[i];
          const c2 = calloutCandidates[j];
          const dx = Math.abs(c1.x - c2.x);
          const dy = Math.abs(c1.y - c2.y);
          const reqX = (c1.boxW + c2.boxW) / 2 + 6;
          const reqY = (c1.boxH + c2.boxH) / 2 + 6;

          if (dx < reqX && dy < reqY) {
            const overlapY = reqY - dy;
            if (c2.targetSide === 'above') {
              c2.y -= overlapY;
            } else {
              c2.y += overlapY;
            }
            moved = true;
          }
        }
      }
      if (!moved) break;
    }

    // Clamp final positions to plot bounds
    calloutCandidates.forEach(c => {
      c.y = Math.max(-20, Math.min(iH - 12, c.y));
      c.x = Math.max(c.boxW / 2 + 4, Math.min(iW - c.boxW / 2 - 4, c.x));
    });

    // Render resolved callout badges and leader stems
    calloutCandidates.forEach(c => {
      const annoG = gCalloutBadges.append('g')
        .attr('transform', `translate(${c.x}, ${c.y})`)
        .style('pointer-events', 'none');

      annoG.append('text')
        .attr('text-anchor', 'middle')
        .attr('dominant-baseline', 'central')
        .attr('font-size', '0.64rem')
        .attr('font-weight', '700')
        .attr('fill', c.calloutTextColor)
        .text(c.calloutText);

      annoG.insert('rect', 'text')
        .attr('x', -c.boxW / 2)
        .attr('y', -c.boxH / 2)
        .attr('width', c.boxW)
        .attr('height', c.boxH)
        .attr('rx', 5)
        .attr('fill', c.calloutBg)
        .attr('stroke', c.calloutBorder)
        .attr('stroke-width', 1.2)
        .attr('opacity', 0.96);

      // Long vertical leader stem line: connecting dot (cx, cy) to badge (x, y)
      const isAbove = c.y < c.cy;
      const stemYStart = isAbove ? c.cy - c.dotR - 1 : c.cy + c.dotR + 1;
      const stemYEnd = isAbove ? c.y + c.boxH / 2 : c.y - c.boxH / 2;

      gCalloutStems.append('line')
        .attr('x1', c.cx).attr('y1', stemYStart)
        .attr('x2', c.x).attr('y2', stemYEnd)
        .attr('stroke', c.calloutBorder)
        .attr('stroke-width', 1.2)
        .attr('stroke-dasharray', '2,2')
        .attr('opacity', 0.75);
    });

    // Render Day Node Circle Groups (rendered on top layer gNodes!)
    timelineData.forEach((d, idx) => {
      const cx = xScale(d.day);
      const cy = yScale(d.riskBefore);
      const absDelta = Math.abs(d.delta);
      const dotR = d.isBaseline ? 5.0 : 6.5;

      let dotColor;
      if (d.isBaseline) {
        dotColor = '#7C8DB5';
      } else if (d.delta > 0) {
        dotColor = '#FF4757'; // Red
      } else if (d.delta < 0) {
        dotColor = isLight ? '#1D4ED8' : '#3B82F6'; // Blue
      } else {
        dotColor = '#7C8DB5';
      }

      const nodeG = gNodes.append('g')
        .attr('class', 'timeline-node-group')
        .style('cursor', 'pointer')
        .attr('pointer-events', 'all');

      nodeG.append('circle')
        .attr('cx', cx).attr('cy', cy).attr('r', 16)
        .attr('fill', 'transparent');

      const circle = nodeG.append('circle')
        .attr('class', 'timeline-node-dot')
        .attr('cx', cx).attr('cy', cy)
        .attr('r', dotR)
        .attr('fill', dotColor)
        .attr('stroke', isLight ? '#ffffff' : '#101F30')
        .attr('stroke-width', 2);

      let isPinned = false;

      // Helper function to position tip dead-center in Panel 2
      function positionTipInPanelCenter() {
        const panelEl = document.querySelector('.panel-timeline');
        if (!panelEl) return;
        const panelRect = panelEl.getBoundingClientRect();
        const panelCenterX = panelRect.left + (panelRect.width / 2);
        const headerEl = panelEl.querySelector('.panel-header');
        const topY = headerEl ? headerEl.getBoundingClientRect().bottom + 8 : panelRect.top + 52;

        tip.style('left', `${panelCenterX}px`)
          .style('top', `${topY}px`)
          .style('transform', 'translateX(-50%)');
      }

      // Function to open / unfold Click-to-Open Popover Card with Close Button (✕)
      function openPopoverCard() {
        const headerTitleText = d.isBaseline ? 'Baseline Profile' : `Day ${d.day > 0 ? '+' + d.day : d.day}`;

        let contentHtml = `
          <div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--border); padding-bottom:5px; margin-bottom:6px;">
            <span style="font-weight:700; color:var(--text); font-size:0.82rem;">${headerTitleText}</span>
            <button class="popover-close-btn" style="background:transparent; border:none; color:var(--text-muted); font-size:1.15rem; font-weight:700; cursor:pointer; padding:0 4px; line-height:1; transition:color 0.2s;" title="Close popover">&times;</button>
          </div>
        `;

        if (d.isBaseline) {
          const baselineEvs = eventsWithScores.filter(e => (e.days_since_tpx ?? 0) <= 0);
          const sortedBaseline = [...baselineEvs].sort((a, b) => Math.abs(b.score || 0) - Math.abs(a.score || 0));
          const evRows = sortedBaseline.map(e => {
            const sc = (e.score || 0) * 100;
            const sign = sc >= 0 ? '+' : '';
            const color = sc > 0 ? '#FF4757' : (sc < 0 ? '#2563EB' : 'var(--text-muted)');
            const valPart = (e.value !== undefined && e.value !== null && e.value !== '') ? `: <strong>${_escHtml(e.value)}</strong>` : '';
            return `<div style="display:flex; justify-content:space-between; margin-top:3px; font-size:0.7rem; gap:12px; align-items:baseline;">
              <span style="color:var(--text); line-height:1.35; word-break:break-word;"><strong>${_escHtml(e.entity)}</strong> &ndash; ${_escHtml(e.attribute)}${valPart}</span>
              <strong style="color:${color}; flex-shrink:0; font-size:0.72rem;">${sign}${sc.toFixed(1)}%</strong>
            </div>`;
          }).join('');

          contentHtml += `
            <div style="font-size:0.75rem; margin-bottom:4px;">Initial ${_scoreTitle(taskKey)}: <strong style="color:${dotColor};">${d.riskBefore}/100</strong></div>
            <div style="margin-top:4px; border-top:1px dashed var(--border); padding-top:4px;">${evRows}</div>
          `;
        } else {
          const sortedEvents = [...d.events].sort((a, b) => Math.abs(b.scaled_delta ?? (b.score || 0) * 100) - Math.abs(a.scaled_delta ?? (a.score || 0) * 100));
          const evRows = sortedEvents.map(e => {
            const sc = e.scaled_delta !== undefined ? e.scaled_delta : (e.score || 0) * 100;
            const sign = sc >= 0 ? '+' : '';
            const color = sc > 0 ? '#FF4757' : (sc < 0 ? '#2563EB' : 'var(--text-muted)');
            const valPart = (e.value !== undefined && e.value !== null && e.value !== '') ? `: <strong>${_escHtml(e.value)}</strong>` : '';
            return `<div style="display:flex; justify-content:space-between; margin-top:3px; font-size:0.7rem; gap:12px; align-items:baseline;">
              <span style="color:var(--text); line-height:1.35; word-break:break-word;"><strong>${_escHtml(e.entity)}</strong> &ndash; ${_escHtml(e.attribute)}${valPart}</span>
              <strong style="color:${color}; flex-shrink:0; font-size:0.72rem;">${sign}${sc.toFixed(1)}%</strong>
            </div>`;
          }).join('');

          const deltaSign = d.delta >= 0 ? '+' : '';
          contentHtml += `
            <div style="font-size:0.75rem;">Cumulated Risk: <strong style="color:${dotColor};">${d.riskBefore}/100 &rarr; ${d.riskAfter}/100</strong> (${deltaSign}${d.delta.toFixed(1)}%)</div>
            <div style="margin-top:4px; border-top:1px dashed var(--border); padding-top:4px;">${evRows}</div>
          `;
        }

        setTimeout(() => {
          tip.classed('visible', true)
            .style('pointer-events', 'auto')
            .html(contentHtml);
          positionTipInPanelCenter();

          tip.select('.popover-close-btn').on('click', (evBtn) => {
            evBtn.stopPropagation();
            tip.classed('visible', false).style('pointer-events', 'none');
            activePinnedData = null;
            circle.attr('r', dotR).attr('stroke-width', 2);
          });
        }, 10);
      }

      // Click event handler: Toggle OFF if clicking active dot, otherwise unfold info popover!
      nodeG
        .on('click', function (ev) {
          ev.stopPropagation();

          if (activePinnedData === d && tip.classed('visible')) {
            // Clicking active dot again closes popover and toggles OFF!
            tip.classed('visible', false).style('pointer-events', 'none');
            activePinnedData = null;
            circle.attr('r', dotR).attr('stroke-width', 2);
          } else {
            // Unfold into full info popover!
            activePinnedData = d;
            g.selectAll('circle.timeline-node-dot').attr('r', dNode => dNode?.isBaseline ? 5.0 : 6.5).attr('stroke-width', 2);
            circle.attr('r', dotR + 3).attr('stroke-width', 3.5);
            openPopoverCard();
          }
        })
        .on('mouseenter mouseover', function () {
          if (!activePinnedData) {
            circle.attr('r', dotR + 2.5).attr('stroke-width', 3);
            tip.classed('visible', true)
              .style('pointer-events', 'none')
              .html(`<div style="font-size:0.70rem; font-weight:600; text-align:center; padding:2px 10px;">Click to pin details</div>`);
            positionTipInPanelCenter();
          }
        })
        .on('mouseleave mouseout', function () {
          if (!activePinnedData) {
            circle.attr('r', dotR).attr('stroke-width', 2);
            tip.classed('visible', false).style('pointer-events', 'none');
          }
        });
    });

    // Cleanly register document click handler without duplicating listeners
    if (window._timelineClosePopoverHandler) {
      document.removeEventListener('click', window._timelineClosePopoverHandler);
    }
    window._timelineClosePopoverHandler = function closeTimelinePopover(ev) {
      if (!ev.target.closest('.timeline-graph-tooltip') && !ev.target.closest('.timeline-node-group')) {
        tip.classed('visible', false).style('pointer-events', 'none');
        activePinnedData = null;
        g.selectAll('circle.timeline-node-dot').attr('r', dNode => dNode?.isBaseline ? 5.0 : 6.5).attr('stroke-width', 2);
      }
    };
    document.addEventListener('click', window._timelineClosePopoverHandler);
  }

  // ================================================================
  // PUBLIC
  // ================================================================
  return {
    initUMAP, updateUMAPCohort, updateUMAPWithPatient, toggleUMAPColorMode,
    initHistogram, updateHistogramWithPatient, updateHistogramData,
    initClusterProfiles, updateClusterProfiles: initClusterProfiles, highlightCluster, resizeCharts, updateThemeColors,
    drawRiskTimelineGraph,
  };
})();
