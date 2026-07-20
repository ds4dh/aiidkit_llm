/**
 * AIIDKIT — charts.js
 * All data visualisations: D3 UMAP scatter, Chart.js KM curves,
 * attribution bar chart, risk histogram, and cluster profiles table.
 * Exposes window.AIIDKIT_CHARTS
 */
'use strict';

window.AIIDKIT_CHARTS = (() => {

  // ---- Fluid font helpers -------------------------------------------
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
  let _umapDotR  = 4.5;   // normal cohort dots
  let _umapEventR = 6.5;  // event-positive dots (slightly larger)
  let _umapStarOuter = 13; // new-patient star outer radius
  let _umapStarInner = 5.5; // new-patient star inner radius

  // Module-level chart instances (preserved for update calls)
  let _umapG = null, _umapScaleX = null, _umapScaleY = null;
  let _umapPatientG = null;
  let _umapColorMode = 'cluster';
  let _umapData = [];
  let _kmChart = null, _attrChart = null, _histChart = null;

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

  function initUMAP(cohortData) {
    const container = document.getElementById('umap-svg-container');
    if (!container) return;
    container.innerHTML = '';
    _umapData = cohortData.patients || [];

    const W = container.clientWidth  || 420;
    const H = container.clientHeight || 300;
    // Match legend width used by _drawUMAPLegend so clip region is consistent
    const _legW = Math.round(_rootFontSize() * 7.5);
    const m = { top: 14, right: _legW + 14, bottom: 14, left: 14 };
    const iW = W - m.left - m.right;
    const iH = H - m.top  - m.bottom;

    // Dot radii: scale with iH so dots stay visually consistent at any viewport width.
    // The chart row has a fixed CSS height (~480px), so iH is roughly stable, giving
    // similar-looking dots at 1080p, 1440p, and 4K.  Clamped to avoid extremes.
    _umapDotR    = Math.max(3.5, Math.min(7,  iH * 0.012));
    _umapEventR  = Math.max(5.0, Math.min(10, iH * 0.016));
    _umapStarOuter = Math.max(11, Math.min(18, iH * 0.034));
    _umapStarInner = _umapStarOuter * 0.42;

    const xs = _umapData.map(d => d.umap_x);
    const ys = _umapData.map(d => d.umap_y);
    const px = (d3.max(xs) - d3.min(xs)) * 0.08;
    const py = (d3.max(ys) - d3.min(ys)) * 0.08;

    _umapScaleX = d3.scaleLinear().domain([d3.min(xs) - px, d3.max(xs) + px]).range([0, iW]);
    _umapScaleY = d3.scaleLinear().domain([d3.min(ys) - py, d3.max(ys) + py]).range([iH, 0]);

    const svg = d3.select(container)
      .append('svg')
      .attr('viewBox', `0 0 ${W} ${H}`)
      .attr('width', '100%').attr('height', '100%');

    svg.append('rect').attr('width', W).attr('height', H).attr('fill', 'var(--umap-bg)');

    svg.append('defs').append('clipPath').attr('id', 'umap-clip')
      .append('rect').attr('x', -15).attr('y', -15).attr('width', iW + 30).attr('height', iH + 30);

    _umapG = svg.append('g').attr('transform', `translate(${m.left},${m.top})`);
    const ptG = _umapG.append('g').attr('clip-path', 'url(#umap-clip)').attr('class', 'pts');

    const zoom = d3.zoom().scaleExtent([0.4, 12]).on('zoom', ev => {
      ptG.attr('transform', ev.transform);
      if (_umapPatientG) _umapPatientG.attr('transform', ev.transform);
    });
    svg.call(zoom);

    const tip = d3.select(container).append('div').attr('class', 'umap-tooltip');

    ptG.selectAll('circle')
      .data(_umapData.sort((a, b) => (a.true_label || 0) - (b.true_label || 0)), d => d.id)
      .join('circle')
      .attr('class', 'umap-pt')
      .attr('cx', d => _umapScaleX(d.umap_x))
      .attr('cy', d => _umapScaleY(d.umap_y))
      .attr('r',  d => d.true_label === 1 ? _umapEventR : _umapDotR)
      .attr('fill', d => {
        if (_umapColorMode === 'cluster') return d.cluster_color || '#888';
        if (_umapColorMode === 'risk') {
          const t = d.risk_score;
          if (t < 0.5) {
            const u = t * 2;
            return `rgb(${Math.round(79 + 176 * u)},${Math.round(142 + 23 * u)},${Math.round(247 - 245 * u)})`;
          } else {
            const u = (t - 0.5) * 2;
            return `rgb(255,${Math.round(165 - 94 * u)},${Math.round(2 + 85 * u)})`;
          }
        }
        if (_umapColorMode === 'imminence') {
          const isLight = document.documentElement.getAttribute('data-theme') === 'light';
          const startColor = isLight ? [200, 200, 200] : [120, 130, 150];
          const endColor = [255, 71, 87]; // #FF4757 (red)
          const imminence = d.event_imminence || 0.0;
          const r = Math.round(startColor[0] + (endColor[0] - startColor[0]) * imminence);
          const g = Math.round(startColor[1] + (endColor[1] - startColor[1]) * imminence);
          const b = Math.round(startColor[2] + (endColor[2] - startColor[2]) * imminence);
          return `rgb(${r},${g},${b})`;
        }
        return d.cluster_color || '#888';
      })
      .attr('opacity', d => d.true_label === 1 ? 1.0 : 0.65)
      .attr('stroke', d => d.true_label === 1 ? (document.documentElement.getAttribute('data-theme') === 'light' ? '#475569' : '#ffffff') : 'none')
      .attr('stroke-width', d => d.true_label === 1 ? 1.8 : 0)
      .style('cursor', 'pointer')
      .on('mouseover', function(ev, d) {
        d3.select(this).attr('opacity', 1).attr('r', d.true_label === 1 ? _umapEventR * 1.25 : _umapDotR * 1.45);
        const bx  = container.getBoundingClientRect();
        const tipW = 195;
        const ex   = ev.clientX - bx.left;
        const ey   = ev.clientY - bx.top;
        
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
          featuresHtml = `<div class="tooltip-features" style="font-size:0.65rem;color:var(--text-muted);margin-top:2px;">Features: ${d.summary}</div>`;
        }

        tip.classed('visible', true)
          .style('left', `${ex + tipW > bx.width ? ex - tipW - 8 : ex + 10}px`)
          .style('top',  `${ey - 12}px`)
          .html(`
            <div class="tooltip-id">${d.id}</div>
            <div class="tooltip-risk">Score: <strong style="color:${d.cluster_color}">${Math.round(d.risk_score * 100)}/100</strong></div>
            <div class="tooltip-risk" style="font-size: 0.65rem; color: var(--text-muted);">Risk: ${d.calibrated_risk != null ? (d.calibrated_risk * 100).toFixed(1) + '%' : (d.risk_score * 100).toFixed(1) + '%'}</div>
            <div class="tooltip-cluster" style="color:${d.cluster_color}">${d.cluster_name}</div>
            <div class="tooltip-summary" style="font-weight: 600;">${summaryLine}</div>
            ${featuresHtml}
          `);
      })
      .on('mouseout', function(ev, d) {
        d3.select(this).attr('opacity', d.true_label === 1 ? 1.0 : 0.65).attr('r', d.true_label === 1 ? _umapEventR : _umapDotR);
        tip.classed('visible', false);
      });

    _umapPatientG = _umapG.append('g').attr('clip-path', 'url(#umap-clip)');
    _drawUMAPLegend(_umapColorMode);
  }

  function _drawUMAPLegend(mode) {
    const container = document.getElementById('umap-svg-container');
    if (!container) return;
    const svg = d3.select(container).select('svg');
    if (svg.empty()) return;

    svg.select('.umap-legend-group').remove();

    const W = container.clientWidth  || 420;
    const H = container.clientHeight || 300;
    // Legend box width scales with root font so text never clips on 4K
    const legW = Math.round(_rootFontSize() * 7.5);
    const m = { top: 14, right: legW + 14, bottom: 14, left: 14 };

    // ---- Sizing -------------------------------------------------------
    const legDotR      = _umapDotR;           // matches scatter dot size
    const legStarOuter = legDotR * 1.3;       // star slightly larger (original ratio)
    const legStarInner = legStarOuter * 0.42;
    const legFontPx    = _scaledPx(0.68);     // px value for SVG font-size attr

    // Row height must fit the tallest symbol (star) plus a small gap
    const rowH = Math.max(legStarOuter * 2 + 6, _scaledPx(1.6));
    const cy   = rowH / 2;              // vertical centre for every row
    const symX = legStarOuter + 3;      // horizontal centre for all symbols
    const textX = symX + legStarOuter + 7; // text starts after widest symbol

    // Determine all items
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const strokeColor = isLight ? '#475569' : '#ffffff';

    let colorItems = [];
    if (mode === 'cluster') {
      colorItems = [
        { label: 'High risk', color: '#FF4757' },
        { label: 'Low risk',  color: '#00C9A7' },
      ];
    } else if (mode === 'risk') {
      colorItems = [
        { label: 'High risk', color: '#FF4757' },
        { label: 'Low risk',  color: '#4F8EF7' },
      ];
    } else if (mode === 'imminence') {
      colorItems = [
        { label: 'Close',      color: '#FF4757' },
        { label: 'Distant/No', color: '#7C8DB5' },
      ];
    }

    const numRows   = colorItems.length + 2;  // + "Event observed" + "New patient"
    const legHeight = numRows * rowH + 8;     // 4px top + 4px bottom padding

    // ---- Legend group -------------------------------------------------
    const lg = svg.append('g')
      .attr('class', 'umap-legend-group')
      .attr('transform', `translate(${W - m.right + 6}, ${m.top + 4})`);

    // Background rect (computed height so nothing is clipped)
    lg.append('rect')
      .attr('x', -4).attr('y', -4)
      .attr('width', legW).attr('height', legHeight)
      .attr('rx', 6)
      .attr('fill', 'var(--legend-bg)').attr('stroke', 'var(--legend-border)');

    // Helper: append a text label centred on cy
    function _legText(parent, label) {
      parent.append('text')
        .attr('x', textX).attr('y', cy)
        .attr('dominant-baseline', 'central')
        .attr('fill', 'var(--legend-text)')
        .attr('font-size', `${legFontPx}px`)
        .attr('font-family', "'Inter',sans-serif")
        .text(label);
    }

    // Colour rows (high/low risk or equivalent)
    colorItems.forEach((e, i) => {
      const g = lg.append('g').attr('transform', `translate(0,${i * rowH})`);
      g.append('circle')
        .attr('cx', symX).attr('cy', cy).attr('r', legDotR)
        .attr('fill', e.color).attr('opacity', 0.85);
      _legText(g, e.label);
    });

    // "Event observed" row — bordered circle
    const evRow = lg.append('g').attr('transform', `translate(0,${colorItems.length * rowH})`);
    evRow.append('circle')
      .attr('cx', symX).attr('cy', cy).attr('r', legDotR)
      .attr('fill', '#7C8DB5').attr('opacity', 0.65)
      .attr('stroke', strokeColor).attr('stroke-width', 1.8);
    _legText(evRow, 'Event observed');

    // "New patient" row — golden star
    const stRow = lg.append('g').attr('transform', `translate(0,${(colorItems.length + 1) * rowH})`);
    stRow.append('path')
      .attr('d', _starPath(symX, cy, legStarOuter, legStarInner, 5))
      .attr('fill', '#FFD700').attr('stroke', 'none');
    _legText(stRow, 'New patient');
  }


  function updateUMAPWithPatient(umapX, umapY) {
    if (!_umapPatientG || !_umapScaleX || !_umapScaleY) return;
    const px = _umapScaleX(umapX);
    const py = _umapScaleY(umapY);
    _umapPatientG.selectAll('*').remove();

    // Scale pulse circle and star with canvas-derived sizes stored at initUMAP
    const pulseR   = _umapStarOuter * 1.08;
    const pulseEnd = _umapStarOuter * 2.5;

    _umapPatientG.append('circle')
      .attr('cx', px).attr('cy', py).attr('r', pulseR)
      .attr('fill', 'none').attr('stroke', '#FFD700').attr('stroke-width', 2)
      .attr('opacity', 0.8)
      .transition().duration(1200).ease(d3.easeExpOut)
      .attr('r', pulseEnd).attr('opacity', 0);

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const starStroke = isLight ? '#475569' : '#ffffff';

    _umapPatientG.append('path')
      .attr('d', _starPath(px, py, _umapStarOuter, _umapStarInner, 5))
      .attr('fill', '#FFD700')
      .attr('stroke', starStroke).attr('stroke-width', 1.8)
      .attr('opacity', 0)
      .transition().duration(450)
      .attr('opacity', 1);
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
    
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const startColor = isLight ? [200, 200, 200] : [120, 130, 150];
    const endColor = [255, 71, 87]; // #FF4757 (red)
    
    _umapG.selectAll('circle.umap-pt')
      .transition().duration(350)
      .attr('fill', d => {
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
          const imminence = d.event_imminence || 0.0;
          const r = Math.round(startColor[0] + (endColor[0] - startColor[0]) * imminence);
          const g = Math.round(startColor[1] + (endColor[1] - startColor[1]) * imminence);
          const b = Math.round(startColor[2] + (endColor[2] - startColor[2]) * imminence);
          return `rgb(${r},${g},${b})`;
        }
        return '#888';
      });
  }

  function getUMAPColorMode() { return _umapColorMode; }

  function updateUMAPHorizon(patients, mode) {
    _umapData = patients;
    if (!_umapG || !_umapData.length) return;

    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    const startColor = isLight ? [200, 200, 200] : [120, 130, 150];
    const endColor = [255, 71, 87]; // #FF4757 (red)

    _umapG.selectAll('circle.umap-pt')
      .data(_umapData, d => d.id)
      .attr('stroke', d => d.true_label === 1 ? (document.documentElement.getAttribute('data-theme') === 'light' ? '#475569' : '#ffffff') : 'none')
      .sort((a, b) => (a.true_label || 0) - (b.true_label || 0))
      .transition().duration(350)
      .attr('r', d => d.true_label === 1 ? _umapEventR : _umapDotR)
      .attr('opacity', d => d.true_label === 1 ? 1.0 : 0.65)
      .attr('stroke-width', d => d.true_label === 1 ? 1.8 : 0)
      .attr('fill', d => {
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
          const imminence = d.event_imminence || 0.0;
          const r = Math.round(startColor[0] + (endColor[0] - startColor[0]) * imminence);
          const g = Math.round(startColor[1] + (endColor[1] - startColor[1]) * imminence);
          const b = Math.round(startColor[2] + (endColor[2] - startColor[2]) * imminence);
          return `rgb(${r},${g},${b})`;
        }
        return '#888';
      });
  }

  // ================================================================
  // FEATURE ATTRIBUTION  (Chart.js v4)
  // ================================================================

  function initAttributionChart(attributions) {
    const canvas = document.getElementById('attr-canvas');
    if (!canvas) return;
    if (_attrChart) { _attrChart.destroy(); _attrChart = null; }
    const { labels, scores, colors } = _prepAttrib(attributions);
    const tc = _getThemeColors();
    _attrChart = new Chart(canvas, {
      type: 'bar',
      data: { labels, datasets: [{ label: 'Attribution', data: scores, backgroundColor: colors, borderRadius: 4, borderSkipped: false }] },
      options: {
        indexAxis: 'y', responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
        plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(6,13,31,0.92)', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, callbacks: { label: ctx => ` ${ctx.parsed.x >= 0 ? '+' : ''}${ctx.parsed.x.toFixed(3)}` } } },
        scales: {
          x: { title: { display: true, text: 'Attribution score (integrated gradients)', color: tc.textMuted, font: { size: _scaledPx(0.71) } }, grid: { color: tc.grid }, ticks: { color: tc.textMuted, font: { size: _scaledPx(0.71) }, callback: v => v >= 0 ? `+${v.toFixed(2)}` : v.toFixed(2) } },
          y: { grid: { display: false }, ticks: { color: tc.text, font: { size: _scaledPx(0.71) }, autoSkip: false } },
        },
      },
    });
  }

  function updateAttributionChart(attributions) {
    if (!_attrChart) { initAttributionChart(attributions); return; }
    const { labels, scores, colors } = _prepAttrib(attributions);
    _attrChart.data.labels = labels;
    _attrChart.data.datasets[0].data = scores;
    _attrChart.data.datasets[0].backgroundColor = colors;
    _attrChart.update();
  }

  function _prepAttrib(attributions) {
    if (!attributions?.length) return { labels: [], scores: [], colors: [] };
    const formatFeat = feat => feat.replace(/^Infection\b/, 'Previous infection');
    const sorted = [...attributions].sort((a, b) => Math.abs(b.score) - Math.abs(a.score)).slice(0, 15);
    return {
      labels: sorted.map(a => `${formatFeat(a.feature)}: ${a.value}`),
      scores: sorted.map(a => a.score),
      colors: sorted.map(a => a.score >= 0 ? 'rgba(255,71,87,0.83)' : 'rgba(79,142,247,0.83)'),
    };
  }

  // ================================================================
  // RISK HISTOGRAM  (Chart.js v4)
  // ================================================================

  function initHistogram(distribution) {
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
        ctx.setLineDash([5, 4]);
        ctx.stroke();

        // 2. Setup text styling for badge — scale with root font size
        const badgeFontPx = _scaledPx(0.64);
        ctx.font = `bold ${badgeFontPx}px Inter, sans-serif`;
        ctx.textBaseline = 'alphabetic';
        const text = '★ NEW PATIENT';
        const textWidth = ctx.measureText(text).width;

        // 3. Compute badge dimensions and position
        const rectW = textWidth + 12; // 6px padding left & right
        const rectH = 16;
        const rectY = yTop - 22;
        let rectX = xPx - rectW / 2;

        // Clamp to prevent horizontal clipping
        if (rectX < scales.x.left) {
          rectX = scales.x.left;
        } else if (rectX + rectW > scales.x.right) {
          rectX = scales.x.right - rectW;
        }

        // 4. Draw rounded badge container
        const radius = 4;
        ctx.beginPath();
        if (typeof ctx.roundRect === 'function') {
          ctx.roundRect(rectX, rectY, rectW, rectH, radius);
        } else {
          ctx.moveTo(rectX + radius, rectY);
          ctx.lineTo(rectX + rectW - radius, rectY);
          ctx.quadraticCurveTo(rectX + rectW, rectY, rectX + rectW, rectY + radius);
          ctx.lineTo(rectX + rectW, rectY + rectH - radius);
          ctx.quadraticCurveTo(rectX + rectW, rectY + rectH, rectX + rectW - radius, rectY + rectH);
          ctx.lineTo(rectX + radius, rectY + rectH);
          ctx.quadraticCurveTo(rectX, rectY + rectH, rectX, rectY + rectH - radius);
          ctx.lineTo(rectX, rectY + radius);
          ctx.quadraticCurveTo(rectX, rectY, rectX + radius, rectY);
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
        ctx.fillStyle = textCol;
        ctx.fill();

        ctx.restore();
      },
    };

    const tc = _getThemeColors();
    _histChart = new Chart(canvas, {
      type: 'bar',
      data: { labels: binLabels, datasets: [{ label: 'Patients', data: counts, backgroundColor: barColors, borderRadius: 3 }] },
      plugins: [histLinePlugin],
      options: {
        responsive: true, maintainAspectRatio: false, animation: { duration: 500 },
        layout: { padding: { top: 25 } },
        plugins: { legend: { display: false }, tooltip: { backgroundColor: 'rgba(6,13,31,0.92)', borderColor: 'rgba(255,255,255,0.1)', borderWidth: 1, callbacks: { label: ctx => ` ${ctx.parsed.y} patients` } }, histLine: { riskScore: null } },
        scales: {
          x: { title: { display: true, text: 'Infection score', color: tc.textMuted, font: { size: _scaledPx(0.71) } }, grid: { display: false }, ticks: { color: tc.textMuted, font: { size: _scaledPx(0.71) }, maxTicksLimit: 6 } },
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

  function updateHistogramData(distribution) {
    if (!_histChart) return;
    const { binLabels, counts } = _binValues(distribution.values, 10);
    _histChart.data.datasets[0].data = counts;
    _histChart.update();
  }

  function _binValues(values, nBins) {
    const counts = new Array(nBins).fill(0);
    const step   = 1 / nBins;
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

    const formatFeat = feat => feat.replace(/^Infection\b/, 'Previous infection');

    try {
      Object.entries(profiles)
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .forEach(([id, prof]) => {
          let staticTags = '';
          let recentTags = '';

          if (Array.isArray(prof.top_features)) {
            // Backward compatibility for old flat array format
            staticTags = prof.top_features
              .map(f => {
                const cls = f.score >= 0 ? 'pos' : 'neg';
                return `<div class="top-feature-tag ${cls}">${formatFeat(f.feature)}: ${f.value}</div>`;
              }).join('') || '<span style="color:#3D4F72">—</span>';
            recentTags = '<span style="color:#3D4F72">—</span>';
          } else if (prof.top_features && typeof prof.top_features === 'object') {
            // New dictionary format with static and recent subsets
            staticTags = (prof.top_features.static || [])
              .map(f => {
                const cls = f.score >= 0 ? 'pos' : 'neg';
                return `<div class="top-feature-tag ${cls}">${formatFeat(f.feature)}: ${f.value}</div>`;
              }).join('') || '<span style="color:#3D4F72">—</span>';

            recentTags = (prof.top_features.recent || [])
              .map(f => {
                const cls = f.score >= 0 ? 'pos' : 'neg';
                return `<div class="top-feature-tag ${cls}">${formatFeat(f.feature)}: ${f.value}</div>`;
              }).join('') || '<span style="color:#3D4F72">—</span>';
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
                <span class="active-patient-badge">★ New Patient</span>
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

  function _short(feature) {
    const parts = feature.split(' - ');
    return parts.length > 1 ? parts.slice(1).join(' - ') : feature;
  }

  function highlightCluster(clusterId) {
    document.querySelectorAll('.cluster-profile-card').forEach(card => {
      card.classList.toggle('active-cluster', card.dataset.clusterId === String(clusterId));
    });
  }

  function resizeCharts() {
    if (_attrChart) _attrChart.resize();
    if (_histChart) _histChart.resize();
  }

  function updateThemeColors() {
    const tc = _getThemeColors();
    // Also refresh global font size in case viewport was resized
    Chart.defaults.color = tc.textMuted;
    Chart.defaults.font.size = _scaledPx(0.78);

    if (_attrChart) {
      const axFontPx = _scaledPx(0.71);
      _attrChart.options.scales.x.grid.color = tc.grid;
      _attrChart.options.scales.x.ticks.color = tc.textMuted;
      _attrChart.options.scales.x.ticks.font = { size: axFontPx };
      _attrChart.options.scales.x.title.color = tc.textMuted;
      _attrChart.options.scales.x.title.font = { size: axFontPx };
      _attrChart.options.scales.y.ticks.color = tc.text;
      _attrChart.options.scales.y.ticks.font = { size: axFontPx };
      _attrChart.update();
    }
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
  // PUBLIC
  // ================================================================
  return {
    initUMAP, updateUMAPWithPatient, toggleUMAPColorMode, updateUMAPHorizon,
    initAttributionChart, updateAttributionChart,
    initHistogram, updateHistogramWithPatient, updateHistogramData,
    initClusterProfiles, updateClusterProfiles: initClusterProfiles, highlightCluster, resizeCharts, updateThemeColors,
  };
})();
