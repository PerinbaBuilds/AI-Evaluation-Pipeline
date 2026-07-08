/* Self-contained SVG charts. No external dependencies.
 *
 * Mark specs: 2px lines with round joins, >=8px end markers ringed in the
 * surface color, bars capped at 24px with a 4px rounded data-end (square at
 * the baseline), 2px surface gaps between touching marks, solid hairline
 * gridlines. Tooltips mirror to keyboard focus; every chart also ships a
 * server-rendered table twin, so no value is gated behind hover.
 */
(function () {
  "use strict";

  /* Light enterprise theme — mirrors the CSS design tokens. */
  var TOKENS = {
    surface: "#ffffff",
    ink: "#14181f",
    ink2: "#58606c",
    muted: "#838b98",
    grid: "#eceef2",
    baseline: "#d3d7de",
    series: ["#2f6feb", "#0ca678"],
  };

  var SVG_NS = "http://www.w3.org/2000/svg";

  function el(name, attrs, parent) {
    var node = document.createElementNS(SVG_NS, name);
    for (var key in attrs) node.setAttribute(key, attrs[key]);
    if (parent) parent.appendChild(node);
    return node;
  }

  /* ------------------------------------------------------------- tooltip */

  var tooltip = null;
  function ensureTooltip() {
    if (!tooltip) {
      tooltip = document.createElement("div");
      tooltip.className = "viz-tooltip";
      tooltip.setAttribute("role", "status");
      document.body.appendChild(tooltip);
    }
    return tooltip;
  }

  function showTip(clientX, clientY, title, lines) {
    var tip = ensureTooltip();
    tip.innerHTML = "";
    var titleEl = document.createElement("div");
    titleEl.className = "tip-title";
    titleEl.textContent = title;
    tip.appendChild(titleEl);
    lines.forEach(function (line) {
      var row = document.createElement("div");
      row.textContent = line;
      tip.appendChild(row);
    });
    tip.classList.add("visible");
    var pad = 12;
    var rect = tip.getBoundingClientRect();
    var x = Math.min(clientX + pad, window.innerWidth - rect.width - pad);
    var y = clientY - rect.height - pad;
    if (y < pad) y = clientY + pad;
    tip.style.left = x + "px";
    tip.style.top = y + "px";
  }

  function hideTip() {
    if (tooltip) tooltip.classList.remove("visible");
  }

  function attachTip(node, title, lines) {
    node.addEventListener("mousemove", function (event) {
      showTip(event.clientX, event.clientY, title, lines);
    });
    node.addEventListener("mouseleave", hideTip);
    node.setAttribute("tabindex", "0");
    node.setAttribute("role", "img");
    node.setAttribute("aria-label", title + ". " + lines.join(". "));
    node.addEventListener("focus", function () {
      var box = node.getBoundingClientRect();
      showTip(box.left + box.width / 2, box.top, title, lines);
    });
    node.addEventListener("blur", hideTip);
  }

  /* --------------------------------------------------------------- scales */

  function niceTicks(maxValue, count) {
    if (maxValue <= 0) maxValue = 1;
    var rough = maxValue / count;
    var magnitude = Math.pow(10, Math.floor(Math.log10(rough)));
    var candidates = [1, 2, 2.5, 5, 10];
    var step = candidates
      .map(function (c) { return c * magnitude; })
      .find(function (s) { return maxValue / s <= count; }) || 10 * magnitude;
    var top = Math.ceil(maxValue / step - 0.001) * step; /* last tick always >= max */
    var ticks = [];
    for (var v = 0; v <= top + step * 0.001; v += step) ticks.push(v);
    return ticks;
  }

  function formatValue(value, format) {
    if (format === "percent") return Math.round(value * 1000) / 10 + "%";
    if (Number.isInteger(value)) return String(value);
    return (Math.round(value * 1000) / 1000).toString();
  }

  /* Rounded data-end, square baseline. Vertical bars round the top. */
  function roundedTopRect(x, y, width, height, radius) {
    var r = Math.min(radius, width / 2, height);
    return (
      "M" + x + "," + (y + height) +
      " L" + x + "," + (y + r) +
      " Q" + x + "," + y + " " + (x + r) + "," + y +
      " L" + (x + width - r) + "," + y +
      " Q" + (x + width) + "," + y + " " + (x + width) + "," + (y + r) +
      " L" + (x + width) + "," + (y + height) + " Z"
    );
  }

  function roundedRightRect(x, y, width, height, radius) {
    var r = Math.min(radius, height / 2, width);
    return (
      "M" + x + "," + y +
      " L" + (x + width - r) + "," + y +
      " Q" + (x + width) + "," + y + " " + (x + width) + "," + (y + r) +
      " L" + (x + width) + "," + (y + height - r) +
      " Q" + (x + width) + "," + (y + height) + " " + (x + width - r) + "," + (y + height) +
      " L" + x + "," + (y + height) + " Z"
    );
  }

  /* ---------------------------------------------------------- line chart */
  /* data: { points: [{label, value, meta?}], format?: "percent" } */

  function lineChart(container, data) {
    container.innerHTML = "";
    var width = Math.max(320, container.clientWidth);
    var plotH = 200;
    var margin = { top: 14, right: 56, bottom: 26, left: 46 };
    var height = plotH + margin.top + margin.bottom;
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height }, container);
    svg.setAttribute("aria-hidden", "false");

    var points = data.points;
    var innerW = width - margin.left - margin.right;
    var maxV = data.format === "percent" ? 1 : Math.max.apply(null, points.map(function (p) { return p.value; }));
    var ticks = data.format === "percent" ? [0, 0.25, 0.5, 0.75, 1] : niceTicks(maxV, 4);
    var topV = ticks[ticks.length - 1];

    function sx(i) {
      return points.length === 1
        ? margin.left + innerW / 2
        : margin.left + (i / (points.length - 1)) * innerW;
    }
    function sy(v) { return margin.top + plotH - (v / topV) * plotH; }

    ticks.forEach(function (t) {
      el("line", { x1: margin.left, x2: width - margin.right, y1: sy(t), y2: sy(t), stroke: TOKENS.grid, "stroke-width": 1 }, svg);
      var lbl = el("text", { x: margin.left - 8, y: sy(t) + 4, "text-anchor": "end", "font-size": 11, fill: TOKENS.muted }, svg);
      lbl.textContent = formatValue(t, data.format);
    });
    el("line", { x1: margin.left, x2: width - margin.right, y1: sy(0), y2: sy(0), stroke: TOKENS.baseline, "stroke-width": 1 }, svg);

    var lineD = points.map(function (p, i) { return (i ? "L" : "M") + sx(i) + "," + sy(p.value); }).join(" ");
    if (points.length > 1) {
      var areaD = lineD + " L" + sx(points.length - 1) + "," + sy(0) + " L" + sx(0) + "," + sy(0) + " Z";
      el("path", { d: areaD, fill: TOKENS.series[0], opacity: 0.1 }, svg);
      el("path", { d: lineD, fill: "none", stroke: TOKENS.series[0], "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round" }, svg);
    }

    points.forEach(function (p, i) {
      var isLast = i === points.length - 1;
      /* invisible >=24px hit target behind the visible marker */
      var hit = el("circle", { cx: sx(i), cy: sy(p.value), r: 14, fill: "transparent" }, svg);
      el("circle", {
        cx: sx(i), cy: sy(p.value), r: isLast ? 5 : 4,
        fill: TOKENS.series[0], stroke: TOKENS.surface, "stroke-width": 2,
        "pointer-events": "none",
      }, svg);
      if (isLast) {
        var endLbl = el("text", { x: sx(i) + 10, y: sy(p.value) + 4, "font-size": 12, "font-weight": 600, fill: TOKENS.ink }, svg);
        endLbl.textContent = formatValue(p.value, data.format);
      }
      attachTip(hit, p.label, [formatValue(p.value, data.format)].concat(p.meta || []));
    });
  }

  /* ------------------------------------------------------------ bar chart */
  /* Vertical bars (histogram). data: { bars: [{label, value}], yLabel? } */

  function barChart(container, data) {
    container.innerHTML = "";
    var width = Math.max(320, container.clientWidth);
    var plotH = 200;
    var margin = { top: 16, right: 12, bottom: 34, left: 40 };
    var height = plotH + margin.top + margin.bottom;
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height }, container);

    var bars = data.bars;
    var innerW = width - margin.left - margin.right;
    var maxV = Math.max.apply(null, bars.map(function (b) { return b.value; }).concat([1]));
    var ticks = niceTicks(maxV, 4);
    var topV = ticks[ticks.length - 1];
    function sy(v) { return margin.top + plotH - (v / topV) * plotH; }

    ticks.forEach(function (t) {
      el("line", { x1: margin.left, x2: width - margin.right, y1: sy(t), y2: sy(t), stroke: TOKENS.grid, "stroke-width": 1 }, svg);
      var lbl = el("text", { x: margin.left - 8, y: sy(t) + 4, "text-anchor": "end", "font-size": 11, fill: TOKENS.muted }, svg);
      lbl.textContent = formatValue(t, null);
    });

    var band = innerW / bars.length;
    var barW = Math.min(24, Math.max(6, band - 2)); /* 2px surface gap minimum */
    var maxIndex = bars.reduce(function (best, b, i) { return b.value > bars[best].value ? i : best; }, 0);

    bars.forEach(function (b, i) {
      var x = margin.left + i * band + (band - barW) / 2;
      var h = (b.value / topV) * plotH;
      if (b.value > 0) {
        el("path", { d: roundedTopRect(x, sy(b.value), barW, h, 4), fill: TOKENS.series[0] }, svg);
      }
      var hit = el("rect", { x: margin.left + i * band, y: margin.top, width: band, height: plotH, fill: "transparent" }, svg);
      attachTip(hit, b.label, [String(b.value) + (data.yLabel ? " " + data.yLabel : "")]);
      if (i === maxIndex && b.value > 0) {
        var cap = el("text", { x: x + barW / 2, y: sy(b.value) - 5, "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, fill: TOKENS.ink }, svg);
        cap.textContent = formatValue(b.value, null);
      }
      if (bars.length <= 6 || i % 2 === 0) {
        var xl = el("text", { x: margin.left + i * band + band / 2, y: margin.top + plotH + 16, "text-anchor": "middle", "font-size": 10.5, fill: TOKENS.muted }, svg);
        xl.textContent = b.label;
      }
    });
    el("line", { x1: margin.left, x2: width - margin.right, y1: sy(0), y2: sy(0), stroke: TOKENS.baseline, "stroke-width": 1 }, svg);
  }

  /* -------------------------------------------------- horizontal bar chart */
  /* data: { bars: [{label, value}], format?: "percent", max?: number } */

  function hbarChart(container, data) {
    container.innerHTML = "";
    var width = Math.max(320, container.clientWidth);
    var rowH = 34;
    var margin = { top: 6, right: 56, bottom: 22, left: 172 };
    var bars = data.bars;
    var height = margin.top + bars.length * rowH + margin.bottom;
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height }, container);

    var innerW = width - margin.left - margin.right;
    var topV = data.max || (data.format === "percent" ? 1 : Math.max.apply(null, bars.map(function (b) { return b.value; }).concat([1])));
    function sx(v) { return margin.left + (v / topV) * innerW; }

    [0, 0.25, 0.5, 0.75, 1].forEach(function (f) {
      var v = f * topV;
      el("line", { x1: sx(v), x2: sx(v), y1: margin.top, y2: height - margin.bottom, stroke: TOKENS.grid, "stroke-width": 1 }, svg);
      var lbl = el("text", { x: sx(v), y: height - 6, "text-anchor": "middle", "font-size": 11, fill: TOKENS.muted }, svg);
      lbl.textContent = formatValue(v, data.format);
    });

    bars.forEach(function (b, i) {
      var y = margin.top + i * rowH + (rowH - 18) / 2;
      var w = Math.max(0, (b.value / topV) * innerW);
      var name = el("text", { x: margin.left - 10, y: y + 13, "text-anchor": "end", "font-size": 12.5, fill: TOKENS.ink2 }, svg);
      name.textContent = b.label.length > 24 ? b.label.slice(0, 23) + "…" : b.label;
      if (w > 0) el("path", { d: roundedRightRect(margin.left, y, w, 18, 4), fill: TOKENS.series[0] }, svg);
      var val = el("text", { x: sx(b.value) + 8, y: y + 13, "font-size": 12, "font-weight": 600, fill: TOKENS.ink }, svg);
      val.textContent = formatValue(b.value, data.format);
      var hit = el("rect", { x: 0, y: margin.top + i * rowH, width: width, height: rowH, fill: "transparent" }, svg);
      attachTip(hit, b.label, [formatValue(b.value, data.format)]);
    });
    el("line", { x1: margin.left, x2: margin.left, y1: margin.top, y2: height - margin.bottom, stroke: TOKENS.baseline, "stroke-width": 1 }, svg);
  }

  /* -------------------------------------------- grouped bars + CI whiskers */
  /* data: { series: [name, name], groups: [{label, values: [a, b], ci: [[lo,hi],[lo,hi]]}] } */

  function groupedBarChart(container, data) {
    container.innerHTML = "";
    var width = Math.max(320, container.clientWidth);
    var plotH = 210;
    var margin = { top: 16, right: 12, bottom: 30, left: 46 };
    var height = plotH + margin.top + margin.bottom;
    var svg = el("svg", { viewBox: "0 0 " + width + " " + height, width: width, height: height }, container);

    var innerW = width - margin.left - margin.right;
    function sy(v) { return margin.top + plotH - v * plotH; } /* domain is [0,1] */

    [0, 0.25, 0.5, 0.75, 1].forEach(function (t) {
      el("line", { x1: margin.left, x2: width - margin.right, y1: sy(t), y2: sy(t), stroke: TOKENS.grid, "stroke-width": 1 }, svg);
      var lbl = el("text", { x: margin.left - 8, y: sy(t) + 4, "text-anchor": "end", "font-size": 11, fill: TOKENS.muted }, svg);
      lbl.textContent = Math.round(t * 100) + "%";
    });

    var groupBand = innerW / data.groups.length;
    var barW = 24;
    var gap = 2; /* surface gap between the pair */

    data.groups.forEach(function (group, gi) {
      var center = margin.left + gi * groupBand + groupBand / 2;
      group.values.forEach(function (value, si) {
        var x = center - barW - gap / 2 + si * (barW + gap);
        var h = Math.max(0, value) * plotH;
        if (h > 0) el("path", { d: roundedTopRect(x, sy(value), barW, h, 4), fill: TOKENS.series[si] }, svg);

        var ci = group.ci && group.ci[si];
        if (ci) {
          var cx = x + barW / 2;
          el("line", { x1: cx, x2: cx, y1: sy(ci[1]), y2: sy(ci[0]), stroke: TOKENS.ink2, "stroke-width": 1.5 }, svg);
          el("line", { x1: cx - 4, x2: cx + 4, y1: sy(ci[1]), y2: sy(ci[1]), stroke: TOKENS.ink2, "stroke-width": 1.5 }, svg);
          el("line", { x1: cx - 4, x2: cx + 4, y1: sy(ci[0]), y2: sy(ci[0]), stroke: TOKENS.ink2, "stroke-width": 1.5 }, svg);
        }

        var lines = [Math.round(value * 1000) / 10 + "%"];
        if (ci) lines.push("95% CI " + Math.round(ci[0] * 1000) / 10 + "% – " + Math.round(ci[1] * 1000) / 10 + "%");
        var hit = el("rect", { x: x - gap, y: margin.top, width: barW + gap * 2, height: plotH, fill: "transparent" }, svg);
        attachTip(hit, data.series[si] + " — " + group.label, lines);

        var cap = el("text", {
          x: x + barW / 2, y: sy(ci ? ci[1] : value) - 6,
          "text-anchor": "middle", "font-size": 11.5, "font-weight": 600, fill: TOKENS.ink,
        }, svg);
        cap.textContent = Math.round(value * 1000) / 10 + "%";
      });
      var gl = el("text", { x: center, y: margin.top + plotH + 20, "text-anchor": "middle", "font-size": 12, fill: TOKENS.ink2 }, svg);
      gl.textContent = group.label;
    });
    el("line", { x1: margin.left, x2: width - margin.right, y1: sy(0), y2: sy(0), stroke: TOKENS.baseline, "stroke-width": 1 }, svg);
  }

  /* --------------------------------------------------------------- driver */

  var RENDERERS = { line: lineChart, bar: barChart, hbar: hbarChart, "grouped-bar": groupedBarChart };

  function renderAll() {
    document.querySelectorAll("[data-chart]").forEach(function (container) {
      var script = container.querySelector("script[type='application/json']");
      if (!script) return;
      var data;
      try { data = JSON.parse(script.textContent); } catch (err) { return; }
      var renderer = RENDERERS[container.getAttribute("data-chart")];
      if (!renderer) return;
      renderer(container, data); /* clears the container */
      container.appendChild(script); /* re-attach the data for the next re-render */
    });
  }

  var resizeTimer = null;
  window.addEventListener("resize", function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(renderAll, 150);
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", renderAll);
  } else {
    renderAll();
  }

  window.EvalPipeCharts = { renderAll: renderAll };
})();
