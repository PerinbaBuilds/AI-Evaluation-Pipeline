/* Lightweight UI behaviours: table filtering, copy-to-clipboard, auto-refresh.
 * No dependencies; progressive-enhancement only (the pages work without JS). */
(function () {
  "use strict";

  // ---- client-side table filter -------------------------------------------
  document.querySelectorAll("input.table-filter").forEach(function (input) {
    var table = document.getElementById(input.getAttribute("data-filter-target"));
    if (!table) return;
    var rows = Array.prototype.slice.call(table.tBodies[0] ? table.tBodies[0].rows : []);
    input.addEventListener("input", function () {
      var needle = input.value.trim().toLowerCase();
      var shown = 0;
      rows.forEach(function (row) {
        var match = row.textContent.toLowerCase().indexOf(needle) !== -1;
        row.hidden = !match;
        if (match) shown++;
      });
      var empty = table.parentNode.querySelector(".filter-empty");
      if (shown === 0 && !empty) {
        empty = document.createElement("p");
        empty.className = "filter-empty text-muted";
        empty.style.padding = "12px 2px";
        empty.textContent = "No rows match “" + input.value + "”.";
        table.parentNode.appendChild(empty);
      } else if (shown > 0 && empty) {
        empty.remove();
      }
    });
  });

  // ---- copy to clipboard ---------------------------------------------------
  document.querySelectorAll("[data-copy]").forEach(function (btn) {
    btn.addEventListener("click", function () {
      var text = btn.getAttribute("data-copy");
      var done = function () {
        btn.classList.add("copied");
        setTimeout(function () { btn.classList.remove("copied"); }, 1200);
      };
      if (navigator.clipboard && navigator.clipboard.writeText) {
        navigator.clipboard.writeText(text).then(done, function () {});
      }
    });
  });

  // ---- light / dark theme toggle ------------------------------------------
  var themeBtn = document.getElementById("theme-toggle");
  if (themeBtn) {
    var root = document.documentElement;
    var label = themeBtn.querySelector(".theme-label");
    var sync = function () {
      var dark = root.getAttribute("data-theme") === "dark";
      themeBtn.setAttribute("aria-pressed", dark ? "true" : "false");
      if (label) label.textContent = dark ? "Light mode" : "Dark mode";
    };
    sync();
    themeBtn.addEventListener("click", function () {
      var dark = root.getAttribute("data-theme") === "dark";
      if (dark) root.removeAttribute("data-theme");
      else root.setAttribute("data-theme", "dark");
      try { localStorage.setItem("evalpipe-theme", dark ? "light" : "dark"); } catch (e) {}
      sync();
      // repaint SVG charts with the new theme's colours
      if (window.EvalPipeCharts && window.EvalPipeCharts.renderAll) window.EvalPipeCharts.renderAll();
    });
  }

  // ---- auto-refresh while a run is executing -------------------------------
  var refresher = document.querySelector("[data-autorefresh]");
  if (refresher) {
    var seconds = parseInt(refresher.getAttribute("data-autorefresh"), 10) || 4;
    setTimeout(function () { window.location.reload(); }, seconds * 1000);
  }
})();
