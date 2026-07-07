/* Real-time model comparison: one prompt across providers, scored by the metric suite. */
(function () {
  "use strict";

  var form = document.getElementById("pg-form");
  if (!form) return;

  var SIDES = ["a", "b", "c"];
  var MODEL_PLACEHOLDERS = {
    mock: "sim-model",
    openai: "gpt-4o-mini",
    anthropic: "your Anthropic model id",
    gemini: "gemini-1.5-flash",
    openai_compatible: "model name",
  };

  function show(nodes, visible) {
    nodes.forEach(function (n) { n.hidden = !visible; });
  }

  function syncSide(side) {
    var type = document.getElementById("pg-type-" + side).value;
    var isMock = type === "mock";
    var isHttp = type === "openai_compatible";
    var isNone = type === "none";
    show(document.querySelectorAll(".pg-mock-only[data-side='" + side + "']"), isMock && !isNone);
    show(document.querySelectorAll(".pg-http-only[data-side='" + side + "']"), isHttp && !isNone);
    show(document.querySelectorAll(".pg-model-field[data-side='" + side + "']"), !isNone);
    var modelInput = document.getElementById("pg-model-" + side);
    if (modelInput && MODEL_PLACEHOLDERS[type]) modelInput.placeholder = "e.g. " + MODEL_PLACEHOLDERS[type];
  }

  SIDES.forEach(function (side) {
    var sel = document.getElementById("pg-type-" + side);
    if (sel) {
      sel.addEventListener("change", function () { syncSide(side); });
      syncSide(side);
    }
  });

  function providerConfig(side) {
    var typeEl = document.getElementById("pg-type-" + side);
    if (!typeEl) return null;
    var type = typeEl.value;
    if (type === "none") return null;
    var model = (document.getElementById("pg-model-" + side).value || "").trim();
    if (type === "mock") {
      return {
        type: "mock",
        model: model || "sim-model",
        quality: parseFloat(document.getElementById("pg-quality-" + side).value) || 0.8,
        latency_ms: 40,
        input_cost_per_1k_tokens: 0.25,
        output_cost_per_1k_tokens: 1.25,
      };
    }
    if (type === "openai_compatible") {
      var cfg = {
        type: "openai_compatible",
        model: model || "model",
        base_url: document.getElementById("pg-url-" + side).value.trim() || "http://localhost:11434/v1",
      };
      var keyEnv = document.getElementById("pg-keyenv-" + side).value.trim();
      if (keyEnv) cfg.api_key_env = keyEnv;
      return cfg;
    }
    // openai / anthropic / gemini — use built-in defaults; model optional
    var out = { type: type };
    if (model) out.model = model;
    else if (type === "anthropic") out.model = "";
    return out;
  }

  function selectedMetrics() {
    var metrics = [];
    document.querySelectorAll(".pg-metric:checked").forEach(function (box) {
      if (box.value === "exact_match") metrics.push({ type: "exact_match", strip_punctuation: true });
      else if (box.value === "token_f1") metrics.push({ type: "token_f1", threshold: 0.6 });
      else if (box.value === "semantic_similarity") metrics.push({ type: "semantic_similarity", threshold: 0.5 });
      else if (box.value === "contains") metrics.push({ type: "contains", mode: "any" });
    });
    return metrics;
  }

  function pct(x) { return Math.round(x * 1000) / 10 + "%"; }
  function r3(x) { return Math.round(x * 1000) / 1000; }

  // Rank successful results: quality first (mean score, then pass), then latency, then cost.
  function rankResults(ok) {
    return ok.slice().sort(function (a, b) {
      var am = a.mean_score, bm = b.mean_score;
      if (am !== null && bm !== null && Math.abs(am - bm) > 1e-9) return bm - am;
      if (!!a.passed !== !!b.passed) return (b.passed ? 1 : 0) - (a.passed ? 1 : 0);
      if (Math.abs(a.latency_ms - b.latency_ms) > 1) return a.latency_ms - b.latency_ms;
      return a.cost_usd - b.cost_usd;
    });
  }

  function conclusionNode(results) {
    var ok = results.filter(function (r) { return !r.error; });
    var wrap = document.createElement("div");
    if (ok.length < 2) {
      if (ok.length === 1 && results.length > 1) {
        wrap.className = "verdict neutral";
        wrap.innerHTML = "<div><div class='title'>Inconclusive</div>" +
          "<div class='detail'>Only one model returned a result; the others errored.</div></div>";
        return wrap;
      }
      return null;
    }

    var ranked = rankResults(ok);
    var best = ranked[0], second = ranked[1];
    var hasScores = best.mean_score !== null && second.mean_score !== null;
    var scoreGap = hasScores && Math.abs(best.mean_score - second.mean_score) > 1e-9;
    var passGap = !!best.passed !== !!second.passed;
    var latGap = Math.abs(best.latency_ms - second.latency_ms) > 1;

    var reason, decisive = true;
    if (scoreGap) reason = "highest mean score (" + r3(best.mean_score) + " vs " + r3(second.mean_score) + ")";
    else if (passGap) reason = "the only model passing every metric";
    else if (hasScores && latGap) reason = "same quality — fastest (" + Math.round(best.latency_ms) + " ms vs " + Math.round(second.latency_ms) + " ms)";
    else if (!hasScores && latGap) reason = "fastest response (" + Math.round(best.latency_ms) + " ms)";
    else if (best.cost_usd + 1e-9 < second.cost_usd) reason = "same quality and speed — lowest cost";
    else { decisive = false; reason = "no measurable difference between the models on these metrics"; }

    wrap.className = "verdict " + (decisive ? "good" : "neutral");
    var trophy = decisive
      ? "<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'><path d='M8 21h8M12 17v4M7 4h10v4a5 5 0 0 1-10 0V4zM5 4H3v2a3 3 0 0 0 3 3M19 4h2v2a3 3 0 0 1-3 3'/></svg>"
      : "<svg width='18' height='18' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round'><circle cx='12' cy='12' r='10'/><path d='M8 12h8'/></svg>";

    var ranking = ranked.map(function (r, i) {
      var bits = [];
      if (r.mean_score !== null) bits.push("mean " + r3(r.mean_score));
      if (r.passed !== null) bits.push(r.passed ? "pass" : "fail");
      bits.push(Math.round(r.latency_ms) + " ms");
      bits.push("$" + r.cost_usd.toFixed(4));
      return "<span class='rank-item'><b>" + (i + 1) + ".</b> " + (r.model || "(model)") +
        " <span class='rank-meta'>" + bits.join(" · ") + "</span></span>";
    }).join("");

    var title = decisive ? ("Winner: " + (best.model || "(model)")) : "It's a tie";
    var errored = results.length - ok.length;
    var detail = (decisive ? "Best by " + reason + "." : reason.charAt(0).toUpperCase() + reason.slice(1) + ".") +
      (errored ? " " + errored + " model(s) errored and were excluded." : "");
    wrap.innerHTML = trophy + "<div style='min-width:0'><div class='title'>" + title + "</div>" +
      "<div class='detail'>" + detail + "</div>" +
      "<div class='ranking'>" + ranking + "</div></div>";
    return wrap;
  }

  function resultCard(result) {
    var card = document.createElement("div");
    card.className = "card result-card";

    var head = document.createElement("div");
    head.className = "result-head";
    var name = document.createElement("div");
    name.className = "result-model";
    name.textContent = result.model || "(model)";
    var tag = document.createElement("span");
    tag.className = "result-tag";
    tag.textContent = result.provider_type || "";
    head.appendChild(name);
    if (result.provider_type) head.appendChild(tag);
    card.appendChild(head);

    if (result.error) {
      var err = document.createElement("div");
      err.className = "pg-error";
      err.textContent = result.error;
      card.appendChild(err);
      return card;
    }

    if (result.passed !== null && result.passed !== undefined) {
      var verdict = document.createElement("div");
      verdict.className = "result-verdict " + (result.passed ? "pass" : "fail");
      verdict.textContent = (result.passed ? "PASS" : "FAIL") +
        (result.mean_score !== null ? " · mean " + (Math.round(result.mean_score * 1000) / 1000) : "");
      card.appendChild(verdict);
    }

    var out = document.createElement("div");
    out.className = "pg-output";
    out.textContent = result.output;
    card.appendChild(out);

    var meta = document.createElement("div");
    meta.className = "pg-meta";
    meta.innerHTML = "<span>Latency <strong>" + Math.round(result.latency_ms) + " ms</strong></span>" +
      "<span>Cost <strong>$" + result.cost_usd.toFixed(5) + "</strong></span>";
    card.appendChild(meta);

    if (result.scores && result.scores.length) {
      var scores = document.createElement("div");
      scores.className = "score-chips";
      result.scores.forEach(function (s) {
        var chip = document.createElement("span");
        chip.className = "score-chip " + (s.passed ? "pass" : "fail");
        chip.title = s.detail || "";
        chip.textContent = s.name + " " + (Math.round(s.score * 100) / 100);
        scores.appendChild(chip);
      });
      card.appendChild(scores);
    }
    return card;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var submit = document.getElementById("pg-submit");
    var errorBox = document.getElementById("pg-error");
    var resultsBox = document.getElementById("pg-results");
    var conclusionBox = document.getElementById("pg-conclusion");
    errorBox.hidden = true;
    conclusionBox.hidden = true;
    conclusionBox.innerHTML = "";

    var providers = SIDES.map(providerConfig).filter(Boolean);
    if (!providers.length) {
      errorBox.textContent = "Choose at least one model.";
      errorBox.hidden = false;
      return;
    }

    submit.disabled = true;
    submit.classList.add("loading");

    fetch("/api/playground", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: document.getElementById("pg-prompt").value,
        reference: document.getElementById("pg-reference").value || null,
        providers: providers,
        evaluators: selectedMetrics(),
      }),
    })
      .then(function (response) {
        if (!response.ok) {
          return response.json().then(function (body) {
            throw new Error(typeof body.detail === "string" ? body.detail : "Request failed (" + response.status + ")");
          });
        }
        return response.json();
      })
      .then(function (body) {
        resultsBox.innerHTML = "";
        resultsBox.hidden = false;
        body.results.forEach(function (result) {
          resultsBox.appendChild(resultCard(result));
        });
        var conclusion = conclusionNode(body.results);
        if (conclusion) {
          conclusionBox.appendChild(conclusion);
          conclusionBox.hidden = false;
        }
      })
      .catch(function (error) {
        errorBox.textContent = error.message;
        errorBox.hidden = false;
      })
      .finally(function () {
        submit.disabled = false;
        submit.classList.remove("loading");
      });
  });
})();
