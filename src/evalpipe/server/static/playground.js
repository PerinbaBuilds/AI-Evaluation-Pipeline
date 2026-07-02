/* Playground: online side-by-side evaluation of one prompt across two providers. */
(function () {
  "use strict";

  var form = document.getElementById("pg-form");
  if (!form) return;

  document.querySelectorAll(".pg-type").forEach(function (select) {
    select.addEventListener("change", function () {
      var side = select.getAttribute("data-side");
      var isMock = select.value === "mock";
      document.querySelectorAll(".pg-mock-only[data-side='" + side + "']").forEach(function (node) {
        node.hidden = !isMock;
      });
      document.querySelectorAll(".pg-http-only[data-side='" + side + "']").forEach(function (node) {
        node.hidden = isMock;
      });
    });
  });

  function providerConfig(side) {
    var type = document.getElementById("pg-type-" + side).value;
    var model = document.getElementById("pg-model-" + side).value.trim() || "model-" + side;
    if (type === "mock") {
      return {
        type: "mock",
        model: model,
        quality: parseFloat(document.getElementById("pg-quality-" + side).value) || 0.8,
        latency_ms: 40,
        input_cost_per_1k_tokens: 0.25,
        output_cost_per_1k_tokens: 1.25,
      };
    }
    var config = {
      type: "openai_compatible",
      model: model,
      base_url: document.getElementById("pg-url-" + side).value.trim(),
    };
    var keyEnv = document.getElementById("pg-keyenv-" + side).value.trim();
    if (keyEnv) config.api_key_env = keyEnv;
    return config;
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    var submit = document.getElementById("pg-submit");
    var errorBox = document.getElementById("pg-error");
    var resultsBox = document.getElementById("pg-results");
    submit.disabled = true;
    errorBox.hidden = true;

    var payload = {
      prompt: document.getElementById("pg-prompt").value,
      reference: document.getElementById("pg-reference").value || null,
      providers: [providerConfig("a"), providerConfig("b")],
    };

    fetch("/api/playground", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
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
        resultsBox.hidden = false;
        ["a", "b"].forEach(function (side, index) {
          var result = body.results[index];
          document.getElementById("pg-result-model-" + side).textContent = result.model;
          var output = document.getElementById("pg-result-output-" + side);
          if (result.error) {
            output.textContent = "Error: " + result.error;
          } else {
            output.textContent = result.output;
          }
          document.getElementById("pg-result-latency-" + side).textContent = Math.round(result.latency_ms) + " ms";
          document.getElementById("pg-result-cost-" + side).textContent = "$" + result.cost_usd.toFixed(5);
        });
      })
      .catch(function (error) {
        errorBox.textContent = error.message;
        errorBox.hidden = false;
      })
      .finally(function () {
        submit.disabled = false;
      });
  });
})();
