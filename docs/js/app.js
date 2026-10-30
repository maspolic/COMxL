/* COMXL — browser front end.
 *
 * Boots Pyodide, installs the comxl wheel shipped in docs/wheels/, loads
 * runner.py as the bridge module, then wires each form to a bridge function.
 * No optimisation logic lives here: this file only moves values in and
 * renders results out.
 */

"use strict";

const PY_PACKAGES = ["numpy", "pandas", "matplotlib", "micropip"];

let pyodide = null;
let runner = null;

/* ── boot ──────────────────────────────────────────────────────────── */

function bootStatus(title, detail) {
  const t = document.getElementById("boot-title");
  const d = document.getElementById("boot-detail");
  if (t) t.textContent = title;
  if (d && detail !== undefined) d.textContent = detail;
}

function bootFailed(message) {
  const card = document.querySelector(".boot-card");
  if (card) card.classList.add("failed");
  bootStatus(
    "Could not start the Python runtime.",
    message + " A reload usually fixes this; the runtime is fetched from a CDN, " +
    "so a blocked or offline connection will stop it."
  );
}

async function boot() {
  try {
    bootStatus("Starting the Python runtime…");
    pyodide = await loadPyodide();

    bootStatus("Loading NumPy, pandas and Matplotlib…");
    await pyodide.loadPackage(PY_PACKAGES);

    bootStatus("Installing the comxl package…");
    const manifest = await fetch("wheels/manifest.json", { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error("wheels/manifest.json is missing.");
        return r.json();
      });

    const micropip = pyodide.pyimport("micropip");
    // Resolve against the page URL so the site works from a project
    // subpath such as /COMXL/ as well as from a user site root.
    const wheelUrl = new URL("wheels/" + manifest.wheel, window.location.href).href;
    await micropip.install(wheelUrl);

    bootStatus("Wiring up the solvers…");
    const runnerSource = await fetch("js/runner.py", { cache: "no-cache" })
      .then((r) => {
        if (!r.ok) throw new Error("js/runner.py is missing.");
        return r.text();
      });
    // runner.py defines its functions in the global namespace, so they are
    // reached through pyodide.globals rather than a module proxy.
    pyodide.runPython(runnerSource);
    runner = pyodide.globals;

    for (const name of ["run_1d", "run_multivariable", "run_lp", "run_kkt",
                        "run_matrix", "run_calculus"]) {
      if (!runner.has(name)) throw new Error(`runner.py did not define ${name}().`);
    }

    const note = document.getElementById("version-note");
    if (note) note.textContent = `comxl ${manifest.version}.`;

    document.getElementById("boot").hidden = true;
    document.getElementById("app").hidden = false;
  } catch (err) {
    console.error(err);
    bootFailed(String(err && err.message ? err.message : err));
  }
}

/* ── calling into Python ───────────────────────────────────────────── */

async function callRunner(fnName, payload) {
  const fn = runner.get(fnName);
  try {
    const proxy = fn(JSON.stringify(payload));
    const value = proxy.toJs({ dict_converter: Object.fromEntries });
    proxy.destroy();
    return value;
  } finally {
    fn.destroy();
  }
}

/* ── rendering ─────────────────────────────────────────────────────── */

function el(tag, attrs = {}, children = []) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") node.className = v;
    else if (k === "html") node.innerHTML = v;
    else if (k === "text") node.textContent = v;
    else node.setAttribute(k, v);
  }
  for (const c of [].concat(children)) {
    if (c) node.appendChild(typeof c === "string" ? document.createTextNode(c) : c);
  }
  return node;
}

function renderTable(table) {
  if (!table || !table.columns || !table.columns.length) return null;
  const thead = el("thead", {}, [
    el("tr", {}, table.columns.map((c) => el("th", { text: c }))),
  ]);
  const tbody = el("tbody", {}, table.rows.map((row) =>
    el("tr", {}, row.map((cell) => el("td", { text: cell })))
  ));
  return el("div", { class: "table-scroll" }, [el("table", {}, [thead, tbody])]);
}

function formatVector(x) {
  if (Array.isArray(x)) {
    return "[" + x.map((v) => Number(v).toPrecision(8).replace(/\.?0+$/, "")).join(",  ") + "]";
  }
  return Number(x).toPrecision(8).replace(/\.?0+$/, "");
}

function renderError(target, message) {
  target.replaceChildren(el("div", { class: "result-error", text: message }));
}

function renderSolverResult(target, res, extras = [], tableLabel = "Iterations") {
  const parts = [];

  const dl = el("dl", {}, [
    el("dt", { text: "Solution x" }),
    el("dd", { text: formatVector(res.x) }),
    el("dt", { text: "Objective f(x)" }),
    el("dd", { text: formatVector(res.f) }),
    el("dt", { text: "Iterations" }),
    el("dd", { text: String(res.iterations) }),
  ]);

  const summary = el(
    "div",
    { class: "summary" + (res.converged ? "" : " not-converged") },
    [
      el("h3", { text: res.method + (res.converged ? " — converged" : " — did not converge") }),
      dl,
      res.message ? el("p", { class: "note", text: res.message }) : null,
    ]
  );
  parts.push(summary);

  for (const extra of extras) if (extra) parts.push(extra);

  if (res.table && res.table.rows && res.table.rows.length) {
    parts.push(el("h4", { text: tableLabel }));
    parts.push(renderTable(res.table));
  }

  if (res.plot) {
    parts.push(el("div", { class: "plot" }, [
      el("img", { src: res.plot, alt: "Plot of the solution" }),
    ]));
  }

  target.replaceChildren(...parts);
}

/* ── form plumbing ─────────────────────────────────────────────────── */

function formValues(form) {
  const out = {};
  for (const field of form.elements) {
    if (!field.name) continue;
    out[field.name] = field.type === "checkbox" ? field.checked : field.value;
  }
  return out;
}

function wire(formId, outputId, fnName, after) {
  const form = document.getElementById(formId);
  const out = document.getElementById(outputId);
  if (!form || !out) return;

  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const button = form.querySelector(".go");
    const label = button.textContent;
    button.disabled = true;
    button.textContent = "Working…";
    out.replaceChildren(el("p", { class: "running", "data-busy": "1",
                                 text: "Running…" }));

    try {
      const result = await callRunner(fnName, formValues(form));
      if (!result.ok) renderError(out, result.error);
      else if (after) after(out, result);
      else renderSolverResult(out, result);
    } catch (err) {
      console.error(err);
      renderError(out, String(err && err.message ? err.message : err));
    } finally {
      button.disabled = false;
      button.textContent = label;
    }
  });

  // example chips
  form.querySelectorAll("[data-ex]").forEach((chip) => {
    chip.addEventListener("click", () => {
      const preset = JSON.parse(chip.dataset.ex);
      for (const [name, value] of Object.entries(preset)) {
        const field = form.elements[name];
        if (!field) continue;
        if (field.type === "checkbox") field.checked = Boolean(value);
        else field.value = value;
      }
      form.dispatchEvent(new Event("submit", { cancelable: true }));
    });
  });
}

/* ── per-panel result renderers ────────────────────────────────────── */

function renderLp(out, res) {
  const extras = [];

  if (res.vertices && res.vertices.rows && res.vertices.rows.length) {
    extras.push(el("h4", { text: "Corner points of the feasible region" }));
    extras.push(renderTable(res.vertices));
  }

  if (res.tableaux && res.tableaux.length) {
    const details = el("details", { class: "tableaux" }, [
      el("summary", { text: `Simplex tableaux (${res.tableaux.length})` }),
    ]);
    res.tableaux.forEach((tab, i) => {
      details.appendChild(el("h5", { text: i === 0 ? "Tableau 0 (initial)" : `Tableau ${i}` }));
      const rendered = renderTable(tab);
      if (rendered) details.appendChild(rendered);
    });
    extras.push(details);
  }

  renderSolverResult(out, res, extras, "Solution");
}

function renderKkt(out, res) {
  const extras = [];
  if (res.active_set) {
    extras.push(el("h4", { text: "Active set at the solution" }));
    extras.push(el("p", { class: "muted", text: res.active_set }));
  }
  renderSolverResult(out, res, extras, "Active-set cases examined");
}

function renderSimple(out, res) {
  const parts = [
    el("div", { class: "summary" }, [
      el("h3", { text: res.title }),
      res.scalar ? el("dl", {}, [
        el("dt", { text: "Value" }), el("dd", { text: res.scalar }),
      ]) : null,
      res.note ? el("p", { class: "note", text: res.note }) : null,
    ]),
  ];
  const table = renderTable(res.table);
  if (table) {
    parts.push(el("h4", { text: "Result" }));
    parts.push(table);
  }
  out.replaceChildren(...parts);
}

/* ── conditional fields ────────────────────────────────────────────── */

const FIELD_SETS = {
  goldensection:   ["bracket", "tol"],
  intervalhalving: ["bracket", "tol"],
  dichotomous:     ["bracket", "tol", "delta"],
  secant:          ["bracket", "tol"],
  fibonacci:       ["bracket", "n"],
  exhaustive:      ["bracket", "n"],
  divide_conquer:  ["bracket", "n"],
  newtonraphson:   ["point", "tol"],
};

function syncOneDimFields() {
  const form = document.getElementById("form-1d");
  const wanted = FIELD_SETS[form.elements.method.value] || ["bracket", "tol"];
  form.querySelectorAll("[data-when]").forEach((group) => {
    group.hidden = !wanted.includes(group.dataset.when);
  });
}

function syncMatrixFields() {
  const form = document.getElementById("form-matrix");
  const needsRhs = form.elements.operation.value === "solve";
  form.querySelectorAll("[data-when=rhs]").forEach((g) => { g.hidden = !needsRhs; });
}

/* ── tabs ──────────────────────────────────────────────────────────── */

function wireTabs() {
  const tabs = Array.from(document.querySelectorAll(".tab"));
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.toggle("is-active", t === tab));
      document.querySelectorAll(".panel").forEach((panel) => {
        panel.classList.toggle("is-active", panel.id === tab.dataset.panel);
      });
    });
  });
}

/* ── start ─────────────────────────────────────────────────────────── */

document.addEventListener("DOMContentLoaded", () => {
  wireTabs();

  wire("form-1d", "out-1d", "run_1d");
  wire("form-multi", "out-multi", "run_multivariable");
  wire("form-lp", "out-lp", "run_lp", renderLp);
  wire("form-kkt", "out-kkt", "run_kkt", renderKkt);
  wire("form-matrix", "out-matrix", "run_matrix", renderSimple);
  wire("form-calc", "out-calc", "run_calculus", renderSimple);

  document.getElementById("form-1d").elements.method
    .addEventListener("change", syncOneDimFields);
  document.getElementById("form-matrix").elements.operation
    .addEventListener("change", syncMatrixFields);
  syncOneDimFields();
  syncMatrixFields();

  boot();
});
