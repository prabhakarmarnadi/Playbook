/* Live data bridge for the Playbook Intelligence UX.
 *
 * On page load, fetch from the FastAPI server at /api/ui/* and populate the
 * (empty) window.* globals declared in data.js. There is no mock fallback —
 * if a required endpoint fails, we render a hard-error overlay and refuse to
 * mount React.
 *
 * Override the API base via ?api=http://localhost:8090 for cross-origin dev.
 * The HTML mount waits on window.__PLAYBOOK_READY; that promise resolves
 * after success or after rendering the error overlay.
 */
(function () {
  "use strict";

  const API_BASE = (function () {
    const params = new URLSearchParams(window.location.search);
    return params.get("api") || "";
  })();

  function url(path) {
    return (API_BASE || "") + path;
  }

  async function mustFetch(path, label) {
    let r;
    try {
      r = await fetch(url(path), { headers: { Accept: "application/json" } });
    } catch (e) {
      throw new Error(`${label}: network error — ${e.message}`);
    }
    if (!r.ok) {
      throw new Error(`${label}: HTTP ${r.status} ${r.statusText}`);
    }
    return await r.json();
  }

  function mergeSeverities(serverSev) {
    if (!serverSev || typeof window.SEV !== "object") return;
    Object.keys(serverSev).forEach((k) => {
      if (window.SEV[k]) {
        window.SEV[k].label = serverSev[k].label;
        window.SEV[k].rank = serverSev[k].rank;
        window.SEV[k].glyph = serverSev[k].glyph;
      }
    });
  }

  function applyMeta(meta) {
    Object.assign(window.CORPUS_META, meta);
  }

  function applyClusters(payload) {
    window.CLUSTERS = (payload && payload.clusters) || [];
  }

  function applyRules(payload) {
    const arr = (payload && payload.rules) || [];
    window.RULES = arr.map((r) => ({
      id: r.id,
      title: r.title,
      description: r.description || "",
      applies_to: r.applies_to,
      field: r.field || null,
      cluster_label: r.cluster_label || null,
      cluster_id: r.cluster_id || null,
      predicate: r.predicate,
      severity: r.severity,
      answer_type: r.answer_type || null,
      escalation_owner: r.escalation_owner || null,
      status: r.status || "draft",
      confidence: typeof r.confidence === "number" ? r.confidence : 1.0,
      source: r.source || {},
      examples:
        r.examples && typeof r.examples === "object"
          ? { pass: r.examples.pass || [], fail: r.examples.fail || [] }
          : { pass: [], fail: [] },
      preferred_language: r.preferred_language || null,
      walkaway_language: r.walkaway_language || null,
      rationale: r.rationale || null,
      lift: r.lift,
      support: r.support,
    }));
  }

  function applyContract(c) {
    if (c && c.id) {
      Object.assign(window.ACTIVE_CONTRACT, c);
    }
  }

  function applyFindings(payload) {
    window.FINDINGS = (payload && payload.findings) || [];
  }

  function applyPortfolio(payload) {
    window.PORTFOLIO = (payload && payload.portfolio) || [];
  }

  function renderError(message, details) {
    const root = document.getElementById("root");
    if (!root) return;
    root.innerHTML = `
      <div style="
        position:fixed; inset:0;
        display:flex; align-items:center; justify-content:center;
        font-family: system-ui, -apple-system, 'Segoe UI', sans-serif;
        background: var(--paper-1, #fafaf7);
      ">
        <div style="
          max-width: 640px; padding: 32px 36px;
          background: white; border: 1px solid #e6e6e1; border-radius: 12px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.04);
        ">
          <div style="font-size: 13px; letter-spacing: 0.08em; text-transform: uppercase;
                       color: #c97c1d; margin-bottom: 12px;">
            Backend unreachable
          </div>
          <div style="font-size: 22px; font-weight: 600; color: #1c1c1a; line-height: 1.2; margin-bottom: 12px;">
            Playbook Intelligence can't reach its data API.
          </div>
          <div style="font-size: 14px; color: #4a4a45; line-height: 1.55; margin-bottom: 18px;">
            ${message}
          </div>
          <pre style="
            font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
            font-size: 12px; background: #fbfbf7; border: 1px solid #ececea;
            border-radius: 6px; padding: 12px; margin: 0 0 20px 0;
            white-space: pre-wrap; word-break: break-word; color: #6b6b65;
          ">${details}</pre>
          <div style="font-size: 13px; color: #6b6b65; line-height: 1.55;">
            Boot the FastAPI server with this DuckDB and retry:
            <pre style="
              font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
              font-size: 12px; background: #fbfbf7; border: 1px solid #ececea;
              border-radius: 6px; padding: 12px; margin: 8px 0 0 0;
              white-space: pre-wrap; word-break: break-word; color: #1c1c1a;
            ">cd .claude/hackathon
DB_PATH=data/demo.duckdb \\
  .venv_playbooks/bin/python -m uvicorn api_server:app \\
  --host 0.0.0.0 --port 8090</pre>
          </div>
        </div>
      </div>
    `;
  }

  async function loadAll() {
    // 1. Severity palette (cheap, lets us check API liveness)
    const sev = await mustFetch("/api/ui/severity_palette", "severity_palette");
    mergeSeverities(sev);

    // 2. Corpus meta — drives top bar AND picks playbook_id for downstream
    const meta = await mustFetch("/api/ui/corpus_meta", "corpus_meta");
    if (!meta.playbook_id) {
      throw new Error(
        "no playbook exists in this DuckDB. Run scripts/run_playbook_miner.py first.",
      );
    }
    applyMeta(meta);
    const playbookId = meta.playbook_id;
    const playbookQ = `?playbook_id=${encodeURIComponent(playbookId)}`;

    // 3. Fan out — clusters, rules, portfolio in parallel
    const [clusters, rules, portfolio] = await Promise.all([
      mustFetch("/api/ui/clusters", "clusters"),
      mustFetch("/api/ui/rules" + playbookQ, "rules"),
      mustFetch(
        "/api/ui/portfolio?limit=24&" + playbookQ.slice(1),
        "portfolio",
      ),
    ]);
    applyClusters(clusters);
    applyRules(rules);
    applyPortfolio(portfolio);

    if (window.RULES.length === 0) {
      throw new Error(
        "playbook has no rules. Run scripts/run_playbook_miner.py to mine drafts.",
      );
    }

    // 4. ACTIVE_CONTRACT + findings — use the first portfolio agreement
    let firstAgreementId = null;
    if (portfolio && portfolio.portfolio && portfolio.portfolio[0]) {
      firstAgreementId = portfolio.portfolio[0].id;
    }
    if (firstAgreementId) {
      const [contract, findings] = await Promise.all([
        mustFetch(
          `/api/ui/contract/${encodeURIComponent(firstAgreementId)}`,
          "contract",
        ),
        mustFetch(
          `/api/ui/findings?agreement_id=${encodeURIComponent(firstAgreementId)}`,
          "findings",
        ),
      ]);
      applyContract(contract);
      applyFindings(findings);
    }

    window.__PLAYBOOK_LIVE = true;
    console.log("[data_loader] live data loaded.", {
      playbook_id: window.CORPUS_META.playbook_id,
      rules: window.RULES.length,
      clusters: window.CLUSTERS.length,
      portfolio: window.PORTFOLIO.length,
      findings: window.FINDINGS.length,
      active_contract: window.ACTIVE_CONTRACT.id,
    });
  }

  // The HTML mount awaits this promise. On error we render an overlay,
  // mark the failure on window, and still resolve so the page doesn't hang.
  window.__PLAYBOOK_READY = loadAll().catch((e) => {
    const baseHint = API_BASE
      ? `Tried ${API_BASE} (override via ?api=...)`
      : "Tried same-origin /api/ui/*";
    console.error("[data_loader] FATAL:", e.message);
    window.__PLAYBOOK_LIVE = false;
    window.__PLAYBOOK_ERROR = e.message;
    renderError(
      `Could not load required data from the backend. ${baseHint}.`,
      e.message,
    );
    // Resolve, do NOT throw — the HTML mount checks __PLAYBOOK_LIVE.
  });
})();
