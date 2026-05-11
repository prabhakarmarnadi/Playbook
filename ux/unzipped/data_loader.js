/* Live data bridge for the Playbook Intelligence UX.
 *
 * On page load, fetch from the FastAPI server at /api/ui/* and overwrite
 * the mock window.* globals defined in data.js. Falls back silently to the
 * mock data if any fetch fails (e.g. when opening the HTML file directly
 * from disk).
 *
 * Loaded AFTER data.js so the mock data is the baseline. We swap fields
 * with live values when the API responses arrive.
 *
 * The HTML uses an async gate (window.__PLAYBOOK_READY) so React waits
 * for live data before mounting. If the API is unreachable, the gate
 * resolves immediately with the mocks.
 */
(function () {
  "use strict";

  const API_BASE = (function () {
    // Same-origin when served from /ui/ on FastAPI. Allow override via
    // ?api=http://localhost:8000 for cross-origin dev.
    const params = new URLSearchParams(window.location.search);
    return params.get("api") || "";
  })();

  function url(path) {
    return (API_BASE || "") + path;
  }

  async function tryFetch(path, label) {
    try {
      const r = await fetch(url(path), {
        headers: { Accept: "application/json" },
      });
      if (!r.ok) {
        console.warn(`[data_loader] ${label}: HTTP ${r.status}`);
        return null;
      }
      return await r.json();
    } catch (e) {
      console.warn(`[data_loader] ${label}: ${e.message}`);
      return null;
    }
  }

  // Severity palette — overwrite SEV's rank/glyph/label from server but keep
  // the CSS variable references local to data.js. Don't replace the whole obj.
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
    if (!meta || !meta.playbook_id) return;
    Object.assign(window.CORPUS_META, meta);
  }

  function applyClusters(payload) {
    if (
      !payload ||
      !Array.isArray(payload.clusters) ||
      payload.clusters.length === 0
    )
      return;
    window.CLUSTERS = payload.clusters;
  }

  function applyRules(payload) {
    if (!payload || !Array.isArray(payload.rules) || payload.rules.length === 0)
      return;
    // The UX rule cards need at least one element with .id. Live data
    // satisfies that — replace wholesale.
    window.RULES = payload.rules.map((r) => ({
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
    if (!c || !c.id) return;
    Object.assign(window.ACTIVE_CONTRACT, c);
  }

  function applyFindings(payload) {
    if (
      !payload ||
      !Array.isArray(payload.findings) ||
      payload.findings.length === 0
    )
      return;
    window.FINDINGS = payload.findings;
  }

  function applyPortfolio(payload) {
    if (
      !payload ||
      !Array.isArray(payload.portfolio) ||
      payload.portfolio.length === 0
    )
      return;
    window.PORTFOLIO = payload.portfolio;
  }

  async function loadAll() {
    // Severity palette first — UI cards rely on SEV being populated.
    const sev = await tryFetch("/api/ui/severity_palette", "severity_palette");
    mergeSeverities(sev);

    // Corpus meta drives the top bar AND picks which playbook id to use.
    const meta = await tryFetch("/api/ui/corpus_meta", "corpus_meta");
    applyMeta(meta);
    const playbookId = meta && meta.playbook_id;

    const clustersP = tryFetch("/api/ui/clusters", "clusters");
    const rulesP = tryFetch(
      "/api/ui/rules" +
        (playbookId ? `?playbook_id=${encodeURIComponent(playbookId)}` : ""),
      "rules",
    );
    const portfolioP = tryFetch(
      "/api/ui/portfolio?limit=24" +
        (playbookId ? `&playbook_id=${encodeURIComponent(playbookId)}` : ""),
      "portfolio",
    );

    // ACTIVE_CONTRACT: try the first agreement from the portfolio first; falls
    // back to ACTIVE_CONTRACT mock id if portfolio is empty.
    const portfolio = await portfolioP;
    applyPortfolio(portfolio);
    let firstAgreementId = window.ACTIVE_CONTRACT && window.ACTIVE_CONTRACT.id;
    if (portfolio && portfolio.portfolio && portfolio.portfolio[0]) {
      firstAgreementId = portfolio.portfolio[0].id;
    }
    const contractP = firstAgreementId
      ? tryFetch(
          `/api/ui/contract/${encodeURIComponent(firstAgreementId)}`,
          "contract",
        )
      : Promise.resolve(null);

    const [clusters, rules, contract] = await Promise.all([
      clustersP,
      rulesP,
      contractP,
    ]);
    applyClusters(clusters);
    applyRules(rules);
    applyContract(contract);

    // Findings for the chosen contract (most recent run).
    if (firstAgreementId) {
      const findings = await tryFetch(
        `/api/ui/findings?agreement_id=${encodeURIComponent(firstAgreementId)}`,
        "findings",
      );
      applyFindings(findings);
    }

    // Tag the page so the React app can show a small "Live" badge if it
    // chooses to (the existing UX ignores this; safe to add).
    window.__PLAYBOOK_LIVE = true;
    console.log("[data_loader] live data loaded.", {
      playbook_id: window.CORPUS_META && window.CORPUS_META.playbook_id,
      rules: window.RULES && window.RULES.length,
      clusters: window.CLUSTERS && window.CLUSTERS.length,
      findings: window.FINDINGS && window.FINDINGS.length,
    });
  }

  // Expose a promise the page can await before mounting React.
  window.__PLAYBOOK_READY = loadAll().catch((e) => {
    console.warn("[data_loader] fallback to mock data:", e.message);
  });
})();
