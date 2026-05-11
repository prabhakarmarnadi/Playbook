/* Playbook Intelligence — runtime data bootstrap.
 *
 * The substantive data (CORPUS_META, CLUSTERS, RULES, ACTIVE_CONTRACT,
 * FINDINGS, PORTFOLIO) is populated by data_loader.js from /api/ui/*.
 * This file ONLY provides:
 *   1. Empty default globals so component references don't crash before
 *      the API responses arrive.
 *   2. The SEV palette — keyed to CSS variables in ds/colors_and_type.css.
 *      Live data overwrites label/rank/glyph on each entry but preserves
 *      the color tokens declared here.
 *
 * The original mock corpus is preserved at data.mock.js.bak for reference.
 */

window.CORPUS_META = {
  playbook_id: null,
  playbook_label: "Loading…",
  domain: "",
  agreement_count: 0,
  mined_at: null,
  drafts_pending: 0,
  rules_active: 0,
  rules_retired: 0,
};

window.CLUSTERS = [];
window.RULES = [];
window.ACTIVE_CONTRACT = {
  id: null,
  name: "",
  counterparty: null,
  uploaded_at: null,
  pages: null,
  word_count: null,
  governing_law: null,
  aligned_at: null,
  duration_ms: null,
};
window.FINDINGS = [];
window.PORTFOLIO = [];

window.SEV = {
  blocker: {
    label: "Blocker",
    color: "var(--status-error-500)",
    bg: "var(--status-error-50)",
    fg: "var(--status-error-700)",
    rank: 0,
    glyph: "x",
  },
  approval_required: {
    label: "Approval required",
    color: "#c97c1d",
    bg: "#fbf0dc",
    fg: "#7f4c0e",
    rank: 1,
    glyph: "alert",
  },
  warn: {
    label: "Warn",
    color: "var(--status-warn-500)",
    bg: "var(--status-warn-50)",
    fg: "var(--status-warn-700)",
    rank: 2,
    glyph: "tri",
  },
  info: {
    label: "Info",
    color: "var(--status-info-500)",
    bg: "var(--status-info-50)",
    fg: "var(--status-info-700)",
    rank: 3,
    glyph: "i",
  },
  pass: {
    label: "Pass",
    color: "var(--status-ok-500)",
    bg: "var(--status-ok-50)",
    fg: "var(--status-ok-700)",
    rank: 4,
    glyph: "ok",
  },
  na: {
    label: "N/A",
    color: "var(--ink-400)",
    bg: "var(--paper-2)",
    fg: "var(--ink-600)",
    rank: 5,
    glyph: "·",
  },
};
