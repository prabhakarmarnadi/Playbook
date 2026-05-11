/* Shared components for Playbook Intelligence */

const { useState, useMemo, useEffect, useRef } = React;

/* ────────────────────────── Top bar ────────────────────────── */
function TopBar({ page, setPage, playbookMeta }) {
  return (
    <div className="topbar">
      <span className="brand-mark">§</span>
      <span className="brand-name">Clausework</span>
      <span style={{ width: 14 }} />
      <span className="crumb">
        Playbooks <span className="sep">/</span> Consulting{" "}
        <span className="sep">/</span>
        <span className="here">{playbookMeta.playbook_id}</span>
      </span>
      <div className="tab-rail">
        <button
          className={`tab ${page === "studio" ? "active" : ""}`}
          onClick={() => setPage("studio")}
        >
          Playbook Studio
          <span className="tab-count">{playbookMeta.drafts_pending}</span>
        </button>
        <button
          className={`tab ${page === "alignment" ? "active" : ""}`}
          onClick={() => setPage("alignment")}
        >
          Contract Alignment
        </button>
      </div>
      <span className="spacer" />
      <div className="kbd-hint">
        <kbd>⌘</kbd>
        <kbd>K</kbd> search
      </div>
      <button className="btn btn-secondary">Export</button>
      <button className="btn btn-primary">Publish</button>
    </div>
  );
}

/* ────────────────────────── Severity badge ────────────────────────── */
function SevBadge({ severity, palette = "warm", size = "md" }) {
  const info = SEV[severity] || SEV.info;
  const colorOverride = severityColor(severity, palette);
  const glyph = sevGlyph(info.glyph);
  return (
    <span
      className={`sev-badge sev-${severity} sev-${size}`}
      style={{
        background: colorOverride.bg,
        color: colorOverride.fg,
        borderColor: colorOverride.border,
      }}
    >
      <span className="sev-glyph" style={{ color: colorOverride.color }}>
        {glyph}
      </span>
      {info.label}
    </span>
  );
}
/* Phosphor icon renderer. Each severity glyph maps to a Phosphor class.
   Renders a real <i className="ph ph-..."/> so the icon picks up the
   currentColor / .sev-glyph styling. */
const _PH_CLASS = {
  x: "ph-x-circle",
  alert: "ph-warning",
  tri: "ph-warning-octagon",
  i: "ph-info",
  ok: "ph-check-circle",
  "·": "ph-circle",
};
function sevGlyph(g) {
  const cls = _PH_CLASS[g] || "ph-circle";
  return React.createElement("i", {
    className: `ph ${cls}`,
    "aria-hidden": "true",
  });
}
function severityColor(sev, palette) {
  const warm = {
    blocker: {
      color: "#b8392a",
      bg: "#fbeeec",
      fg: "#7c2417",
      border: "#f4d2cc",
    },
    approval_required: {
      color: "#c97c1d",
      bg: "#fbf0dc",
      fg: "#7f4c0e",
      border: "#f1ddb1",
    },
    warn: { color: "#b07f1e", bg: "#fbf3df", fg: "#76520f", border: "#f3e2b3" },
    info: { color: "#3d6889", bg: "#eaf0f5", fg: "#244360", border: "#c8d6e3" },
    pass: { color: "#527030", bg: "#ecf1e6", fg: "#354b1b", border: "#cfdcbd" },
  };
  const traffic = {
    blocker: {
      color: "#d63b1f",
      bg: "#fde6e1",
      fg: "#7c1d0f",
      border: "#f4c4b8",
    },
    approval_required: {
      color: "#e8901a",
      bg: "#fdefd8",
      fg: "#7a4708",
      border: "#f5d6a3",
    },
    warn: { color: "#e6b800", bg: "#fbf3cf", fg: "#5c4900", border: "#f0e09a" },
    info: { color: "#2576c4", bg: "#e2eef9", fg: "#0e3a66", border: "#bcd6ed" },
    pass: { color: "#2f8a4a", bg: "#dff0e3", fg: "#0f4a22", border: "#b8dcc1" },
  };
  const ink = {
    blocker: {
      color: "#1c1a17",
      bg: "#f1ede2",
      fg: "#1c1a17",
      border: "#1c1a17",
    },
    approval_required: {
      color: "#1c1a17",
      bg: "#f7f5ee",
      fg: "#1c1a17",
      border: "#46413a",
    },
    warn: { color: "#1c1a17", bg: "#f7f5ee", fg: "#46413a", border: "#6b6358" },
    info: { color: "#46413a", bg: "#f7f5ee", fg: "#46413a", border: "#b0a999" },
    pass: { color: "#6b6358", bg: "#fbfaf6", fg: "#6b6358", border: "#d6cfbe" },
  };
  return ({ warm, traffic, ink }[palette] || warm)[sev] || warm.info;
}

/* ────────────────────────── Source provenance chip ────────────────────────── */
function ProvenanceChip({ source }) {
  if (source.miner === "coverage") {
    return (
      <span className="prov-chip">
        <b>Coverage</b> {Math.round(source.ratio * 100)}% · n={source.n}
      </span>
    );
  }
  if (source.miner === "distribution") {
    return (
      <span className="prov-chip">
        <b>Distribution</b> p10–p90 [{source.p10}, {source.p90}] · median{" "}
        {source.median}
      </span>
    );
  }
  if (source.miner === "categorical") {
    return (
      <span className="prov-chip">
        <b>Mode</b> "{source.mode}" {Math.round(source.frequency * 100)}% · n=
        {source.n}
      </span>
    );
  }
  if (source.miner === "outlier") {
    return (
      <span className="prov-chip">
        <b>Outlier</b> {Math.round(source.outlier_pct * 100)}% off-curve · n=
        {source.n}
      </span>
    );
  }
  if (source.miner === "contrastive") {
    return (
      <span className="prov-chip">
        <b>Lift</b> ratio {source.lift_ratio.toFixed(2)}× · n={source.n}
      </span>
    );
  }
  if (source.migrated_from) {
    return (
      <span className="prov-chip migrated">
        <b>Migrated</b> from {source.migrated_from}
      </span>
    );
  }
  return <span className="prov-chip">—</span>;
}

/* ────────────────────────── Predicate renderer ────────────────────────── */
function Predicate({ node, depth = 0 }) {
  if (!node || typeof node !== "object")
    return <span className="pred-lit">{JSON.stringify(node)}</span>;
  const { op, args } = node;
  if (op === "field.between") {
    return (
      <span className="pred-line">
        <span className="pred-op">field.between</span>(
        <span className="pred-id">{args[0]}</span>,
        <span className="pred-num">{args[1]}</span>,
        <span className="pred-num">{args[2]}</span>)
      </span>
    );
  }
  if (op === "field.eq") {
    return (
      <span className="pred-line">
        <span className="pred-op">field.eq</span>(
        <span className="pred-id">{args[0]}</span>,
        <span className="pred-str">"{args[1]}"</span>)
      </span>
    );
  }
  if (op === "clause.classified_as") {
    return (
      <span className="pred-line">
        <span className="pred-op">clause.classified_as</span>(
        <span className="pred-str">"{args[0]}"</span>)
      </span>
    );
  }
  if (op === "not") {
    return (
      <span className="pred-line">
        <span className="pred-op">not</span>(
        <Predicate node={args[0]} depth={depth + 1} />)
      </span>
    );
  }
  if (op === "any_of" || op === "and" || op === "or") {
    return (
      <span className="pred-block">
        <span className="pred-op">{op}</span>(
        <div className="pred-children">
          {args.map((a, i) => (
            <div key={i} className="pred-child">
              <Predicate node={a} depth={depth + 1} />
              {i < args.length - 1 && <span className="pred-comma">,</span>}
            </div>
          ))}
        </div>
        )
      </span>
    );
  }
  if (op === "if_then") {
    return (
      <span className="pred-block">
        <span className="pred-op">if</span>{" "}
        <Predicate node={args[0]} depth={depth + 1} />
        <div className="pred-children">
          <span className="pred-op">then</span>{" "}
          <Predicate node={args[1]} depth={depth + 1} />
        </div>
      </span>
    );
  }
  return (
    <span className="pred-line">
      <span className="pred-op">{op}</span>(…)
    </span>
  );
}

/* ────────────────────────── Mini sparkbar / distribution ────────────────────────── */
function MiniHistogram({ source }) {
  if (source.miner !== "distribution") return null;
  const { p10, p25, median, p75, p90 } = source;
  const lo = p10,
    hi = p90,
    span = hi - lo || 1;
  const x = (v) => ((v - lo) / span) * 100;
  return (
    <svg viewBox="0 0 100 24" className="mini-hist" preserveAspectRatio="none">
      <line
        x1="0"
        y1="12"
        x2="100"
        y2="12"
        stroke="var(--ink-200)"
        strokeWidth="1"
      />
      <rect
        x={x(p25)}
        y="6"
        width={x(p75) - x(p25)}
        height="12"
        fill="var(--accent-100)"
        stroke="var(--accent-300)"
        strokeWidth="0.5"
      />
      <line
        x1={x(median)}
        y1="3"
        x2={x(median)}
        y2="21"
        stroke="var(--accent-500)"
        strokeWidth="1.5"
      />
      <line
        x1={x(p10)}
        y1="8"
        x2={x(p10)}
        y2="16"
        stroke="var(--ink-400)"
        strokeWidth="1"
      />
      <line
        x1={x(p90)}
        y1="8"
        x2={x(p90)}
        y2="16"
        stroke="var(--ink-400)"
        strokeWidth="1"
      />
      <text
        x="0"
        y="23"
        fontSize="6"
        fill="var(--ink-600)"
        fontFamily="var(--font-mono)"
      >
        {p10}
      </text>
      <text
        x="100"
        y="23"
        fontSize="6"
        fill="var(--ink-600)"
        textAnchor="end"
        fontFamily="var(--font-mono)"
      >
        {p90}
      </text>
    </svg>
  );
}

function CategoricalBars({ source }) {
  if (source.miner !== "categorical" || !source.distribution) return null;
  const entries = Object.entries(source.distribution);
  const total = entries.reduce((s, [, v]) => s + v, 0);
  return (
    <div className="cat-bars">
      {entries.map(([label, v]) => {
        const pct = (v / total) * 100;
        const isMode = label === source.mode;
        return (
          <div key={label} className={`cat-bar ${isMode ? "mode" : ""}`}>
            <div className="cat-bar-track">
              <div className="cat-bar-fill" style={{ width: `${pct}%` }} />
            </div>
            <div className="cat-bar-label">
              <span>{label}</span>
              <span className="cat-bar-pct">{Math.round(pct)}%</span>
            </div>
          </div>
        );
      })}
    </div>
  );
}

/* ────────────────────────── Lift / support / N indicator ────────────────────────── */
function TrustMeter({ rule }) {
  const lift = rule.lift;
  const support = rule.support;
  const n = rule.source.n || 0;
  const conf = rule.confidence;
  return (
    <div className="trust-row">
      <div
        className="trust-cell"
        title="Lift = how much more this pattern occurs within-cluster vs global"
      >
        <span className="trust-num">
          {lift == null ? "—" : lift.toFixed(1) + "×"}
        </span>
        <span className="trust-lbl">lift</span>
      </div>
      <div className="trust-cell" title="Fraction of corpus this rule matches">
        <span className="trust-num">
          {support == null ? "—" : Math.round(support * 100) + "%"}
        </span>
        <span className="trust-lbl">support</span>
      </div>
      <div
        className="trust-cell"
        title="Corpus sample size used to mine this rule"
      >
        <span className="trust-num">{n}</span>
        <span className="trust-lbl">n</span>
      </div>
      <div className="trust-cell" title="Mined-rule confidence">
        <span className="trust-num">{Math.round(conf * 100)}%</span>
        <span className="trust-lbl">conf.</span>
      </div>
    </div>
  );
}

/* ────────────────────────── Status badges ────────────────────────── */
function StatusBadge({ status }) {
  const map = {
    draft: {
      bg: "var(--paper-3)",
      fg: "var(--ink-700)",
      border: "var(--ink-300)",
      label: "Draft",
    },
    active: {
      bg: "#ecf1e6",
      fg: "#354b1b",
      border: "#cfdcbd",
      label: "Active",
    },
    retired: {
      bg: "var(--paper-2)",
      fg: "var(--ink-500)",
      border: "var(--ink-200)",
      label: "Retired",
    },
  };
  const s = map[status] || map.draft;
  return (
    <span
      className="status-badge"
      style={{ background: s.bg, color: s.fg, borderColor: s.border }}
    >
      {s.label}
    </span>
  );
}

/* ────────────────────────── Export to window ────────────────────────── */
Object.assign(window, {
  TopBar,
  SevBadge,
  ProvenanceChip,
  Predicate,
  MiniHistogram,
  CategoricalBars,
  TrustMeter,
  StatusBadge,
  severityColor,
  sevGlyph,
});
