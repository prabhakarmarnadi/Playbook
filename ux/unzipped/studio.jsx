/* Playbook Studio — triage drafts. 3 layout variants (Tweaks toggles). */

const { useState: useStateS, useMemo: useMemoS } = React;

function PlaybookStudio({ tweaks, openRule, focusRule, setFocusRule }) {
  const layout = tweaks.studio_layout;
  const density = tweaks.density;
  const palette = tweaks.severity_palette;

  const [filterSev, setFilterSev] = useStateS("all");
  const [filterApplies, setFilterApplies] = useStateS("all");
  const [filterSource, setFilterSource] = useStateS("all");
  const [filterCluster, setFilterCluster] = useStateS("all");
  const [accepted, setAccepted] = useStateS({});
  const [retired, setRetired] = useStateS({});

  const rules = useMemoS(
    () =>
      RULES.filter((r) => {
        if (filterSev !== "all" && r.severity !== filterSev) return false;
        if (filterApplies !== "all" && r.applies_to !== filterApplies)
          return false;
        if (filterSource !== "all") {
          const m =
            r.source.miner || (r.source.migrated_from ? "migrated" : "");
          if (m !== filterSource) return false;
        }
        if (filterCluster !== "all" && r.cluster_id !== filterCluster)
          return false;
        return true;
      }),
    [filterSev, filterApplies, filterSource, filterCluster],
  );

  const focus = rules.find((r) => r.id === focusRule) || rules[0];

  function ruleState(r) {
    if (accepted[r.id]) return "active";
    if (retired[r.id]) return "retired";
    return r.status;
  }

  return (
    <div className={`studio studio-${layout} density-${density}`}>
      {/* ─── Toolbar ─── */}
      <div className="studio-toolbar">
        <div className="tb-group">
          <span className="eyebrow">Triage</span>
          <span className="tb-count">
            {rules.length} drafts · {Object.keys(accepted).length} accepted this
            session
          </span>
        </div>
        <div className="tb-group filters">
          <FilterPill
            label="Severity"
            value={filterSev}
            onChange={setFilterSev}
            options={[
              ["all", "All"],
              ["blocker", "Blocker"],
              ["approval_required", "Approval"],
              ["warn", "Warn"],
              ["info", "Info"],
            ]}
          />
          <FilterPill
            label="Applies to"
            value={filterApplies}
            onChange={setFilterApplies}
            options={[
              ["all", "All"],
              ["domain", "Domain"],
              ["cluster", "Cluster"],
              ["field", "Field"],
              ["composite", "Composite"],
              ["cross_field", "Cross-field"],
            ]}
          />
          <FilterPill
            label="Source"
            value={filterSource}
            onChange={setFilterSource}
            options={[
              ["all", "All"],
              ["coverage", "Coverage"],
              ["distribution", "Distribution"],
              ["categorical", "Categorical"],
              ["contrastive", "Contrastive"],
              ["outlier", "Outlier"],
              ["migrated", "Migrated"],
            ]}
          />
        </div>
        <div className="tb-group right">
          <button className="btn btn-ghost">Migrate legacy benchmarks</button>
          <button className="btn btn-secondary">
            Bulk accept ({rules.length})
          </button>
        </div>
      </div>

      {/* ─── Layout: slim (default), card-feed, kanban, table ─── */}
      {layout === "slim" && (
        <div className="studio-slim-body">
          <ClusterRail
            clusters={CLUSTERS}
            rules={RULES}
            active={filterCluster}
            onPick={(cid) => setFilterCluster(cid)}
          />
          <div className="feed-col">
            {rules.length === 0 && (
              <div
                style={{
                  padding: "32px 22px",
                  color: "var(--ink-500)",
                  fontSize: 13,
                }}
              >
                No rules match the current filters.
              </div>
            )}
            {rules.map((r) => (
              <RuleCard
                key={r.id}
                rule={r}
                state={ruleState(r)}
                palette={palette}
                focused={focus && focus.id === r.id}
                onFocus={() => setFocusRule(r.id)}
                onAccept={() => setAccepted({ ...accepted, [r.id]: true })}
                onRetire={() => setRetired({ ...retired, [r.id]: true })}
              />
            ))}
          </div>
          <div className="canvas-col">
            {focus && (
              <RuleDetail
                rule={focus}
                state={ruleState(focus)}
                palette={palette}
                onAccept={() => setAccepted({ ...accepted, [focus.id]: true })}
                onRetire={() => setRetired({ ...retired, [focus.id]: true })}
              />
            )}
          </div>
        </div>
      )}

      {layout === "card-feed" && (
        <div className="studio-body two-pane">
          <div className="feed-col">
            {rules.map((r) => (
              <RuleCard
                key={r.id}
                rule={r}
                state={ruleState(r)}
                palette={palette}
                focused={focus && focus.id === r.id}
                onFocus={() => setFocusRule(r.id)}
                onAccept={() => setAccepted({ ...accepted, [r.id]: true })}
                onRetire={() => setRetired({ ...retired, [r.id]: true })}
              />
            ))}
          </div>
          <div className="detail-col">
            {focus && (
              <RuleDetail
                rule={focus}
                state={ruleState(focus)}
                palette={palette}
                onAccept={() => setAccepted({ ...accepted, [focus.id]: true })}
                onRetire={() => setRetired({ ...retired, [focus.id]: true })}
              />
            )}
          </div>
        </div>
      )}

      {layout === "kanban" && (
        <div className="studio-body kanban-body">
          {[
            {
              key: "draft",
              label: "Draft",
              rules: rules.filter((r) => ruleState(r) === "draft"),
            },
            {
              key: "review",
              label: "In Review",
              rules: rules.filter(
                (r) =>
                  ruleState(r) === "draft" &&
                  (r.confidence < 0.85 ||
                    r.severity === "approval_required" ||
                    r.severity === "blocker"),
              ),
            },
            {
              key: "active",
              label: "Approved",
              rules: rules.filter((r) => ruleState(r) === "active"),
            },
            {
              key: "retired",
              label: "Retired",
              rules: rules.filter((r) => ruleState(r) === "retired"),
            },
          ].map((col) => (
            <div key={col.key} className="kanban-col">
              <div className="kanban-head">
                <span>{col.label}</span>
                <span className="kanban-count">{col.rules.length}</span>
              </div>
              <div className="kanban-list">
                {col.rules.length === 0 && (
                  <div className="kanban-empty">No rules</div>
                )}
                {col.rules.map((r) => (
                  <div
                    key={r.id}
                    className="kanban-card"
                    onClick={() => setFocusRule(r.id)}
                  >
                    <div className="kc-head">
                      <SevBadge
                        severity={r.severity}
                        size="sm"
                        palette={palette}
                      />
                      <span className="kc-applies">{r.applies_to}</span>
                    </div>
                    <div className="kc-title">{r.title}</div>
                    <div className="kc-trust">
                      <span>
                        {r.lift == null ? "—" : r.lift.toFixed(1) + "×"} lift
                      </span>
                      <span className="kc-dot">·</span>
                      <span>n={r.source.n || "—"}</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      {layout === "table" && (
        <div className="studio-body two-pane">
          <div className="feed-col">
            <table className="rules-table">
              <thead>
                <tr>
                  <th>Severity</th>
                  <th>Title</th>
                  <th>Applies</th>
                  <th>Source</th>
                  <th className="rt-num">Lift</th>
                  <th className="rt-num">n</th>
                  <th className="rt-num">Conf.</th>
                  <th></th>
                </tr>
              </thead>
              <tbody>
                {rules.map((r) => (
                  <tr
                    key={r.id}
                    className={focus && focus.id === r.id ? "selected" : ""}
                    onClick={() => setFocusRule(r.id)}
                  >
                    <td>
                      <SevBadge
                        severity={r.severity}
                        size="sm"
                        palette={palette}
                      />
                    </td>
                    <td className="rt-title">
                      {r.title}
                      <div className="rt-sub mono">
                        {r.cluster_label} · {ruleState(r)}
                      </div>
                    </td>
                    <td>
                      <span className="rt-tag">{r.applies_to}</span>
                    </td>
                    <td>
                      <ProvenanceChip source={r.source} />
                    </td>
                    <td className="rt-num">
                      {r.lift == null ? "—" : r.lift.toFixed(1) + "×"}
                    </td>
                    <td className="rt-num">{r.source.n || "—"}</td>
                    <td className="rt-num">
                      {Math.round(r.confidence * 100)}%
                    </td>
                    <td>
                      <button className="btn btn-ghost btn-xs">Open</button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="detail-col">
            {focus && (
              <RuleDetail
                rule={focus}
                state={ruleState(focus)}
                palette={palette}
                onAccept={() => setAccepted({ ...accepted, [focus.id]: true })}
                onRetire={() => setRetired({ ...retired, [focus.id]: true })}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/* ────────────────────────── Filter pill ────────────────────────── */
function FilterPill({ label, value, onChange, options }) {
  return (
    <label className="filter-pill">
      <span className="fp-label">{label}</span>
      <select value={value} onChange={(e) => onChange(e.target.value)}>
        {options.map(([v, l]) => (
          <option key={v} value={v}>
            {l}
          </option>
        ))}
      </select>
    </label>
  );
}

/* ────────────────────────── Rule card (feed-row) ────────────────────────── */
function RuleCard({
  rule,
  state,
  palette,
  focused,
  onFocus,
  onAccept,
  onRetire,
}) {
  return (
    <div
      className={`rule-card ${focused ? "focused" : ""} state-${state}`}
      onClick={onFocus}
    >
      <div
        className="rc-edge"
        style={{ background: severityColor(rule.severity, palette).color }}
      />
      <div className="rc-body">
        <div className="rc-top">
          <SevBadge severity={rule.severity} palette={palette} />
          <span className="rc-applies">{rule.applies_to}</span>
          <span className="rc-cluster mono">{rule.cluster_label}</span>
          <span className="rc-spacer" />
          <StatusBadge status={state} />
        </div>
        <h3 className="rc-title">{rule.title}</h3>
        <div className="rc-prov">
          <ProvenanceChip source={rule.source} />
          {rule.source.miner === "distribution" && (
            <MiniHistogram source={rule.source} />
          )}
        </div>
        <TrustMeter rule={rule} />
        <div className="rc-actions">
          <button
            className="btn btn-primary btn-sm"
            onClick={(e) => {
              e.stopPropagation();
              onAccept();
            }}
            disabled={state === "active"}
          >
            Accept
          </button>
          <button
            className="btn btn-secondary btn-sm"
            onClick={(e) => {
              e.stopPropagation();
              onFocus();
            }}
          >
            Edit
          </button>
          <button
            className="btn btn-ghost btn-sm"
            onClick={(e) => {
              e.stopPropagation();
              onRetire();
            }}
            disabled={state === "retired"}
          >
            Retire
          </button>
        </div>
      </div>
    </div>
  );
}

/* ────────────────────────── Rule detail panel ────────────────────────── */
function RuleDetail({ rule, state, palette, onAccept, onRetire }) {
  const [tab, setTab] = useStateS("examples");
  const [showJson, setShowJson] = useStateS(false);
  const [editSev, setEditSev] = useStateS(rule.severity);
  const [editOwner, setEditOwner] = useStateS(rule.escalation_owner || "");

  // reset edits when rule changes
  React.useEffect(() => {
    setEditSev(rule.severity);
    setEditOwner(rule.escalation_owner || "");
    setTab("examples");
  }, [rule.id]);

  return (
    <div className="rule-detail">
      <div className="rd-head">
        <div className="rd-head-top">
          <span className="rd-id mono">{rule.id}</span>
          <StatusBadge status={state} />
        </div>
        <h2 className="rd-title">{rule.title}</h2>
        <div className="rd-meta">
          <span className="mono">{rule.applies_to}</span>
          <span className="rd-sep">·</span>
          <span>{rule.cluster_label}</span>
          <span className="rd-sep">·</span>
          <ProvenanceChip source={rule.source} />
        </div>
      </div>

      <div className="rd-tabs">
        {["examples", "predicate", "language", "provenance", "rationale"].map(
          (t) => (
            <button
              key={t}
              className={`rd-tab ${tab === t ? "active" : ""}`}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ),
        )}
      </div>

      <div className="rd-body">
        {tab === "examples" && <ExamplesPane rule={rule} palette={palette} />}
        {tab === "predicate" && (
          <PredicatePane
            rule={rule}
            showJson={showJson}
            setShowJson={setShowJson}
          />
        )}
        {tab === "language" && <LanguagePane rule={rule} />}
        {tab === "provenance" && <ProvenancePane rule={rule} />}
        {tab === "rationale" && <RationalePane rule={rule} />}
      </div>

      <div className="rd-edit">
        <div className="rd-edit-row">
          <label className="rd-edit-label">Severity</label>
          <select
            value={editSev}
            onChange={(e) => setEditSev(e.target.value)}
            className="rd-select"
          >
            <option value="blocker">Blocker</option>
            <option value="approval_required">Approval required</option>
            <option value="warn">Warn</option>
            <option value="info">Info</option>
          </select>
          <label className="rd-edit-label">Escalation</label>
          <input
            className="rd-input"
            value={editOwner}
            onChange={(e) => setEditOwner(e.target.value)}
          />
        </div>
        <div className="rd-actions">
          <button
            className="btn btn-primary"
            onClick={onAccept}
            disabled={state === "active"}
          >
            Accept &amp; publish
          </button>
          <button className="btn btn-secondary">Edit predicate</button>
          <button
            className="btn btn-ghost"
            onClick={onRetire}
            disabled={state === "retired"}
          >
            Retire
          </button>
        </div>
      </div>
    </div>
  );
}

/* ────────────────────────── Detail tabs ────────────────────────── */
function ExamplesPane({ rule, palette }) {
  return (
    <div className="ex-pane">
      <div className="ex-col">
        <div className="ex-head pass">
          <SevBadge severity="pass" palette={palette} size="sm" />
          <span>Examples that pass ({rule.examples.pass.length})</span>
        </div>
        {rule.examples.pass.map((ex, i) => (
          <div key={i} className="ex-card pass">
            <div className="ex-meta">
              <span className="mono">{ex.agreement}</span>
              {ex.value && <span className="ex-val">{ex.value}</span>}
            </div>
            <p className="ex-snippet prose-document">{ex.snippet}</p>
            <a className="ex-jump" href="#">
              Jump to evidence ↗
            </a>
          </div>
        ))}
      </div>
      <div className="ex-col">
        <div className="ex-head fail">
          <SevBadge severity={rule.severity} palette={palette} size="sm" />
          <span>Examples that fail ({rule.examples.fail.length})</span>
        </div>
        {rule.examples.fail.map((ex, i) => (
          <div key={i} className="ex-card fail">
            <div className="ex-meta">
              <span className="mono">{ex.agreement}</span>
              {ex.value && <span className="ex-val">{ex.value}</span>}
            </div>
            <p className="ex-snippet prose-document">{ex.snippet}</p>
            <a className="ex-jump" href="#">
              Jump to evidence ↗
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}

function PredicatePane({ rule, showJson, setShowJson }) {
  return (
    <div className="pred-pane">
      <div className="pred-controls">
        <label className="json-toggle">
          <input
            type="checkbox"
            checked={showJson}
            onChange={(e) => setShowJson(e.target.checked)}
          />
          Show raw JSON
        </label>
        <span className="mono pred-applies">applies_to: {rule.applies_to}</span>
      </div>
      {!showJson ? (
        <div className="pred-structured">
          <Predicate node={rule.predicate} />
        </div>
      ) : (
        <pre className="pred-json mono">
          {JSON.stringify(rule.predicate, null, 2)}
        </pre>
      )}
      <div className="pred-explain">
        At alignment time this predicate evaluates over each contract's
        extracted fields and clause classifications. Returns{" "}
        <span className="mono">true</span> when the contract conforms.
      </div>
    </div>
  );
}

function LanguagePane({ rule }) {
  return (
    <div className="lang-pane">
      {rule.preferred_language && (
        <div className="lang-block">
          <div className="lang-label">
            <span className="eyebrow">Preferred</span>
            <button className="btn btn-ghost btn-xs">Edit</button>
          </div>
          <div className="lang-prose prose-document">
            {rule.preferred_language}
          </div>
        </div>
      )}
      {rule.walkaway_language && (
        <div className="lang-block walkaway">
          <div className="lang-label">
            <span className="eyebrow">Walkaway / anti-pattern</span>
          </div>
          <div className="lang-prose prose-document">
            {rule.walkaway_language}
          </div>
        </div>
      )}
      {!rule.preferred_language && !rule.walkaway_language && (
        <div className="lang-empty">
          No standard language captured for this rule yet.{" "}
          <a href="#">Add preferred clause</a>
        </div>
      )}
    </div>
  );
}

function ProvenancePane({ rule }) {
  const s = rule.source;
  return (
    <div className="prov-pane">
      {s.miner === "categorical" && <CategoricalBars source={s} />}
      {s.miner === "distribution" && (
        <div className="prov-dist">
          <DistributionChart source={s} />
          <div className="prov-stats">
            <Stat label="median" value={s.median} />
            <Stat label="p10" value={s.p10} />
            <Stat label="p25" value={s.p25} />
            <Stat label="p75" value={s.p75} />
            <Stat label="p90" value={s.p90} />
            <Stat label="n" value={s.n} />
            <Stat label="outliers" value={s.outliers ?? "—"} />
          </div>
        </div>
      )}
      {s.miner === "coverage" && (
        <div className="prov-cov">
          <CoverageRing ratio={s.ratio} />
          <div className="prov-stats">
            <Stat label="coverage" value={Math.round(s.ratio * 100) + "%"} />
            <Stat label="contracts" value={`${s.n}/${s.total}`} />
          </div>
        </div>
      )}
      {s.miner === "contrastive" && (
        <div className="prov-contr">
          <div className="contr-row">
            <span>cluster lift</span>
            <div className="contr-bar">
              <div
                style={{ width: `${Math.min(s.cluster_lift * 15, 100)}%` }}
              />
            </div>
            <span className="mono">{s.cluster_lift.toFixed(2)}×</span>
          </div>
          <div className="contr-row">
            <span>global lift</span>
            <div className="contr-bar">
              <div
                style={{
                  width: `${Math.min(s.global_lift * 15, 100)}%`,
                  background: "var(--ink-400)",
                }}
              />
            </div>
            <span className="mono">{s.global_lift.toFixed(2)}×</span>
          </div>
          <div className="contr-row strong">
            <span>ratio</span>
            <div className="contr-bar">
              <div
                style={{
                  width: `${Math.min(s.lift_ratio * 25, 100)}%`,
                  background: "var(--accent-500)",
                }}
              />
            </div>
            <span className="mono">{s.lift_ratio.toFixed(2)}×</span>
          </div>
        </div>
      )}
      {s.miner === "outlier" && (
        <div className="prov-out">
          Outlier scan: <b>{Math.round(s.outlier_pct * 100)}%</b> of clauses
          deviate from cluster centroid (n={s.n}).
        </div>
      )}
      {s.migrated_from && (
        <div className="prov-migrated">
          <div>
            Migrated from <b>{s.migrated_from}</b> on {s.migrated_at}.
          </div>
          <div className="prov-explain mono">
            benchmark.similarity ≥ 0.82 → carried over as a typed rule.
          </div>
        </div>
      )}
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <div className="stat">
      <div className="stat-num mono">{value}</div>
      <div className="stat-lbl">{label}</div>
    </div>
  );
}

function RationalePane({ rule }) {
  return (
    <div className="rat-pane">
      <p className="rat-text">{rule.rationale}</p>
      <div className="rat-meta">
        <div>
          <span className="eyebrow">Tags</span>
          <div className="rat-tags">
            {(rule.tags || ["consulting", rule.applies_to]).map((t) => (
              <span key={t} className="rat-tag">
                {t}
              </span>
            ))}
          </div>
        </div>
        <div>
          <span className="eyebrow">Escalation</span>
          <div className="mono">{rule.escalation_owner || "—"}</div>
        </div>
        <div>
          <span className="eyebrow">Answer type</span>
          <div className="mono">{rule.answer_type}</div>
        </div>
      </div>
    </div>
  );
}

/* ────────────────────────── Custom provenance viz ────────────────────────── */
function DistributionChart({ source }) {
  const { p10, p25, median, p75, p90 } = source;
  const lo = p10 - (p90 - p10) * 0.1;
  const hi = p90 + (p90 - p10) * 0.1;
  const span = hi - lo || 1;
  const x = (v) => ((v - lo) / span) * 100;
  return (
    <svg viewBox="0 0 100 50" className="dist-chart" preserveAspectRatio="none">
      <defs>
        <pattern
          id="hatch"
          patternUnits="userSpaceOnUse"
          width="3"
          height="3"
          patternTransform="rotate(45)"
        >
          <line
            x1="0"
            y1="0"
            x2="0"
            y2="3"
            stroke="var(--accent-300)"
            strokeWidth="0.6"
          />
        </pattern>
      </defs>
      <rect
        x="0"
        y="20"
        width="100"
        height="14"
        fill="var(--paper-2)"
        stroke="var(--ink-200)"
        strokeWidth="0.3"
      />
      <rect
        x={x(p10)}
        y="22"
        width={x(p25) - x(p10)}
        height="10"
        fill="url(#hatch)"
        stroke="var(--accent-300)"
        strokeWidth="0.4"
      />
      <rect
        x={x(p25)}
        y="20"
        width={x(p75) - x(p25)}
        height="14"
        fill="var(--accent-100)"
        stroke="var(--accent-400)"
        strokeWidth="0.5"
      />
      <rect
        x={x(p75)}
        y="22"
        width={x(p90) - x(p75)}
        height="10"
        fill="url(#hatch)"
        stroke="var(--accent-300)"
        strokeWidth="0.4"
      />
      <line
        x1={x(median)}
        y1="14"
        x2={x(median)}
        y2="40"
        stroke="var(--accent-500)"
        strokeWidth="1.5"
      />
      <text
        x={x(median)}
        y="10"
        fontSize="5"
        fill="var(--accent-700)"
        textAnchor="middle"
        fontFamily="var(--font-mono)"
      >
        median {median}
      </text>
      <text
        x={x(p10)}
        y="48"
        fontSize="4.5"
        fill="var(--ink-600)"
        textAnchor="middle"
        fontFamily="var(--font-mono)"
      >
        p10 {p10}
      </text>
      <text
        x={x(p90)}
        y="48"
        fontSize="4.5"
        fill="var(--ink-600)"
        textAnchor="middle"
        fontFamily="var(--font-mono)"
      >
        p90 {p90}
      </text>
    </svg>
  );
}

function CoverageRing({ ratio }) {
  const r = 22,
    c = 2 * Math.PI * r;
  const dash = c * ratio;
  return (
    <svg viewBox="0 0 60 60" className="cov-ring">
      <circle
        cx="30"
        cy="30"
        r={r}
        fill="none"
        stroke="var(--paper-3)"
        strokeWidth="4"
      />
      <circle
        cx="30"
        cy="30"
        r={r}
        fill="none"
        stroke="var(--accent-500)"
        strokeWidth="4"
        strokeDasharray={`${dash} ${c}`}
        transform="rotate(-90 30 30)"
        strokeLinecap="butt"
      />
      <text
        x="30"
        y="33"
        textAnchor="middle"
        fontSize="11"
        fontWeight="600"
        fill="var(--ink-900)"
        fontFamily="var(--font-mono)"
      >
        {Math.round(ratio * 100)}%
      </text>
    </svg>
  );
}

/* ────────────────────────── ClusterRail (slim layout) ──────────────────────────
   Legora-style left navigation: cluster taxonomy with rule counts per cluster.
   Click "All clusters" to clear the filter; click a cluster to scope the feed.
*/
function ClusterRail({ clusters, rules, active, onPick }) {
  // Count rules per cluster_id and per applies_to fallback (some rules bind via label).
  const counts = useMemoS(() => {
    const m = {};
    (rules || []).forEach((r) => {
      if (r.cluster_id) m[r.cluster_id] = (m[r.cluster_id] || 0) + 1;
    });
    return m;
  }, [rules]);

  const totalRules = (rules || []).length;
  const sorted = [...(clusters || [])].sort(
    (a, b) => (b.chunk_count || 0) - (a.chunk_count || 0),
  );

  return (
    <div className="rail">
      <div className="rail-head">Workspace</div>
      <div
        className={`rail-row ${active === "all" ? "active" : ""}`}
        onClick={() => onPick("all")}
      >
        <span className="rl-label" style={{ fontSize: 13 }}>
          All clusters
        </span>
        <span className="rl-meta">{totalRules}</span>
      </div>

      <div className="rail-section">Clusters · by size</div>
      {sorted.map((c) => {
        const n = counts[c.id] || 0;
        return (
          <div
            key={c.id}
            className={`rail-row ${active === c.id ? "active" : ""}`}
            onClick={() => onPick(c.id)}
            title={`${c.chunk_count} chunks · quality ${(c.quality || 0).toFixed(2)}`}
          >
            <span className="rl-label">{c.label}</span>
            <span className="rl-meta">{n > 0 ? n : "·"}</span>
          </div>
        );
      })}
    </div>
  );
}

window.PlaybookStudio = PlaybookStudio;
