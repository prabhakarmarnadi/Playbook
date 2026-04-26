# EVoC Field Discovery & Extraction: 3-Way Method Comparison

## Experiment Setup

| Parameter | Value |
|---|---|
| **Model** | Azure OpenAI `gpt-5.4-mini` (`fdn-ai-oai-eastus2`) |
| **Dataset** | CUAD 510 contracts (`cuad_510_demo.duckdb`) |
| **Embedding model** | `all-MiniLM-L6-v2` (for Latent Briefing scoring) |
| **API version** | `2024-12-01-preview` |
| **Max fields per cluster** | 15 |
| **Max sample chunks** | 30 |

### Methods Compared

1. **Standard GPT** — Single-shot LLM call to discover extraction fields from sample text
2. **DSPy RLM** — DSPy Reasoning Language Model with iterative REPL-style field discovery (3–6 steps)
3. **RLM + Latent Briefing (LB)** — RLM augmented with trajectory-compressed context via MiniLM-L6-v2 segment scoring

---

## Macro Domain Cluster Results

**Source:** `clusters` table, 270 clusters with ≥15 chunks, 5 sampled  
**Extraction:** 8 chunks per cluster

### Clusters Evaluated

| Cluster | Label | Chunks |
|---|---|---|
| 1 | Franchise agreement | 2,167 |
| 2 | Offering Allocation Compliance | 1,021 |
| 3 | Website Co-Branding Agreement | 484 |
| 4 | Defined Terms and Exceptions | 397 |
| 5 | Stock option grants | 262 |

### Aggregate Results (Macro)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Avg fields/cluster | 12.2 | 8.6 | 8.0 |
| Fill rate | 0.301 | 0.328 | **0.421** |
| Avg confidence | 0.810 | 0.797 | **0.807** |
| Verbatim match rate | 0.807 | 0.818 | **0.870** |
| Grounded rate | 0.884 | 0.880 | **0.894** |
| Hallucination rate | 0.066 | **0.030** | 0.057 |
| Avg discovery time (s) | **3.6** | 21.2 | 21.4 |
| Avg extraction time (s) | 8.4 | 8.6 | **7.5** |
| Failures | 0 | 1 | 0 |

**Macro Winner:** RLM + LB leads on fill rate (0.421), verbatim match (0.870), and grounding (0.894). DSPy RLM has the lowest hallucination (0.030). Standard GPT is fastest for discovery (3.6s).

---

## Clause-Level Cluster Results

**Source:** `cluster_layers` table (`clustering_run='clause'`, `layer_index=0`), 22 non-outlier clusters, 5 sampled  
**Extraction:** 8 clauses per cluster  
**Labels:** LLM-generated from sample clauses

### Clusters Evaluated

| Cluster ID | Label | Clauses |
|---|---|---|
| 14 | Representations and Warranties | 1,304 |
| 9 | Termination and Compensation Terms | 1,013 |
| 7 | Affiliate Program Terms | 437 |
| 0 | Signature Page | 192 |
| 8 | Payment and Revenue Definitions | 192 |

### Aggregate Results (Clause)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Avg fields/cluster | **12.2** | 10.6 | 10.0 |
| Fill rate | 0.415 | 0.351 | **0.445** |
| Avg confidence | **0.716** | 0.683 | 0.670 |
| Verbatim match rate | **0.488** | 0.410 | 0.366 |
| Grounded rate | 0.781 | **0.853** | 0.754 |
| Hallucination rate | 0.194 | **0.108** | 0.156 |
| Avg discovery time (s) | **3.4** | 39.8 | 35.3 |
| Avg extraction time (s) | 10.1 | 9.6 | **9.0** |
| Failures | 0 | 0 | 0 |

### Per-Cluster Breakdown (Clause)

#### Cluster 14: Representations and Warranties (1,304 clauses)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Fields discovered | 12 | 4 | 15 |
| Fill rate | 0.417 | 0.500 | 0.275 |
| Confidence | 0.795 | 0.772 | 0.779 |
| Verbatim match | 0.750 | 0.750 | 0.727 |
| Hallucination | 0.050 | **0.000** | **0.000** |
| Discovery time (s) | 3.38 | 43.99 | 66.65 |
| Extraction time (s) | 10.72 | 8.49 | 9.05 |

#### Cluster 9: Termination and Compensation Terms (1,013 clauses)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Fields discovered | 12 | 15 | 4 |
| Fill rate | 0.229 | 0.292 | **0.531** |
| Confidence | 0.645 | 0.677 | 0.656 |
| Verbatim match | 0.136 | 0.257 | 0.176 |
| Hallucination | 0.364 | 0.286 | **0.176** |
| Discovery time (s) | 3.50 | 28.06 | 34.96 |
| Extraction time (s) | 10.41 | 10.40 | 7.26 |

#### Cluster 7: Affiliate Program Terms (437 clauses)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Fields discovered | 12 | 5 | 9 |
| Fill rate | **0.385** | 0.150 | 0.292 |
| Confidence | **0.768** | 0.633 | 0.683 |
| Verbatim match | **0.676** | 0.000 | 0.286 |
| Hallucination | 0.027 | **0.000** | 0.143 |
| Discovery time (s) | 3.53 | 63.35 | 20.98 |
| Extraction time (s) | 9.51 | 8.29 | 7.65 |

#### Cluster 0: Signature Page (192 clauses)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Fields discovered | 10 | 14 | 7 |
| Fill rate | **0.812** | 0.420 | 0.821 |
| Confidence | 0.629 | 0.598 | 0.560 |
| Verbatim match | 0.308 | 0.404 | 0.261 |
| Hallucination | 0.492 | **0.000** | 0.109 |
| Discovery time (s) | 2.74 | 23.96 | 18.14 |
| Extraction time (s) | 9.22 | 11.61 | 11.03 |

#### Cluster 8: Payment and Revenue Definitions (192 clauses)

| Metric | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Fields discovered | 15 | 15 | 15 |
| Fill rate | 0.233 | 0.392 | **0.433** |
| Confidence | **0.745** | 0.736 | 0.702 |
| Verbatim match | **0.571** | 0.638 | 0.446 |
| Hallucination | **0.036** | 0.255 | 0.092 |
| Discovery time (s) | 3.84 | 39.53 | 35.54 |
| Extraction time (s) | 10.51 | 9.01 | 9.38 |

---

## Clause vs Macro Comparison

| Metric | Level | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|---|
| Fill rate | Macro | 0.301 | 0.328 | 0.421 |
| | Clause | 0.415 (+0.114) | 0.351 (+0.023) | 0.445 (+0.024) |
| Verbatim match | Macro | 0.807 | 0.818 | 0.870 |
| | Clause | 0.488 (−0.319) | 0.410 (−0.408) | 0.366 (−0.504) |
| Hallucination | Macro | 0.066 | 0.030 | 0.057 |
| | Clause | 0.194 (+0.128) | 0.108 (+0.078) | 0.156 (+0.099) |
| Grounded rate | Macro | 0.884 | 0.880 | 0.894 |
| | Clause | 0.781 (−0.103) | 0.853 (−0.027) | 0.754 (−0.140) |
| Discovery time (s) | Macro | 3.6 | 21.2 | 21.4 |
| | Clause | 3.4 (−0.2) | 39.8 (+18.6) | 35.3 (+13.9) |

---

## Grounding Distribution (Clause-Level, All Extractions)

| Level | Standard GPT | DSPy RLM | RLM + LB |
|---|---|---|---|
| Verbatim | 94 | 70 | 59 |
| High overlap | 49 | 52 | 60 |
| Partial overlap | 3 | 4 | 4 |
| Weak overlap | 2 | 3 | 7 |
| Hallucinated | 44 | 22 | 24 |

---

## Key Observations

1. **Fill rate improves at clause level** — All three methods show higher fill rates on clause clusters vs macro clusters. Clauses are shorter, more focused text units, making extraction easier.

2. **Verbatim match drops significantly at clause level** — All methods lose 30–50% verbatim match rate. Clause text is more fragmented; extractions tend to be semantic paraphrases rather than exact spans.

3. **Hallucination increases at clause level** — Standard GPT is most affected (0.066 → 0.194), particularly on "Signature Page" clusters where boolean fields like `witness_presence` and `notary_acknowledgment` are commonly hallucinated. DSPy RLM remains most robust (0.030 → 0.108).

4. **DSPy RLM is consistently the safest method** — Lowest hallucination rate in both macro (0.030) and clause (0.108) experiments. Its iterative REPL approach produces more conservative, grounded fields.

5. **RLM + LB leads on fill rate** — Best fill rate in both macro (0.421) and clause (0.445). The Latent Briefing compression helps focus field discovery on the most salient aspects.

6. **Standard GPT is 10× faster for discovery** — ~3.5s vs ~35–40s for RLM methods. For latency-sensitive applications, it remains the pragmatic choice.

7. **Cluster type matters** — "Representations and Warranties" had near-zero hallucination for RLM methods, while "Signature Page" was problematic for Standard GPT (49.2% hallucination) due to boolean field generation.

8. **RLM discovery time increases at clause level** — RLM methods take ~15–19s longer per cluster at clause level, likely due to more REPL steps needed for shorter, more ambiguous text.

---

## Winner Summary

| Category | Macro Winner | Clause Winner |
|---|---|---|
| Best fill rate | RLM + LB (0.421) | RLM + LB (0.445) |
| Lowest hallucination | DSPy RLM (0.030) | DSPy RLM (0.108) |
| Best verbatim match | RLM + LB (0.870) | Standard GPT (0.488) |
| Most grounded | RLM + LB (0.894) | DSPy RLM (0.853) |
| Fastest discovery | Standard GPT (3.6s) | Standard GPT (3.4s) |
| Most fields | Standard GPT (12.2) | Standard GPT (12.2) |
