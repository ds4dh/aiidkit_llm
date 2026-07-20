"""
AIIDKIT Decision Support Tool — Mock Data Generator
====================================================
Generates 250 clinically-plausible synthetic patients for demonstration.
Uses the exact t-EAV schema and clinical vocabulary of the real AIIDKIT
dataset. No real patient data is exposed.

When MODEL_PATH is set in config.py this module is bypassed entirely and
real inference runs instead.

Architecture
------------
generate_cohort_data(seed)  →  dict   (the /api/cohort payload)
mock_predict(events, horizon, cohort)  →  dict  (the /api/predict payload)
"""

import numpy as np
from scipy.special import expit
import config as CONFIG
from collections import defaultdict

# ---------------------------------------------------------------------------
# CLUSTER PROFILES
# Three archetypal patient phenotypes with distinct risk signatures.
# ---------------------------------------------------------------------------

CLUSTER_PROFILES = [
    {
        "id": 0,
        "name": "High risk",
        "color": "#FF4757",
        "n": 120,
        "base_risk": 0.45,
        "risk_std": 0.12,
        "event_rate": 0.32,
        "umap_center": (1.5, 2.3),
        "umap_spread": 1.2,
        # Risk-elevating features (positive attribution)
        "risk_features": [
            ("Medication",  "Tacrolimus",        "Higher",  0.18),
            ("Lab result",  "WBC",               "Lowest",  0.16),
            ("Lab result",  "Lymphocyte count",  "Lowest",  0.14),
            ("Infection",   "CMV",               "1",       0.13),
            ("Lab result",  "CRP",               "Higher",  0.12),
            ("Comorbidity", "Diabetes mellitus", "1",       0.10),
            ("Lab result",  "eGFR",              "Lowest",  0.09),
        ],
        # Protective features (negative attribution)
        "protective_features": [
            ("Medication", "Cotrimoxazole",  "1", -0.09),
            ("Medication", "Valganciclovir", "1", -0.08),
        ],
        # Typical clinical profile for sequence generation
        "event_weights": {
            ("Medication",    "Tacrolimus",           ("Higher", "Highest", "High")):   0.9,
            ("Medication",    "Mycophenolate Mofetil",("1",)):                  0.9,
            ("Medication",    "Glucocorticoid",       ("Middle", "High")):      0.8,
            ("Medication",    "Cotrimoxazole",        ("0",)):                  0.6,
            ("Medication",    "Valganciclovir",       ("0", "1")):              0.5,
            ("Lab result",    "Creatinine",           ("High", "Higher")):      0.9,
            ("Lab result",    "eGFR",                 ("Low", "Lowest")):       0.9,
            ("Lab result",    "Tacrolimus level",     ("Higher", "Highest")):   0.9,
            ("Lab result",    "WBC",                  ("Lowest", "Lower")):     0.8,
            ("Lab result",    "Lymphocyte count",     ("Lowest", "Lower")):     0.7,
            ("Lab result",    "CRP",                  ("High", "Higher")):      0.7,
            ("Infection",     "CMV",                  ("1",)):                  0.6,
            ("Clinical event","Rejection event",      ("0", "1")):              0.4,
            ("Comorbidity",   "Diabetes mellitus",    ("1",)):                  0.6,
            ("Comorbidity",   "Hypertension",         ("1",)):                  0.8,
            ("Donor",         "CMV D/R status",       ("D+/R+", "D+/R-")):     0.6,
        },
    },
    {
        "id": 1,
        "name": "Low risk",
        "color": "#00C9A7",
        "n": 130,
        "base_risk": 0.08,
        "risk_std": 0.04,
        "event_rate": 0.06,
        "umap_center": (-3.0, -2.0),
        "umap_spread": 1.2,
        "risk_features": [
            ("Lab result",  "Creatinine", "High", 0.05),
        ],
        "protective_features": [
            ("Medication",  "Cotrimoxazole",       "1",              -0.12),
            ("Medication",  "Valganciclovir",      "1",              -0.09),
            ("Lab result",  "eGFR",                "High",           -0.08),
            ("Lab result",  "WBC",                 "Middle",         -0.06),
            ("Donor",       "Donor type",          "Living related", -0.07),
            ("Medication",  "Tacrolimus",          "Middle",         -0.05),
        ],
        "event_weights": {
            ("Medication",    "Tacrolimus",           ("Low", "Middle")):       0.9,
            ("Medication",    "Mycophenolate Mofetil",("1",)):                  0.9,
            ("Medication",    "Glucocorticoid",       ("Lowest", "Low")):       0.8,
            ("Medication",    "Cotrimoxazole",        ("1",)):                  0.9,
            ("Medication",    "Valganciclovir",       ("1",)):                  0.7,
            ("Lab result",    "Creatinine",           ("Low", "Middle")):       0.9,
            ("Lab result",    "eGFR",                 ("High", "Higher")):      0.9,
            ("Lab result",    "Tacrolimus level",     ("Low", "Middle")):       0.9,
            ("Lab result",    "WBC",                  ("Middle",)):             0.8,
            ("Lab result",    "CRP",                  ("Low", "Middle")):       0.7,
            ("Comorbidity",   "Diabetes mellitus",    ("0",)):                  0.8,
            ("Comorbidity",   "Hypertension",         ("0", "1")):              0.6,
            ("Donor",         "Donor type",           ("Living related", "DBD")):0.7,
            ("Donor",         "CMV D/R status",       ("D-/R-", "D-/R+")):     0.6,
        },
    },
]

# Common baseline events added for all patients at day 0
BASELINE_EVENTS = [
    ("Clinical event", "Transplantation event", ["1"]),
    ("Patient",        "Age at transplant",     ["Lower", "Low", "Middle", "High", "Higher"]),
    ("Patient",        "Sex",                   ["Male", "Female"]),
    ("Patient",        "Dialysis duration",     ["Low", "Middle", "High"]),
]

# Follow-up time points where labs / meds are re-assessed (days since Tpx)
FOLLOW_UP_DAYS = [0, 7, 14, 30, 60, 90]


# ---------------------------------------------------------------------------
# HELPER: Kaplan-Meier estimator
# ---------------------------------------------------------------------------

def _km_estimate(durations, events, eval_times):
    """
    Right-censored Kaplan-Meier survival estimate with Greenwood CI.
    Returns (survival, ci_lower, ci_upper) at each eval_time.
    """
    durations = np.asarray(durations, dtype=float)
    events    = np.asarray(events,    dtype=int)
    n_total   = len(durations)

    # Unique event times
    event_times = np.sort(np.unique(durations[events == 1]))

    km_t = [0.0]
    km_s = [1.0]
    km_g = [0.0]   # Greenwood's accumulator

    s = 1.0
    g = 0.0
    for t in event_times:
        at_risk = int((durations >= t).sum())
        d       = int(((durations == t) & (events == 1)).sum())
        if at_risk > 0 and d > 0:
            s *= (at_risk - d) / at_risk
            if at_risk * (at_risk - d) > 0:
                g += d / (at_risk * (at_risk - d))
        km_t.append(float(t))
        km_s.append(float(s))
        km_g.append(float(g))

    # Step-function lookup at each eval_time
    survival, ci_lo, ci_hi = [], [], []
    for ev_t in eval_times:
        idx = 0
        for i, tt in enumerate(km_t):
            if tt <= ev_t:
                idx = i
        s_val = km_s[idx]
        g_val = km_g[idx]
        se    = s_val * np.sqrt(g_val)
        survival.append(float(np.clip(s_val, 0, 1)))
        ci_lo.append(float(np.clip(s_val - 1.96 * se, 0, 1)))
        ci_hi.append(float(np.clip(s_val + 1.96 * se, 0, 1)))

    return survival, ci_lo, ci_hi


def _log_rank_p(t1, e1, t2, e2):
    """Compute log-rank test statistic and p-value (chi-squared, df=1)."""
    from scipy.stats import chi2
    all_times = np.sort(np.unique(
        np.concatenate([t1[e1 == 1], t2[e2 == 1]])
    ))
    O1 = O2 = E1 = E2 = 0.0
    for t in all_times:
        n1 = (t1 >= t).sum()
        n2 = (t2 >= t).sum()
        d1 = ((t1 == t) & (e1 == 1)).sum()
        d2 = ((t2 == t) & (e2 == 1)).sum()
        n  = n1 + n2
        d  = d1 + d2
        if n > 0:
            O1 += d1;  O2 += d2
            E1 += n1 * d / n
            E2 += n2 * d / n
    if E1 < 1e-9 or E2 < 1e-9:
        return 1.0
    chi2_stat = (O1 - E1)**2 / E1 + (O2 - E2)**2 / E2
    return float(1 - chi2.cdf(chi2_stat, df=1))


# ---------------------------------------------------------------------------
# PATIENT SEQUENCE GENERATOR
# ---------------------------------------------------------------------------

def _generate_patient_sequence(profile, rng, follow_up_days):
    """Return a list of EAV event dicts for one patient."""
    events = []

    # Baseline events (day 0 — transplantation day)
    for (entity, attribute, values) in BASELINE_EVENTS:
        val = rng.choice(values)
        events.append({
            "entity": entity, "attribute": attribute,
            "value": val, "days_since_tpx": 0.0,
        })

    # Donor data (day 0)
    for (key, attr, values), prob in profile["event_weights"].items():
        if key == "Donor" and rng.random() < prob:
            val = rng.choice(values)
            events.append({
                "entity": key, "attribute": attr,
                "value": val, "days_since_tpx": 0.0,
            })

    # Follow-up events (meds + labs per time point)
    for day in follow_up_days[1:]:
        # Jitter time slightly
        t = float(day) + rng.uniform(-2, 2)
        t = max(1.0, t)
        for (key, attr, values), prob in profile["event_weights"].items():
            if key in ("Medication", "Lab result") and rng.random() < prob:
                val = rng.choice(values)
                events.append({
                    "entity": key, "attribute": attr,
                    "value": val, "days_since_tpx": round(t, 1),
                })

        # Occasional infection or clinical events
        for (key, attr, values), prob in profile["event_weights"].items():
            if key in ("Infection", "Clinical event", "Comorbidity") and rng.random() < prob * 0.4:
                val = rng.choice(values)
                events.append({
                    "entity": key, "attribute": attr,
                    "value": val, "days_since_tpx": round(t, 1),
                })

    # Sort by time
    events.sort(key=lambda e: e["days_since_tpx"])
    return events


def _patient_summary(events):
    """Build a short human-readable summary from the event sequence."""
    snippets = []
    for e in events:
        attr = e["attribute"]
        val  = e["value"]
        if attr == "Tacrolimus" and val in ("Higher", "Highest"):
            snippets.append("Tacrolimus\u2191")
        if attr == "WBC" and val in ("Lowest", "Lower"):
            snippets.append("Leucopenia")
        if attr == "CMV" and val == "1":
            snippets.append("CMV+")
        if attr == "Cotrimoxazole" and val == "1":
            snippets.append("Cotrimoxazole\u2713")
        if attr == "Rejection event" and val == "1":
            snippets.append("Rejection")
        if attr == "eGFR" and val in ("Lowest", "Lower"):
            snippets.append("Low eGFR")
        if attr == "CRP" and val in ("High", "Higher", "Highest"):
            snippets.append("CRP\u2191")
        if attr == "Diabetes mellitus" and val == "1":
            snippets.append("DM")
    seen = []
    for s in snippets:
        if s not in seen:
            seen.append(s)
    return ", ".join(seen[:5]) if seen else "Standard profile"


# ---------------------------------------------------------------------------
# MAIN COHORT GENERATOR
# ---------------------------------------------------------------------------

def generate_cohort_data(fup: int = 90, seed: int = 42) -> dict:
    """
    Generate all cohort data returned by GET /api/cohort.

    Returns a dict ready to be jsonified.
    """
    rng = np.random.default_rng(seed)

    # Dynamically resolve follow-up days based on selected FUP
    all_fup_days = [0, 7, 14, 30, 60, 90, 180, 360, 720]
    follow_up_days = [d for d in all_fup_days if d <= fup]

    # 1. Create the base patient records (static fields)
    base_patients = []
    patient_seqs = []
    for profile in CLUSTER_PROFILES:
        n = profile["n"]
        cx, cy = profile["umap_center"]
        spread = profile["umap_spread"]
        ux = rng.normal(cx, spread, n)
        uy = rng.normal(cy, spread, n)
        
        for i in range(n):
            seq = _generate_patient_sequence(profile, rng, follow_up_days)
            base_patients.append({
                "umap_x": float(ux[i]),
                "umap_y": float(uy[i]),
                "profile": profile,
                "summary": _patient_summary(seq),
            })
            patient_seqs.append(seq)

    # Shuffling index so patient IDs are randomized
    shuffled_indices = list(range(len(base_patients)))
    rng.shuffle(shuffled_indices)

    # Generate labels and risk scores for each horizon dynamically
    horizons_data = {}
    
    # Pre-generate risk scores and labels for all patients so we can maintain consistency
    patient_metrics = []
    for bp in base_patients:
        prof = bp["profile"]
        # Monotonal risks
        r30 = float(np.clip(rng.normal(prof["base_risk"], prof["risk_std"]), 0.01, 0.99))
        r60 = float(np.clip(r30 * 1.12, 0.01, 0.99))
        r90 = float(np.clip(r30 * 1.22, 0.01, 0.99))
        
        # Generate consistent time to event
        if prof["id"] == 0:  # High risk
            u = rng.random()
            if u < 0.32:
                t_event = int(rng.uniform(5, 30))
            elif u < 0.42:
                t_event = int(rng.uniform(31, 60))
            elif u < 0.48:
                t_event = int(rng.uniform(61, 90))
            elif u < 0.60:
                t_event = int(rng.uniform(91, 360))
            else:
                t_event = None
        else:  # Low risk
            u = rng.random()
            if u < 0.06:
                t_event = int(rng.uniform(5, 30))
            elif u < 0.09:
                t_event = int(rng.uniform(31, 60))
            elif u < 0.12:
                t_event = int(rng.uniform(61, 90))
            elif u < 0.18:
                t_event = int(rng.uniform(91, 360))
            else:
                t_event = None
                
        # derive monotonic true labels
        l30 = 1 if (t_event is not None and t_event <= 30) else 0
        l60 = 1 if (t_event is not None and t_event <= 60) else 0
        l90 = 1 if (t_event is not None and t_event <= 90) else 0
        
        imminence = float(np.exp(-t_event / 365.0)) if t_event is not None else 0.0
        
        patient_metrics.append({
            "risks": {30: r30, 60: r60, 90: r90},
            "labels": {30: l30, 60: l60, 90: l90},
            "days_to_event": t_event,
            "event_imminence": round(imminence, 4)
        })

    # For each horizon, build the cohort data
    for h in [30, 60, 90]:
        h_patients = []
        all_risks = []
        
        for idx in shuffled_indices:
            bp = base_patients[idx]
            metrics = patient_metrics[idx]
            prof = bp["profile"]
            
            risk = metrics["risks"][h]
            label = metrics["labels"][h]
            days_to_event = metrics["days_to_event"]
            event_imminence = metrics["event_imminence"]
            
            # Determine UI cluster based on config method
            if getattr(CONFIG, "CLUSTERING_METHOD", "model_risk") == "clusterer":
                best_cluster = prof
            else:
                if risk >= CONFIG.RISK_THRESHOLD:
                    best_cluster = CLUSTER_PROFILES[0]
                else:
                    best_cluster = CLUSTER_PROFILES[1]
            
            pid = f"PAT_{len(h_patients)+1:03d}"
            logit_v = np.log(risk / (1.0 - risk))
            cal_risk = float(np.clip(expit(logit_v - 1.5), 0.01, 0.95))
            
            h_patients.append({
                "id":            pid,
                "umap_x":        bp["umap_x"],
                "umap_y":        bp["umap_y"],
                "risk_score":    risk,
                "calibrated_risk": round(cal_risk, 4),
                "cluster":       best_cluster["id"],
                "cluster_name":  best_cluster["name"],
                "cluster_color": best_cluster["color"],
                "true_label":    label,
                "summary":       bp["summary"],
                "days_to_event": days_to_event,
                "event_imminence": event_imminence,
                "top_features":  [
                    {"feature": f"{e} - {a}", "value": v, "score": s}
                    for (e, a, v, s) in prof["risk_features"][:3]
                ]
            })
            all_risks.append(risk)

        # Risk distribution
        rv = np.array(all_risks)
        risk_distribution = {
            "values":      [round(float(x), 4) for x in rv],
            "mean":        round(float(rv.mean()), 4),
            "std":         round(float(rv.std()),  4),
            "percentiles": {
                "p25": round(float(np.percentile(rv, 25)), 4),
                "p50": round(float(np.percentile(rv, 50)), 4),
                "p75": round(float(np.percentile(rv, 75)), 4),
                "p90": round(float(np.percentile(rv, 90)), 4),
            },
        }

        # Count patient-level features in each cluster for prevalence calculations
        cluster_sizes = {0: 0, 1: 0}
        for p in h_patients:
            cluster_sizes[p["cluster"]] = cluster_sizes.get(p["cluster"], 0) + 1
            
        static_counts = {0: defaultdict(int), 1: defaultdict(int)}
        recent_counts = {0: defaultdict(int), 1: defaultdict(int)}
        all_static_features = set()
        all_recent_features = set()

        for idx in shuffled_indices:
            c_id = h_patients[shuffled_indices.index(idx)]["cluster"]
            seq = patient_seqs[idx]
            
            patient_static = set()
            patient_recent = set()
            
            # Find the max time in the sequence
            valid_times = [float(ev.get("days_since_tpx", 0.0)) for ev in seq if ev.get("entity") and ev.get("attribute")]
            max_t = max(valid_times) if valid_times else 0.0
            
            for ev in seq:
                ent = ev.get("entity", "")
                attr = ev.get("attribute", "")
                val = ev.get("value", "")
                t_val = float(ev.get("days_since_tpx", 0.0))
                
                if not ent or not attr or not val:
                    continue
                if attr in ["Transplantation event", "Transplant procedure"]:
                    continue
                    
                feat = (ent, attr, val)
                
                # Check if static: Patient/Donor entity or time <= 0.0
                is_static = (ent in ["Patient", "Donor"] or t_val <= 0.0)
                # Check if recent: not static and occurred within the last month of the observation window
                is_recent = (not is_static and t_val > 0.0 and t_val >= (max_t - 30.0))
                
                if is_static:
                    patient_static.add(feat)
                elif is_recent:
                    patient_recent.add(feat)
                    
            for feat in patient_static:
                static_counts[c_id][feat] += 1
                all_static_features.add(feat)
            for feat in patient_recent:
                recent_counts[c_id][feat] += 1
                all_recent_features.add(feat)

        # Cluster profiles
        cluster_profiles = {}
        for profile in CLUSTER_PROFILES:
            c_id = profile["id"]
            other_c_id = 1 - c_id
            c_size = cluster_sizes.get(c_id, 0)
            other_size = cluster_sizes.get(other_c_id, 0)
            
            c_labels = [p["true_label"] for p in h_patients if p["cluster"] == c_id]
            c_event_rate = float(np.mean(c_labels)) if c_labels else profile["event_rate"] * (1.25 if h==60 else 1.45 if h==90 else 1.0)
            c_event_rate = min(0.99, c_event_rate)
            
            # Static diffs
            static_diffs = []
            for feat in all_static_features:
                p_c = (static_counts[c_id][feat] / c_size) if c_size > 0 else 0.0
                p_other = (static_counts[other_c_id][feat] / other_size) if other_size > 0 else 0.0
                diff = p_c - p_other
                if diff > 0.0:
                    static_diffs.append((feat, diff))
            sorted_static = sorted(static_diffs, key=lambda x: x[1], reverse=True)[:10]
            top_static = []
            for (ent, attr, val), diff in sorted_static:
                score = diff if c_id == 0 else -diff
                top_static.append({
                    "feature": f"{ent} - {attr}",
                    "value": val,
                    "score": score
                })

            # Recent diffs
            recent_diffs = []
            for feat in all_recent_features:
                p_c = (recent_counts[c_id][feat] / c_size) if c_size > 0 else 0.0
                p_other = (recent_counts[other_c_id][feat] / other_size) if other_size > 0 else 0.0
                diff = p_c - p_other
                if diff > 0.0:
                    recent_diffs.append((feat, diff))
            sorted_recent = sorted(recent_diffs, key=lambda x: x[1], reverse=True)[:10]
            top_recent = []
            for (ent, attr, val), diff in sorted_recent:
                score = diff if c_id == 0 else -diff
                top_recent.append({
                    "feature": f"{ent} - {attr}",
                    "value": val,
                    "score": score
                })
                
            cluster_profiles[str(profile["id"])] = {
                "name":        profile["name"],
                "n_patients":  c_size,
                "event_rate":  round(c_event_rate, 4),
                "color":       profile["color"],
                "top_features": {
                    "static": top_static,
                    "recent": top_recent
                },
            }

        horizons_data[str(h)] = {
            "patients": h_patients,
            "risk_distribution": risk_distribution,
            "cluster_profiles": cluster_profiles,
        }

    # Compute global attributions from _RISK_FEATURE_SCORES
    global_attrs = []
    for (ent, attr, val), score in _RISK_FEATURE_SCORES.items():
        global_attrs.append({
            "feature": f"{ent} - {attr}",
            "value": val,
            "score": score
        })
    global_attributions = sorted(global_attrs, key=lambda x: abs(x["score"]), reverse=True)[:15]

    return {
        "horizons": horizons_data,
        "global_attributions": global_attributions,
    }


# ---------------------------------------------------------------------------
# MOCK PREDICTION
# ---------------------------------------------------------------------------

_RISK_FEATURE_SCORES: dict[tuple, float] = {
    # Key: (entity, attribute, value) → attribution score
    ("Medication",  "Tacrolimus",        "Highest"):     0.20,
    ("Medication",  "Tacrolimus",        "Higher"):      0.18,
    ("Medication",  "Tacrolimus",        "High"):        0.12,
    ("Lab result",  "WBC",               "Lowest"):      0.16,
    ("Lab result",  "WBC",               "Lower"):       0.10,
    ("Lab result",  "Lymphocyte count",  "Lowest"):      0.14,
    ("Lab result",  "Lymphocyte count",  "Lower"):       0.09,
    ("Infection",   "CMV",               "1"):           0.13,
    ("Lab result",  "CRP",               "Higher"):      0.12,
    ("Lab result",  "CRP",               "Highest"):     0.14,
    ("Lab result",  "CRP",               "High"):        0.08,
    ("Comorbidity", "Diabetes mellitus", "1"):           0.10,
    ("Lab result",  "eGFR",              "Lowest"):      0.09,
    ("Lab result",  "eGFR",              "Lower"):       0.06,
    ("Infection",   "Bacterial infection","1"):          0.11,
    ("Infection",   "UTI",               "1"):           0.08,
    ("Clinical event","Rejection event", "1"):           0.09,
    # Protective
    ("Medication",  "Cotrimoxazole",     "1"):          -0.12,
    ("Medication",  "Valganciclovir",    "1"):          -0.09,
    ("Lab result",  "eGFR",             "High"):        -0.08,
    ("Lab result",  "eGFR",             "Higher"):      -0.10,
    ("Lab result",  "eGFR",             "Highest"):     -0.12,
    ("Donor",       "Donor type",        "Living related"): -0.07,
    ("Medication",  "Tacrolimus",       "Low"):         -0.06,
    ("Medication",  "Tacrolimus",       "Lowest"):      -0.08,
    ("Lab result",  "WBC",             "High"):         -0.05,
    ("Lab result",  "WBC",             "Middle"):       -0.04,
    ("Lab result",  "Creatinine",       "Lowest"):      -0.10,
    ("Lab result",  "Creatinine",       "Low"):         -0.07,
    ("Lab result",  "Tacrolimus level", "Lowest"):      -0.09,
    ("Lab result",  "Tacrolimus level", "Low"):         -0.06,
}

# Horizon multipliers for monotone risk (30d < 60d < 90d)
_HORIZON_MULT = {30: 1.0, 60: 1.12, 90: 1.22}


def _risk_category(score: float) -> tuple[str, str]:
    if score < CONFIG.RISK_THRESHOLD:   return "Low risk",  "#00C9A7"
    return                              "High risk", "#FF4757"


def _deterministic_seed(events: list[dict]) -> int:
    import hashlib
    # Sort events to ensure order-independence
    sorted_events = sorted(
        events,
        key=lambda e: (
            float(e.get("days_since_tpx", 0.0)),
            str(e.get("entity", "")),
            str(e.get("attribute", "")),
            str(e.get("value", ""))
        )
    )
    event_str = "".join(
        f"{e.get('entity')}-{e.get('attribute')}-{e.get('value')}-{e.get('days_since_tpx')}"
        for e in sorted_events
    )
    return int(hashlib.md5(event_str.encode("utf-8")).hexdigest(), 16) & 0xffffffff


def mock_predict(events: list[dict], horizon: int, cohort: dict) -> dict:
    """
    Simulate a model prediction for a new patient.

    Parameters
    ----------
    events  : list of dicts with keys entity, attribute, value, days_since_tpx
    horizon : prediction horizon in days (30, 60, or 90)
    cohort  : the cohort dict returned by generate_cohort_data()

    Returns
    -------
    dict ready to be jsonified as the /api/predict response.
    """
    # --- Compute attributions per event ---
    attribution_map: dict[str, dict] = {}   # key: "entity-attribute: value"
    raw_score = 0.0

    events_with_scores = []
    for ev in events:
        key = (ev.get("entity", ""), ev.get("attribute", ""), ev.get("value", ""))
        sc  = _RISK_FEATURE_SCORES.get(key, 0.0)
        raw_score += sc
        feat_key = f"{key[0]} - {key[1]}: {key[2]}"
        if feat_key not in attribution_map:
            attribution_map[feat_key] = {
                "feature": f"{key[0]} - {key[1]}",
                "value":   key[2],
                "score":   sc,
            }
        events_with_scores.append({**ev, "score": sc})

    # Detect synthetic patients to calibrate their mock risk score in DEMO MODE
    is_synth_low = False
    is_synth_mod = False
    is_synth_high = False
    if events:
        first_attr = events[0].get("attribute", "") or ""
        if len(events) > 500:
            if "myocardial" in first_attr.lower() or "contusion" in first_attr.lower():
                # Synthetic low risk (patient 1) -> target ~10%
                raw_score += -0.73
                is_synth_low = True
            elif "cause of death" in first_attr.lower() or first_attr.lower() == "cause of death":
                # Synthetic moderate risk (patient 2) -> target ~32%
                raw_score += -0.25
                is_synth_mod = True
        elif 250 < len(events) < 400:
            if "twin" in first_attr.lower():
                # Synthetic high risk (patient 3) -> target ~65%
                raw_score += 0.21
                is_synth_high = True

    # Map raw attributions to [0.05, 0.95] via sigmoid
    base_risk_30d = float(np.clip(expit(raw_score * 3.0), 0.05, 0.95))

    # Apply horizon multiplier (monotone)
    risk_scores = {
        "30d": round(float(np.clip(base_risk_30d,        0.05, 0.95)), 4),
        "60d": round(float(np.clip(base_risk_30d * 1.12, 0.05, 0.95)), 4),
        "90d": round(float(np.clip(base_risk_30d * 1.22, 0.05, 0.95)), 4),
    }
    selected_risk = risk_scores[f"{horizon}d"]
    
    # Simulate calibrated risk by shifting mock logits by -1.5
    calibrated_risks = {}
    for k, v in risk_scores.items():
        logit_v = np.log(v / (1.0 - v))
        calibrated_risks[k] = round(float(np.clip(expit(logit_v - 1.5), 0.01, 0.95)), 4)
        
    selected_calibrated_risk = calibrated_risks[f"{horizon}d"]
    category, color = _risk_category(selected_risk)

    # Mock days_to_event and event_imminence based on synthetic detection or risk
    if is_synth_low:
        q_t_event = None
        q_imminence = 0.0
    elif is_synth_mod:
        q_t_event = 120
        q_imminence = float(np.exp(-120.0 / 365.0))
    elif is_synth_high:
        q_t_event = 25
        q_imminence = float(np.exp(-25.0 / 365.0))
    else:
        has_event = selected_risk >= CONFIG.RISK_THRESHOLD
        if has_event:
            q_t_event = int(np.clip(120 - selected_risk * 100, 10, 360))
            q_imminence = float(np.exp(-q_t_event / 365.0))
        else:
            q_t_event = None
            q_imminence = 0.0

    # --- Assign cluster based on config method ---
    if getattr(CONFIG, "CLUSTERING_METHOD", "model_risk") == "clusterer":
        # Nearest profile by risk proximity
        best_cluster = min(
            CLUSTER_PROFILES,
            key=lambda p: abs(p["base_risk"] - selected_risk),
        )
    else:
        if selected_risk >= CONFIG.RISK_THRESHOLD:
            best_cluster = CLUSTER_PROFILES[0]
        else:
            best_cluster = CLUSTER_PROFILES[1]
    cluster_id   = best_cluster["id"]
    cluster_name = best_cluster["name"]
    cluster_color = best_cluster["color"]

    # UMAP position near cluster center (with deterministic noise)
    import random
    seed_val = _deterministic_seed(events)
    local_random = random.Random(seed_val)
    cx, cy = best_cluster["umap_center"]
    umap_x = round(float(cx + local_random.gauss(0, 0.3)), 3)
    umap_y = round(float(cy + local_random.gauss(0, 0.3)), 3)

    # --- Percentile rank in cohort ---
    active_cohort = cohort["horizons"][str(horizon)]
    cohort_risks = np.array([p["risk_score"] for p in active_cohort["patients"]])
    percentile   = int(np.round((cohort_risks < selected_risk).mean() * 100))

    # --- Top attributions (sorted by abs score, top 15) ---
    attributions = sorted(
        attribution_map.values(),
        key=lambda x: abs(x["score"]),
        reverse=True,
    )[:15]

    # --- Similar patients (nearest in UMAP space) ---
    umap_pts  = np.array([[p["umap_x"], p["umap_y"]] for p in active_cohort["patients"]])
    query_pt  = np.array([umap_x, umap_y])
    dists     = np.linalg.norm(umap_pts - query_pt, axis=1)
    top_idx   = np.argsort(dists)[:3]
    similar   = []
    for i in top_idx:
        p = active_cohort["patients"][i]
        similar.append({
            "id": p["id"],
            "risk_score": p["risk_score"],
            "summary": p["summary"],
        })

    # --- Clinical narrative ---
    pos_feats = [a for a in attributions if a["score"] > 0.05]
    neg_feats = [a for a in attributions if a["score"] < -0.05]
    risk_txt  = (
        f"This patient's predicted {horizon}-day infection score is "
        f"{selected_risk*100:.0f}/100 ({category}), placing them at the "
        f"{percentile}th percentile of the reference cohort."
    )
    if pos_feats:
        names = ", ".join(
            f"{f['feature']}: {f['value']}" for f in pos_feats[:3]
        )
        risk_txt += f" Key risk drivers: {names}."
    if neg_feats:
        names = ", ".join(
            f"{f['feature']}: {f['value']}" for f in neg_feats[:2]
        )
        risk_txt += f" Protective factors present: {names}."
    risk_txt += (
        f" The patient clusters with the '{cluster_name}' phenotype "
        f"(event rate ≈ {int(best_cluster['event_rate']*100)}%)."
    )

    return {
        "risk_scores":       risk_scores,
        "calibrated_risks":  calibrated_risks,
        "selected_horizon":  horizon,
        "risk_score":        selected_risk,
        "calibrated_risk":   selected_calibrated_risk,
        "risk_category":     category,
        "risk_color":        color,
        "percentile":        percentile,
        "cluster":           cluster_id,
        "cluster_name":      cluster_name,
        "cluster_color":     cluster_color,
        "umap_x":            umap_x,
        "umap_y":            umap_y,
        "days_to_event":     q_t_event,
        "event_imminence":   round(q_imminence, 4),
        "attributions":      [
            {**a, "score": round(a["score"], 4)} for a in attributions
        ],
        "events_with_scores": events_with_scores,
        "similar_patients":   similar,
        "narrative":          risk_txt,
    }
