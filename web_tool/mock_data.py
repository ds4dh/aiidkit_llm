"""
AIIDKIT Decision Support Tool — Mock Data Generator
====================================================
Generates 250 clinically-plausible synthetic patients for demonstration using
the exact v3.9 t-EAV entity/attribute schema and supporting arbitrary unified
task & horizon configurations (including all 13 canonical tasks across 7 horizons).
"""

from collections import defaultdict
import copy
import hashlib
import random
import numpy as np
from scipy.special import expit

import config as CONFIG
from src.tasks import format_horizon_label, get_task_meta


STATIC_ENTITY_SET = {
    "Patient info", "Donor info", "Transplant info", "Mismatch info",
    "Serology info", "Patient serology", "Immunology", "History",
}

CLUSTER_PROFILES = [
    {
        "id": 0,
        "name": "High risk",
        "color": "#FF4757",
        "n": 120,
        "base_risk": 0.42,
        "risk_std": 0.11,
        "event_rate": 0.28,
        "umap_center": (1.5, 2.3),
        "umap_spread": 1.2,
        "risk_features": [
            ("Patient lab",  "Tacrolimus level (µg/l)", "Higher",  0.18),
            ("Patient lab",  "Leukocytes (G/l)",        "Lowest",  0.16),
            ("Patient lab",  "Lymphocytes (G/l)",       "Lowest",  0.14),
            ("Infection",    "Pathogen",                "CMV",     0.13),
            ("Patient lab",  "CRP (mg/l)",              "Higher",  0.12),
            ("Comorbidity",  "Diagnosis",               "Diabetes mellitus", 0.10),
            ("Patient lab",  "eGFR (mL/min/1.73m²)",    "Lowest",  0.09),
        ],
        "event_weights": {
            ("Medication",   "Start (immunosuppression)",    ("Tacrolimus", "Mycophenolate Mofetil", "Glucocorticoid")): 0.9,
            ("Medication",   "Stop (infection prophylaxis)", ("Cotrimoxazole", "Valganciclovir")): 0.55,
            ("Patient lab",  "Creatinine (µmol/l)",          ("High", "Higher")): 0.9,
            ("Patient lab",  "eGFR (mL/min/1.73m²)",         ("Low", "Lowest")): 0.9,
            ("Patient lab",  "Tacrolimus level (µg/l)",      ("Higher", "Highest")): 0.85,
            ("Patient lab",  "Leukocytes (G/l)",             ("Lowest", "Lower")): 0.8,
            ("Patient lab",  "Lymphocytes (G/l)",            ("Lowest", "Lower")): 0.75,
            ("Patient lab",  "CRP (mg/l)",                   ("High", "Higher")): 0.75,
            ("Infection",    "Pathogen",                     ("CMV", "E. coli", "Klebsiella pneumoniae")): 0.55,
            ("Organ event",  "Diagnosis",                    ("Biops proven rj", "Graft pyelonephritis")): 0.45,
            ("Comorbidity",  "Diagnosis",                    ("Diabetes mellitus", "Hypertension")): 0.65,
            ("Donor info",   "Type",                         ("DBD", "DCD")): 0.7,
            ("Serology info","CMV",                          ("D+/R-", "D+/R+")): 0.65,
        },
    },
    {
        "id": 1,
        "name": "Low risk",
        "color": "#00C9A7",
        "n": 130,
        "base_risk": 0.08,
        "risk_std": 0.035,
        "event_rate": 0.05,
        "umap_center": (-3.0, -2.0),
        "umap_spread": 1.2,
        "risk_features": [
            ("Patient lab", "Creatinine (µmol/l)", "Middle", 0.03),
        ],
        "event_weights": {
            ("Medication",   "Start (immunosuppression)",    ("Tacrolimus", "Mycophenolate Mofetil")): 0.9,
            ("Medication",   "Start (infection prophylaxis)",("Cotrimoxazole", "Valganciclovir")): 0.85,
            ("Patient lab",  "Creatinine (µmol/l)",          ("Low", "Middle")): 0.9,
            ("Patient lab",  "eGFR (mL/min/1.73m²)",         ("High", "Higher")): 0.9,
            ("Patient lab",  "Tacrolimus level (µg/l)",      ("Low", "Middle")): 0.9,
            ("Patient lab",  "Leukocytes (G/l)",             ("Middle",)): 0.85,
            ("Patient lab",  "CRP (mg/l)",                   ("Lowest", "Low")): 0.8,
            ("Donor info",   "Type",                         ("Living related", "Living unrelated")): 0.75,
            ("Serology info","CMV",                          ("D-/R-", "D-/R+")): 0.7,
        },
    },
]

BASELINE_EVENTS = [
    ("Transplant info", "Transplant event",          ["Kidney"]),
    ("Patient info",    "Age at transplant (years)", ["Lower", "Low", "Middle", "High", "Higher"]),
    ("Patient info",    "Sex",                       ["Male", "Female"]),
    ("Transplant info", "Dialysis Type (Pre-Tx)",    ["HD", "PD"]),
]

_RISK_FEATURE_SCORES: dict[tuple[str, str, str], float] = {
    ("Patient lab",  "Tacrolimus level (µg/l)",       "Highest"): 0.20,
    ("Patient lab",  "Tacrolimus level (µg/l)",       "Higher"):  0.17,
    ("Patient lab",  "Tacrolimus level (µg/l)",       "High"):    0.11,
    ("Patient lab",  "Leukocytes (G/l)",              "Lowest"):  0.16,
    ("Patient lab",  "Leukocytes (G/l)",              "Lower"):   0.10,
    ("Patient lab",  "Lymphocytes (G/l)",             "Lowest"):  0.14,
    ("Patient lab",  "Lymphocytes (G/l)",             "Lower"):   0.09,
    ("Infection",    "Pathogen",                      "CMV"):     0.13,
    ("Infection",    "Pathogen",                      "E. coli"): 0.12,
    ("Infection",    "Pathogen",                      "Klebsiella pneumoniae"): 0.13,
    ("Patient lab",  "CRP (mg/l)",                    "Highest"): 0.15,
    ("Patient lab",  "CRP (mg/l)",                    "Higher"):  0.12,
    ("Patient lab",  "CRP (mg/l)",                    "High"):    0.08,
    ("Comorbidity",  "Diagnosis",                     "Diabetes mellitus"): 0.10,
    ("Patient lab",  "eGFR (mL/min/1.73m²)",          "Lowest"):  0.11,
    ("Patient lab",  "eGFR (mL/min/1.73m²)",          "Lower"):   0.07,
    ("Organ event",  "Diagnosis",                     "Biops proven rj"): 0.12,
    ("Organ event",  "Diagnosis",                     "Graft pyelonephritis"): 0.15,
    ("Medication",   "Stop (infection prophylaxis)",  "Cotrimoxazole"): 0.09,
    ("Medication",   "Stop (infection prophylaxis)",  "Valganciclovir"): 0.09,
    # Protective
    ("Medication",   "Start (infection prophylaxis)", "Cotrimoxazole"): -0.11,
    ("Medication",   "Start (infection prophylaxis)", "Valganciclovir"): -0.09,
    ("Patient lab",  "eGFR (mL/min/1.73m²)",          "High"):    -0.08,
    ("Patient lab",  "eGFR (mL/min/1.73m²)",          "Higher"):  -0.10,
    ("Patient lab",  "eGFR (mL/min/1.73m²)",          "Highest"): -0.12,
    ("Donor info",   "Type",                          "Living related"): -0.07,
    ("Patient lab",  "Creatinine (µmol/l)",           "Lowest"):  -0.09,
    ("Patient lab",  "Creatinine (µmol/l)",           "Low"):     -0.06,
    ("Patient lab",  "CRP (mg/l)",                    "Lowest"):  -0.05,
}


def _horizon_multiplier(horizon: int) -> float:
    """Smooth monotone growth factor across arbitrary horizons (30d .. 1800d)."""
    return float(1.0 + 0.18 * np.log1p(max(0.0, (float(horizon) - 30.0) / 30.0)))


def _apply_horizon_multiplier(base_prob_30d: float, horizon: int) -> float:
    """Monotonically scales a 30d probability in log-odds space so P(H1) <= P(H2)."""
    p = float(np.clip(base_prob_30d, 0.005, 0.985))
    if horizon <= 30:
        return round(p, 4)
    shift = 0.34 * np.log(float(horizon) / 30.0)
    logit_p = np.log(p / (1.0 - p))
    return round(float(np.clip(expit(logit_p + shift), p, 0.99)), 4)


def build_clinical_narrative(
    *,
    task_key: str,
    task_label: str,
    selected_horizon: int,
    horizons: list[int],
    risk_scores: dict[str, float],
    calibrated_risks: dict[str, float],
    cluster_name: str,
) -> str:
    """Builds an outcome-aware, multi-horizon clinical summary sentence."""
    meta = get_task_meta(task_key)
    clinical_noun = meta["clinical_noun"]
    score_noun = "infection score" if meta["is_infection"] else "risk score"
    h_lbl = format_horizon_label(selected_horizon, style="long")
    sel_s = int(round(risk_scores.get(f"{selected_horizon}d", 0.0) * 100))
    sel_c = float(calibrated_risks.get(f"{selected_horizon}d", 0.0)) * 100.0

    other_hs = [h for h in horizons if h != selected_horizon]
    ref_hs = [other_hs[0], other_hs[-1]] if len(other_hs) >= 2 else other_hs
    ref_parts = [
        f"{format_horizon_label(oh, style='short')} {int(round(risk_scores.get(f'{oh}d', 0.0) * 100))}/100 "
        f"({calibrated_risks.get(f'{oh}d', 0.0) * 100.0:.1f}%)"
        for oh in ref_hs
    ]
    ref_sentence = (
        f" Reference trajectories span {' to '.join(ref_parts)} across the prediction window."
        if ref_parts
        else ""
    )
    return (
        f"At the {h_lbl} horizon, the patient's {score_noun} for {task_label} is "
        f"{sel_s}/100 (calibrated probability: {sel_c:.1f}%), placing them in the "
        f"'{cluster_name}' subgroup.{ref_sentence} Overall, the patient is stratified as "
        f"{cluster_name} for post-transplant {clinical_noun}."
    )


def _generate_patient_sequence(profile, rng, follow_up_days):
    events = []
    for entity, attribute, values in BASELINE_EVENTS:
        events.append({
            "entity": entity,
            "attribute": attribute,
            "value": str(rng.choice(values)),
            "days_since_tpx": 0.0,
        })

    for (ent, attr, values), prob in profile["event_weights"].items():
        if ent in STATIC_ENTITY_SET and rng.random() < prob:
            events.append({
                "entity": ent,
                "attribute": attr,
                "value": str(rng.choice(values)),
                "days_since_tpx": 0.0,
            })

    for day in follow_up_days[1:]:
        t = round(max(1.0, float(day) + float(rng.uniform(-2.0, 2.0))), 1)
        for (ent, attr, values), prob in profile["event_weights"].items():
            if ent in ("Medication", "Patient lab") and rng.random() < prob:
                events.append({
                    "entity": ent,
                    "attribute": attr,
                    "value": str(rng.choice(values)),
                    "days_since_tpx": t,
                })
            elif ent in ("Infection", "Organ event", "Comorbidity") and rng.random() < prob * 0.4:
                events.append({
                    "entity": ent,
                    "attribute": attr,
                    "value": str(rng.choice(values)),
                    "days_since_tpx": t,
                })

    events.sort(key=lambda e: e["days_since_tpx"])
    return events


def _patient_summary(events):
    snippets = []
    for e in events:
        ent, attr, val = e.get("entity", ""), e.get("attribute", ""), e.get("value", "")
        if "Tacrolimus" in attr and val in ("High", "Higher", "Highest"):
            snippets.append("Tacrolimus↑")
        elif "Leukocytes" in attr and val in ("Lowest", "Lower"):
            snippets.append("Leukopenia")
        elif ent == "Infection" and val not in ("0", "False", "No"):
            snippets.append(f"{val}+")
        elif ent == "Medication" and "Start" in attr and val == "Cotrimoxazole":
            snippets.append("Cotrimoxazole✓")
        elif ent == "Organ event" and "rj" in val.lower():
            snippets.append("Rejection")
        elif "eGFR" in attr and val in ("Lowest", "Lower"):
            snippets.append("Low eGFR")
        elif "CRP" in attr and val in ("High", "Higher", "Highest"):
            snippets.append("CRP↑")
        elif val == "Diabetes mellitus":
            snippets.append("DM")
    dedup = list(dict.fromkeys(snippets))
    return ", ".join(dedup[:4]) if dedup else "Standard profile"


def generate_cohort_data(fup: int = 90, seed: int = 42) -> dict:
    rng = np.random.default_rng(seed)
    horizons = list(getattr(CONFIG, "AVAILABLE_HORIZONS", [30, 60, 90, 180, 360, 720, 1800]))
    all_fup_days = [0, 7, 14, 30, 60, 90, 180, 360, 720]
    follow_up_days = [d for d in all_fup_days if d <= fup] or [0]

    base_patients = []
    patient_seqs = []
    fup_phase = float(fup) / 90.0
    for profile in CLUSTER_PROFILES:
        n = profile["n"]
        cx, cy = profile["umap_center"]
        spread = profile["umap_spread"]
        ux = rng.normal(cx, spread, n)
        uy = rng.normal(cy, spread, n)
        drift_x = rng.normal(0.18 if profile["id"] == 0 else -0.15, 0.12, n)
        drift_y = rng.normal(0.15 if profile["id"] == 0 else -0.12, 0.12, n)

        for i in range(n):
            seq = _generate_patient_sequence(profile, rng, follow_up_days)
            base_patients.append({
                "umap_x": float(ux[i] + drift_x[i] * np.tanh(fup_phase * 0.6)),
                "umap_y": float(uy[i] + drift_y[i] * np.tanh(fup_phase * 0.6)),
                "profile": profile,
                "summary": _patient_summary(seq),
            })
            patient_seqs.append(seq)

    shuffled_indices = list(range(len(base_patients)))
    rng.shuffle(shuffled_indices)

    max_h = max(horizons)
    patient_metrics = []
    for bp in base_patients:
        prof = bp["profile"]
        r30 = float(np.clip(rng.normal(prof["base_risk"], prof["risk_std"]), 0.02, 0.95))
        risks_by_h = {h: _apply_horizon_multiplier(r30, h) for h in horizons}

        u = float(rng.random())
        if prof["id"] == 0:
            t_event = int(rng.exponential(scale=280.0)) + 4 if u < 0.72 else None
        else:
            t_event = int(rng.exponential(scale=950.0)) + 15 if u < 0.25 else None
        if t_event is not None and t_event > max_h:
            t_event = None

        labels_by_h = {h: (1 if (t_event is not None and t_event <= h) else 0) for h in horizons}
        imminence = float(np.exp(-t_event / 365.0)) if t_event is not None else 0.0
        patient_metrics.append({
            "risks": risks_by_h,
            "labels": labels_by_h,
            "days_to_event": t_event,
            "event_imminence": round(imminence, 4),
        })

    horizons_data = {}
    for h in horizons:
        h_patients = []
        all_risks = []

        for out_idx, idx in enumerate(shuffled_indices):
            bp = base_patients[idx]
            metrics = patient_metrics[idx]
            prof = bp["profile"]
            risk = metrics["risks"][h]
            label = metrics["labels"][h]

            if getattr(CONFIG, "CLUSTERING_METHOD", "clusterer") == "clusterer":
                best_cluster = prof
            else:
                best_cluster = CLUSTER_PROFILES[0] if risk >= CONFIG.RISK_THRESHOLD else CLUSTER_PROFILES[1]

            logit_v = np.log(np.clip(risk, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(risk, 1e-4, 1.0 - 1e-4)))
            cal_risk = float(np.clip(expit(logit_v - 1.35), 0.005, 0.96))

            h_patients.append({
                "id":              f"PAT_{out_idx + 1:03d}",
                "umap_x":          round(bp["umap_x"], 3),
                "umap_y":          round(bp["umap_y"], 3),
                "risk_score":      risk,
                "calibrated_risk": round(cal_risk, 4),
                "cluster":         best_cluster["id"],
                "cluster_name":    best_cluster["name"],
                "cluster_color":   best_cluster["color"],
                "true_label":      label,
                "summary":         bp["summary"],
                "days_to_event":   metrics["days_to_event"],
                "event_imminence": metrics["event_imminence"],
            })
            all_risks.append(risk)

        rv = np.array(all_risks)
        risk_distribution = {
            "values": [round(float(x), 4) for x in rv],
            "mean": round(float(rv.mean()), 4),
            "std": round(float(rv.std()), 4),
            "percentiles": {
                "p25": round(float(np.percentile(rv, 25)), 4),
                "p50": round(float(np.percentile(rv, 50)), 4),
                "p75": round(float(np.percentile(rv, 75)), 4),
                "p90": round(float(np.percentile(rv, 90)), 4),
            },
        }

        cluster_sizes = {0: 0, 1: 0}
        for p in h_patients:
            cluster_sizes[p["cluster"]] = cluster_sizes.get(p["cluster"], 0) + 1

        static_counts = {0: defaultdict(int), 1: defaultdict(int)}
        recent_counts = {0: defaultdict(int), 1: defaultdict(int)}
        all_static_features = set()
        all_recent_features = set()

        for out_idx, idx in enumerate(shuffled_indices):
            c_id = h_patients[out_idx]["cluster"]
            seq = patient_seqs[idx]
            p_static = set()
            p_recent = set()
            valid_times = [float(ev.get("days_since_tpx", 0.0)) for ev in seq if ev.get("entity") and ev.get("attribute")]
            max_t = max(valid_times) if valid_times else 0.0

            for ev in seq:
                ent = ev.get("entity", "")
                attr = ev.get("attribute", "")
                val = ev.get("value", "")
                t_val = float(ev.get("days_since_tpx", 0.0))
                if not ent or not attr or not val or attr in ("Transplant event", "Transplant procedure"):
                    continue
                feat = (ent, attr, val)
                is_static = ent in STATIC_ENTITY_SET or t_val <= 0.0
                if is_static:
                    p_static.add(feat)
                elif t_val >= max(0.1, max_t - 30.0):
                    p_recent.add(feat)

            for feat in p_static:
                static_counts[c_id][feat] += 1
                all_static_features.add(feat)
            for feat in p_recent:
                recent_counts[c_id][feat] += 1
                all_recent_features.add(feat)

        cluster_profiles = {}
        for profile in CLUSTER_PROFILES:
            c_id = profile["id"]
            other_c_id = 1 - c_id
            c_size = cluster_sizes.get(c_id, 0)
            other_size = cluster_sizes.get(other_c_id, 0)
            c_labels = [p["true_label"] for p in h_patients if p["cluster"] == c_id and p["true_label"] >= 0]
            c_event_rate = min(0.99, float(np.mean(c_labels)) if c_labels else 0.0)

            def _top_enriched(counts_dict, universe):
                diffs = []
                for feat in universe:
                    p_c = (counts_dict[c_id][feat] / c_size) if c_size > 0 else 0.0
                    p_opp = (counts_dict[other_c_id][feat] / other_size) if other_size > 0 else 0.0
                    p_glob = (counts_dict[c_id][feat] + counts_dict[other_c_id][feat]) / float(max(1, len(h_patients)))
                    if p_glob > 0.85:
                        continue
                    diff = p_c - p_opp
                    if diff >= 0.04 and (p_opp == 0 or (p_c / p_opp) >= 1.15):
                        diffs.append((feat, diff, diff * (p_c / (p_opp + 0.05))))
                return [
                    {"feature": f"{ent} - {attr}", "value": val, "score": (diff if c_id == 0 else -diff)}
                    for (ent, attr, val), diff, _ in sorted(diffs, key=lambda x: x[2], reverse=True)[:10]
                ]

            cluster_profiles[str(c_id)] = {
                "name":       profile["name"],
                "n_patients": c_size,
                "event_rate": round(c_event_rate, 4),
                "color":      profile["color"],
                "top_features": {
                    "static": _top_enriched(static_counts, all_static_features),
                    "recent": _top_enriched(recent_counts, all_recent_features),
                },
            }

        horizons_data[str(h)] = {
            "patients": h_patients,
            "risk_distribution": risk_distribution,
            "cluster_profiles": cluster_profiles,
        }

    tasks_data = {}
    available_tasks = getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}])
    task_horizons_cfg = getattr(CONFIG, "TASK_HORIZONS", {})
    for t_info in available_tasks:
        t_key = t_info["key"]
        t_hrzs = [str(h) for h in task_horizons_cfg.get(t_key, horizons) if str(h) in horizons_data]
        filtered_horizons = {hk: horizons_data[hk] for hk in (t_hrzs or list(horizons_data.keys()))}
        scale = float(get_task_meta(t_key).get("mock_scale", 0.80))
        if abs(scale - 1.0) < 1e-6:
            tasks_data[t_key] = {"horizons": filtered_horizons}
        else:
            t_horizons = copy.deepcopy(filtered_horizons)
            for h_obj in t_horizons.values():
                for p in h_obj["patients"]:
                    p["risk_score"] = round(float(np.clip(p["risk_score"] * scale, 0.01, 0.99)), 4)
                    p["calibrated_risk"] = round(float(np.clip(p["calibrated_risk"] * scale, 0.005, 0.98)), 4)
                rv = np.array([p["risk_score"] for p in h_obj["patients"]])
                h_obj["risk_distribution"] = {
                    "values": [round(float(x), 4) for x in rv],
                    "mean": round(float(rv.mean()), 4),
                    "std": round(float(rv.std()), 4),
                    "percentiles": {
                        "p25": round(float(np.percentile(rv, 25)), 4),
                        "p50": round(float(np.percentile(rv, 50)), 4),
                        "p75": round(float(np.percentile(rv, 75)), 4),
                        "p90": round(float(np.percentile(rv, 90)), 4),
                    },
                }
                for c_prof in h_obj["cluster_profiles"].values():
                    c_prof["event_rate"] = round(float(min(0.99, c_prof["event_rate"] * scale)), 4)
            tasks_data[t_key] = {"horizons": t_horizons}

    default_horizons = tasks_data.get(CONFIG.TASK, {"horizons": horizons_data})["horizons"]
    return {
        "fup": fup,
        "global_umap_bounds": {"min_x": -6.5, "max_x": 5.5, "min_y": -5.5, "max_y": 6.0},
        "horizons": default_horizons,
        "tasks": tasks_data,
    }


def _deterministic_seed(events: list[dict]) -> int:
    sorted_events = sorted(
        events,
        key=lambda e: (
            float(e.get("days_since_tpx", 0.0)),
            str(e.get("entity", "")),
            str(e.get("attribute", "")),
            str(e.get("value", "")),
        ),
    )
    event_str = "".join(
        f"{e.get('entity')}-{e.get('attribute')}-{e.get('value')}-{e.get('days_since_tpx')}"
        for e in sorted_events
    )
    return int(hashlib.md5(event_str.encode("utf-8")).hexdigest(), 16) & 0xFFFFFFFF


def _truncate_events_for_fup(events: list[dict], fup: int | None) -> list[dict]:
    if fup is None or not events:
        return list(events)
    filtered = [e for e in events if float(e.get("days_since_tpx", 0.0)) <= float(fup)]
    if not filtered:
        min_day = min(float(e.get("days_since_tpx", 0.0)) for e in events)
        filtered = [e for e in events if float(e.get("days_since_tpx", 0.0)) <= min_day]
    return filtered


def _mock_stage_coords_and_scores(events_at_stage: list[dict], fup_val: int) -> tuple[float, float, float]:
    raw_score = 0.0
    for ev in events_at_stage:
        key = (ev.get("entity", ""), ev.get("attribute", ""), ev.get("value", ""))
        raw_score += _RISK_FEATURE_SCORES.get(key, 0.0)
    if events_at_stage:
        event_str = " ".join(
            f"{e.get('entity', '')} {e.get('attribute', '')} {e.get('value', '') or e.get('value_binned', '')}"
            for e in events_at_stage
        ).lower()
        if "pyelonephritis" in event_str:
            raw_score += 0.21
        elif "quinolone" in event_str or "pckd" in event_str:
            raw_score -= 0.65

    base_risk_30d_main = float(np.clip(expit(raw_score * 3.0), 0.04, 0.94))
    risk_90d = _apply_horizon_multiplier(base_risk_30d_main, 90)
    local_random = random.Random(_deterministic_seed(events_at_stage) ^ int(fup_val * 31))
    alpha = float(np.clip((risk_90d - 0.10) / 0.65, 0.0, 1.0))
    cx = -3.0 + alpha * 4.5
    cy = -2.0 + alpha * 4.3
    return (
        round(float(cx + local_random.gauss(0, 0.22)), 3),
        round(float(cy + local_random.gauss(0, 0.22)), 3),
        base_risk_30d_main,
    )


def mock_predict(
    events: list[dict],
    horizon: int,
    cohort: dict,
    selected_task: str | None = None,
    fup: int | None = None,
    trajectory_fups: list[int] | None = None,
) -> dict:
    horizons = list(getattr(CONFIG, "AVAILABLE_HORIZONS", [30, 60, 90, 180, 360, 720, 1800]))
    full_events = sorted(list(events), key=lambda e: float(e.get("days_since_tpx", 0.0)))
    active_fup = fup if fup is not None else CONFIG.DEFAULT_FUP
    stage_events = _truncate_events_for_fup(full_events, active_fup)

    attribution_map: dict[str, dict] = {}
    events_with_scores = []
    for ev in stage_events:
        key = (ev.get("entity", ""), ev.get("attribute", ""), ev.get("value", ""))
        sc = _RISK_FEATURE_SCORES.get(key, 0.0)
        feat_key = f"{key[0]} - {key[1]}: {key[2]}"
        if feat_key not in attribution_map:
            attribution_map[feat_key] = {"feature": f"{key[0]} - {key[1]}", "value": key[2], "score": sc}
        events_with_scores.append({**ev, "score": sc})

    umap_x, umap_y, base_risk_30d_main = _mock_stage_coords_and_scores(stage_events, active_fup)
    attributions = sorted(attribution_map.values(), key=lambda x: abs(x["score"]), reverse=True)[:15]

    available_tasks = getattr(CONFIG, "AVAILABLE_TASKS", [{"key": CONFIG.TASK, "label": CONFIG.TASK}])
    cohort_tasks = cohort.get("tasks", {})
    tasks_predictions = {}

    task_horizons_cfg = getattr(CONFIG, "TASK_HORIZONS", {})
    for t_info in available_tasks:
        t_key = t_info["key"]
        t_label = t_info["label"]
        t_horizons = list(task_horizons_cfg.get(t_key, horizons))
        scale = float(get_task_meta(t_key).get("mock_scale", 0.80))
        b30 = float(np.clip(base_risk_30d_main * scale, 0.02, 0.94))

        risk_scores = {f"{h}d": _apply_horizon_multiplier(b30, h) for h in t_horizons}
        calibrated_risks = {}
        for hk, hv in risk_scores.items():
            lv = np.log(np.clip(hv, 1e-4, 1.0 - 1e-4) / (1.0 - np.clip(hv, 1e-4, 1.0 - 1e-4)))
            calibrated_risks[hk] = round(float(np.clip(expit(lv - 1.35), 0.005, 0.96)), 4)

        t_cohort = cohort_tasks.get(t_key, cohort)
        first_cohort_pts = t_cohort["horizons"][str(t_horizons[0])]["patients"]
        if first_cohort_pts:
            umap_pts = np.array([[p["umap_x"], p["umap_y"]] for p in first_cohort_pts])
            dists = np.sum((umap_pts - np.array([umap_x, umap_y])) ** 2, axis=1)
            top_idx = np.argsort(dists)[:3]
        else:
            top_idx = np.array([], dtype=int)

        by_horizon = {}
        for h in t_horizons:
            selected_risk = risk_scores[f"{h}d"]
            selected_calibrated_risk = calibrated_risks[f"{h}d"]
            best_cluster = min(CLUSTER_PROFILES, key=lambda p: abs(p["base_risk"] * scale - selected_risk))
            cluster_id = best_cluster["id"]
            cluster_name = best_cluster["name"]
            cluster_color = best_cluster["color"]

            active_cohort = t_cohort["horizons"][str(h)]
            c_pts = active_cohort["patients"]
            cohort_risks = np.array([p["risk_score"] for p in c_pts]) if c_pts else np.array([0.5])
            percentile = int(np.round((cohort_risks < selected_risk).mean() * 100))
            similar = [
                {"id": c_pts[i]["id"], "risk_score": c_pts[i]["risk_score"], "summary": c_pts[i]["summary"]}
                for i in top_idx if i < len(c_pts)
            ]

            risk_txt = build_clinical_narrative(
                task_key=t_key,
                task_label=t_label,
                selected_horizon=h,
                horizons=t_horizons,
                risk_scores=risk_scores,
                calibrated_risks=calibrated_risks,
                cluster_name=cluster_name,
            )

            by_horizon[str(h)] = {
                "task":               t_key,
                "task_label":         t_label,
                "risk_scores":        risk_scores,
                "calibrated_risks":   calibrated_risks,
                "selected_horizon":   h,
                "risk_score":         selected_risk,
                "calibrated_risk":    selected_calibrated_risk,
                "risk_category":      cluster_name,
                "risk_color":         cluster_color,
                "percentile":         percentile,
                "cluster":            cluster_id,
                "cluster_name":       cluster_name,
                "cluster_color":      cluster_color,
                "umap_x":             umap_x,
                "umap_y":             umap_y,
                "days_to_event":      25 if cluster_id == 0 else None,
                "event_imminence":    round(float(np.exp(-25.0 / 365.0)), 4) if cluster_id == 0 else 0.0,
                "attributions":       [{**a, "score": round(a["score"] * scale, 4)} for a in attributions],
                "events_with_scores": [{**ev, "score": round(ev["score"] * scale, 4)} for ev in events_with_scores],
                "similar_patients":   similar,
                "narrative":          risk_txt,
                "ig_computed":        True,
            }

        sel_h_str = str(horizon if horizon in t_horizons else t_horizons[0])
        tasks_predictions[t_key] = {
            **by_horizon[sel_h_str],
            "fup": active_fup,
            "ig_computed": True,
            "by_horizon": by_horizon,
        }

    if trajectory_fups is None:
        trajectory_fups = [f for f in getattr(CONFIG, "AVAILABLE_FUPS", [0, 30, 60, 90]) if f <= active_fup]
    trajectory = []
    for f_step in trajectory_fups:
        evs_f = _truncate_events_for_fup(full_events, f_step)
        ux_f, uy_f, r30_f = _mock_stage_coords_and_scores(evs_f, f_step)
        t_scores = {}
        for t_info in available_tasks:
            t_k = t_info["key"]
            t_hrzs = list(task_horizons_cfg.get(t_k, horizons))
            sc = float(get_task_meta(t_k).get("mock_scale", 0.80))
            b30 = float(np.clip(r30_f * sc, 0.02, 0.94))
            rs = {f"{h}d": _apply_horizon_multiplier(b30, h) for h in t_hrzs}
            cs = {
                hk: round(float(np.clip(expit(np.log(hv / (1.0 - hv)) - 1.35), 0.005, 0.96)), 4)
                for hk, hv in rs.items()
            }
            t_scores[t_k] = {"risk_scores": rs, "calibrated_risks": cs}
        trajectory.append({
            "fup": f_step,
            "umap_x": ux_f,
            "umap_y": uy_f,
            "n_events": len(evs_f),
            "tasks": t_scores,
        })

    active_task = selected_task if (selected_task and selected_task in tasks_predictions) else CONFIG.TASK
    return {
        **tasks_predictions[active_task],
        "fup": active_fup,
        "trajectory": trajectory,
        "tasks": tasks_predictions,
    }
