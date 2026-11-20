"""Canonical clinical task, horizon, and feature-naming metadata for AIIDKIT.

Shared across training (`src/`, `scripts/`), downstream evaluation/analyses,
and the interactive web decision-support tool (`web_tool/`).
"""
from __future__ import annotations

import re

# Default 18 validation FUP landmarks spanning all 5 post-transplant windows (0..3600d)
DEFAULT_VALIDATION_FUPS: list[int] = [
    0, 30, 60, 90, 120, 150, 180, 240, 300, 360,
    540, 720, 900, 1080, 1440, 1800, 2700, 3600,
]

ACUTE_HORIZONS: list[int] = [30, 60, 90]
LONG_TERM_HORIZONS: list[int] = [360, 720, 1800]
UNIFIED_HORIZONS: list[int] = [30, 60, 90, 180, 360, 720, 1800]

# Canonical metadata for all AIIDKIT clinical outcomes
TASK_CATALOG: dict[str, dict] = {
    # --- Bacterial Infections ---
    "infection_bacteria": {
        "label": "All Bacteria (General)",
        "group": "Bacterial Infections",
        "domain": "Bacteria",
        "clinical_noun": "a bacterial infection",
        "is_infection": True,
        "mock_scale": 1.00,
    },
    "infection_grp_enterobacterales": {
        "label": "Enterobacterales",
        "group": "Bacterial Infections",
        "domain": "Bacteria",
        "clinical_noun": "an Enterobacterales infection",
        "is_infection": True,
        "mock_scale": 0.72,
    },
    "infection_grp_gram_positive": {
        "label": "Gram-positive Bacteria",
        "group": "Bacterial Infections",
        "domain": "Bacteria",
        "clinical_noun": "a Gram-positive bacterial infection",
        "is_infection": True,
        "mock_scale": 0.45,
    },
    "infection_grp_gn_nonfermenters": {
        "label": "Gram-negative Non-fermenters",
        "group": "Bacterial Infections",
        "domain": "Bacteria",
        "clinical_noun": "a Gram-negative non-fermenter infection",
        "is_infection": True,
        "mock_scale": 0.22,
    },
    "infection_grp_clostridium": {
        "label": "Clostridioides spp.",
        "group": "Bacterial Infections",
        "domain": "Bacteria",
        "clinical_noun": "a Clostridioides infection",
        "is_infection": True,
        "mock_scale": 0.18,
    },
    # --- Viral Infections ---
    "infection_virus": {
        "label": "All Viruses (General)",
        "group": "Viral Infections",
        "domain": "Viruses",
        "clinical_noun": "a viral infection",
        "is_infection": True,
        "mock_scale": 0.85,
    },
    "infection_grp_herpes_viruses": {
        "label": "Herpesviruses (HSV / VZV / EBV)",
        "group": "Viral Infections",
        "domain": "Viruses",
        "clinical_noun": "a Herpesvirus infection",
        "is_infection": True,
        "mock_scale": 0.38,
    },
    "infection_grp_cmv": {
        "label": "Cytomegalovirus (CMV)",
        "group": "Viral Infections",
        "domain": "Viruses",
        "clinical_noun": "a Cytomegalovirus (CMV) infection",
        "is_infection": True,
        "mock_scale": 0.30,
    },
    "infection_grp_bkpyv": {
        "label": "BK Polyomavirus (BKPyV)",
        "group": "Viral Infections",
        "domain": "Viruses",
        "clinical_noun": "a BK Polyomavirus (BKPyV) infection",
        "is_infection": True,
        "mock_scale": 0.26,
    },
    # --- Fungal Infections ---
    "infection_fungi": {
        "label": "All Fungi (General)",
        "group": "Fungal Infections",
        "domain": "Fungi",
        "clinical_noun": "a fungal infection",
        "is_infection": True,
        "mock_scale": 0.32,
    },
    "infection_grp_candida": {
        "label": "Candida spp.",
        "group": "Fungal Infections",
        "domain": "Fungi",
        "clinical_noun": "a Candida fungal infection",
        "is_infection": True,
        "mock_scale": 0.22,
    },
    "infection_grp_pneumocystis": {
        "label": "Pneumocystis jirovecii",
        "group": "Fungal Infections",
        "domain": "Fungi",
        "clinical_noun": "a Pneumocystis jirovecii infection",
        "is_infection": True,
        "mock_scale": 0.12,
    },
    "infection_grp_aspergillus": {
        "label": "Aspergillus spp.",
        "group": "Fungal Infections",
        "domain": "Fungi",
        "clinical_noun": "an Aspergillus infection",
        "is_infection": True,
        "mock_scale": 0.10,
    },
    # --- Patient & Allograft Survival Outcomes ---
    "death": {
        "label": "Patient Mortality (Death)",
        "group": "Clinical Outcomes",
        "domain": "Death",
        "clinical_noun": "all-cause mortality",
        "is_infection": False,
        "mock_scale": 0.24,
    },
    "graft_loss": {
        "label": "Allograft Loss",
        "group": "Clinical Outcomes",
        "domain": "Graft Loss",
        "clinical_noun": "allograft loss",
        "is_infection": False,
        "mock_scale": 0.28,
    },
    # --- Top-level Pan-Infection Umbrella (optional) ---
    "infection": {
        "label": "Any Infection (Overall)",
        "group": "Bacterial Infections",
        "domain": "Infection",
        "clinical_noun": "an infection",
        "is_infection": True,
        "mock_scale": 1.10,
    },
}

# The 13 well-powered clinical tasks used in `multitask_unified`
UNIFIED_TASKS: list[str] = [
    "infection_bacteria",
    "infection_grp_enterobacterales",
    "infection_grp_gram_positive",
    "infection_grp_gn_nonfermenters",
    "infection_grp_clostridium",
    "infection_virus",
    "infection_grp_herpes_viruses",
    "infection_grp_cmv",
    "infection_grp_bkpyv",
    "infection_fungi",
    "infection_grp_candida",
    "death",
    "graft_loss",
]

# Legacy 9-task bacterial + viral subset (`multitask_all`)
PAN_BV_TASKS: list[str] = UNIFIED_TASKS[:9]


def normalize_task_key(task_key: str) -> str:
    """Canonicalize survival shorthand (`grp_cmv`) to binary label key (`infection_grp_cmv`)."""
    t = str(task_key).strip()
    if t.startswith("grp_"):
        return f"infection_{t}"
    return t


def to_survival_task_key(task_key: str) -> str:
    """Convert canonical binary task key (`infection_grp_cmv`) to survival label key (`grp_cmv`)."""
    t = str(task_key).strip()
    if t.startswith("infection_grp_"):
        return t[len("infection_"):]
    return t


def get_task_meta(task_key: str) -> dict:
    """Return canonical task metadata dict with graceful fallback for unknown task keys."""
    norm_key = normalize_task_key(task_key)
    if norm_key in TASK_CATALOG:
        return dict(TASK_CATALOG[norm_key])
    clean = norm_key.replace("infection_grp_", "").replace("infection_", "").replace("_", " ").title()
    is_inf = norm_key not in ("death", "graft_loss")
    return {
        "label": clean,
        "group": "Clinical Outcomes" if not is_inf else "Other Infections",
        "domain": clean,
        "clinical_noun": f"a {clean} infection" if is_inf else clean.lower(),
        "is_infection": is_inf,
        "mock_scale": 0.35,
    }


def build_available_tasks_list(task_keys: list[str]) -> list[dict[str, str]]:
    """Build UI-ready `[{'key': ..., 'label': ..., 'group': ..., 'is_infection': ...}]` list."""
    out = []
    for k in task_keys:
        meta = get_task_meta(k)
        out.append({
            "key": k,
            "label": meta["label"],
            "group": meta["group"],
            "domain": meta["domain"],
            "clinical_noun": meta["clinical_noun"],
            "is_infection": meta["is_infection"],
        })
    return out


def format_horizon_label(days: int, style: str = "short") -> str:
    """Format a prediction horizon in days into a clean clinical label.

    Styles:
      - 'compact': '30d', '60d', '90d', '6m', '1y', '2y', '5y'
      - 'short':   '30d', '60d', '90d', '6m (180d)', '1y (360d)', '2y (720d)', '5y (1800d)'
      - 'long':    '30-day', '60-day', '90-day', '6-month (180d)', '1-year (360d)', '2-year (720d)', '5-year (1800d)'
    """
    d = int(days)
    exact_years = {360: 1, 365: 1, 720: 2, 730: 2, 1080: 3, 1095: 3, 1800: 5, 1825: 5, 3600: 10, 3650: 10}
    if d == 180:
        if style == "compact":
            return "6m"
        if style == "long":
            return "6-month (180d)"
        return "6m (180d)"
    if d in exact_years:
        y = exact_years[d]
        if style == "compact":
            return f"{y}y"
        if style == "long":
            return f"{y}-year ({d}d)"
        return f"{y}y ({d}d)"
    if style == "long":
        return f"{d}-day"
    return f"{d}d"


def build_horizon_labels_map(
    horizons: list[int],
    style: str | None = "long",
) -> dict[int, str] | dict[str, dict[str, str]]:
    """Build `{int(h): formatted_str}` for a requested `style` (or full nested map when `style=None`)."""
    if style is not None:
        return {int(h): format_horizon_label(int(h), style) for h in horizons}
    return {
        str(int(h)): {
            "compact": format_horizon_label(int(h), "compact"),
            "short": format_horizon_label(int(h), "short"),
            "long": format_horizon_label(int(h), "long"),
        }
        for h in horizons
    }


def format_eav_feature_name(ent_name: str, attr_name: str, entity_id: int | None = None) -> str:
    """Format decoded `(entity, attribute)` token strings into a readable clinical feature label."""
    if ent_name == "Infection":
        ent_name = "Previous infection"
    if entity_id is not None and entity_id < 5:
        return attr_name
    return f"{ent_name} - {attr_name}" if ent_name else attr_name


_ABBREV_REPLACEMENTS = {
    "patient": "pt.",
    "transplant": "tpx.",
    "immunology": "immu.",
    "medication": "med.",
    "hemoglobin": "hgb.",
    "previous": "prev.",
    "infection": "inf.",
    "diagnosis": "diag.",
    "event": "evt.",
    "resistance": "res.",
    "pathogen": "path.",
    "comorbidity": "comorb.",
    "pulmonary": "pulm.",
    "bacteria": "bact.",
    "clinically": "clin.",
    "clinical": "clin.",
}
_ABBREV_PATTERN = re.compile(r"\b(" + "|".join(_ABBREV_REPLACEMENTS.keys()) + r")\b", re.IGNORECASE)


def abbreviate_feature_name(text: str) -> str:
    """Replace full clinical terms with concise abbreviations while preserving original casing."""
    def _replace(match: re.Match) -> str:
        word = match.group(0)
        rep = _ABBREV_REPLACEMENTS.get(word.lower(), word)
        if word.istitle():
            return rep.capitalize()
        if word.isupper():
            return rep.upper()
        return rep

    return _ABBREV_PATTERN.sub(_replace, text)
