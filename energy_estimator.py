from __future__ import annotations

ENERGY = {
    "light": {"wh_per_1k": 0.001, "example": "7B model (Mistral-7B)"},
    "medium": {"wh_per_1k": 0.009, "example": "13–34B model (Mixtral-8x7B)"},
    "heavy": {"wh_per_1k": 0.032, "example": "70B+ model (Llama-3-70B)"},
}
GRID_CO2 = 385

TIER_MAP = {
    "code": "medium",
    "reasoning": "heavy",
    "data": "medium",
    "creative": "medium",
    "writing": "light",
    "summarization": "light",
    "qa": "light",
    "translation": "light",
    "general": "light",
}


def get_tier(task: str, complexity_score: float) -> str:
    base = TIER_MAP.get(task, "light")
    if complexity_score >= 4.0:
        return "heavy"
    if complexity_score >= 2.5 and base == "light":
        return "medium"
    return base


def estimate(tokens: int, tier: str) -> dict:
    wh = (tokens / 1000) * ENERGY[tier]["wh_per_1k"]
    co2_g = (wh / 1000) * GRID_CO2
    return {
        "wh": round(wh, 5),
        "co2_g": round(co2_g, 5),
        "model": ENERGY[tier]["example"],
        "tier": tier,
    }


def savings(orig_tok: int, comp_tok: int, heavy_tier: str, selected_tier: str) -> dict:
    baseline = estimate(orig_tok, heavy_tier)
    actual = estimate(comp_tok, selected_tier)
    pct_saved = round((1 - actual["wh"] / max(baseline["wh"], 1e-9)) * 100, 1)
    return {
        "wh_saved": round(baseline["wh"] - actual["wh"], 5),
        "co2_g_saved": round(baseline["co2_g"] - actual["co2_g"], 5),
        "pct_saved": max(0.0, pct_saved),
        "note": "Source: Luccioni et al. 2023. Values are approximations.",
    }


def simple_complexity(token_count: int, task: str) -> float:
    base = {"reasoning": 3.5, "code": 2.5, "data": 2.5, "creative": 2.0}.get(task, 1.5)
    length_bonus = min(token_count / 200, 2.0)
    return round(min(base + length_bonus, 5.0), 1)
