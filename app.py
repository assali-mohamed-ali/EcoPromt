import streamlit as st

from clarity_gate import ClarityResult, assess, enrich
from energy_estimator import estimate, get_tier, savings, simple_complexity
from task_classifier import predict as classify_task
from token_economist import EconomyReport, analyze

st.set_page_config(page_title="EcoPrompt AI", page_icon="🌱", layout="wide")
st.title("🌱 EcoPrompt AI")
st.caption("Reduce token waste. Save energy. Get better AI responses.")


# ── Sidebar ───────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Prompt")
    user_prompt = st.text_area(
        "Enter your prompt",
        height=160,
        placeholder="e.g. fix it  /  write something  /  help me",
    )
    run = st.button("Analyze & Optimize", type="primary", use_container_width=True)

if not run or not user_prompt.strip():
    st.info("Enter a prompt in the sidebar and click Analyze.")
    st.stop()


# ── Step 1 : Clarity Gate ─────────────────────────────────────────
st.subheader("Step 1 — Clarity Check")
clarity: ClarityResult = assess(user_prompt)

col1, col2 = st.columns([1, 2])
with col1:
    color = "red" if clarity.probability > 0.6 else "orange" if clarity.probability > 0.3 else "green"
    st.metric("Ambiguity score", f"{clarity.probability:.2f}")
    st.progress(clarity.probability)

if clarity.is_ambiguous:
    st.warning("Your prompt is ambiguous. Answer a few questions to help the model understand you better.")
    answers = {}
    with st.form("clarify"):
        for q in clarity.questions:
            selected = st.multiselect(q["text"], q["options"], key=q["id"])
            answers[q["id"]] = selected
        submitted = st.form_submit_button("Clarify →")
    if submitted:
        user_prompt = enrich(user_prompt, answers)
        st.success("Prompt enriched with your answers.")
        with st.expander("View enriched prompt"):
            st.code(user_prompt)
else:
    st.success("Prompt is clear — no clarification needed.")


# ── Step 2 : Token Economy Analysis ───────────────────────────────
st.divider()
st.subheader("Step 2 — Token Economy Report")

report: EconomyReport = analyze(user_prompt)

# Summary metrics
m1, m2, m3, m4 = st.columns(4)
m1.metric("Original tokens", report.original_tokens)
m2.metric("Compressed tokens", report.compressed_tokens)
m3.metric("Tokens saved", report.token_reduction)
m4.metric("Reduction", f"{report.reduction_pct}%")

# Findings in tabs
t1, t2, t3, t4, t5 = st.tabs(["Fillers removed", "Repetitions", "Language overhead", "Costly patterns", "Redundant sentences"])

with t1:
    if report.fillers_removed:
        for f in report.fillers_removed:
            st.write(f"- **{f['pattern']}** ({f['reason']}) — found {f['count']}×")
    else:
        st.write("No filler words detected.")

with t2:
    reps = report.repetitions
    if reps["repeated_words"]:
        st.write(f"**{reps['total_wasted_tokens']} tokens wasted** by repetition:")
        for r in reps["repeated_words"]:
            st.write(f"- `{r['word']}` repeated {r['count']}× → {r['tokens_wasted']} extra token(s)")
    else:
        st.write("No significant word repetition detected.")

with t3:
    lang = report.language_analysis
    if lang["token_overhead_warning"]:
        st.warning(lang["explanation"])
        st.write(f"Languages detected: {', '.join(lang['languages_detected'])}")
    else:
        st.write("Single language detected — no tokenization overhead.")

with t4:
    if report.costly_patterns:
        for p in report.costly_patterns:
            st.write(f"**{p['name']}** (×{p['count']})")
            st.caption(f"{p['explanation']}")
            st.caption(f"💡 {p['suggestion']}")
    else:
        st.write("No costly token patterns found.")

with t5:
    red = report.redundant_sentences
    if red["redundant_pairs"]:
        st.write(f"Estimated **{red['estimated_removable_tokens']} removable tokens** from redundancy:")
        for pair in red["redundant_pairs"]:
            st.write(f"- Similarity {pair['similarity']}: *\"{pair['s1']}...\"* ≈ *\"{pair['s2']}...\"*")
    else:
        st.write("No redundant sentences detected.")

# Optimized prompt preview
st.divider()
st.subheader("Optimized Prompt")
col_a, col_b = st.columns(2)
with col_a:
    st.caption("Original")
    st.code(report.original_prompt)
with col_b:
    st.caption("Compressed")
    st.code(report.compressed_prompt)


# ── Step 3 : Task + Routing ───────────────────────────────────────
st.divider()
st.subheader("Step 3 — Task Classification & Model Routing")

task_result = classify_task(report.compressed_prompt)
task = task_result["task"]
complexity = simple_complexity(report.compressed_tokens, task)
tier = get_tier(task, complexity)

c1, c2, c3 = st.columns(3)
c1.metric("Task type", task)
c2.metric("Complexity score", f"{complexity} / 5")
c3.metric("Recommended tier", tier.upper())

tier_models = {
    "light": "Mistral-7B-Instruct",
    "medium": "Mixtral-8x7B-Instruct",
    "heavy": "Llama-3-70B-Instruct",
}
st.info(f"Recommended open-source model: **{tier_models[tier]}** — run locally with Ollama or HuggingFace.")


# ── Step 4 : Energy & CO₂ savings ────────────────────────────────
st.divider()
st.subheader("Step 4 — Energy & CO₂ Savings")

result = savings(
    orig_tok=report.original_tokens,
    comp_tok=report.compressed_tokens,
    heavy_tier="heavy",
    selected_tier=tier,
)

e1, e2, e3 = st.columns(3)
e1.metric("Energy saved", f"{result['wh_saved']:.4f} Wh")
e2.metric("CO₂ avoided", f"{result['co2_g_saved']:.4f} g")
e3.metric("% saved", f"{result['pct_saved']}%")

st.caption(result["note"])
st.caption("Grid intensity: 385 gCO₂/kWh (IEA 2023 global average)")
