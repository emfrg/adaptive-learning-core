"""Algorithm Explorer - Interactive demos of the core algorithms."""

import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

from adaptive_learning_core.models.card import CardState, LearnerStereotype
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.algorithms.mastery_update import MasteryUpdater
from adaptive_learning_core.algorithms.knowledge_decay import KnowledgeDecay
from adaptive_learning_core.algorithms.card_selector import CardSelector
from adaptive_learning_core.bloom.classifier import HeuristicBloomClassifier
from adaptive_learning_core.bloom.taxonomy import BLOOM_LEVEL_INFO
from adaptive_learning_core.irt.models import Rasch1PL, Model2PL, Model3PL, ItemParameters


st.title("🔬 Algorithm Explorer")
st.markdown("""
Explore the core algorithms that power the adaptive learning system.
Each section lets you interact with the algorithms and see their behavior.
""")

# Algorithm selection
algo = st.selectbox(
    "Select Algorithm",
    [
        "Mastery Update (Eq. 3)",
        "Knowledge Decay (Eq. 7-8)",
        "Card Selection (Eq. 4-5)",
        "Bloom's Taxonomy Classification",
        "Item Response Theory (IRT)",
    ],
)

st.divider()

# ============================================================================
# Mastery Update
# ============================================================================
if algo == "Mastery Update (Eq. 3)":
    st.subheader("📈 Mastery Update Algorithm")

    st.latex(r"M(c) = M(c) + \alpha \times C_{c,q} \times \delta")

    st.markdown("""
    **Parameters:**
    - **M(c)** = Current mastery score for card c
    - **α** = Learning rate (depends on pace: slow=0.05, medium=0.10, fast=0.15)
    - **C_{c,q}** = Contribution weight of card c to question q
    - **δ** = +1 (correct) or -1 (incorrect)
    """)

    col1, col2 = st.columns(2)

    with col1:
        current_mastery = st.slider(
            "Current Mastery M(c)",
            0.0, 1.0, 0.3, 0.05,
            help="The card's current mastery score",
        )

        contribution = st.slider(
            "Contribution Weight C_{c,q}",
            0.0, 1.0, 0.8, 0.1,
            help="How much this card contributes to the question",
        )

    with col2:
        pace = st.radio(
            "Learning Pace",
            ["slow", "medium", "fast"],
            index=1,
        )

        is_correct = st.toggle("Correct Answer", value=True)

    # Calculate
    alpha_map = {"slow": 0.05, "medium": 0.10, "fast": 0.15}
    alpha = alpha_map[pace]
    delta = 1 if is_correct else -1
    change = alpha * contribution * delta
    new_mastery = max(0.0, min(1.0, current_mastery + change))

    # Display results
    st.divider()
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("α (Alpha)", f"{alpha:.2f}")
    with col2:
        st.metric("δ (Delta)", f"{delta:+d}")
    with col3:
        st.metric("Change", f"{change:+.3f}")
    with col4:
        st.metric("New Mastery", f"{new_mastery:.3f}")

    # Visual
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Before", "After"],
        y=[current_mastery, new_mastery],
        marker_color=["#3b82f6", "#22c55e" if is_correct else "#ef4444"],
        text=[f"{current_mastery:.2f}", f"{new_mastery:.2f}"],
        textposition="outside",
    ))
    fig.update_layout(
        title="Mastery Update",
        yaxis_range=[0, 1.1],
        yaxis_title="Mastery Score",
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Knowledge Decay
# ============================================================================
elif algo == "Knowledge Decay (Eq. 7-8)":
    st.subheader("⏰ Knowledge Decay Algorithm")

    st.latex(r"M(c) = M(c) \times e^{-\beta \times \lambda \times \tau}")

    st.markdown("""
    **Parameters:**
    - **β** = Decay rate (default: 0.1)
    - **λ** = Decay multiplier:
      - 0 if τ < 4 hours (no decay)
      - 1 if same calendar day
      - 2 if different calendar day
    - **τ** = Time elapsed (hours)
    """)

    col1, col2 = st.columns(2)

    with col1:
        initial_mastery = st.slider(
            "Initial Mastery",
            0.1, 1.0, 0.8, 0.1,
        )

        beta = st.slider(
            "Decay Rate (β)",
            0.01, 0.3, 0.1, 0.01,
        )

    with col2:
        hours = st.slider(
            "Hours Since Last Study (τ)",
            0, 168, 24,
            help="Time elapsed since last interaction",
        )

        same_day = st.toggle(
            "Same Calendar Day",
            value=False,
            help="Whether the study session is on the same day",
        )

    # Calculate
    if hours < 4:
        lambda_val = 0
    elif same_day:
        lambda_val = 1
    else:
        lambda_val = 2

    decay_factor = np.exp(-beta * lambda_val * hours / 24)
    new_mastery = initial_mastery * decay_factor

    # Display
    st.divider()
    st.subheader("Results")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("λ (Lambda)", lambda_val)
    with col2:
        st.metric("Decay Factor", f"{decay_factor:.4f}")
    with col3:
        st.metric("Score Lost", f"{initial_mastery - new_mastery:.4f}")
    with col4:
        st.metric("New Mastery", f"{new_mastery:.4f}")

    # Decay curve
    hours_range = np.linspace(0, 168, 100)
    same_day_decay = initial_mastery * np.exp(-beta * 1 * hours_range / 24)
    diff_day_decay = initial_mastery * np.exp(-beta * 2 * hours_range / 24)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hours_range, y=same_day_decay,
        mode="lines", name="Same Day (λ=1)",
        line=dict(color="blue"),
    ))
    fig.add_trace(go.Scatter(
        x=hours_range, y=diff_day_decay,
        mode="lines", name="Different Day (λ=2)",
        line=dict(color="red"),
    ))
    fig.add_trace(go.Scatter(
        x=[hours], y=[new_mastery],
        mode="markers", name="Current",
        marker=dict(size=15, color="green"),
    ))
    fig.update_layout(
        title="Knowledge Decay Over Time",
        xaxis_title="Hours Since Last Study",
        yaxis_title="Mastery Score",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Card Selection
# ============================================================================
elif algo == "Card Selection (Eq. 4-5)":
    st.subheader("🎯 Card Selection Algorithm")

    st.latex(r"P(c) = \frac{1 - M(c) + \gamma}{Z}")

    st.markdown("""
    **Parameters:**
    - **P(c)** = Selection probability for card c
    - **M(c)** = Mastery score
    - **γ (gamma)** = Prior probability for unseen cards (default: 0.1)
    - **Z** = Normalization constant

    **Gamma Boosting (Eq. 5):** γ_new = γ × δ when quiz_score > θ
    """)

    col1, col2 = st.columns(2)

    with col1:
        num_cards = st.slider("Number of Cards", 3, 10, 5)
        gamma = st.slider("Gamma (γ)", 0.0, 0.5, 0.1, 0.05)

    with col2:
        include_unseen = st.toggle("Include Unseen Cards", value=True)
        quiz_score = st.slider("Last Quiz Score", 0.0, 1.0, 0.75, 0.05)

    # Generate random card states
    np.random.seed(42)
    mastery_scores = np.random.uniform(0, 0.8, num_cards)
    if include_unseen:
        mastery_scores[0] = 0.0  # First card is unseen

    # Calculate probabilities
    probs = []
    for m in mastery_scores:
        g = gamma if m == 0 else 0
        probs.append(1 - m + g)

    total = sum(probs)
    probs = [p / total for p in probs]

    # Check gamma boost
    theta = 0.8
    if quiz_score > theta:
        boosted_gamma = gamma * 1.5
        boost_applied = True
    else:
        boosted_gamma = gamma
        boost_applied = False

    # Display
    st.divider()
    st.subheader("Selection Probabilities")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Boost Threshold (θ)", f"{theta:.0%}")
    with col2:
        if boost_applied:
            st.metric("Gamma Boosted!", f"{gamma:.2f} → {boosted_gamma:.2f}")
        else:
            st.metric("No Boost", f"γ = {gamma:.2f}")

    # Chart
    data = []
    for i, (m, p) in enumerate(zip(mastery_scores, probs)):
        data.append({
            "Card": f"Card {i+1}",
            "Mastery": m,
            "Probability": p,
            "Unseen": "Yes" if m == 0 else "No",
        })

    df = pd.DataFrame(data)

    fig = px.bar(
        df, x="Card", y="Probability",
        color="Mastery", color_continuous_scale="RdYlGn_r",
        text=df["Probability"].apply(lambda x: f"{x:.2%}"),
    )
    fig.update_layout(title="Card Selection Probabilities (lower mastery = higher probability)")
    st.plotly_chart(fig, use_container_width=True)


# ============================================================================
# Bloom's Taxonomy
# ============================================================================
elif algo == "Bloom's Taxonomy Classification":
    st.subheader("🌸 Bloom's Taxonomy Classification")

    st.markdown("""
    The **Heuristic Bloom Classifier** uses keyword matching to determine
    the cognitive level of a question.

    | Level | Name | Example Verbs |
    |-------|------|---------------|
    | 1 | Remember | define, list, recall, identify |
    | 2 | Understand | explain, describe, summarize |
    | 3 | Apply | calculate, solve, demonstrate |
    | 4 | Analyze | compare, contrast, differentiate |
    | 5 | Evaluate | judge, assess, critique |
    | 6 | Create | design, develop, propose |
    """)

    question = st.text_input(
        "Enter a question to classify:",
        value="Explain how photosynthesis works.",
    )

    if question:
        classifier = HeuristicBloomClassifier()
        result = classifier.classify_with_confidence(question)

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Level", result.level.name)
        with col2:
            st.metric("Value", result.level.value)
        with col3:
            st.metric("Confidence", f"{result.confidence:.0%}")

        # Show level info
        info = BLOOM_LEVEL_INFO.get(result.level)
        if info:
            st.info(f"**{info.name}**: {info.description}")

        # Show matched keywords
        st.subheader("Matched Keywords")
        keywords = classifier.LEVEL_KEYWORDS.get(result.level, [])
        question_lower = question.lower()
        matched = [kw for kw in keywords if kw in question_lower]

        if matched:
            st.success(f"Matched: {', '.join(matched)}")
        else:
            st.warning("No direct keyword match. Level inferred from context.")


# ============================================================================
# IRT
# ============================================================================
elif algo == "Item Response Theory (IRT)":
    st.subheader("📊 Item Response Theory (IRT)")

    st.markdown("""
    **IRT models** estimate the probability of a correct response based on
    learner ability (θ) and item parameters.
    """)

    model_type = st.radio(
        "Select Model",
        ["1PL (Rasch)", "2PL", "3PL"],
        horizontal=True,
    )

    col1, col2 = st.columns(2)

    with col1:
        difficulty = st.slider(
            "Difficulty (b)",
            -3.0, 3.0, 0.0, 0.1,
            help="Higher = harder question",
        )

    with col2:
        if model_type in ["2PL", "3PL"]:
            discrimination = st.slider(
                "Discrimination (a)",
                0.1, 3.0, 1.0, 0.1,
                help="Higher = better differentiates abilities",
            )
        else:
            discrimination = 1.0

        if model_type == "3PL":
            guessing = st.slider(
                "Guessing (c)",
                0.0, 0.5, 0.25, 0.05,
                help="Probability of guessing correctly",
            )
        else:
            guessing = 0.0

    # Create model and item
    if model_type == "1PL (Rasch)":
        model = Rasch1PL()
    elif model_type == "2PL":
        model = Model2PL()
    else:
        model = Model3PL()

    item = ItemParameters(
        item_id="demo",
        difficulty=difficulty,
        discrimination=discrimination,
        guessing=guessing,
    )

    # Calculate probabilities across ability range
    abilities = np.linspace(-3, 3, 100)
    probs = [model.probability(theta, item) for theta in abilities]

    # Plot Item Characteristic Curve (ICC)
    st.divider()
    st.subheader("Item Characteristic Curve (ICC)")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=abilities, y=probs,
        mode="lines",
        name="P(correct)",
        line=dict(color="blue", width=3),
    ))

    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray")
    fig.add_vline(x=difficulty, line_dash="dash", line_color="red",
                  annotation_text=f"b={difficulty:.1f}")

    if model_type == "3PL":
        fig.add_hline(y=guessing, line_dash="dot", line_color="green",
                      annotation_text=f"c={guessing:.2f}")

    fig.update_layout(
        xaxis_title="Learner Ability (θ)",
        yaxis_title="P(Correct)",
        yaxis_range=[0, 1],
        xaxis_range=[-3, 3],
    )
    st.plotly_chart(fig, use_container_width=True)

    # Probability table
    st.subheader("Probability by Ability Level")

    ability_levels = [-2.0, -1.0, 0.0, 1.0, 2.0]
    data = []
    for theta in ability_levels:
        p = model.probability(theta, item)
        data.append({
            "Ability (θ)": f"{theta:+.1f}",
            "P(Correct)": f"{p:.3f}",
            "Interpretation": "Very Low" if theta < -1 else ("Low" if theta < 0 else ("Average" if theta < 1 else ("High" if theta < 2 else "Very High"))),
        })

    st.dataframe(pd.DataFrame(data), hide_index=True)

    # Formula
    with st.expander("📐 Model Formula"):
        if model_type == "1PL (Rasch)":
            st.latex(r"P(X=1|\theta, b) = \frac{1}{1 + e^{-(\theta - b)}}")
        elif model_type == "2PL":
            st.latex(r"P(X=1|\theta, a, b) = \frac{1}{1 + e^{-a(\theta - b)}}")
        else:
            st.latex(r"P(X=1|\theta, a, b, c) = c + \frac{1-c}{1 + e^{-a(\theta - b)}}")
