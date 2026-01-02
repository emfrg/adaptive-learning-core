"""Plotly chart components for the Progress Dashboard."""

from collections import Counter
from typing import Any

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from adaptive_learning_core.models.card import CardState, LearnerStereotype
from adaptive_learning_core.models.question import Question, BloomLevel


# Color scheme for stereotypes
STEREOTYPE_COLORS = {
    LearnerStereotype.NOVICE: "#ef4444",      # Red
    LearnerStereotype.BEGINNER: "#f97316",     # Orange
    LearnerStereotype.INTERMEDIATE: "#eab308", # Yellow
    LearnerStereotype.ADVANCED: "#22c55e",     # Green
    LearnerStereotype.EXPERT: "#3b82f6",       # Blue
}

# Color scheme for Bloom levels
BLOOM_COLORS = {
    BloomLevel.REMEMBER: "#94a3b8",
    BloomLevel.UNDERSTAND: "#60a5fa",
    BloomLevel.APPLY: "#34d399",
    BloomLevel.ANALYZE: "#fbbf24",
    BloomLevel.EVALUATE: "#f97316",
    BloomLevel.CREATE: "#a855f7",
}


def mastery_bar_chart(card_states: dict[str, CardState]) -> go.Figure:
    """Create a bar chart of mastery scores per card.

    Args:
        card_states: Dictionary mapping card IDs to CardState objects.

    Returns:
        Plotly figure with mastery bar chart.
    """
    if not card_states:
        fig = go.Figure()
        fig.add_annotation(text="No data available", showarrow=False)
        return fig

    # Sort by mastery score
    sorted_states = sorted(card_states.values(), key=lambda s: s.mastery_score, reverse=True)

    data = []
    for state in sorted_states:
        data.append({
            "Card": state.card_id[:20] + "..." if len(state.card_id) > 20 else state.card_id,
            "Mastery": state.mastery_score,
            "Stereotype": state.stereotype.name,
        })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Card",
        y="Mastery",
        color="Stereotype",
        color_discrete_map={s.name: STEREOTYPE_COLORS[s] for s in LearnerStereotype},
        title="Card Mastery Scores",
    )

    # Add threshold line
    fig.add_hline(
        y=0.8,
        line_dash="dash",
        line_color="green",
        annotation_text="Mastery Threshold (80%)",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_range=[0, 1],
        yaxis_title="Mastery Score",
        showlegend=True,
    )

    return fig


def bloom_distribution_pie(questions: list[Question]) -> go.Figure:
    """Create a pie chart of question distribution by Bloom level.

    Args:
        questions: List of Question objects.

    Returns:
        Plotly figure with Bloom level distribution.
    """
    if not questions:
        fig = go.Figure()
        fig.add_annotation(text="No questions available", showarrow=False)
        return fig

    counts = Counter(q.bloom_level for q in questions)

    labels = [level.name for level in counts.keys()]
    values = list(counts.values())
    colors = [BLOOM_COLORS[level] for level in counts.keys()]

    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        marker_colors=colors,
        hole=0.3,
    )])

    fig.update_layout(
        title="Questions by Bloom's Taxonomy Level",
        showlegend=True,
    )

    return fig


def mastery_history_chart(history: list[dict[str, Any]]) -> go.Figure:
    """Create a line chart of mastery progression over time.

    Args:
        history: List of dicts with 'timestamp', 'card_id', 'mastery_score'.

    Returns:
        Plotly figure with mastery history.
    """
    if not history:
        fig = go.Figure()
        fig.add_annotation(text="No history available", showarrow=False)
        return fig

    df = pd.DataFrame(history)

    fig = px.line(
        df,
        x="timestamp",
        y="mastery_score",
        color="card_id",
        title="Mastery Progression Over Time",
        markers=True,
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        yaxis_title="Mastery Score",
        xaxis_title="Time",
    )

    return fig


def quiz_performance_chart(quiz_history: list[dict[str, Any]]) -> go.Figure:
    """Create a bar chart of quiz performance over time.

    Args:
        quiz_history: List of dicts with 'quiz_number', 'score', 'num_questions'.

    Returns:
        Plotly figure with quiz performance.
    """
    if not quiz_history:
        fig = go.Figure()
        fig.add_annotation(text="No quiz history available", showarrow=False)
        return fig

    df = pd.DataFrame(quiz_history)

    fig = px.bar(
        df,
        x="quiz_number",
        y="score",
        title="Quiz Performance",
        text=df["score"].apply(lambda x: f"{x:.0%}"),
    )

    fig.update_layout(
        yaxis_range=[0, 1],
        yaxis_title="Score",
        xaxis_title="Quiz Number",
    )

    fig.update_traces(textposition="outside")

    return fig


def decay_preview_chart(
    initial_mastery: float = 0.8,
    beta: float = 0.1,
    max_hours: int = 168,
) -> go.Figure:
    """Create a chart showing knowledge decay over time.

    Args:
        initial_mastery: Starting mastery score.
        beta: Decay rate parameter.
        max_hours: Maximum hours to show.

    Returns:
        Plotly figure with decay preview.
    """
    import numpy as np

    hours = np.linspace(0, max_hours, 100)

    # Same day decay (lambda=1)
    same_day = initial_mastery * np.exp(-beta * 1.0 * hours / 24)

    # Different day decay (lambda=2)
    diff_day = initial_mastery * np.exp(-beta * 2.0 * hours / 24)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=hours,
        y=same_day,
        mode="lines",
        name="Same Day (λ=1)",
        line=dict(color="blue"),
    ))

    fig.add_trace(go.Scatter(
        x=hours,
        y=diff_day,
        mode="lines",
        name="Different Day (λ=2)",
        line=dict(color="red"),
    ))

    fig.add_hline(y=0.8, line_dash="dash", line_color="green", annotation_text="Expert")
    fig.add_hline(y=0.6, line_dash="dash", line_color="orange", annotation_text="Advanced")
    fig.add_hline(y=0.4, line_dash="dash", line_color="yellow", annotation_text="Intermediate")

    fig.update_layout(
        title=f"Knowledge Decay Preview (β={beta}, Initial={initial_mastery})",
        xaxis_title="Hours Since Last Study",
        yaxis_title="Mastery Score",
        yaxis_range=[0, 1],
        showlegend=True,
    )

    return fig


def card_selection_probability_chart(
    card_states: dict[str, CardState],
    gamma: float = 0.1,
) -> go.Figure:
    """Create a chart showing card selection probabilities.

    Args:
        card_states: Dictionary of card states.
        gamma: Prior probability for unseen cards.

    Returns:
        Plotly figure with selection probabilities.
    """
    if not card_states:
        fig = go.Figure()
        fig.add_annotation(text="No cards available", showarrow=False)
        return fig

    # Calculate probabilities (Equation 4)
    probs = {}
    total = 0.0

    for card_id, state in card_states.items():
        g = gamma if state.is_unseen else 0.0
        p = (1 - state.mastery_score + g)
        probs[card_id] = p
        total += p

    # Normalize
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}

    # Sort by probability
    sorted_probs = sorted(probs.items(), key=lambda x: x[1], reverse=True)

    data = []
    for card_id, prob in sorted_probs:
        state = card_states[card_id]
        data.append({
            "Card": card_id[:15] + "..." if len(card_id) > 15 else card_id,
            "Probability": prob,
            "Mastery": state.mastery_score,
        })

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x="Card",
        y="Probability",
        color="Mastery",
        color_continuous_scale="RdYlGn_r",  # Red=high mastery (low prob), Green=low mastery (high prob)
        title=f"Card Selection Probabilities (γ={gamma})",
    )

    fig.update_layout(
        xaxis_tickangle=-45,
        yaxis_title="Selection Probability",
    )

    return fig
