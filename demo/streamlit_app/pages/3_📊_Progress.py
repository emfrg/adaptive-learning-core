"""Progress Dashboard - Visualize learning progress."""

import sys
from pathlib import Path

# Add demo directory to path for imports
demo_dir = Path(__file__).parent.parent.parent
if str(demo_dir) not in sys.path:
    sys.path.insert(0, str(demo_dir))

import streamlit as st
import pandas as pd

from streamlit_app.components.charts import (
    mastery_bar_chart,
    bloom_distribution_pie,
    mastery_history_chart,
    quiz_performance_chart,
    decay_preview_chart,
)


st.title("📊 Progress Dashboard")

# Check if we have data
if "card_states" not in st.session_state or not st.session_state.card_states:
    st.warning("No learning data available yet!")
    st.info("Start a learning session to track your progress.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("📄 Upload Document"):
            st.switch_page("pages/1_📄_Upload_Document.py")
    with col2:
        if st.button("🎓 Start Learning"):
            st.switch_page("pages/2_🎓_Learning_Session.py")
    st.stop()

# Overview metrics
st.subheader("📈 Overview")

col1, col2, col3, col4 = st.columns(4)

card_states = st.session_state.card_states
avg_mastery = sum(s.mastery_score for s in card_states.values()) / len(card_states)
mastered = sum(1 for s in card_states.values() if s.mastery_score >= 0.8)
total_quizzes = len(st.session_state.get("quiz_history", []))
total_correct = sum(q.get("correct", 0) for q in st.session_state.get("quiz_history", []))
total_questions = sum(q.get("num_questions", 0) for q in st.session_state.get("quiz_history", []))

with col1:
    st.metric("Total Cards", len(card_states))

with col2:
    st.metric("Average Mastery", f"{avg_mastery:.0%}")

with col3:
    st.metric("Cards Mastered", f"{mastered}/{len(card_states)}")

with col4:
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    st.metric("Quiz Accuracy", f"{overall_accuracy:.0%}")

st.divider()

# Tabs for different views
tab1, tab2, tab3, tab4 = st.tabs(["📊 Mastery", "📝 Quizzes", "📚 Bloom Levels", "⏰ Decay Preview"])

with tab1:
    st.subheader("Card Mastery Scores")

    # Mastery bar chart
    fig = mastery_bar_chart(card_states)
    st.plotly_chart(fig, use_container_width=True)

    # Detailed table
    with st.expander("📋 Detailed Card States"):
        data = []
        for card_id, state in sorted(card_states.items(), key=lambda x: -x[1].mastery_score):
            data.append({
                "Card ID": card_id,
                "Mastery": f"{state.mastery_score:.2%}",
                "Stereotype": state.stereotype.name,
                "Times Seen": state.times_seen,
                "Accuracy": f"{state.accuracy:.0%}" if state.times_seen > 0 else "N/A",
            })
        st.dataframe(pd.DataFrame(data), hide_index=True)

    # Mastery history
    if st.session_state.get("mastery_history"):
        st.subheader("Mastery Progression Over Time")
        fig = mastery_history_chart(st.session_state.mastery_history)
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Quiz Performance")

    quiz_history = st.session_state.get("quiz_history", [])

    if quiz_history:
        # Quiz performance chart
        fig = quiz_performance_chart(quiz_history)
        st.plotly_chart(fig, use_container_width=True)

        # Stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Quizzes", total_quizzes)
        with col2:
            avg_score = sum(q["score"] for q in quiz_history) / len(quiz_history)
            st.metric("Average Score", f"{avg_score:.0%}")
        with col3:
            best_score = max(q["score"] for q in quiz_history)
            st.metric("Best Score", f"{best_score:.0%}")

        # Detailed history
        with st.expander("📋 Quiz History"):
            data = []
            for q in quiz_history:
                data.append({
                    "Quiz": q["quiz_number"],
                    "Score": f"{q['score']:.0%}",
                    "Correct": f"{q['correct']}/{q['num_questions']}",
                    "Time": q.get("timestamp", "N/A"),
                })
            st.dataframe(pd.DataFrame(data), hide_index=True)
    else:
        st.info("No quiz history yet. Complete a quiz to see your performance.")

with tab3:
    st.subheader("Question Distribution by Bloom's Taxonomy")

    questions = st.session_state.get("questions", [])

    if questions:
        fig = bloom_distribution_pie(questions)
        st.plotly_chart(fig, use_container_width=True)

        # Bloom level info
        with st.expander("ℹ️ About Bloom's Taxonomy"):
            st.markdown("""
            **Bloom's Taxonomy** classifies questions by cognitive complexity:

            | Level | Name | Description |
            |-------|------|-------------|
            | 1 | **Remember** | Recall facts and basic concepts |
            | 2 | **Understand** | Explain ideas or concepts |
            | 3 | **Apply** | Use information in new situations |
            | 4 | **Analyze** | Draw connections among ideas |
            | 5 | **Evaluate** | Justify a decision or action |
            | 6 | **Create** | Produce new or original work |

            Higher levels build on lower levels. The adaptive system matches
            question difficulty to your current mastery level.
            """)
    else:
        st.info("No questions generated yet.")

with tab4:
    st.subheader("Knowledge Decay Preview")

    st.markdown("""
    Knowledge decays over time according to Ebbinghaus's Forgetting Curve:

    `M(c) = M(c) × e^{-β × λ × τ}`

    Where:
    - **β** = decay rate (default: 0.1)
    - **λ** = decay multiplier (1 for same day, 2 for different day)
    - **τ** = time elapsed in hours
    """)

    col1, col2 = st.columns(2)

    with col1:
        initial_mastery = st.slider(
            "Initial Mastery",
            min_value=0.1,
            max_value=1.0,
            value=0.8,
            step=0.1,
        )

    with col2:
        beta = st.slider(
            "Decay Rate (β)",
            min_value=0.01,
            max_value=0.3,
            value=0.1,
            step=0.01,
        )

    fig = decay_preview_chart(initial_mastery, beta)
    st.plotly_chart(fig, use_container_width=True)

    st.info("""
    **Key insight:** Regular review (same day) slows decay significantly
    compared to returning after days away. This is why spaced repetition works!
    """)

# Export options
st.divider()
st.subheader("📥 Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("📋 Export Mastery Data (CSV)"):
        data = []
        for card_id, state in card_states.items():
            data.append({
                "card_id": card_id,
                "mastery_score": state.mastery_score,
                "stereotype": state.stereotype.name,
                "times_seen": state.times_seen,
                "times_correct": state.times_correct,
            })
        df = pd.DataFrame(data)
        csv = df.to_csv(index=False)
        st.download_button(
            "⬇️ Download CSV",
            csv,
            "mastery_data.csv",
            "text/csv",
        )

with col2:
    if st.button("📋 Export Quiz History (CSV)"):
        quiz_history = st.session_state.get("quiz_history", [])
        if quiz_history:
            df = pd.DataFrame(quiz_history)
            csv = df.to_csv(index=False)
            st.download_button(
                "⬇️ Download CSV",
                csv,
                "quiz_history.csv",
                "text/csv",
            )
        else:
            st.warning("No quiz history to export.")
