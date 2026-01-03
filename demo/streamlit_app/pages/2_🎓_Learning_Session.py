"""Learning Session page - Interactive adaptive learning with quizzes."""

import json
import random
import re
import asyncio
from datetime import datetime, timezone
from typing import Optional

import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from adaptive_learning_core.models.card import Card, CardState, LearnerStereotype
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.models.learner import LearningGoal, PacePreference
from adaptive_learning_core.algorithms.mastery_update import MasteryUpdater
from adaptive_learning_core.algorithms.knowledge_decay import KnowledgeDecay
from adaptive_learning_core.algorithms.card_selector import CardSelector
from adaptive_learning_core.algorithms.question_selector import QuestionSelector
from adaptive_learning_core.bloom.taxonomy import BLOOM_LEVEL_INFO, get_action_verbs


# Question generation prompt template
QUESTION_GEN_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert educational content creator specializing in multiple-choice questions.

Generate {num_questions} multiple-choice questions from the given context at Bloom's Taxonomy level {bloom_level} ({bloom_name}).

Bloom Level {bloom_level} ({bloom_name}):
- Description: {bloom_description}
- Action verbs to use: {action_verbs}

Guidelines:
1. Questions must be directly answerable from the context
2. Use appropriate action verbs for the Bloom level
3. Create 3 plausible distractors (wrong answers)
4. Provide a clear explanation for the correct answer
5. Ensure distractors are clearly incorrect but tempting

IMPORTANT: Output ONLY valid JSON. Do not include quotes within strings - use single quotes or paraphrase instead. No markdown, no explanation, just the JSON array.

Output format:
[
  {{
    "stem": "The question text?",
    "correct_answer": "The correct answer",
    "distractors": ["Wrong 1", "Wrong 2", "Wrong 3"],
    "explanation": "Why the correct answer is right",
    "bloom_level": {bloom_level}
  }}
]"""),
    ("human", """Context:
{context}

Generate {num_questions} questions at Bloom level {bloom_level}. Output ONLY the JSON array, no other text.""")
])


def run_async(coro):
    """Run async coroutine in Streamlit."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def parse_json_response(response: str) -> list[dict]:
    """Parse JSON from LLM response, handling markdown blocks and common issues.

    Args:
        response: Raw LLM response string

    Returns:
        Parsed list of question dictionaries
    """
    # Remove markdown code blocks if present
    response = re.sub(r'^```json\s*', '', response.strip())
    response = re.sub(r'^```\s*', '', response)
    response = re.sub(r'\s*```$', '', response)

    # Find JSON array in response
    match = re.search(r'\[[\s\S]*\]', response)
    if not match:
        return []

    json_str = match.group()

    # Try direct parsing first
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass

    # Try to fix common issues: escape unescaped quotes within string values
    # This handles cases like: "explanation": "The term "foo" means..."
    try:
        # More aggressive fix: find string values and escape internal quotes
        def fix_string_value(s: str) -> str:
            """Escape unescaped quotes inside a JSON string."""
            # This is a simplified approach - escape quotes not at string boundaries
            if len(s) < 2:
                return s
            inner = s[1:-1]  # Remove surrounding quotes
            # Escape any unescaped quotes (not preceded by backslash)
            fixed_inner = re.sub(r'(?<!\\)"', r'\\"', inner)
            return f'"{fixed_inner}"'

        # Find all string values and try to fix them
        # Pattern to match key-value pairs
        fixed_json = json_str
        # Replace smart quotes with regular quotes
        fixed_json = fixed_json.replace('"', '"').replace('"', '"')
        fixed_json = fixed_json.replace(''', "'").replace(''', "'")

        result = json.loads(fixed_json)
        return result
    except json.JSONDecodeError:
        pass

    # Last resort: try ast.literal_eval which is more lenient
    try:
        import ast
        result = ast.literal_eval(json_str)
        if isinstance(result, list):
            return result
    except (ValueError, SyntaxError):
        pass

    return []


async def generate_questions_from_cards(
    llm,
    cards: list[Card],
    num_questions_per_card: int = 2,
    target_bloom_level: int = 2,
) -> list[Question]:
    """Generate questions from cards using the configured LLM.

    Args:
        llm: LangChain chat model
        cards: List of cards to generate questions from
        num_questions_per_card: Questions to generate per card
        target_bloom_level: Bloom's Taxonomy level (1-6)

    Returns:
        List of generated Question objects
    """
    chain = QUESTION_GEN_PROMPT | llm | StrOutputParser()

    bloom_info = BLOOM_LEVEL_INFO.get(BloomLevel(target_bloom_level))
    if not bloom_info:
        bloom_info = BLOOM_LEVEL_INFO[BloomLevel.UNDERSTAND]

    action_verbs = ", ".join(get_action_verbs(BloomLevel(target_bloom_level))[:5])

    all_questions = []

    for card in cards:
        try:
            response = await chain.ainvoke({
                "context": card.text,
                "num_questions": num_questions_per_card,
                "bloom_level": target_bloom_level,
                "bloom_name": bloom_info.name,
                "bloom_description": bloom_info.description,
                "action_verbs": action_verbs,
            })

            # Parse JSON from response
            result = parse_json_response(response)

            # Convert to Question objects
            for i, q_data in enumerate(result):
                question = Question(
                    id=f"{card.id}_q_{i}",
                    module_id=card.module_id,
                    stem=q_data["stem"],
                    correct_answer=q_data["correct_answer"],
                    distractors=q_data["distractors"],
                    explanation=q_data["explanation"],
                    bloom_level=BloomLevel(q_data.get("bloom_level", target_bloom_level)),
                    linked_card_ids=[card.id],
                    card_contribution_weights={card.id: 1.0},
                )
                all_questions.append(question)

        except Exception as e:
            st.warning(f"Failed to generate questions for card {card.id}: {e}")
            continue

    return all_questions


st.title("🎓 Learning Session")

# Check if we have cards loaded
if "cards" not in st.session_state or not st.session_state.cards:
    st.warning("No learning material loaded!")
    st.info("Go to **Upload Document** to load a document or use sample data.")
    if st.button("📄 Go to Upload"):
        st.switch_page("pages/1_📄_Upload_Document.py")
    st.stop()

# Initialize session state for learning
if "pace" not in st.session_state:
    st.session_state.pace = "medium"
if "target_level" not in st.session_state:
    st.session_state.target_level = 4  # Expert
if "current_cycle" not in st.session_state:
    st.session_state.current_cycle = 0
if "session_phase" not in st.session_state:
    st.session_state.session_phase = "calibration"
if "selected_card_ids" not in st.session_state:
    st.session_state.selected_card_ids = []
if "quiz_questions" not in st.session_state:
    st.session_state.quiz_questions = []
if "quiz_answers" not in st.session_state:
    st.session_state.quiz_answers = {}
if "quiz_history" not in st.session_state:
    st.session_state.quiz_history = []
if "mastery_history" not in st.session_state:
    st.session_state.mastery_history = []


def get_learning_goal_info(goal: LearningGoal) -> tuple[int, str]:
    """Map learning goal to target level and description."""
    goal_mapping = {
        LearningGoal.MEMORIZE: (0, "Novice - Remember key facts"),
        LearningGoal.COMPREHEND: (1, "Beginner - Understand concepts"),
        LearningGoal.USE: (2, "Intermediate - Apply knowledge"),
        LearningGoal.EXAMINE: (3, "Advanced - Analyze relationships"),
        LearningGoal.ASSESS: (4, "Expert - Evaluate and critique"),
        LearningGoal.DEVELOP: (4, "Expert - Create new ideas"),
    }
    return goal_mapping.get(goal, (2, "Intermediate"))


# ============================================================================
# PHASE: Calibration
# ============================================================================
if st.session_state.session_phase == "calibration":
    st.subheader("📋 Learning Calibration")
    st.markdown("""
    Before we begin, let's set up your learning preferences. This helps the
    system adapt to your pace and goals.
    """)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**🎯 Learning Goal**")
        goal = st.radio(
            "What do you want to achieve?",
            options=[
                ("I want to memorize key facts", LearningGoal.MEMORIZE),
                ("I want to understand concepts", LearningGoal.COMPREHEND),
                ("I want to apply knowledge", LearningGoal.USE),
                ("I want to analyze relationships", LearningGoal.EXAMINE),
                ("I want to evaluate ideas", LearningGoal.ASSESS),
                ("I want to create new things", LearningGoal.DEVELOP),
            ],
            format_func=lambda x: x[0],
            index=2,  # Default to "use"
            label_visibility="collapsed",
        )

    with col2:
        st.markdown("**⚡ Learning Pace**")
        pace = st.radio(
            "How fast do you want to progress?",
            options=[
                ("Slow - Careful study (α=0.05)", "slow"),
                ("Medium - Balanced (α=0.10)", "medium"),
                ("Fast - Rapid progress (α=0.15)", "fast"),
            ],
            format_func=lambda x: x[0],
            index=1,  # Default to medium
            label_visibility="collapsed",
        )

    st.divider()

    # Cards per cycle
    cards_per_cycle = st.slider(
        "Cards per learning cycle",
        min_value=1,
        max_value=min(10, len(st.session_state.cards)),
        value=min(3, len(st.session_state.cards)),
        help="Number of cards to study in each cycle",
    )

    if st.button("🚀 Start Learning", type="primary"):
        # Save calibration
        st.session_state.learning_goal = goal[1]
        st.session_state.pace = pace[1]
        st.session_state.target_level = get_learning_goal_info(goal[1])[0]
        st.session_state.cards_per_cycle = cards_per_cycle
        st.session_state.session_phase = "learning"
        st.session_state.current_cycle = 1
        st.rerun()


# ============================================================================
# PHASE: Learning Cycle
# ============================================================================
elif st.session_state.session_phase == "learning":

    # Show session info
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Cycle", st.session_state.current_cycle)
    with col2:
        avg_mastery = sum(s.mastery_score for s in st.session_state.card_states.values()) / len(st.session_state.card_states)
        st.metric("Avg. Mastery", f"{avg_mastery:.0%}")
    with col3:
        mastered = sum(1 for s in st.session_state.card_states.values() if s.mastery_score >= 0.8)
        st.metric("Cards Mastered", f"{mastered}/{len(st.session_state.card_states)}")
    with col4:
        st.metric("Pace", st.session_state.pace.title())

    st.divider()

    # Step 1: Apply decay (if returning)
    if st.session_state.current_cycle > 1:
        decay = KnowledgeDecay()
        now = datetime.now(timezone.utc)
        for state in st.session_state.card_states.values():
            if state.last_seen_at:
                decay.apply_decay(state, now, in_place=True)

    # Step 2: Select cards
    if not st.session_state.selected_card_ids:
        selector = CardSelector()
        result = selector.select_cards(
            list(st.session_state.card_states.values()),
            n=st.session_state.cards_per_cycle,
        )
        st.session_state.selected_card_ids = result.selected_card_ids

    # Step 3: Show cards for review
    st.subheader("📚 Study These Cards")

    selected_cards = [c for c in st.session_state.cards if c.id in st.session_state.selected_card_ids]

    for i, card in enumerate(selected_cards):
        state = st.session_state.card_states.get(card.id)
        mastery = state.mastery_score if state else 0.0

        with st.expander(f"Card {i+1} (Mastery: {mastery:.0%})", expanded=True):
            st.write(card.text)

    st.divider()

    # Step 4: Quiz
    st.subheader("📝 Quiz Time!")

    # Check if we have questions
    if "questions" not in st.session_state or not st.session_state.questions:
        st.warning("No questions available!")

        if st.session_state.get("llm_configured"):
            st.info("Generate questions from your learning material using AI.")

            col1, col2 = st.columns(2)
            with col1:
                questions_per_card = st.slider(
                    "Questions per card",
                    min_value=1,
                    max_value=5,
                    value=2,
                    help="Number of questions to generate for each card",
                )
            with col2:
                bloom_target = st.selectbox(
                    "Target Bloom Level",
                    options=[1, 2, 3, 4, 5, 6],
                    index=1,  # Default to UNDERSTAND
                    format_func=lambda x: f"{x} - {BloomLevel(x).name}",
                    help="Cognitive complexity level for questions",
                )

            if st.button("🔄 Generate Questions with AI", type="primary"):
                with st.spinner("Generating questions... This may take a minute."):
                    try:
                        questions = run_async(
                            generate_questions_from_cards(
                                llm=st.session_state.llm,
                                cards=st.session_state.cards,
                                num_questions_per_card=questions_per_card,
                                target_bloom_level=bloom_target,
                            )
                        )
                        if questions:
                            st.session_state.questions = questions
                            st.success(f"Generated {len(questions)} questions!")
                            st.rerun()
                        else:
                            st.error("No questions were generated. Try again.")
                    except Exception as e:
                        st.error(f"Error generating questions: {e}")

            st.divider()
            st.caption("Or use simple demo questions (no AI):")
        else:
            st.info("""
            Questions need to be generated using an LLM. Either:
            1. Configure your API key in .env file and restart
            2. Use the Sample Data which includes pre-made questions
            """)

        # Fallback: create simple questions from the cards
        if st.button("📝 Use Simple Questions (Demo Mode)"):
            questions = []
            for card in selected_cards:
                # Create a simple REMEMBER-level question
                q = Question(
                    id=f"q_{card.id}",
                    module_id=card.module_id,
                    stem=f"What is described in this excerpt?\n\n\"{card.text[:150]}...\"",
                    correct_answer="This excerpt describes the content accurately",
                    distractors=[
                        "This excerpt is about something else",
                        "This excerpt contains errors",
                        "This excerpt is incomplete",
                    ],
                    explanation="The excerpt accurately describes the content from the learning material.",
                    bloom_level=BloomLevel.REMEMBER,
                    linked_card_ids=[card.id],
                    card_contribution_weights={card.id: 1.0},
                )
                questions.append(q)
            st.session_state.questions = questions
            st.rerun()

        st.stop()

    # Select questions for this quiz
    if not st.session_state.quiz_questions:
        # Get questions linked to selected cards
        available_questions = [
            q for q in st.session_state.questions
            if any(cid in st.session_state.selected_card_ids for cid in q.linked_card_ids)
        ]

        if available_questions:
            # Take up to 3 questions
            st.session_state.quiz_questions = random.sample(
                available_questions,
                min(3, len(available_questions)),
            )
        else:
            st.warning("No questions available for the selected cards.")
            st.stop()

    # Display quiz questions
    quiz_key = f"quiz_{st.session_state.current_cycle}"

    with st.form(key=quiz_key):
        answers = {}

        for i, q in enumerate(st.session_state.quiz_questions):
            st.markdown(f"**Question {i+1}** ({q.bloom_level.name})")
            st.write(q.stem)

            # Shuffle options
            options = q.all_options.copy()
            random.shuffle(options)

            selected = st.radio(
                "Select your answer:",
                options,
                key=f"q_{i}_{quiz_key}",
                label_visibility="collapsed",
            )
            answers[q.id] = selected

            st.divider()

        submitted = st.form_submit_button("✅ Submit Answers", type="primary")

        if submitted:
            st.session_state.quiz_answers = answers
            st.session_state.session_phase = "results"
            st.rerun()


# ============================================================================
# PHASE: Results
# ============================================================================
elif st.session_state.session_phase == "results":
    st.subheader("📊 Quiz Results")

    # Calculate score
    correct = 0
    total = len(st.session_state.quiz_questions)
    updater = MasteryUpdater()

    results = []
    for q in st.session_state.quiz_questions:
        selected = st.session_state.quiz_answers.get(q.id)
        is_correct = selected == q.correct_answer
        if is_correct:
            correct += 1

        # Show result
        if is_correct:
            st.success(f"✅ **Correct!** {q.stem[:50]}...")
        else:
            st.error(f"❌ **Incorrect.** {q.stem[:50]}...")
            st.write(f"Your answer: {selected}")
            st.write(f"Correct answer: {q.correct_answer}")

        with st.expander("Explanation"):
            st.write(q.explanation)

        # Update mastery for linked cards
        for card_id in q.linked_card_ids:
            if card_id in st.session_state.card_states:
                old_state = st.session_state.card_states[card_id]
                old_score = old_state.mastery_score

                result = updater.update_single(
                    old_state,
                    q,
                    is_correct=is_correct,
                    pace=st.session_state.pace,
                )

                # Track history
                st.session_state.mastery_history.append({
                    "timestamp": datetime.now(),
                    "card_id": card_id,
                    "mastery_score": result.new_score,
                    "old_score": old_score,
                    "question_id": q.id,
                    "is_correct": is_correct,
                })

        st.divider()

    # Show score
    score = correct / total if total > 0 else 0
    st.session_state.quiz_history.append({
        "quiz_number": st.session_state.current_cycle,
        "score": score,
        "num_questions": total,
        "correct": correct,
        "timestamp": datetime.now(),
    })

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Score", f"{correct}/{total}")
    with col2:
        st.metric("Percentage", f"{score:.0%}")
    with col3:
        # Check if gamma boosting applies
        if score >= 0.8:
            st.metric("Bonus", "🔥 Gamma Boost!")
        else:
            st.metric("Bonus", "None")

    st.divider()

    # Show updated mastery
    st.subheader("📈 Updated Mastery")

    for card_id in st.session_state.selected_card_ids:
        state = st.session_state.card_states.get(card_id)
        if state:
            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(state.mastery_score, text=f"{card_id}: {state.mastery_score:.0%}")
            with col2:
                st.write(state.stereotype.name)

    st.divider()

    # Check if ready for module exam
    avg_mastery = sum(s.mastery_score for s in st.session_state.card_states.values()) / len(st.session_state.card_states)
    mastered = sum(1 for s in st.session_state.card_states.values() if s.mastery_score >= 0.8)

    if mastered == len(st.session_state.card_states):
        st.success("🎉 **Congratulations!** You've mastered all cards!")
        st.balloons()
    elif avg_mastery >= 0.8:
        st.info("📝 You're ready for the module exam!")

    # Continue options
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Continue Learning", type="primary"):
            st.session_state.session_phase = "learning"
            st.session_state.current_cycle += 1
            st.session_state.selected_card_ids = []
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.rerun()

    with col2:
        if st.button("📊 View Progress"):
            st.switch_page("pages/3_📊_Progress.py")

    with col3:
        if st.button("🏠 End Session"):
            st.session_state.session_phase = "calibration"
            st.session_state.current_cycle = 0
            st.session_state.selected_card_ids = []
            st.session_state.quiz_questions = []
            st.session_state.quiz_answers = {}
            st.rerun()
