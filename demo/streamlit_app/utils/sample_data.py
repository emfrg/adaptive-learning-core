"""Sample data for the Streamlit demo.

Provides a fallback Biology 101 module with pre-created cards and questions,
similar to the CLI demo.
"""

import random
from datetime import datetime, timezone
from typing import Optional

from adaptive_learning_core.models.card import Card, CardState
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.models.module import Module


def get_sample_module() -> tuple[Module, list[Card], list[Question]]:
    """Create sample module with cards and questions.

    Returns:
        Tuple of (Module, list of Cards, list of Questions).
    """
    module = Module(
        id="mod_biology_101",
        name="Biology 101: Introduction to Cells",
        description="Fundamental concepts about cell biology",
        card_ids=[f"card_{i}" for i in range(1, 11)],
        question_ids=[f"q_{i}" for i in range(1, 8)],
    )

    cards = [
        Card(
            id="card_1",
            module_id=module.id,
            text="The cell is the basic structural and functional unit of all living organisms. "
            "Cells are the smallest units of life that can replicate independently.",
        ),
        Card(
            id="card_2",
            module_id=module.id,
            text="The cell membrane (plasma membrane) is a lipid bilayer that surrounds the cell, "
            "controlling what enters and exits. It is selectively permeable.",
        ),
        Card(
            id="card_3",
            module_id=module.id,
            text="The nucleus is the control center of the cell, containing genetic material (DNA). "
            "It is surrounded by a double membrane called the nuclear envelope.",
        ),
        Card(
            id="card_4",
            module_id=module.id,
            text="Mitochondria are the powerhouses of the cell, producing ATP through cellular respiration. "
            "They have their own DNA and double membrane.",
        ),
        Card(
            id="card_5",
            module_id=module.id,
            text="Ribosomes are protein factories that translate mRNA into proteins. They can be free "
            "in the cytoplasm or attached to the endoplasmic reticulum.",
        ),
        Card(
            id="card_6",
            module_id=module.id,
            text="The endoplasmic reticulum (ER) is a network of membranes. Rough ER has ribosomes "
            "and synthesizes proteins; smooth ER synthesizes lipids.",
        ),
        Card(
            id="card_7",
            module_id=module.id,
            text="The Golgi apparatus modifies, packages, and ships proteins and lipids. "
            "It consists of flattened membrane sacs called cisternae.",
        ),
        Card(
            id="card_8",
            module_id=module.id,
            text="Lysosomes contain digestive enzymes that break down waste materials and cellular debris. "
            "They are involved in autophagy and apoptosis.",
        ),
        Card(
            id="card_9",
            module_id=module.id,
            text="The cytoskeleton provides structural support and enables cell movement. "
            "It consists of microfilaments, intermediate filaments, and microtubules.",
        ),
        Card(
            id="card_10",
            module_id=module.id,
            text="Plant cells have additional structures: cell wall (cellulose), chloroplasts "
            "(photosynthesis), and a large central vacuole (storage and turgor pressure).",
        ),
    ]

    questions = [
        Question(
            id="q_1",
            module_id=module.id,
            stem="What is the basic structural and functional unit of all living organisms?",
            correct_answer="Cell",
            distractors=["Atom", "Molecule", "Tissue"],
            explanation="The cell is defined as the basic structural and functional unit of life.",
            bloom_level=BloomLevel.REMEMBER,
            linked_card_ids=["card_1"],
            card_contribution_weights={"card_1": 1.0},
        ),
        Question(
            id="q_2",
            module_id=module.id,
            stem="What is the primary function of the cell membrane?",
            correct_answer="Control what enters and exits the cell",
            distractors=["Store genetic material", "Produce energy", "Synthesize proteins"],
            explanation="The cell membrane is selectively permeable, controlling molecular traffic.",
            bloom_level=BloomLevel.UNDERSTAND,
            linked_card_ids=["card_2"],
            card_contribution_weights={"card_2": 1.0},
        ),
        Question(
            id="q_3",
            module_id=module.id,
            stem="Which organelle is known as the control center of the cell?",
            correct_answer="Nucleus",
            distractors=["Mitochondria", "Ribosome", "Golgi apparatus"],
            explanation="The nucleus contains DNA and controls cell activities.",
            bloom_level=BloomLevel.REMEMBER,
            linked_card_ids=["card_3"],
            card_contribution_weights={"card_3": 1.0},
        ),
        Question(
            id="q_4",
            module_id=module.id,
            stem="If a cell needs more energy, which organelle would increase in number?",
            correct_answer="Mitochondria",
            distractors=["Lysosomes", "Golgi apparatus", "Vacuoles"],
            explanation="Mitochondria produce ATP; more energy demand leads to more mitochondria.",
            bloom_level=BloomLevel.APPLY,
            linked_card_ids=["card_4"],
            card_contribution_weights={"card_4": 1.0},
        ),
        Question(
            id="q_5",
            module_id=module.id,
            stem="Compare the functions of rough ER and smooth ER.",
            correct_answer="Rough ER synthesizes proteins (has ribosomes), smooth ER synthesizes lipids",
            distractors=[
                "Both synthesize proteins",
                "Rough ER stores water, smooth ER stores lipids",
                "They have identical functions",
            ],
            explanation="Rough ER has ribosomes for protein synthesis; smooth ER lacks ribosomes and makes lipids.",
            bloom_level=BloomLevel.ANALYZE,
            linked_card_ids=["card_5", "card_6"],
            card_contribution_weights={"card_5": 0.5, "card_6": 0.5},
        ),
        Question(
            id="q_6",
            module_id=module.id,
            stem="Evaluate the importance of lysosomes in cellular health.",
            correct_answer="Critical for waste removal and recycling damaged organelles",
            distractors=[
                "Not important, cells can survive without them",
                "Only important in plant cells",
                "Only used during cell division",
            ],
            explanation="Lysosomes perform autophagy and remove waste, essential for cell health.",
            bloom_level=BloomLevel.EVALUATE,
            linked_card_ids=["card_8"],
            card_contribution_weights={"card_8": 1.0},
        ),
        Question(
            id="q_7",
            module_id=module.id,
            stem="Design an experiment to demonstrate the function of chloroplasts.",
            correct_answer="Compare plant growth/oxygen production in light vs. dark conditions",
            distractors=[
                "Observe cells under a microscope",
                "Measure cell membrane thickness",
                "Count the number of ribosomes",
            ],
            explanation="Chloroplasts perform photosynthesis, which requires light.",
            bloom_level=BloomLevel.CREATE,
            linked_card_ids=["card_10"],
            card_contribution_weights={"card_10": 1.0},
        ),
    ]

    return module, cards, questions


def create_card_states(
    cards: list[Card],
    learner_id: str,
    random_mastery: bool = False,
) -> dict[str, CardState]:
    """Create card states for a learner.

    Args:
        cards: List of cards.
        learner_id: Learner identifier.
        random_mastery: If True, assign random initial mastery scores.

    Returns:
        Dictionary mapping card IDs to CardState objects.
    """
    states = {}
    for card in cards:
        mastery = random.uniform(0, 0.6) if random_mastery else 0.0
        times_seen = random.randint(0, 5) if random_mastery else 0

        states[card.id] = CardState(
            card_id=card.id,
            learner_id=learner_id,
            mastery_score=mastery,
            times_seen=times_seen,
            times_correct=int(times_seen * random.uniform(0.5, 0.9)) if times_seen > 0 else 0,
        )

    return states


def initialize_session_with_sample_data(learner_id: str = "demo_user") -> None:
    """Initialize Streamlit session state with sample Biology 101 data.

    Args:
        learner_id: Learner identifier.
    """
    import streamlit as st

    module, cards, questions = get_sample_module()

    st.session_state.module = module
    st.session_state.cards = cards
    st.session_state.questions = questions
    st.session_state.card_states = create_card_states(cards, learner_id, random_mastery=False)
    st.session_state.learner_id = learner_id
    st.session_state.current_cycle = 0
    st.session_state.quiz_history = []
    st.session_state.mastery_history = []
