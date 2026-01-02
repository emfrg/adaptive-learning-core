#!/usr/bin/env python3
"""Interactive CLI demo for the Adaptive Learning Core library.

This demo showcases the core algorithms:
- Mastery Update (Equation 3)
- Knowledge Decay (Equations 7-8)
- Card Selection (Equations 4-5)
- Question Selection (Rules 2-3)
- Item Response Theory (1PL, 2PL, 3PL)
- Bloom's Taxonomy classification

Run with: uv run python demo/cli_demo.py
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Confirm, IntPrompt, Prompt
from rich.table import Table
from rich.text import Text

from adaptive_learning_core import (
    # Models
    Card,
    CardState,
    Question,
    BloomLevel,
    Module,
    ModuleProgress,
    ModuleStatus,
    LearnerProfile,
    PacePreference,
    LearningGoal,
    # Algorithms
    MasteryUpdater,
    KnowledgeDecay,
    CardSelector,
    QuestionLinker,
    QuestionSelector,
    # IRT
    Rasch1PL,
    Model2PL,
    Model3PL,
    ItemParameters,
    DiscriminationAnalyzer,
    # Bloom
    BloomLevelInfo,
    BLOOM_LEVEL_INFO,
    HeuristicBloomClassifier,
)

app = typer.Typer(help="Adaptive Learning Core Demo")
console = Console()


# =============================================================================
# Sample Data Generation
# =============================================================================

def create_sample_module() -> tuple[Module, list[Card], list[Question]]:
    """Create sample module with cards and questions."""
    module = Module(
        id="mod_biology_101",
        name="Biology 101: Introduction to Cells",
        description="Fundamental concepts about cell biology",
        card_ids=[f"card_{i}" for i in range(1, 11)],
        question_ids=[f"q_{i}" for i in range(1, 16)],
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
    cards: list[Card], learner_id: str, random_mastery: bool = False
) -> dict[str, CardState]:
    """Create card states for a learner."""
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


# =============================================================================
# Demo Commands
# =============================================================================


@app.command()
def mastery_update():
    """Demo: Mastery Update Algorithm (Equation 3)."""
    console.print(Panel.fit(
        "[bold blue]Mastery Update Algorithm Demo[/bold blue]\n\n"
        "Equation 3: M(c) = M(c) + [bold]alpha[/bold] x [bold]C_{c,q}[/bold] x [bold]delta[/bold]\n\n"
        "Where:\n"
        "  - M(c) = mastery score for card c\n"
        "  - alpha = progression rate (from pace preference)\n"
        "  - C_{c,q} = contribution weight of card to question\n"
        "  - delta = +1 (correct) or -1 (incorrect)"
    ))

    module, cards, questions = create_sample_module()
    card_states = create_card_states(cards[:3], "learner_1")
    updater = MasteryUpdater()

    # Show initial state
    table = Table(title="Initial Card States")
    table.add_column("Card ID", style="cyan")
    table.add_column("Mastery", justify="right")
    table.add_column("Stereotype", style="green")

    for card_id, state in card_states.items():
        table.add_row(card_id, f"{state.mastery_score:.3f}", state.stereotype.name)
    console.print(table)

    # Simulate answering a question
    question = questions[4]  # Question linked to card_5 and card_6 (but we use card_1, card_2, card_3)
    question_simple = questions[0]  # Question linked to card_1

    console.print(f"\n[yellow]Question:[/yellow] {question_simple.stem}")
    console.print(f"[dim]Linked to: {question_simple.linked_card_ids}[/dim]")

    for pace in ["slow", "medium", "fast"]:
        # Reset state for demo
        card_states["card_1"].mastery_score = 0.3
        result = updater.update_single(
            card_states["card_1"],
            question_simple,
            is_correct=True,
            pace=pace,  # type: ignore
        )
        console.print(
            f"\n[green]Correct answer with {pace} pace:[/green]"
            f"\n  alpha={result.alpha_used:.3f}, contribution={result.contribution_weight:.3f}"
            f"\n  Score: {result.old_score:.3f} -> {result.new_score:.3f} "
            f"([green]+{result.score_change:.3f}[/green])"
        )

    # Incorrect answer
    card_states["card_1"].mastery_score = 0.5
    result = updater.update_single(
        card_states["card_1"],
        question_simple,
        is_correct=False,
        pace="medium",
    )
    console.print(
        f"\n[red]Incorrect answer (medium pace):[/red]"
        f"\n  Score: {result.old_score:.3f} -> {result.new_score:.3f} "
        f"([red]{result.effective_change:.3f}[/red])"
    )


@app.command()
def knowledge_decay():
    """Demo: Knowledge Decay Algorithm (Equations 7-8)."""
    console.print(Panel.fit(
        "[bold blue]Knowledge Decay Algorithm Demo[/bold blue]\n\n"
        "Equation 7: M(c) = M(c) x [bold]e^{-beta x lambda x tau}[/bold]\n\n"
        "Equation 8 (Decay Multiplier lambda):\n"
        "  lambda = 0 if tau < 4 hours (no decay)\n"
        "  lambda = 1 if tau >= 4 hours, same day\n"
        "  lambda = 2 if tau >= 4 hours, different day"
    ))

    decay = KnowledgeDecay()
    now = datetime.now(timezone.utc)

    # Create card state with known mastery
    card_state = CardState(
        card_id="card_1",
        learner_id="learner_1",
        mastery_score=0.8,
        last_seen_at=now - timedelta(hours=2),
    )

    table = Table(title="Decay Over Time (Starting Mastery: 0.80)")
    table.add_column("Time Since Last Seen", style="cyan")
    table.add_column("Lambda", justify="center")
    table.add_column("Decay Factor", justify="right")
    table.add_column("New Mastery", justify="right", style="yellow")

    scenarios = [
        ("2 hours", timedelta(hours=2), True),
        ("4 hours (same day)", timedelta(hours=4), True),
        ("8 hours (same day)", timedelta(hours=8), True),
        ("1 day", timedelta(days=1), False),
        ("3 days", timedelta(days=3), False),
        ("7 days", timedelta(days=7), False),
    ]

    for label, delta, same_day in scenarios:
        card_state.mastery_score = 0.8
        if same_day:
            card_state.last_seen_at = now - delta
        else:
            card_state.last_seen_at = now - delta

        result = decay.apply_decay(card_state, now, in_place=False)
        table.add_row(
            label,
            f"{result.decay_multiplier:.1f}",
            f"{result.decay_factor:.4f}",
            f"{result.new_score:.4f}",
        )

    console.print(table)

    # Preview decay
    console.print("\n[bold]Decay Preview Function:[/bold]")
    console.print(f"  Current score: 0.80")
    for hours in [8, 24, 48, 168]:
        future_score = decay.preview_decay(0.8, hours, same_day=False)
        console.print(f"  After {hours:3d} hours: {future_score:.4f}")


@app.command()
def card_selection():
    """Demo: Card Selection Algorithm (Equations 4-5)."""
    console.print(Panel.fit(
        "[bold blue]Card Selection Algorithm Demo[/bold blue]\n\n"
        "Equation 4: P(c) = (1 - M(c) + [bold]gamma[/bold]) / Z\n"
        "Equation 5: gamma_new = gamma x [bold]delta[/bold] (when score > theta)\n\n"
        "Cards with lower mastery have higher selection probability.\n"
        "Gamma ensures unseen cards aren't neglected."
    ))

    module, cards, questions = create_sample_module()
    card_states = create_card_states(cards, "learner_1", random_mastery=True)
    selector = CardSelector()

    # Show card states sorted by mastery
    table = Table(title="Card States (sorted by mastery)")
    table.add_column("Card ID", style="cyan")
    table.add_column("Mastery", justify="right")
    table.add_column("Selection Prob", justify="right", style="green")

    sorted_states = sorted(card_states.values(), key=lambda s: s.mastery_score)
    probabilities = selector.calculate_probabilities(sorted_states)

    for state in sorted_states:
        prob = probabilities.get(state.card_id, 0)
        table.add_row(
            state.card_id,
            f"{state.mastery_score:.3f}",
            f"{prob:.4f}",
        )
    console.print(table)

    # Demonstrate selection
    console.print("\n[bold]Selecting 3 cards for next learning cycle:[/bold]")
    result = selector.select_cards(list(card_states.values()), n=3, random_state=42)
    for i, card_id in enumerate(result.selected_card_ids, 1):
        state = card_states[card_id]
        console.print(
            f"  {i}. {card_id} (mastery: {state.mastery_score:.3f}, "
            f"prob: {result.selection_probabilities[card_id]:.4f})"
        )

    # Gamma boosting
    console.print("\n[bold]Gamma Boosting (Equation 5):[/bold]")
    console.print(f"  Initial gamma: {result.gamma_used:.3f}")

    boosted_gamma = selector.maybe_boost_gamma(result.gamma_used, quiz_score=0.9)
    console.print(f"  After 90% quiz score: gamma = {boosted_gamma:.3f} (boosted!)")

    not_boosted = selector.maybe_boost_gamma(result.gamma_used, quiz_score=0.7)
    console.print(f"  After 70% quiz score: gamma = {not_boosted:.3f} (no boost)")


@app.command()
def irt_models():
    """Demo: Item Response Theory Models (1PL, 2PL, 3PL)."""
    console.print(Panel.fit(
        "[bold blue]Item Response Theory (IRT) Demo[/bold blue]\n\n"
        "1PL (Rasch): P(X=1) = 1 / (1 + e^{-(theta-b)})\n"
        "2PL: P(X=1) = 1 / (1 + e^{-a(theta-b)})\n"
        "3PL: P(X=1) = c + (1-c) / (1 + e^{-a(theta-b)})\n\n"
        "Where:\n"
        "  - theta = learner ability\n"
        "  - b = item difficulty\n"
        "  - a = discrimination\n"
        "  - c = guessing parameter"
    ))

    # Create models
    rasch = Rasch1PL()
    model_2pl = Model2PL()
    model_3pl = Model3PL()

    # Create item parameters
    easy_item = ItemParameters(difficulty=-1.0, discrimination=1.0, guessing=0.0)
    medium_item = ItemParameters(difficulty=0.0, discrimination=1.5, guessing=0.0)
    hard_item = ItemParameters(difficulty=1.5, discrimination=2.0, guessing=0.25)

    # Calculate probabilities for different ability levels
    table = Table(title="Probability of Correct Response by Ability Level")
    table.add_column("Ability (theta)", style="cyan", justify="center")
    table.add_column("Easy (b=-1)", justify="right")
    table.add_column("Medium (b=0)", justify="right")
    table.add_column("Hard (b=1.5, c=0.25)", justify="right")

    abilities = [-2.0, -1.0, 0.0, 1.0, 2.0]
    for theta in abilities:
        p_easy = rasch.probability(theta, easy_item)
        p_medium = model_2pl.probability(theta, medium_item)
        p_hard = model_3pl.probability(theta, hard_item)
        table.add_row(
            f"{theta:+.1f}",
            f"{p_easy:.3f}",
            f"{p_medium:.3f}",
            f"{p_hard:.3f}",
        )
    console.print(table)

    # Discrimination analysis
    console.print("\n[bold]Discrimination Analysis:[/bold]")
    analyzer = DiscriminationAnalyzer()

    discrimination_values = [0.3, 0.7, 1.2, 1.8, 2.5]
    for disc in discrimination_values:
        category = analyzer._categorize_discrimination(disc)
        console.print(f"  a={disc:.1f}: {category.value}")


@app.command()
def bloom_taxonomy():
    """Demo: Bloom's Taxonomy Classification."""
    console.print(Panel.fit(
        "[bold blue]Bloom's Taxonomy Demo[/bold blue]\n\n"
        "6 Cognitive Levels (lowest to highest):\n"
        "  1. Remember - Recall facts\n"
        "  2. Understand - Explain ideas\n"
        "  3. Apply - Use information\n"
        "  4. Analyze - Draw connections\n"
        "  5. Evaluate - Justify decisions\n"
        "  6. Create - Produce new work"
    ))

    table = Table(title="Bloom's Taxonomy Levels")
    table.add_column("Level", style="cyan", justify="center")
    table.add_column("Name", style="bold")
    table.add_column("Stereotype", style="green")
    table.add_column("Example Verbs", style="dim")

    for level in BloomLevel:
        info = BLOOM_LEVEL_INFO.get(level, BloomLevelInfo(
            level=level, name=level.name, description="", action_verbs=[],
            question_stems=[], example_tasks=[]
        ))
        verbs = ", ".join(level.action_verbs[:4])
        stereotype_level = level.to_stereotype_level()
        stereotype_names = ["Novice", "Beginner", "Intermediate", "Advanced", "Expert"]
        table.add_row(
            str(level.value),
            level.name,
            stereotype_names[stereotype_level],
            verbs,
        )
    console.print(table)

    # Heuristic classification demo
    console.print("\n[bold]Heuristic Classification (keyword-based):[/bold]")
    classifier = HeuristicBloomClassifier()

    sample_questions = [
        "Define photosynthesis.",
        "Explain how photosynthesis works.",
        "Calculate the rate of photosynthesis given...",
        "Compare aerobic and anaerobic respiration.",
        "Evaluate the efficiency of C4 photosynthesis.",
        "Design an experiment to measure photosynthesis.",
    ]

    for q in sample_questions:
        result = classifier.classify(q)
        console.print(f"  [dim]{q}[/dim]")
        console.print(f"    -> {result.level.name} (confidence: {result.confidence:.2f})")


@app.command()
def full_demo():
    """Run a complete interactive learning session demo."""
    console.print(Panel.fit(
        "[bold green]Complete Adaptive Learning Demo[/bold green]\n\n"
        "This demo simulates a learning session:\n"
        "1. Set up learner profile\n"
        "2. Apply knowledge decay\n"
        "3. Select cards for learning cycle\n"
        "4. Quiz on selected cards\n"
        "5. Update mastery scores\n"
        "6. Show progress"
    ))

    # Setup
    module, cards, questions = create_sample_module()
    learner = LearnerProfile(
        id="learner_demo",
        display_name="Demo Learner",
        default_pace=PacePreference.MEDIUM,
        default_goal=LearningGoal.COMPREHEND,
    )

    console.print(f"\n[bold]Learner:[/bold] {learner.display_name}")
    console.print(f"[bold]Module:[/bold] {module.name}")
    console.print(f"[bold]Pace:[/bold] {learner.default_pace.value}")
    console.print(f"[bold]Goal:[/bold] {learner.default_goal.value}")

    # Initialize card states with some prior knowledge
    card_states = create_card_states(cards, learner.id, random_mastery=True)

    # Step 1: Apply decay
    console.print("\n[bold cyan]Step 1: Applying Knowledge Decay[/bold cyan]")
    decay = KnowledgeDecay()
    now = datetime.now(timezone.utc)

    # Simulate some cards were seen hours ago
    for i, state in enumerate(card_states.values()):
        if state.times_seen > 0:
            state.last_seen_at = now - timedelta(hours=random.randint(5, 48))

    decay_results = decay.apply_decay_batch(list(card_states.values()), now)
    decayed_count = sum(1 for r in decay_results if r.decay_applied)
    console.print(f"  Decayed {decayed_count}/{len(decay_results)} cards")

    # Step 2: Select cards
    console.print("\n[bold cyan]Step 2: Selecting Cards for Learning Cycle[/bold cyan]")
    selector = CardSelector()
    selection_result = selector.select_cards(list(card_states.values()), n=3)
    console.print(f"  Selected {selection_result.num_selected} cards:")
    for card_id in selection_result.selected_card_ids:
        state = card_states[card_id]
        console.print(f"    - {card_id} (mastery: {state.mastery_score:.3f})")

    # Step 3: Quiz
    console.print("\n[bold cyan]Step 3: Quiz Time![/bold cyan]")

    # Get questions for selected cards
    available_questions = [q for q in questions if any(
        cid in selection_result.selected_card_ids for cid in q.linked_card_ids
    )][:3]

    updater = MasteryUpdater()
    total_correct = 0

    for i, q in enumerate(available_questions, 1):
        console.print(f"\n[yellow]Q{i}:[/yellow] {q.stem}")
        console.print(f"[dim]Bloom Level: {q.bloom_level.name}[/dim]")

        # Simulate answer (random for demo)
        is_correct = random.random() > 0.3  # 70% chance correct
        if is_correct:
            console.print("[green]Correct![/green]")
            total_correct += 1
        else:
            console.print("[red]Incorrect.[/red]")
            console.print(f"[dim]Answer: {q.correct_answer}[/dim]")

        # Update mastery for linked cards
        for card_id in q.linked_card_ids:
            if card_id in card_states:
                result = updater.update_single(
                    card_states[card_id],
                    q,
                    is_correct=is_correct,
                    pace="medium",
                )
                console.print(
                    f"  [dim]{card_id}: {result.old_score:.3f} -> {result.new_score:.3f}[/dim]"
                )

    # Step 4: Progress Summary
    console.print("\n[bold cyan]Step 4: Progress Summary[/bold cyan]")
    quiz_score = total_correct / len(available_questions) if available_questions else 0

    table = Table(title="Final Card States")
    table.add_column("Card", style="cyan")
    table.add_column("Mastery", justify="right")
    table.add_column("Stereotype", style="green")
    table.add_column("Accuracy", justify="right")

    for state in sorted(card_states.values(), key=lambda s: -s.mastery_score):
        table.add_row(
            state.card_id,
            f"{state.mastery_score:.3f}",
            state.stereotype.name,
            f"{state.accuracy:.0%}" if state.times_seen > 0 else "N/A",
        )
    console.print(table)

    console.print(f"\n[bold]Quiz Score:[/bold] {quiz_score:.0%} ({total_correct}/{len(available_questions)})")

    # Gamma boosting check
    if quiz_score > 0.8:
        console.print("[green]Great performance! Gamma will be boosted for next cycle.[/green]")

    avg_mastery = sum(s.mastery_score for s in card_states.values()) / len(card_states)
    console.print(f"[bold]Average Module Mastery:[/bold] {avg_mastery:.2%}")


@app.command()
def info():
    """Show information about the Adaptive Learning Core library."""
    from adaptive_learning_core import __version__

    console.print(Panel.fit(
        f"[bold green]Adaptive Learning Core v{__version__}[/bold green]\n\n"
        "A Python library implementing adaptive learning algorithms based on:\n"
        "  - Bloom's Taxonomy for cognitive classification\n"
        "  - Mastery Learning for progression tracking\n"
        "  - Item Response Theory for question evaluation\n"
        "  - LangGraph for agent orchestration\n"
        "  - LangChain for LLM operations and RAG\n\n"
        "[bold]Key Equations:[/bold]\n"
        "  Eq. 3: Mastery Update\n"
        "  Eq. 4-5: Card Selection\n"
        "  Eq. 7-8: Knowledge Decay\n\n"
        "[bold]GitHub:[/bold] https://github.com/emfrg/adaptive-learning-core"
    ))


if __name__ == "__main__":
    app()
