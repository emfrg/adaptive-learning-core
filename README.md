# Adaptive Learning Core

An Adaptive Learning System implementing personalized e-learning with Bloom's Taxonomy, Mastery Learning, Item Response Theory, and LangGraph orchestration.

## Overview

This system provides the algorithmic foundation for adaptive learning that:

- **Tracks learner mastery** using continuous mastery scores with time-based decay
- **Selects learning content** probabilistically based on mastery gaps
- **Generates questions** using RAG (Retrieval-Augmented Generation) from source materials
- **Classifies question difficulty** using Bloom's Taxonomy (6 cognitive levels)
- **Evaluates question quality** using IRT discrimination, LLM-as-judge, and RAGAS metrics

## Key Algorithms

### From the Research Paper

| Equation | Description | Implementation |
|----------|-------------|----------------|
| Eq. 1 | `cos_sim(q,c) = (q·c)/(‖q‖×‖c‖)` | Question-card similarity |
| Eq. 2 | `C_{c,q} = exp(sim) / Σexp(sim)` | Softmax contribution weights |
| Eq. 3 | `M(c) = M(c) + α × C_{c,q} × δ` | Mastery update rule |
| Eq. 4 | `P(c) = (1 - M(c) + γ) / Z` | Card selection probability |
| Eq. 5 | `γ_new = γ × δ` | Gamma boosting |
| Eq. 7-8 | `M(c) = M(c) × e^{-βλτ}` | Knowledge decay (Ebbinghaus) |

### Item Response Theory (IRT)

Full implementation of psychometric models for question quality estimation:

- **1PL (Rasch)**: `P(θ) = 1 / (1 + e^{-(θ-b)})`
- **2PL**: `P(θ) = 1 / (1 + e^{-a(θ-b)})`
- **3PL**: `P(θ) = c + (1-c) / (1 + e^{-a(θ-b)})`

## Paper
For detailed system presentation and user interface design, see the [full report](INFOMAIS_Group13-Final%20Paper.pdf)

## Installation

```bash
# Using uv (recommended)
uv sync

# With optional dependencies
uv sync --extra openai --extra anthropic

# For development (includes pytest, ruff, mypy)
uv sync --group dev
```

## Quick Start

```python
from adaptive_learning_core import (
    Card, CardState, LearnerStereotype,
    Question, BloomLevel,
    MasteryUpdater, KnowledgeDecay, CardSelector,
    DEFAULT_CONFIG,
)

# Create cards from learning material
card = Card(
    id="card_001",
    content="Photosynthesis converts light energy into chemical energy...",
    module_id="biology_101",
)

# Initialize card state for a learner
state = CardState(
    card_id=card.id,
    mastery=0.3,
    stereotype=LearnerStereotype.BEGINNER,
)

# Apply knowledge decay after time away
decay = KnowledgeDecay(config=DEFAULT_CONFIG.decay)
decayed_state = decay.apply_decay(state, hours_elapsed=48)

# Select cards for learning based on mastery gaps
selector = CardSelector(config=DEFAULT_CONFIG.selection)
selected = selector.select_cards(
    cards=[card],
    states={card.id: decayed_state},
    n_cards=5,
)

# Update mastery after quiz
updater = MasteryUpdater(config=DEFAULT_CONFIG.mastery)
result = updater.update_mastery(
    states={card.id: state},
    question_contributions={card.id: 0.8},
    score=0.75,
    is_correct=True,
)
```

## Module Structure

```
adaptive_learning_core/
├── models/          # Data models (Card, Question, Quiz, Module, Learner)
├── algorithms/      # Core algorithms (mastery, decay, selection, linking)
├── bloom/           # Bloom's Taxonomy classification
├── rag/             # RAG pipeline (chunking, embeddings, generation)
├── agents/          # LangGraph agents for orchestration
├── irt/             # Item Response Theory models (1PL, 2PL, 3PL)
└── evaluation/      # Quality evaluation (LLM-judge, RAGAS)
```

## Configuration

All parameters are configurable:

```python
from adaptive_learning_core import AdaptiveLearningConfig, MasteryConfig

config = AdaptiveLearningConfig(
    mastery=MasteryConfig(
        alpha_slow=0.05,    # Learning rate for slow pace
        alpha_medium=0.10,  # Learning rate for medium pace
        alpha_fast=0.15,    # Learning rate for fast pace
        boost_factor=1.5,   # δ - boost when score > threshold
        boost_threshold=0.8 # θ - score threshold for boosting
    ),
    # ... other configs
)
```

## Requirements

- Python 3.11+
- NumPy, SciPy, scikit-learn
- Optional: sentence-transformers, openai, anthropic


## Acknowledgments

Based on a university project for the Adaptive Interactive Systems course at Utrecht University (2024).

Original contributors:
- Sebastian Daniëls
- Emmanuel Fragkiadakis
- Ivan Oskam
- Lili Tordai

Continued development and maintenance by Emmanuel (Manos) Fragkiadakis.

## License

MIT License
