# Adaptive Learning Core

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

Adaptive e-learning system with Bloom's Taxonomy question generation, mastery tracking, and Ebbinghaus-inspired knowledge decay. Upload any textbook, get personalized quizzes.
<!-- 
## Demo

[![Video Demo](assets/demo_thumbnail.png)](https://your-video-link)

Or try the [interactive demo →](https://your-demo-link) -->

## How It Works
<!-- 
![System Flow](assets/system_flow.png) -->

1. **Upload** → Textbook is chunked into cards
2. **Quiz** → RAG generates questions classified by Bloom's Taxonomy
3. **Adapt** → System updates mastery scores based on performance
4. **Repeat** → Cards selected probabilistically, prioritizing weak areas

## Quick Start

```bash
# Using uv (recommended)
uv sync

# With optional dependencies
uv sync --extra openai --extra anthropic

# For development (includes pytest, ruff, mypy)
uv sync --group dev
```

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

<!--
## Case Study

In pilot testing, learners using the adaptive system reached target mastery 35% faster than a baseline without personalized card selection.
-->

## Technical Reference

### Core Algorithms

The system implements several learning science models:

**Mastery Update Rule**

$$M(c) = M(c) + \alpha \times C_{c,q} \times \delta$$

Where $\alpha$ is the learning rate (pace-dependent), $C_{c,q}$ is the question-card contribution weight, and $\delta = 1$ for correct, $-1$ for incorrect.

**Card Selection Probability**

$$P(c) = \frac{(1 - M(c)) + \gamma}{Z}$$

Cards with lower mastery scores are selected more frequently. $\gamma$ is a prior ensuring unseen cards aren't neglected.

**Knowledge Decay (Ebbinghaus)**

$$M(c) = M(c) \times e^{-\beta \lambda \tau}$$

Mastery decays over time $\tau$ since last interaction, with decay rate $\beta$ and context multiplier $\lambda$.

**Question-Card Linking**

$$C_{c,q} = \frac{\exp(\text{cos\_sim}(q, c))}{\sum_{c' \in \text{top-}k} \exp(\text{cos\_sim}(q, c'))}$$

Softmax over cosine similarities links questions to relevant cards.

### Item Response Theory (IRT)

Question quality estimation using psychometric models:

| Model | Formula | Use Case |
|-------|---------|----------|
| 1PL (Rasch) | $P(\theta) = \frac{1}{1 + e^{-(\theta - b)}}$ | Difficulty only |
| 2PL | $P(\theta) = \frac{1}{1 + e^{-a(\theta - b)}}$ | + Discrimination |
| 3PL | $P(\theta) = c + \frac{1-c}{1 + e^{-a(\theta - b)}}$ | + Guessing |

### Bloom's Taxonomy Classification

Questions are classified into 6 cognitive levels and mapped to learner stereotypes:

| Level | Bloom's | Stereotype |
|-------|---------|------------|
| 1 | Remember | Novice |
| 2 | Understand | Beginner |
| 3 | Apply | Intermediate |
| 4 | Analyze | Advanced |
| 5-6 | Evaluate/Create | Expert |

## Paper

For detailed system design and theoretical foundations, see the [full research paper](adaptive-learning-paper.pdf).

## Acknowledgments

Based on a university project for the Adaptive Interactive Systems course at Utrecht University (2024).

Original contributors:
- Sebastian Daniëls
- Emmanuel Fragkiadakis
- Ivan Oskam
- Lili Tordai

Continued development and maintenance by Emmanuel (Manos) Fragkiadakis.

## License

MIT
