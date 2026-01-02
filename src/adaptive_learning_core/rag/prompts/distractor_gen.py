"""Prompts for generating and improving distractors (wrong answers).

Distractors are the incorrect answer options in multiple-choice questions.
Good distractors are plausible but clearly wrong, testing true understanding.
"""

DISTRACTOR_GENERATION_PROMPT = """Given a question and its correct answer, generate {num_distractors} plausible but incorrect answer options (distractors).

QUESTION: {question_stem}
CORRECT ANSWER: {correct_answer}

CONTEXT (from source material):
{context}

REQUIREMENTS:
1. Distractors must be clearly incorrect
2. Distractors should be plausible (common misconceptions, partial truths, related concepts)
3. Distractors should be similar in length and format to the correct answer
4. Avoid obviously absurd options
5. Each distractor should represent a different type of error

TYPES OF GOOD DISTRACTORS:
- Common misconceptions about the topic
- Partial truths (correct for a different context)
- Related but incorrect concepts
- Answers that would be correct for a similar question
- Factual errors that sound plausible

OUTPUT FORMAT (JSON):
{{
    "distractors": [
        {{"text": "Distractor 1", "rationale": "Why this is a good distractor"}},
        {{"text": "Distractor 2", "rationale": "Why this is a good distractor"}},
        {{"text": "Distractor 3", "rationale": "Why this is a good distractor"}}
    ]
}}

Generate distractors now:"""


DISTRACTOR_IMPROVEMENT_PROMPT = """Review and improve the following distractors for a multiple-choice question.

QUESTION: {question_stem}
CORRECT ANSWER: {correct_answer}
CURRENT DISTRACTORS: {distractors}

CONTEXT (source material):
{context}

EVALUATE EACH DISTRACTOR:
1. Is it clearly incorrect? (Must be wrong)
2. Is it plausible? (Could a learner reasonably choose it?)
3. Is it similar in format to the correct answer?
4. Does it avoid being obviously absurd?
5. Is it distinct from other distractors?

For each weak distractor, suggest an improvement.

OUTPUT FORMAT (JSON):
{{
    "evaluations": [
        {{
            "original": "The distractor text",
            "is_good": true/false,
            "issues": ["List of problems if any"],
            "improved": "Improved version if needed"
        }}
    ],
    "final_distractors": ["Final list of best distractors"]
}}

Evaluate and improve:"""


DISTRACTOR_FROM_MISCONCEPTIONS_PROMPT = """Generate distractors based on common misconceptions about the topic.

TOPIC: {topic}
QUESTION: {question_stem}
CORRECT ANSWER: {correct_answer}

SOURCE MATERIAL:
{context}

Think about what mistakes learners commonly make when learning about {topic}:
1. What do beginners often confuse?
2. What partial understandings might lead to wrong answers?
3. What related but different concepts might be mistaken for the correct one?

Generate {num_distractors} distractors based on these misconceptions.

OUTPUT FORMAT (JSON):
{{
    "distractors": [
        {{
            "text": "Distractor text",
            "misconception": "What misconception this represents",
            "why_learners_choose_this": "Why a learner might pick this"
        }}
    ]
}}

Generate misconception-based distractors:"""
