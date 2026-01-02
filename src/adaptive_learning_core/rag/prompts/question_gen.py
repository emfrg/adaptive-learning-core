"""Prompts for question generation from learning materials.

These prompts are used by the QuestionGenerator to create
multiple-choice questions from card content.
"""

QUESTION_GENERATION_PROMPT = """You are an expert educational content creator.
Generate {num_questions} multiple-choice questions based on the following learning material.

LEARNING MATERIAL:
---
{content}
---

REQUIREMENTS:
1. Each question should test understanding of key concepts from the material
2. Questions should vary in cognitive complexity (from recall to analysis)
3. Each question needs exactly {num_distractors} plausible but incorrect distractors
4. Distractors should be clearly wrong but not obviously absurd
5. Include a brief explanation of why the correct answer is right

OUTPUT FORMAT (JSON):
{{
    "questions": [
        {{
            "stem": "The question text ending with a question mark?",
            "correct_answer": "The correct answer",
            "distractors": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
            "explanation": "Brief explanation of why the correct answer is right"
        }}
    ]
}}

Generate the questions now:"""


BLOOM_TARGETED_PROMPT = """You are an expert educational content creator specializing in Bloom's Taxonomy.
Generate a question at the {bloom_level} level of cognitive complexity.

BLOOM'S TAXONOMY LEVEL: {bloom_level}
Description: {bloom_description}
Suggested action verbs: {action_verbs}

LEARNING MATERIAL:
---
{content}
---

REQUIREMENTS:
1. The question MUST specifically target the {bloom_level} cognitive level
2. Use appropriate action verbs for this level
3. Include exactly {num_distractors} plausible but incorrect distractors
4. Distractors should be at a similar cognitive level
5. Include explanation referencing why this tests {bloom_level} skills

OUTPUT FORMAT (JSON):
{{
    "questions": [
        {{
            "stem": "A {bloom_level}-level question?",
            "correct_answer": "The correct answer",
            "distractors": ["Wrong answer 1", "Wrong answer 2", "Wrong answer 3"],
            "explanation": "This tests {bloom_level} because..."
        }}
    ]
}}

Generate the question now:"""


MULTI_BLOOM_PROMPT = """You are an expert educational content creator specializing in Bloom's Taxonomy.
Generate questions at multiple cognitive levels from the following material.

TARGET LEVELS:
{target_levels}

LEARNING MATERIAL:
---
{content}
---

For each target level, generate ONE question that specifically tests that cognitive level.
Use appropriate action verbs and question structures for each level.

OUTPUT FORMAT (JSON):
{{
    "questions": [
        {{
            "stem": "Question text?",
            "correct_answer": "The correct answer",
            "distractors": ["Wrong 1", "Wrong 2", "Wrong 3"],
            "explanation": "Why this answer is correct",
            "bloom_level": "REMEMBER|UNDERSTAND|APPLY|ANALYZE|EVALUATE|CREATE"
        }}
    ]
}}

Generate the questions now:"""


QUESTION_IMPROVEMENT_PROMPT = """Review and improve the following generated question.

ORIGINAL QUESTION:
Stem: {stem}
Correct Answer: {correct_answer}
Distractors: {distractors}

SOURCE MATERIAL:
{content}

ISSUES TO CHECK:
1. Is the question clear and unambiguous?
2. Is there exactly one correct answer?
3. Are distractors plausible but clearly incorrect?
4. Is the question grounded in the source material?
5. Is the difficulty level appropriate?

OUTPUT FORMAT (JSON):
{{
    "improved_stem": "The improved question?",
    "improved_correct_answer": "Improved correct answer",
    "improved_distractors": ["Improved wrong 1", "Improved wrong 2", "Improved wrong 3"],
    "changes_made": ["List of changes made"],
    "quality_score": 0.0 to 1.0
}}

Improve the question:"""
