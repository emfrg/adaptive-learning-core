"""Prompts for Bloom's Taxonomy classification.

These prompts are used by the BloomClassifier to classify questions
according to Bloom's revised taxonomy of cognitive domains.
"""

BLOOM_CLASSIFICATION_PROMPT = """You are an expert in educational assessment and Bloom's Taxonomy.
Your task is to classify the cognitive level of the following question.

BLOOM'S TAXONOMY LEVELS (from lowest to highest):
1. REMEMBER - Recall facts and basic concepts
   Verbs: define, list, name, identify, recall, recognize

2. UNDERSTAND - Explain ideas or concepts
   Verbs: explain, describe, summarize, interpret, compare, classify

3. APPLY - Use information in new situations
   Verbs: apply, use, solve, demonstrate, calculate, implement

4. ANALYZE - Draw connections among ideas
   Verbs: analyze, examine, differentiate, distinguish, investigate

5. EVALUATE - Justify a stand or decision
   Verbs: evaluate, assess, judge, critique, justify, defend

6. CREATE - Produce new or original work
   Verbs: create, design, develop, formulate, propose, construct

QUESTION:
{question}

CORRECT ANSWER:
{answer}

INSTRUCTIONS:
1. Analyze what cognitive process is required to answer this question
2. Consider the action verb used in the question stem
3. Consider the type of thinking needed (recall vs. understanding vs. application, etc.)
4. Determine the primary Bloom's level being tested

Respond with ONLY the level name in this exact format:
LEVEL: <level_name>

For example:
LEVEL: REMEMBER
LEVEL: UNDERSTAND
LEVEL: APPLY
LEVEL: ANALYZE
LEVEL: EVALUATE
LEVEL: CREATE"""


BLOOM_ANALYSIS_PROMPT = """You are an expert in educational assessment and Bloom's Taxonomy.
Provide a detailed analysis of the cognitive level of the following question.

BLOOM'S TAXONOMY LEVELS:
1. REMEMBER - Retrieve relevant knowledge from long-term memory
2. UNDERSTAND - Construct meaning from instructional messages
3. APPLY - Carry out or use a procedure in a given situation
4. ANALYZE - Break material into parts and determine relationships
5. EVALUATE - Make judgments based on criteria and standards
6. CREATE - Put elements together to form a coherent whole

QUESTION:
{question}

CORRECT ANSWER:
{answer}

CONTEXT (source material):
{context}

Provide your analysis in this JSON format:
{{
    "primary_level": "The main Bloom's level (e.g., APPLY)",
    "secondary_level": "Secondary level if applicable, or null",
    "confidence": 0.0 to 1.0,
    "reasoning": "Brief explanation of why this classification was chosen",
    "action_verb_detected": "The key verb indicating cognitive level",
    "cognitive_process": "Description of the thinking process required"
}}"""


BLOOM_BATCH_CLASSIFICATION_PROMPT = """You are an expert in educational assessment and Bloom's Taxonomy.
Classify each of the following questions according to Bloom's Taxonomy.

BLOOM'S TAXONOMY LEVELS:
1. REMEMBER - Recall facts (define, list, name, identify)
2. UNDERSTAND - Explain ideas (explain, describe, summarize)
3. APPLY - Use in new situations (apply, solve, calculate)
4. ANALYZE - Draw connections (analyze, compare, differentiate)
5. EVALUATE - Make judgments (evaluate, assess, critique)
6. CREATE - Produce new work (create, design, develop)

QUESTIONS:
{questions}

Respond with a JSON array of classifications:
[
    {{"question_index": 0, "level": "REMEMBER"}},
    {{"question_index": 1, "level": "APPLY"}},
    ...
]"""


BLOOM_VALIDATION_PROMPT = """Review the following question and its Bloom's Taxonomy classification.
Determine if the classification is accurate.

QUESTION:
{question}

CLAIMED LEVEL: {claimed_level}
LEVEL DESCRIPTION: {level_description}

Is this classification accurate? Consider:
1. Does the question's action verb match this level?
2. Does the cognitive process required match this level?
3. Could it be classified at a different level?

Respond in JSON format:
{{
    "is_accurate": true or false,
    "suggested_level": "The correct level if different, or same as claimed",
    "explanation": "Brief explanation"
}}"""
