"""LangGraph-based Question Generation Agent.

This agent implements a RAG pipeline for generating educational MCQs:

    ┌─────────────────────────────────────────────────────────────┐
    │                QUESTION GENERATION GRAPH                      │
    └─────────────────────────────────────────────────────────────┘

    START
      │
      ▼
    ┌──────────────────┐
    │ chunk_documents  │ ◄── LangChain TextSplitter
    └─────┬────────────┘
          │
          ▼
    ┌──────────────┐
    │ embed_chunks │ ◄── Sentence Transformers
    └─────┬────────┘
          │
          ▼
    ┌──────────────────┐
    │ retrieve_context │ ◄── Vector similarity
    └─────┬────────────┘
          │
          ▼
    ┌────────────────────┐
    │ generate_questions │ ◄── LLM generation
    └─────┬──────────────┘
          │
          ├──── retry? ────┐
          │                │
          ▼                │
    ┌────────────────────┐ │
    │ validate_questions │ │
    └─────┬──────────────┘ │
          │                │
          │   (need more)  │
          └────────────────┘
          │
          ▼
        END
"""

import json
from typing import Any, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from adaptive_learning_core.agents.state import QuestionGenState
from adaptive_learning_core.agents.nodes import (
    chunk_documents,
    embed_chunks,
    retrieve_context,
    validate_questions,
    should_retry_generation,
)
from adaptive_learning_core.models.question import Question, BloomLevel
from adaptive_learning_core.bloom.taxonomy import get_action_verbs, BLOOM_LEVEL_INFO


# Question generation prompt
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

Output as JSON array with this structure:
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

Generate {num_questions} questions at Bloom level {bloom_level}.""")
])


def create_generate_node(llm: BaseChatModel):
    """Create a generation node with the given LLM.

    Args:
        llm: LangChain chat model

    Returns:
        Node function for question generation
    """
    parser = JsonOutputParser()

    async def generate_questions(state: QuestionGenState) -> dict[str, Any]:
        """Generate questions from context using LLM."""
        retrieved_chunks = state.get("retrieved_chunks", [])
        target_bloom = state.get("target_bloom_level", 1)
        num_questions = state.get("num_questions", 5)
        module_id = state.get("module_id", "default")

        if not retrieved_chunks:
            return {
                "generated_questions": [],
                "messages": [AIMessage(content="No context available for generation")],
            }

        # Combine chunks into context
        context = "\n\n".join(c.get("text", "") for c in retrieved_chunks)

        # Get Bloom level info
        bloom_info = BLOOM_LEVEL_INFO.get(target_bloom, BLOOM_LEVEL_INFO[1])
        action_verbs = ", ".join(get_action_verbs(target_bloom)[:5])

        # Generate using LLM
        chain = QUESTION_GEN_PROMPT | llm | parser

        try:
            result = await chain.ainvoke({
                "context": context,
                "num_questions": num_questions,
                "bloom_level": target_bloom,
                "bloom_name": bloom_info.name,
                "bloom_description": bloom_info.description,
                "action_verbs": action_verbs,
            })

            # Convert to Question objects
            questions = []
            for i, q_data in enumerate(result):
                question = Question(
                    id=f"{module_id}_q_{i}",
                    module_id=module_id,
                    stem=q_data["stem"],
                    correct_answer=q_data["correct_answer"],
                    distractors=q_data["distractors"],
                    explanation=q_data["explanation"],
                    bloom_level=BloomLevel(q_data.get("bloom_level", target_bloom)),
                )
                questions.append(question)

            existing = state.get("generated_questions", [])

            return {
                "generated_questions": existing + questions,
                "generation_attempts": state.get("generation_attempts", 0) + 1,
                "messages": [AIMessage(content=f"Generated {len(questions)} questions")],
            }

        except Exception as e:
            return {
                "generation_attempts": state.get("generation_attempts", 0) + 1,
                "error": str(e),
                "messages": [AIMessage(content=f"Generation error: {e}")],
            }

    return generate_questions


def create_question_gen_graph(llm: BaseChatModel) -> StateGraph:
    """Create the question generation state graph.

    Args:
        llm: LangChain chat model for generation

    Returns:
        Compiled StateGraph for question generation
    """
    graph = StateGraph(QuestionGenState)

    # Add nodes
    graph.add_node("chunk", chunk_documents)
    graph.add_node("embed", embed_chunks)
    graph.add_node("retrieve", retrieve_context)
    graph.add_node("generate", create_generate_node(llm))
    graph.add_node("validate", validate_questions)

    # Add edges
    graph.add_edge(START, "chunk")
    graph.add_edge("chunk", "embed")
    graph.add_edge("embed", "retrieve")
    graph.add_edge("retrieve", "generate")

    # Conditional: retry or validate
    graph.add_conditional_edges(
        "generate",
        should_retry_generation,
        {
            "generate": "retrieve",  # Retry with different context
            "validate": "validate",
        }
    )

    graph.add_edge("validate", END)

    return graph


class QuestionGenAgent:
    """Agent for generating educational questions from source material.

    Uses LangGraph for workflow and LangChain for LLM operations.

    Example:
        >>> from langchain_openai import ChatOpenAI
        >>>
        >>> llm = ChatOpenAI(model="gpt-4")
        >>> agent = QuestionGenAgent(llm=llm)
        >>>
        >>> questions = await agent.generate(
        ...     source_text=textbook_content,
        ...     num_questions=10,
        ...     target_bloom_level=3,  # Apply
        ...     module_id="biology_101",
        ... )
    """

    def __init__(
        self,
        llm: BaseChatModel,
        checkpointer: Optional[Any] = None,
    ):
        """Initialize the question generation agent.

        Args:
            llm: LangChain chat model
            checkpointer: Optional LangGraph checkpointer
        """
        self.llm = llm
        self.checkpointer = checkpointer or MemorySaver()

        # Create and compile graph
        self.graph = create_question_gen_graph(llm)
        self.compiled = self.graph.compile(checkpointer=self.checkpointer)

    async def generate(
        self,
        source_text: str = "",
        source_documents: Optional[list[dict]] = None,
        num_questions: int = 5,
        target_bloom_level: int = 1,
        module_id: str = "default",
        thread_id: str = "default",
    ) -> list[Question]:
        """Generate questions from source material.

        Args:
            source_text: Raw text content
            source_documents: List of document dicts with content
            num_questions: Number of questions to generate
            target_bloom_level: Bloom's Taxonomy level (1-6)
            module_id: Module identifier for questions
            thread_id: Thread ID for checkpointing

        Returns:
            List of generated Question objects
        """
        initial_state = QuestionGenState(
            source_text=source_text,
            source_documents=source_documents or [],
            num_questions=num_questions,
            target_bloom_level=target_bloom_level,
            module_id=module_id,
            chunks=[],
            chunk_embeddings=[],
            retrieved_chunks=[],
            generated_questions=[],
            validated_questions=[],
            rejected_questions=[],
            generation_attempts=0,
            messages=[],
        )

        config = {"configurable": {"thread_id": thread_id}}
        result = await self.compiled.ainvoke(initial_state, config)

        return result.get("validated_questions", [])

    async def generate_batch(
        self,
        source_text: str,
        bloom_levels: list[int],
        questions_per_level: int = 3,
        module_id: str = "default",
    ) -> dict[int, list[Question]]:
        """Generate questions at multiple Bloom levels.

        Args:
            source_text: Source material
            bloom_levels: List of Bloom levels to generate for
            questions_per_level: Questions per level
            module_id: Module identifier

        Returns:
            Dict mapping bloom_level -> list of questions
        """
        results = {}
        for level in bloom_levels:
            questions = await self.generate(
                source_text=source_text,
                num_questions=questions_per_level,
                target_bloom_level=level,
                module_id=module_id,
                thread_id=f"{module_id}_level_{level}",
            )
            results[level] = questions

        return results

    def get_supported_bloom_levels(self) -> list[dict[str, Any]]:
        """Get info about supported Bloom levels.

        Returns:
            List of Bloom level info dicts
        """
        return [
            {
                "level": level,
                "name": info.name,
                "description": info.description,
                "action_verbs": info.action_verbs[:5],
            }
            for level, info in BLOOM_LEVEL_INFO.items()
        ]
