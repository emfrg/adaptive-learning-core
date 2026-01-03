"""Upload Document page - PDF/text upload and chunking."""

import uuid
from io import BytesIO

import streamlit as st

from adaptive_learning_core.models.card import Card, CardState
from adaptive_learning_core.models.module import Module
from adaptive_learning_core.rag.chunker import (
    TokenChunker,
    SentenceChunker,
    SemanticChunker,
    ChunkingConfig,
)

# Try to import pypdf
try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


st.title("📄 Upload Document")
st.markdown("""
Upload a PDF or paste text to create learning cards. The document will be
split into chunks that become individual learning cards.
""")

# Initialize session state
if "raw_text" not in st.session_state:
    st.session_state.raw_text = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "cards" not in st.session_state:
    st.session_state.cards = []
if "module" not in st.session_state:
    st.session_state.module = None

# Tabs for input method
tab1, tab2, tab3 = st.tabs(["📤 Upload PDF", "📝 Paste Text", "📚 Sample Data"])

with tab1:
    if not PDF_AVAILABLE:
        st.warning("pypdf not installed. Use 'uv add pypdf' to enable PDF upload.")
    else:
        uploaded_file = st.file_uploader(
            "Choose a PDF file",
            type=["pdf"],
            help="Upload a PDF document to extract text and create learning cards.",
        )

        if uploaded_file is not None:
            with st.spinner("Extracting text from PDF..."):
                try:
                    reader = PdfReader(BytesIO(uploaded_file.read()))
                    text = ""
                    for page in reader.pages:
                        text += page.extract_text() + "\n\n"

                    st.session_state.raw_text = text
                    st.session_state.uploaded_filename = uploaded_file.name
                    st.success(f"Extracted {len(text)} characters from {len(reader.pages)} pages")
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")

with tab2:
    text_input = st.text_area(
        "Paste your learning material here",
        height=300,
        placeholder="Enter or paste the text you want to learn from...",
    )

    if st.button("Use This Text", type="primary"):
        if text_input.strip():
            st.session_state.raw_text = text_input
            st.session_state.uploaded_filename = "pasted_text"
            st.success(f"Loaded {len(text_input)} characters")
        else:
            st.warning("Please enter some text first")

with tab3:
    st.markdown("""
    Don't have a document? Use our sample **Biology 101** module with
    pre-created cards and questions about cell biology.
    """)

    if st.button("Load Sample Data", type="primary"):
        import sys
        from pathlib import Path
        # Add demo directory to path for imports
        demo_dir = Path(__file__).parent.parent.parent
        if str(demo_dir) not in sys.path:
            sys.path.insert(0, str(demo_dir))
        from streamlit_app.utils.sample_data import get_sample_module, create_card_states

        module, cards, questions = get_sample_module()

        st.session_state.module = module
        st.session_state.cards = cards
        st.session_state.questions = questions
        st.session_state.card_states = create_card_states(cards, "demo_user")
        st.session_state.learner_id = "demo_user"
        st.session_state.raw_text = "\n\n".join([c.text for c in cards])

        st.success(f"Loaded {len(cards)} cards and {len(questions)} questions!")
        st.info("Go to **Learning Session** to start learning!")

# Show text preview and chunking options if text is loaded
if st.session_state.raw_text:
    st.divider()
    st.subheader("📖 Text Preview")

    with st.expander("View extracted text", expanded=False):
        st.text(st.session_state.raw_text[:2000] + "..." if len(st.session_state.raw_text) > 2000 else st.session_state.raw_text)

    st.write(f"**Total characters:** {len(st.session_state.raw_text)}")

    # Chunking configuration
    st.divider()
    st.subheader("⚙️ Chunking Configuration")

    col1, col2 = st.columns(2)

    with col1:
        chunking_strategy = st.selectbox(
            "Chunking Strategy",
            ["Token-based", "Sentence-based", "Semantic"],
            help="""
            - **Token-based**: Fixed token size with overlap
            - **Sentence-based**: Groups sentences together
            - **Semantic**: Splits at logical boundaries (sections, paragraphs)
            """,
        )

    with col2:
        chunk_size = st.slider(
            "Chunk Size (tokens)",
            min_value=100,
            max_value=1000,
            value=500,
            step=50,
            help="Target number of tokens per chunk",
        )

    overlap = st.slider(
        "Overlap (tokens)",
        min_value=0,
        max_value=200,
        value=50,
        step=10,
        help="Number of overlapping tokens between chunks",
    )

    # Process button
    if st.button("🔄 Create Cards", type="primary"):
        with st.spinner("Chunking document..."):
            config = ChunkingConfig(
                chunk_size=chunk_size,
                chunk_overlap=overlap,
            )

            # Select chunker based on strategy
            if chunking_strategy == "Token-based":
                chunker = TokenChunker(config)
            elif chunking_strategy == "Sentence-based":
                chunker = SentenceChunker(config)
            else:
                chunker = SemanticChunker(config)

            # Generate source ID
            source_id = st.session_state.get("uploaded_filename", "document")

            # Chunk the text
            chunks = chunker.chunk(st.session_state.raw_text, source_id)
            st.session_state.chunks = chunks

            # Create module
            module_id = f"mod_{uuid.uuid4().hex[:8]}"
            module = Module(
                id=module_id,
                name=f"Module: {source_id}",
                description=f"Created from {source_id}",
                source_filename=source_id,
                card_ids=[f"card_{i}" for i in range(len(chunks))],
            )
            st.session_state.module = module

            # Convert chunks to cards
            cards = [chunk.to_card(module_id) for chunk in chunks]
            st.session_state.cards = cards

            # Initialize card states
            learner_id = "demo_user"
            card_states = {}
            for card in cards:
                card_states[card.id] = CardState(
                    card_id=card.id,
                    learner_id=learner_id,
                    mastery_score=0.0,
                    times_seen=0,
                )
            st.session_state.card_states = card_states
            st.session_state.learner_id = learner_id

            # Clear any existing questions (need to regenerate)
            st.session_state.questions = []
            st.session_state.question_gen_complete = False

            st.success(f"Created {len(cards)} learning cards!")

# Show cards preview
if st.session_state.cards:
    st.divider()
    st.subheader("📚 Learning Cards Preview")

    st.write(f"**Total cards:** {len(st.session_state.cards)}")

    # Show first few cards
    for i, card in enumerate(st.session_state.cards[:5]):
        with st.expander(f"Card {i+1}: {card.id}", expanded=(i == 0)):
            st.write(card.text)

    if len(st.session_state.cards) > 5:
        st.info(f"... and {len(st.session_state.cards) - 5} more cards")

    # Next steps
    st.divider()
    st.subheader("✅ Next Steps")

    if st.session_state.get("llm_configured"):
        st.success("API key configured! Go to **Learning Session** to generate questions and start learning.")
    else:
        st.warning("Configure your API key in the sidebar to generate questions.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("➡️ Go to Learning Session"):
            st.switch_page("pages/2_🎓_Learning_Session.py")
    with col2:
        if st.button("📊 View Progress"):
            st.switch_page("pages/3_📊_Progress.py")
