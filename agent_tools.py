
from pydantic import BaseModel, Field
from functools import lru_cache

import chromadb
from sentence_transformers import SentenceTransformer
from langchain.tools import tool

from feedback_tools import (
    summarize_feedback,
    generate_feedback_report,
)


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATABASE_FOLDER = "chroma_db"
COLLECTION_NAME = "university_info"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Lower cosine distance means a closer semantic match.
MAX_COSINE_DISTANCE = 0.50

RESULTS_COUNT = 5

CONDUCT_SOURCE = "data/pdfs/Code-of-Conduct-for-Students.pdf"

CONDUCT_KEYWORDS = [
    "code of conduct",
    "student conduct",
    "student responsibilities",
    "student responsibility",
    "responsibilities of students",
    "responsibility of students",
    "student duties",
    "student obligations",
    "rules for students",
    "conduct expectations",
    "student misconduct",
    "student rights",
    "disciplinary rules",
]


# --------------------------------------------------
# 2. Tool input schema
# --------------------------------------------------

class UniversitySearchInput(BaseModel):
    question: str | dict = Field(
        description=(
            "The university-related question to search for. "
            "Provide a plain string or a structured string "
            "containing a value field."
        )
    )


# --------------------------------------------------
# 3. Load reusable resources
# --------------------------------------------------

@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the embedding model only once."""

    print("Loading MiniLM embedding model...")

    return SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )


@lru_cache(maxsize=1)
def get_chroma_collection():
    """Connect to the existing university information collection."""

    client = chromadb.PersistentClient(
        path=DATABASE_FOLDER
    )

    return client.get_collection(
        name=COLLECTION_NAME
    )


# --------------------------------------------------
# 4. University information retrieval
# --------------------------------------------------

@tool(args_schema=UniversitySearchInput)
def search_university_information(question: str | dict) -> str:
    """
    Search university PDF documents for information relevant to
    questions about university rules, courses, exams, academic
    calendars, and student responsibilities.

    Returns matching passages with their source filenames and
    page numbers.
    """

    # Handle Qwen's structured string representation.
    if isinstance(question, dict):
        question = question.get("value", "")

    # Validate the question.
    if not isinstance(question, str) or not question.strip():
        return "Please provide a valid question to search."

    try:
        # Get cached resources.
        model = get_embedding_model()
        collection = get_chroma_collection()

        total_chunks = collection.count()

        if total_chunks == 0:
            return "The university knowledge base is empty."

        # Generate the question embedding.
        question_embedding = model.encode(
            question.strip(),
            normalize_embeddings=True,
        ).tolist()

        # Check whether this is a student conduct question.
        question_lower = question.lower()

        is_conduct_question = any(
            keyword in question_lower
            for keyword in CONDUCT_KEYWORDS
        )

        # Prepare semantic search arguments.
        query_arguments = {
            "query_embeddings": [question_embedding],
            "n_results": min(RESULTS_COUNT, total_chunks),
            "include": [
                "documents",
                "metadatas",
                "distances",
            ],
        }

        # Restrict conduct questions to the relevant PDF.
        if is_conduct_question:
            query_arguments["where"] = {
                "source": CONDUCT_SOURCE
            }

        # Perform semantic search.
        results = collection.query(
            **query_arguments
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        if not documents:
            if is_conduct_question:
                return (
                    "No student conduct passages were found "
                    "in the Code of Conduct document."
                )

            return "No relevant university information was found."

        # Keep only sufficiently relevant passages.
        relevant_passages = []

        for document, metadata, distance in zip(
            documents,
            metadatas,
            distances,
        ):
            if distance > MAX_COSINE_DISTANCE:
                continue

            source = metadata.get(
                "source",
                "Unknown document",
            )

            page = metadata.get(
                "page",
                "Unknown",
            )

            relevant_passages.append({
                "source": source,
                "page": page,
                "text": document,
                "distance": distance,
            })

        if not relevant_passages:
            return (
                "I could not find sufficiently relevant "
                "information in the university documents "
                "to answer this question. Please try "
                "rephrasing it or consult the university."
            )

        # Format the retrieved passages.
        formatted_passages = []

        for passage in relevant_passages:
            formatted_passages.append(
                f"Source: {passage['source']}\n"
                f"Page: {passage['page']}\n"
                f"Text:\n{passage['text']}"
            )

        return "\n\n---\n\n".join(
            formatted_passages
        )

    except Exception as error:
        return (
            "University information search encountered "
            f"an error: {str(error)}"
        )


# --------------------------------------------------
# 5. Student feedback tools
# --------------------------------------------------

@tool
def summarize_student_feedback() -> str:
    """
    Analyze the student feedback CSV and return evidence-based
    themes, counts, positive comments, and concerns.
    """

    return summarize_feedback()


@tool
def generate_student_report() -> str:
    """
    Generate a factual student feedback report with comment
    counts, original feedback, suggested actions, and limitations.
    """

    return generate_feedback_report()


# --------------------------------------------------
# 6. Register CampusAI tools
# --------------------------------------------------

CAMPUSAI_TOOLS = [
    search_university_information,
    summarize_student_feedback,
    generate_student_report,
]
