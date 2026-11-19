
from pathlib import Path
import hashlib
import re

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# 1. Configuration
PDF_FOLDER = Path("data/pdfs")
DATABASE_FOLDER = "chroma_db"
COLLECTION_NAME = "university_info"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
BATCH_SIZE = 100


# 2. Split text into overlapping chunks
def split_into_chunks(text, chunk_size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be between 0 and chunk_size.")

    # Clean unnecessary whitespace
    text = re.sub(r"\s+", " ", text).strip()

    words = text.split()

    chunks = []
    start = 0

    while start < len(words):

        end = start
        current_length = 0

        # Build a chunk within the character limit
        while end < len(words):

            word_length = len(words[end])

            if end > start:
                word_length += 1  # Space between words

            if current_length + word_length > chunk_size and end > start:
                break

            current_length += word_length
            end += 1

        chunk = " ".join(words[start:end])

        if chunk:
            chunks.append(chunk)

        if end >= len(words):
            break

        # Calculate the starting position of the next chunk
        # using the requested overlap
        next_start = end
        overlap_length = 0

        while next_start > start:

            previous_word = words[next_start - 1]

            added_length = len(previous_word)

            if overlap_length > 0:
                added_length += 1

            if overlap_length + added_length > overlap:
                break

            overlap_length += added_length
            next_start -= 1

        # Prevent an infinite loop
        if next_start == start:
            next_start = end

        start = next_start

    return chunks


# 3. Extract PDF text and metadata
def collect_documents():

    documents = []
    metadatas = []
    processed_files = 0

    if not PDF_FOLDER.exists():
        raise FileNotFoundError(
            f"PDF folder not found: {PDF_FOLDER}"
        )

    pdf_files = sorted(PDF_FOLDER.glob("*.pdf"))

    if not pdf_files:
        raise ValueError(
            "No PDF files found inside data/pdfs."
        )

    print(f"Found {len(pdf_files)} PDF files.")

    for pdf_file in pdf_files:

        print(f"\nReading PDF: {pdf_file.name}")

        try:
            reader = PdfReader(str(pdf_file))

            if reader.is_encrypted:
                print(f"Skipping encrypted PDF: {pdf_file.name}")
                continue

            file_has_text = False

            for page_number, page in enumerate(reader.pages, start=1):

                page_text = page.extract_text() or ""

                if not page_text.strip():
                    print(
                        f"Warning: No selectable text on "
                        f"page {page_number} of {pdf_file.name}"
                    )
                    continue

                page_chunks = split_into_chunks(page_text)

                for chunk_index, chunk in enumerate(page_chunks):

                    documents.append(chunk)

                    metadatas.append({
                        "source": str(pdf_file),
                        "source_type": "pdf",
                        "page": page_number,
                        "chunk_index": chunk_index,
                    })

                    file_has_text = True

            if file_has_text:
                processed_files += 1
            else:
                print(f"Warning: No text extracted from {pdf_file.name}")

        except Exception as error:
            print(
                f"Could not process {pdf_file.name}: {error}"
            )

    if not documents:
        raise ValueError(
            "No text was extracted from the PDFs. "
            "Check the PDF files and their text format."
        )

    print(f"\nTotal chunks collected: {len(documents)}")
    print(f"PDF files with extracted text: {processed_files}")

    return documents, metadatas, processed_files


# 4. Generate unique IDs for chunks
def generate_chunk_ids(metadatas):

    ids = []

    for metadata in metadatas:

        unique_text = (
            f"{metadata['source']}-"
            f"{metadata['page']}-"
            f"{metadata['chunk_index']}"
        )

        chunk_id = hashlib.sha256(
            unique_text.encode("utf-8")
        ).hexdigest()

        ids.append(chunk_id)

    return ids


# 5. Store documents in ChromaDB
def store_in_chromadb(documents, metadatas, embeddings):

    client = chromadb.PersistentClient(
        path=DATABASE_FOLDER
    )

    # Reuse the existing collection if available.
    # Otherwise, create a new one.
    try:
        collection = client.get_collection(
            name=COLLECTION_NAME
        )

        print("\nExisting ChromaDB collection found.")

    except Exception:
        collection = client.create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

        print("\nCreated a new ChromaDB collection.")

    ids = generate_chunk_ids(metadatas)

    # Retrieve existing IDs so outdated chunks can be removed
    existing_data = collection.get(
        include=[]
    )

    existing_ids = set(existing_data["ids"])
    new_ids = set(ids)

    # Update the collection in batches
    for start in range(0, len(documents), BATCH_SIZE):

        end = min(start + BATCH_SIZE, len(documents))

        collection.upsert(
            ids=ids[start:end],
            documents=documents[start:end],
            embeddings=embeddings[start:end].tolist(),
            metadatas=metadatas[start:end],
        )

        print(
            f"Stored chunks {start + 1} to {end}"
        )

    # Delete outdated chunks only after all new chunks are stored
    outdated_ids = list(existing_ids - new_ids)

    if outdated_ids:

        for start in range(0, len(outdated_ids), BATCH_SIZE):

            collection.delete(
                ids=outdated_ids[start:start + BATCH_SIZE]
            )

        print(f"Removed {len(outdated_ids)} outdated chunks.")

    print("\nChromaDB updated successfully!")
    print(f"Total indexed chunks: {collection.count()}")


# 6. Main ingestion pipeline
def main():

    print("=" * 50)
    print("       CAMPUSAI - DOCUMENT INGESTION")
    print("=" * 50)

    # Extract PDF content
    documents, metadatas, processed_files = collect_documents()

    # Load embedding model
    print("\nLoading MiniLM embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    # Generate embeddings
    print("\nGenerating embeddings...")

    embeddings = model.encode(
        documents,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    print("Embeddings generated successfully.")

    # Store in ChromaDB
    print("\nUpdating ChromaDB...")

    store_in_chromadb(
        documents,
        metadatas,
        embeddings,
    )

    print("\n" + "=" * 50)
    print("       INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 50)

    print(f"PDF files processed: {processed_files}")
    print(f"Chunks indexed: {len(documents)}")
    print(f"Database location: {DATABASE_FOLDER}/")


if __name__ == "__main__":
    main()

