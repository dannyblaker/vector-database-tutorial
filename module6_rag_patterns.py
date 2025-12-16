"""
Module 6: Retrieval Augmented Generation (RAG) Patterns
=======================================================

This module teaches you about RAG - a powerful pattern for building AI applications
that combines retrieval from vector databases with language models.

Learning Objectives:
- Understand the RAG architecture
- Implement document chunking strategies
- Build a question-answering system
- Handle context windows effectively
- Learn RAG best practices and patterns
- Understand limitations and solutions
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from colorama import Fore, Style
from pathlib import Path
from tabulate import tabulate
import textwrap


def print_section(title):
    """Helper function to print section headers"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{Style.RESET_ALL}\n")


def load_job_ads():
    """Load job advertisements"""
    corpus_path = Path("example_corpus")
    job_ads = {}

    for file_path in sorted(corpus_path.glob("job_ad_*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            job_ads[file_path.stem] = f.read()

    return job_ads


def lesson_1_what_is_rag():
    """Lesson 1: Understanding RAG"""
    print_section("Lesson 1: What is RAG?")

    print("RAG (Retrieval Augmented Generation) is a pattern for building AI systems")
    print("that combines:\n")

    print("1. RETRIEVAL: Finding relevant documents from a knowledge base")
    print("2. AUGMENTATION: Adding retrieved docs to the LLM prompt")
    print("3. GENERATION: LLM generates answer using retrieved context\n")

    print("=" * 80)
    print("RAG ARCHITECTURE")
    print("=" * 80)
    print("""
    User Question
         ↓
    [1. Embed Question]
         ↓
    [2. Search Vector DB] → Returns top-k relevant documents
         ↓
    [3. Build Prompt]
         ↓
         Question + Retrieved Docs → [4. LLM] → Answer
    """)

    print("=" * 80)
    print("\n❌ WITHOUT RAG (Problems):")
    print("   • LLM only knows what it was trained on")
    print("   • No access to your private data")
    print("   • Can't answer questions about recent events")
    print("   • May hallucinate answers")
    print("   • Can't cite sources\n")

    print("✓ WITH RAG (Solutions):")
    print("   • Access to your specific knowledge base")
    print("   • Up-to-date information")
    print("   • Grounded in actual documents")
    print("   • Can cite sources")
    print("   • Reduces hallucination\n")

    print("💡 Common RAG Use Cases:")
    print("   • Enterprise knowledge bases (internal docs, wikis)")
    print("   • Customer support chatbots")
    print("   • Legal/medical document Q&A")
    print("   • Code documentation assistants")
    print("   • Research paper analysis\n")

    print("In this tutorial, we'll build a job-search assistant using RAG!\n")


def lesson_2_document_chunking():
    """Lesson 2: Chunking strategies"""
    print_section("Lesson 2: Document Chunking")

    print("Documents are often too long to:")
    print("   • Embed as single units (loses granularity)")
    print("   • Fit in LLM context windows")
    print("\nSolution: Break documents into chunks!\n")

    job_ads = load_job_ads()
    sample_doc = list(job_ads.values())[0]

    print("Example document (first 300 chars):")
    print("-" * 80)
    print(sample_doc[:300] + "...")
    print("-" * 80)
    print(f"\nDocument length: {len(sample_doc)} characters\n")

    # Strategy 1: Fixed-size chunking
    print("1. FIXED-SIZE CHUNKING")
    print("   Split by number of characters/tokens with overlap\n")

    def chunk_by_chars(text, chunk_size=200, overlap=50):
        """Chunk text by characters with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    chunks = chunk_by_chars(sample_doc, chunk_size=200, overlap=50)

    print(f"   Chunk size: 200 chars, Overlap: 50 chars")
    print(f"   Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks[:3], 1):
        print(f"   Chunk {i}:")
        print(f"   {chunk[:100]}...")
        print()

    print("   ✓ Pros: Simple, consistent size")
    print("   ✗ Cons: May split mid-sentence\n")

    # Strategy 2: Sentence-based chunking
    print("2. SENTENCE-BASED CHUNKING")
    print("   Split by sentences, combine until size limit\n")

    def chunk_by_sentences(text, max_chars=300):
        """Chunk by sentences"""
        # Simple sentence splitting
        sentences = text.replace('!', '.').replace('?', '.').split('.')
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        current_chunk = []
        current_size = 0

        for sentence in sentences:
            sentence_size = len(sentence)
            if current_size + sentence_size > max_chars and current_chunk:
                chunks.append('. '.join(current_chunk) + '.')
                current_chunk = [sentence]
                current_size = sentence_size
            else:
                current_chunk.append(sentence)
                current_size += sentence_size

        if current_chunk:
            chunks.append('. '.join(current_chunk) + '.')

        return chunks

    sent_chunks = chunk_by_sentences(sample_doc, max_chars=300)

    print(f"   Max size: 300 chars")
    print(f"   Number of chunks: {len(sent_chunks)}\n")

    for i, chunk in enumerate(sent_chunks[:2], 1):
        print(f"   Chunk {i} ({len(chunk)} chars):")
        print(f"   {chunk[:100]}...")
        print()

    print("   ✓ Pros: Preserves sentence boundaries")
    print("   ✗ Cons: Variable chunk sizes\n")

    # Strategy 3: Semantic chunking
    print("3. SEMANTIC CHUNKING")
    print("   Split based on topic/section boundaries")
    print("   • Look for headings, paragraphs")
    print("   • Use embedding similarity to detect topic changes")
    print("   • More sophisticated but better quality\n")

    print("💡 Chunking Best Practices:")
    print("   ✓ Include overlap to preserve context")
    print("   ✓ Chunk size: 200-500 tokens typically")
    print("   ✓ Test different strategies for your data")
    print("   ✓ Store chunk metadata (source doc, position)")
    print("   ✓ Consider using specialized chunking libraries\n")

    return chunks, sent_chunks


def lesson_3_building_rag_system():
    """Lesson 3: Building a RAG system"""
    print_section("Lesson 3: Building a RAG System")

    print("Let's build a complete RAG system for job search!\n")

    job_ads = load_job_ads()

    # Step 1: Chunk documents
    print("Step 1: Chunking documents")

    def chunk_document(text, chunk_size=400, overlap=100):
        """Simple chunking with overlap"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            if chunk.strip():
                chunks.append(chunk.strip())
            start = end - overlap
        return chunks

    all_chunks = []
    chunk_metadata = []

    for job_id, content in job_ads.items():
        chunks = chunk_document(content)
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            chunk_metadata.append({
                "source": job_id,
                "chunk_index": i,
                "total_chunks": len(chunks)
            })

    print(
        f"  ✓ Created {len(all_chunks)} chunks from {len(job_ads)} documents\n")

    # Step 2: Create embeddings and store
    print("Step 2: Embedding and storing chunks")

    model = SentenceTransformer('all-MiniLM-L6-v2')
    client = chromadb.PersistentClient(path="./chroma_db")

    try:
        client.delete_collection("job_chunks_rag")
    except:
        pass

    collection = client.create_collection("job_chunks_rag")

    embeddings = model.encode(all_chunks, show_progress_bar=True)
    print()

    chunk_ids = [f"chunk_{i}" for i in range(len(all_chunks))]

    collection.add(
        ids=chunk_ids,
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=chunk_metadata
    )

    print(f"  ✓ Stored {len(all_chunks)} chunks in vector database\n")

    # Step 3: Build retrieval function
    print("Step 3: Implementing retrieval")

    def retrieve_context(query, k=3):
        """Retrieve top-k relevant chunks"""
        query_embedding = model.encode(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )

        return results

    # Step 4: Build prompt function
    print("Step 4: Creating prompt template")

    def build_rag_prompt(query, retrieved_chunks):
        """Build prompt with retrieved context"""
        context = "\n\n".join([
            f"[Document {i+1}]\n{chunk}"
            for i, chunk in enumerate(retrieved_chunks)
        ])

        prompt = f"""You are a helpful assistant that answers questions about job postings.
Use the following context to answer the question. If you cannot answer based on the context, say so.

Context:
{context}

Question: {query}

Answer:"""

        return prompt

    print("  ✓ Prompt template ready\n")

    # Step 5: Demo the system
    print("Step 5: Testing the RAG system")
    print("=" * 80)

    questions = [
        "What programming languages are required?",
        "Are there any remote positions?",
        "What experience level is needed?"
    ]

    for question in questions:
        print(f"\nQuestion: {question}")
        print("-" * 80)

        # Retrieve
        results = retrieve_context(question, k=3)

        print("\nRetrieved Context:")
        for i, (chunk, metadata) in enumerate(zip(results['documents'][0],
                                                  results['metadatas'][0]), 1):
            print(
                f"\n  [{i}] From {metadata['source']} (chunk {metadata['chunk_index']})")
            print(f"  {chunk[:150]}...")

        # Build prompt
        prompt = build_rag_prompt(question, results['documents'][0])

        print(f"\nPrompt length: {len(prompt)} characters")
        print("\nNote: In production, this prompt would be sent to an LLM (GPT-4, Claude, etc.)")
        print("The LLM would generate an answer based on the retrieved context.")
        print()

    print("\n💡 RAG System Components:")
    print("   ✓ Document chunking")
    print("   ✓ Embedding generation")
    print("   ✓ Vector storage")
    print("   ✓ Retrieval function")
    print("   ✓ Prompt construction")
    print("   ✓ LLM integration (not shown, requires API)\n")


def lesson_4_advanced_rag_patterns():
    """Lesson 4: Advanced RAG patterns"""
    print_section("Lesson 4: Advanced RAG Patterns")

    print("Advanced techniques to improve RAG systems:\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("job_chunks_rag")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Pattern 1: Query Expansion
    print("1. QUERY EXPANSION")
    print("   Generate multiple versions of the query\n")

    original_query = "Python jobs"
    expanded_queries = [
        "Python jobs",
        "Python developer positions",
        "Software engineer with Python experience",
        "Python programming roles"
    ]

    print(f"   Original: '{original_query}'")
    print(f"   Expanded: {expanded_queries}\n")

    # Retrieve for each and combine
    all_results = []
    for query in expanded_queries:
        query_emb = model.encode(query)
        results = collection.query(
            query_embeddings=[query_emb.tolist()],
            n_results=2
        )
        all_results.extend(results['ids'][0])

    # Deduplicate
    unique_results = list(dict.fromkeys(all_results))
    print(f"   Retrieved {len(all_results)} total chunks")
    print(f"   Unique chunks after deduplication: {len(unique_results)}\n")

    print("   ✓ Improves recall")
    print("   ✓ Catches different phrasings\n")

    # Pattern 2: Hypothetical Document Embeddings (HyDE)
    print("2. HYPOTHETICAL DOCUMENT EMBEDDINGS (HyDE)")
    print("   Generate hypothetical answer, then search with it\n")

    query = "What skills are needed?"
    hypothetical_answer = """
    The position requires skills in Python programming, web development,
    database management, and experience with cloud platforms.
    """

    print(f"   Query: '{query}'")
    print(f"   Hypothetical answer: {hypothetical_answer.strip()}\n")

    # Search with hypothetical answer
    hyp_emb = model.encode(hypothetical_answer)
    results = collection.query(
        query_embeddings=[hyp_emb.tolist()],
        n_results=3
    )

    print(f"   Retrieved chunks: {len(results['ids'][0])}")
    print("   ✓ Sometimes more effective than query alone\n")

    # Pattern 3: Re-ranking
    print("3. TWO-STAGE RETRIEVAL + RE-RANKING")
    print("   1. Fast retrieval of top-k candidates")
    print("   2. Accurate re-ranking of candidates\n")

    query = "senior developer"
    query_emb = model.encode(query)

    # Stage 1: Broad retrieval
    results = collection.query(
        query_embeddings=[query_emb.tolist()],
        n_results=10
    )

    print(f"   Stage 1: Retrieved {len(results['ids'][0])} candidates")

    # Stage 2: Re-rank by additional criteria
    reranked = []
    for chunk, metadata, distance in zip(results['documents'][0],
                                         results['metadatas'][0],
                                         results['distances'][0]):
        # Example: boost chunks from certain sources
        score = 1 - distance
        if metadata['chunk_index'] == 0:  # First chunk often has overview
            score *= 1.2

        reranked.append((chunk, metadata, score))

    reranked.sort(key=lambda x: x[2], reverse=True)

    print(f"   Stage 2: Re-ranked and selected top 3")
    for i, (_, metadata, score) in enumerate(reranked[:3], 1):
        print(
            f"     {i}. {metadata['source']} chunk {metadata['chunk_index']} (score: {score:.4f})")

    print()

    # Pattern 4: Parent Document Retrieval
    print("4. PARENT DOCUMENT RETRIEVAL")
    print("   • Embed small chunks for precision")
    print("   • Return larger parent sections for context\n")

    print("   Example:")
    print("     Search: 'Python experience'")
    print("     Match: Small chunk mentioning Python")
    print("     Return: Entire 'Requirements' section\n")

    print("   ✓ Best of both: precise matching + full context\n")

    print("💡 More Advanced Patterns:")
    print("   • Self-querying: Let LLM generate metadata filters")
    print("   • Multi-query: Generate diverse search queries")
    print("   • Fusion: Combine results from multiple strategies")
    print("   • Iterative retrieval: Retrieve → Generate → Retrieve again")
    print("   • Agentic RAG: LLM decides when/what to retrieve\n")


def lesson_5_rag_challenges():
    """Lesson 5: RAG challenges and solutions"""
    print_section("Lesson 5: RAG Challenges and Solutions")

    print("Common challenges and how to address them:\n")

    challenges = [
        {
            "challenge": "Lost in the middle",
            "description": "LLMs pay less attention to middle of context",
            "solutions": [
                "Put most relevant chunks at beginning/end",
                "Reduce number of retrieved chunks",
                "Use reranking to order by relevance"
            ]
        },
        {
            "challenge": "Context window limits",
            "description": "LLM has maximum token limit",
            "solutions": [
                "Retrieve fewer but more relevant chunks",
                "Summarize retrieved content",
                "Use longer context models (GPT-4, Claude)",
                "Implement iterative retrieval"
            ]
        },
        {
            "challenge": "Retrieval noise",
            "description": "Retrieved chunks may not be relevant",
            "solutions": [
                "Set minimum similarity threshold",
                "Use hybrid search (semantic + keyword)",
                "Implement reranking",
                "Fine-tune embedding model on your data"
            ]
        },
        {
            "challenge": "Chunking artifacts",
            "description": "Important info split across chunks",
            "solutions": [
                "Use overlapping chunks",
                "Try different chunk sizes",
                "Use semantic chunking",
                "Return parent document context"
            ]
        },
        {
            "challenge": "Stale information",
            "description": "Documents become outdated",
            "solutions": [
                "Regular re-indexing schedule",
                "Track document versions",
                "Add timestamp metadata",
                "Incremental updates"
            ]
        },
        {
            "challenge": "Multi-hop reasoning",
            "description": "Answer requires info from multiple sources",
            "solutions": [
                "Iterative retrieval",
                "Graph-based retrieval",
                "Agentic approach",
                "Retrieve more chunks"
            ]
        }
    ]

    for i, item in enumerate(challenges, 1):
        print(f"{i}. {item['challenge'].upper()}")
        print(f"   Problem: {item['description']}")
        print("   Solutions:")
        for sol in item['solutions']:
            print(f"     • {sol}")
        print()

    print("💡 RAG Best Practices Summary:")
    print("   ✓ Test different chunking strategies")
    print("   ✓ Monitor retrieval quality")
    print("   ✓ Use metadata effectively")
    print("   ✓ Implement logging and analytics")
    print("   ✓ Have fallbacks for low-quality retrievals")
    print("   ✓ Consider hybrid approaches")
    print("   ✓ Continuously evaluate and improve\n")


def lesson_6_rag_evaluation():
    """Lesson 6: Evaluating RAG systems"""
    print_section("Lesson 6: Evaluating RAG Systems")

    print("How to measure RAG system performance:\n")

    print("1. RETRIEVAL METRICS")
    print()

    metrics_retrieval = [
        ["Precision@k", "% of retrieved docs that are relevant", "Quality"],
        ["Recall@k", "% of relevant docs that were retrieved", "Coverage"],
        ["MRR", "Mean Reciprocal Rank - position of first relevant", "Ranking"],
        ["NDCG", "Normalized Discounted Cumulative Gain", "Ranking"]
    ]

    print(tabulate(metrics_retrieval,
                   headers=['Metric', 'Description', 'Measures'],
                   tablefmt='grid'))

    print("\n2. GENERATION METRICS")
    print()

    metrics_generation = [
        ["Faithfulness", "Answer grounded in retrieved context", "Hallucination"],
        ["Answer Relevance", "Answer addresses the question", "Quality"],
        ["Context Relevance", "Retrieved docs are relevant", "Retrieval"],
        ["BLEU/ROUGE", "Similarity to reference answers", "Accuracy"]
    ]

    print(tabulate(metrics_generation,
                   headers=['Metric', 'Description', 'Measures'],
                   tablefmt='grid'))

    print("\n3. END-TO-END EVALUATION")
    print()
    print("   Create test sets with:")
    print("     • Questions")
    print("     • Expected answers (gold standard)")
    print("     • Relevant documents")
    print()
    print("   Measure:")
    print("     • Retrieval accuracy")
    print("     • Answer accuracy")
    print("     • Latency")
    print("     • Cost per query\n")

    print("4. HUMAN EVALUATION")
    print()
    print("   Have humans rate:")
    print("     • Answer correctness (1-5)")
    print("     • Answer completeness (1-5)")
    print("     • Relevance (1-5)")
    print("     • Helpfulness (1-5)\n")

    print("💡 Evaluation Tools:")
    print("   • RAGAS: RAG Assessment framework")
    print("   • TruLens: Observability for LLM apps")
    print("   • LangSmith: LangChain debugging")
    print("   • Custom test harnesses\n")

    print("💡 Continuous Monitoring:")
    print("   ✓ Log all queries and responses")
    print("   ✓ Track retrieval scores")
    print("   ✓ Monitor latency")
    print("   ✓ Collect user feedback")
    print("   ✓ A/B test improvements")
    print("   ✓ Regular offline evaluation\n")


def run_module():
    """Run all lessons in Module 6"""
    print(f"\n{Fore.GREEN}{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "MODULE 6: RETRIEVAL AUGMENTED GENERATION (RAG)".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}{Style.RESET_ALL}\n")

    print("This module teaches you about RAG - the foundation of modern AI apps.\n")

    # Run lessons
    lesson_1_what_is_rag()
    lesson_2_document_chunking()
    lesson_3_building_rag_system()
    lesson_4_advanced_rag_patterns()
    lesson_5_rag_challenges()
    lesson_6_rag_evaluation()

    print(f"\n{Fore.GREEN}{'='*80}")
    print("MODULE 6 COMPLETE! ✓")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    print("📚 What you learned:")
    print("  ✓ RAG architecture and motivation")
    print("  ✓ Document chunking strategies")
    print("  ✓ Building a complete RAG system")
    print("  ✓ Advanced patterns: query expansion, HyDE, reranking")
    print("  ✓ Common challenges and solutions")
    print("  ✓ Evaluation metrics and monitoring\n")

    print("🎉 CONGRATULATIONS!")
    print("   You've completed the Vector Database Tutorial!")
    print("   You now understand vectors, embeddings, vector databases, and RAG!\n")


if __name__ == "__main__":
    run_module()
