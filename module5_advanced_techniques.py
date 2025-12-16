"""
Module 5: Advanced Vector Database Techniques
=============================================

This module covers advanced querying, filtering, and optimization techniques
for vector databases.

Learning Objectives:
- Master complex metadata filtering
- Implement hybrid search (semantic + keyword)
- Use multiple collections effectively
- Optimize query performance
- Handle real-world scenarios
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from colorama import Fore, Style
from pathlib import Path
from tabulate import tabulate
import re
from collections import Counter


def print_section(title):
    """Helper function to print section headers"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{Style.RESET_ALL}\n")


def load_job_ads():
    """Load job advertisements with enhanced metadata extraction"""
    corpus_path = Path("example_corpus")
    job_ads = {}

    for file_path in sorted(corpus_path.glob("job_ad_*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

            # Extract metadata
            metadata = {
                "filename": file_path.stem,
                "word_count": len(content.split()),
                "char_count": len(content),
                "has_remote": "remote" in content.lower(),
                "has_python": "python" in content.lower(),
                "has_javascript": "javascript" in content.lower() or "js" in content.lower(),
                "has_java": " java " in content.lower(),
                "experience_level": extract_experience_level(content),
            }

            job_ads[file_path.stem] = {
                "content": content,
                "metadata": metadata
            }

    return job_ads


def extract_experience_level(text):
    """Extract experience level from job ad"""
    text_lower = text.lower()

    if any(word in text_lower for word in ["senior", "sr.", "lead", "principal"]):
        return "senior"
    elif any(word in text_lower for word in ["junior", "jr.", "entry", "graduate"]):
        return "junior"
    elif "mid-level" in text_lower or "intermediate" in text_lower:
        return "mid"
    else:
        return "unknown"


def lesson_1_complex_filtering():
    """Lesson 1: Complex metadata filtering"""
    print_section("Lesson 1: Advanced Metadata Filtering")

    # Setup
    job_ads = load_job_ads()

    client = chromadb.PersistentClient(path="./chroma_db")

    # Create new collection with rich metadata
    try:
        client.delete_collection("jobs_advanced")
    except:
        pass

    collection = client.create_collection("jobs_advanced")

    print("Loading job ads with rich metadata...")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    ids = []
    documents = []
    metadatas = []

    for job_id, job_data in job_ads.items():
        ids.append(job_id)
        documents.append(job_data["content"])
        metadatas.append(job_data["metadata"])

    embeddings = model.encode(documents, show_progress_bar=True)
    print()

    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"✓ Added {len(ids)} jobs with metadata\n")

    # Show sample metadata
    sample = collection.get(limit=1, include=["metadatas"])
    print("Sample metadata structure:")
    print(f"  {sample['metadatas'][0]}\n")

    # Complex queries
    print("1. SIMPLE FILTER: Find Python jobs")
    query_embedding = model.encode(["developer position"])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={"has_python": True}
    )

    print(f"  Found {len(results['ids'][0])} Python jobs:")
    for job_id in results['ids'][0]:
        print(f"    • {job_id}")

    print("\n2. COMBINED FILTER: Senior Python developers")
    query_embedding = model.encode(["software engineer"])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={
            "$and": [
                {"has_python": True},
                {"experience_level": "senior"}
            ]
        }
    )

    print(f"  Found {len(results['ids'][0])} senior Python jobs:")
    for job_id, metadata in zip(results['ids'][0], results['metadatas'][0]):
        print(f"    • {job_id} - {metadata['experience_level']}")

    print("\n3. OR FILTER: Python OR JavaScript jobs")
    query_embedding = model.encode(["web developer"])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={
            "$or": [
                {"has_python": True},
                {"has_javascript": True}
            ]
        }
    )

    print(f"  Found {len(results['ids'][0])} Python/JS jobs:")
    for job_id, metadata in zip(results['ids'][0], results['metadatas'][0]):
        techs = []
        if metadata['has_python']:
            techs.append("Python")
        if metadata['has_javascript']:
            techs.append("JavaScript")
        print(f"    • {job_id} - {', '.join(techs)}")

    print("\n4. RANGE FILTER: Jobs with 100-300 words")
    query_embedding = model.encode(["job opportunity"])[0].tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        where={
            "$and": [
                {"word_count": {"$gte": 100}},
                {"word_count": {"$lte": 300}}
            ]
        }
    )

    print(f"  Found {len(results['ids'][0])} jobs:")
    for job_id, metadata in zip(results['ids'][0], results['metadatas'][0]):
        print(f"    • {job_id} - {metadata['word_count']} words")

    print("\n💡 Supported Operators:")
    print("   $eq, $ne - Equal, Not equal")
    print("   $gt, $gte, $lt, $lte - Comparison operators")
    print("   $and, $or - Logical operators")
    print("   $in, $nin - List membership\n")

    return client, collection


def lesson_2_hybrid_search():
    """Lesson 2: Hybrid search (semantic + keyword)"""
    print_section("Lesson 2: Hybrid Search")

    print("Hybrid search combines:")
    print("  1. Semantic search (embeddings)")
    print("  2. Keyword search (exact matches)")
    print("  3. Metadata filtering\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("jobs_advanced")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Get all documents for keyword search
    all_docs = collection.get(include=["documents", "metadatas"])

    def hybrid_search(query, keyword=None, filters=None, n_results=3):
        """Perform hybrid search"""
        print(f"  Query: '{query}'")
        if keyword:
            print(f"  Keyword: '{keyword}'")
        if filters:
            print(f"  Filters: {filters}")
        print()

        # Step 1: Semantic search
        query_embedding = model.encode(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 3,  # Get more for filtering
            where=filters
        )

        # Step 2: Keyword filtering if specified
        if keyword:
            filtered_ids = []
            filtered_distances = []
            filtered_docs = []
            filtered_meta = []

            keyword_lower = keyword.lower()

            for i, (doc_id, doc, dist, meta) in enumerate(zip(
                results['ids'][0],
                results['documents'][0],
                results['distances'][0],
                results['metadatas'][0]
            )):
                if keyword_lower in doc.lower():
                    filtered_ids.append(doc_id)
                    filtered_distances.append(dist)
                    filtered_docs.append(doc)
                    filtered_meta.append(meta)

            # Limit to n_results
            final_results = {
                'ids': [filtered_ids[:n_results]],
                'distances': [filtered_distances[:n_results]],
                'documents': [filtered_docs[:n_results]],
                'metadatas': [filtered_meta[:n_results]]
            }
        else:
            final_results = {
                'ids': [results['ids'][0][:n_results]],
                'distances': [results['distances'][0][:n_results]],
                'documents': [results['documents'][0][:n_results]],
                'metadatas': [results['metadatas'][0][:n_results]]
            }

        return final_results

    # Example 1: Pure semantic
    print("1. PURE SEMANTIC SEARCH")
    results = hybrid_search("software development role")

    for i, (job_id, dist) in enumerate(zip(results['ids'][0], results['distances'][0]), 1):
        print(f"    {i}. {job_id} (distance: {dist:.4f})")

    # Example 2: Semantic + keyword
    print("\n2. SEMANTIC + KEYWORD")
    results = hybrid_search("developer position", keyword="python")

    for i, (job_id, dist) in enumerate(zip(results['ids'][0], results['distances'][0]), 1):
        print(f"    {i}. {job_id} (distance: {dist:.4f})")

    # Example 3: Semantic + metadata + keyword
    print("\n3. SEMANTIC + METADATA + KEYWORD")
    results = hybrid_search(
        "engineering role",
        keyword="remote",
        filters={"has_python": True}
    )

    for i, (job_id, dist, meta) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        print(f"    {i}. {job_id} (distance: {dist:.4f})")
        print(
            f"       Python: {meta['has_python']}, Remote: {meta['has_remote']}")

    print("\n💡 Hybrid Search Benefits:")
    print("   ✓ Combines semantic understanding with exact matching")
    print("   ✓ More precise results")
    print("   ✓ Better user control")
    print("   ✓ Can enforce business rules via filters\n")


def lesson_3_reranking():
    """Lesson 3: Re-ranking search results"""
    print_section("Lesson 3: Re-ranking Search Results")

    print("Re-ranking improves initial results using additional signals:\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("jobs_advanced")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    query = "software developer"
    print(f"Query: '{query}'\n")

    # Initial retrieval
    query_embedding = model.encode(query)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=10
    )

    print("INITIAL RESULTS (by semantic similarity):")
    print("-" * 80)

    initial_data = []
    for i, (job_id, dist, meta) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    ), 1):
        initial_data.append([i, job_id, f"{dist:.4f}", meta['word_count']])

    print(tabulate(initial_data,
                   headers=['Rank', 'Job ID', 'Distance', 'Words'],
                   tablefmt='grid'))

    # Re-rank by multiple factors
    print("\nRE-RANKING with composite score:")
    print("  Score = (1 - distance) * 0.7 + (word_count/500) * 0.3\n")

    reranked = []
    for job_id, dist, meta in zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    ):
        # Composite score
        semantic_score = 1 - dist  # Lower distance = higher score
        # Prefer longer (more detailed)
        length_score = min(meta['word_count'] / 500, 1.0)

        composite_score = semantic_score * 0.7 + length_score * 0.3

        reranked.append({
            'id': job_id,
            'semantic': semantic_score,
            'length': length_score,
            'composite': composite_score,
            'word_count': meta['word_count']
        })

    reranked.sort(key=lambda x: x['composite'], reverse=True)

    reranked_data = []
    for i, item in enumerate(reranked[:5], 1):
        reranked_data.append([
            i,
            item['id'],
            f"{item['composite']:.4f}",
            f"{item['semantic']:.4f}",
            f"{item['length']:.4f}",
            item['word_count']
        ])

    print(tabulate(reranked_data,
                   headers=['Rank', 'Job ID', 'Final',
                            'Semantic', 'Length', 'Words'],
                   tablefmt='grid'))

    print("\n💡 Re-ranking Strategies:")
    print("   • Combine multiple signals (relevance + recency + popularity)")
    print("   • Use business rules (boost premium listings)")
    print("   • Apply personalization (user preferences)")
    print("   • Cross-encoder models for better accuracy")
    print("   • A/B test different ranking formulas\n")


def lesson_4_multi_collection_search():
    """Lesson 4: Searching across multiple collections"""
    print_section("Lesson 4: Multi-Collection Search")

    print("Organizing data into multiple collections:\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create separate collections for different job types
    job_ads = load_job_ads()

    # Categorize jobs
    tech_jobs = {}
    other_jobs = {}

    for job_id, job_data in job_ads.items():
        content_lower = job_data["content"].lower()
        if any(word in content_lower for word in ["developer", "engineer", "programming", "software"]):
            tech_jobs[job_id] = job_data
        else:
            other_jobs[job_id] = job_data

    print(f"Categorized jobs:")
    print(f"  Tech jobs: {len(tech_jobs)}")
    print(f"  Other jobs: {len(other_jobs)}\n")

    # Create collections
    collections_to_create = [
        ("tech_jobs", tech_jobs),
        ("other_jobs", other_jobs)
    ]

    created_collections = {}

    for col_name, jobs_dict in collections_to_create:
        if len(jobs_dict) == 0:
            continue

        try:
            client.delete_collection(col_name)
        except:
            pass

        collection = client.create_collection(col_name)

        ids = list(jobs_dict.keys())
        documents = [job_data["content"] for job_data in jobs_dict.values()]
        metadatas = [job_data["metadata"] for job_data in jobs_dict.values()]

        embeddings = model.encode(documents)

        collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )

        created_collections[col_name] = collection
        print(f"✓ Created collection '{col_name}' with {len(ids)} documents")

    print("\nSearching across multiple collections:\n")

    query = "python developer"
    query_embedding = model.encode(query)

    # Search each collection
    all_results = []

    for col_name, collection in created_collections.items():
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=3
        )

        for job_id, dist in zip(results['ids'][0], results['distances'][0]):
            all_results.append({
                'collection': col_name,
                'job_id': job_id,
                'distance': dist
            })

    # Combine and sort
    all_results.sort(key=lambda x: x['distance'])

    print(f"Results for '{query}' across all collections:")
    print("-" * 80)

    for i, result in enumerate(all_results[:5], 1):
        print(f"  {i}. [{result['collection']}] {result['job_id']}")
        print(f"     Distance: {result['distance']:.4f}")

    print("\n💡 Multi-Collection Benefits:")
    print("   ✓ Organize by document type or category")
    print("   ✓ Different embedding models per collection")
    print("   ✓ Separate access control")
    print("   ✓ Independent scaling")
    print("   ✓ Easier data management\n")


def lesson_5_query_optimization():
    """Lesson 5: Query optimization techniques"""
    print_section("Lesson 5: Query Optimization")

    print("Techniques to make queries faster and more efficient:\n")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("jobs_advanced")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("1. LIMIT RESULTS (n_results)")
    print("   Don't fetch more than you need\n")

    query = "developer"
    query_embedding = model.encode(query)

    import time

    for n in [5, 10, 20]:
        start = time.time()
        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n
        )
        elapsed = (time.time() - start) * 1000
        print(f"   n_results={n}: {elapsed:.2f}ms")

    print("\n2. INCLUDE ONLY WHAT YOU NEED")
    print("   Specify include parameter\n")

    # Full data
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=["documents", "metadatas", "distances", "embeddings"]
    )
    time_full = (time.time() - start) * 1000

    # IDs only
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        include=["distances"]
    )
    time_ids = (time.time() - start) * 1000

    print(f"   Full data: {time_full:.2f}ms")
    print(f"   IDs + distances only: {time_ids:.2f}ms")
    print(f"   Speedup: {time_full/time_ids:.1f}x")

    print("\n3. CACHE QUERY EMBEDDINGS")
    print("   Pre-compute embeddings for common queries\n")

    common_queries = [
        "python developer",
        "web developer",
        "software engineer"
    ]

    # Simulate caching
    query_cache = {}

    print("   Building cache...")
    for q in common_queries:
        query_cache[q] = model.encode(q)
        print(f"     Cached: '{q}'")

    print("\n   Cache hit is instant (no encoding needed)!")

    print("\n4. USE METADATA FILTERS EARLY")
    print("   Narrow search space before semantic search\n")

    # Without filter
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    time_no_filter = (time.time() - start) * 1000

    # With filter
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5,
        where={"has_python": True}
    )
    time_with_filter = (time.time() - start) * 1000

    print(f"   Without filter: {time_no_filter:.2f}ms")
    print(f"   With filter: {time_with_filter:.2f}ms")

    print("\n💡 Optimization Checklist:")
    print("   ✓ Use appropriate n_results")
    print("   ✓ Include only needed fields")
    print("   ✓ Cache frequent query embeddings")
    print("   ✓ Add metadata indexes")
    print("   ✓ Use filters to reduce search space")
    print("   ✓ Batch similar queries")
    print("   ✓ Monitor query performance\n")


def run_module():
    """Run all lessons in Module 5"""
    print(f"\n{Fore.GREEN}{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "MODULE 5: ADVANCED VECTOR DATABASE TECHNIQUES".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}{Style.RESET_ALL}\n")

    print("This module covers advanced techniques for vector databases.\n")

    # Run lessons
    lesson_1_complex_filtering()
    lesson_2_hybrid_search()
    lesson_3_reranking()
    lesson_4_multi_collection_search()
    lesson_5_query_optimization()

    print(f"\n{Fore.GREEN}{'='*80}")
    print("MODULE 5 COMPLETE! ✓")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    print("📚 What you learned:")
    print("  ✓ Complex metadata filtering with logical operators")
    print("  ✓ Hybrid search (semantic + keyword + filters)")
    print("  ✓ Re-ranking strategies for better results")
    print("  ✓ Multi-collection architecture")
    print("  ✓ Query optimization techniques")
    print("  ✓ Production-ready search patterns\n")


if __name__ == "__main__":
    run_module()
