"""
Module 3: Vector Similarity and Semantic Search
===============================================

This module teaches you how to use vector similarity to perform semantic search,
which is the foundation of modern search engines and retrieval systems.

Learning Objectives:
- Master different distance metrics (cosine, euclidean, dot product)
- Build a semantic search engine from scratch
- Understand ranking and retrieval
- Compare semantic search vs keyword search
- Implement search with our job ads corpus
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from colorama import Fore, Style
from pathlib import Path
from tabulate import tabulate
import time


def print_section(title):
    """Helper function to print section headers"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{Style.RESET_ALL}\n")


def load_job_ads():
    """Load job advertisements from example corpus"""
    corpus_path = Path("example_corpus")
    job_ads = {}

    for file_path in sorted(corpus_path.glob("job_ad_*.txt")):
        with open(file_path, 'r', encoding='utf-8') as f:
            job_ads[file_path.stem] = f.read()

    return job_ads


def lesson_1_distance_metrics():
    """Lesson 1: Understanding distance metrics"""
    print_section("Lesson 1: Distance Metrics for Similarity")

    print("There are several ways to measure vector similarity:\n")

    # Create example vectors
    v1 = np.array([1.0, 2.0, 3.0])
    v2 = np.array([1.5, 2.5, 3.5])
    v3 = np.array([-1.0, -2.0, -3.0])

    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Vector 3: {v3}\n")

    # 1. Cosine Similarity
    print("1. COSINE SIMILARITY (Most common for text)")
    print("   Measures the angle between vectors")
    print("   Range: -1 (opposite) to 1 (identical)")
    print("   Formula: cos(θ) = (A·B) / (||A|| × ||B||)\n")

    def cosine_sim(a, b):
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    cos_12 = cosine_sim(v1, v2)
    cos_13 = cosine_sim(v1, v3)

    print(f"   Cosine(v1, v2) = {cos_12:.4f} → Very similar direction")
    print(f"   Cosine(v1, v3) = {cos_13:.4f} → Opposite directions\n")

    # 2. Euclidean Distance
    print("2. EUCLIDEAN DISTANCE")
    print("   Straight-line distance between points")
    print("   Range: 0 (identical) to ∞")
    print("   Formula: √(Σ(ai - bi)²)\n")

    euclidean_12 = np.linalg.norm(v1 - v2)
    euclidean_13 = np.linalg.norm(v1 - v3)

    print(f"   Euclidean(v1, v2) = {euclidean_12:.4f}")
    print(f"   Euclidean(v1, v3) = {euclidean_13:.4f}\n")

    # 3. Dot Product
    print("3. DOT PRODUCT")
    print("   Direct multiplication and sum")
    print("   Range: -∞ to ∞")
    print("   Formula: Σ(ai × bi)\n")

    dot_12 = np.dot(v1, v2)
    dot_13 = np.dot(v1, v3)

    print(f"   Dot(v1, v2) = {dot_12:.4f}")
    print(f"   Dot(v1, v3) = {dot_13:.4f}\n")

    # Comparison
    print("💡 Which to use?")
    print("   ✓ COSINE: Best for text (ignores magnitude, focuses on direction)")
    print("   • Euclidean: Good for spatial data")
    print("   • Dot Product: Fast, but sensitive to magnitude\n")

    print("For semantic search with text embeddings, USE COSINE SIMILARITY!\n")

    return v1, v2, v3


def lesson_2_building_search_engine():
    """Lesson 2: Building a basic semantic search engine"""
    print_section("Lesson 2: Building a Semantic Search Engine")

    print("Let's build a simple but powerful search engine!\n")

    # Load data
    job_ads = load_job_ads()
    job_names = list(job_ads.keys())
    job_texts = list(job_ads.values())

    print(f"📚 Loaded {len(job_ads)} job advertisements\n")

    # Load model and create embeddings
    print("Step 1: Create embeddings for all documents")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("  → Encoding documents...")
    doc_embeddings = model.encode(job_texts, show_progress_bar=True)
    print(
        f"  ✓ Created {len(doc_embeddings)} embeddings ({doc_embeddings.shape[1]} dimensions)\n")

    # Search function
    def search(query, top_k=3):
        """Perform semantic search"""
        print(f"\n🔍 Query: '{query}'")
        print("  → Encoding query...")
        query_embedding = model.encode(query)

        print("  → Calculating similarities...")
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

        print("  → Ranking results...\n")
        # Get top k results
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append({
                'rank': rank,
                'document': job_names[idx],
                'similarity': similarities[idx],
                'preview': job_texts[idx][:150].replace('\n', ' ')
            })

        return results

    # Example searches
    print("Step 2: Perform semantic searches\n")

    queries = [
        "Python developer with machine learning experience",
        "Web developer position",
        "Software engineer role"
    ]

    for query in queries:
        results = search(query, top_k=3)

        print(f"Results for: '{query}'")
        print("-" * 80)
        for r in results:
            print(f"\n{r['rank']}. {r['document']}")
            print(f"   Similarity: {r['similarity']:.4f}")
            print(f"   Preview: {r['preview']}...")
        print()

    print("\n💡 Notice how the search understands MEANING, not just keywords!")
    print("   'Python developer' finds relevant jobs even without exact match.\n")

    return model, doc_embeddings, job_names, job_texts


def lesson_3_semantic_vs_keyword():
    """Lesson 3: Semantic search vs keyword search"""
    print_section("Lesson 3: Semantic Search vs Keyword Search")

    job_ads = load_job_ads()
    job_names = list(job_ads.keys())
    job_texts = list(job_ads.values())

    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(job_texts)

    print("Let's compare semantic search with traditional keyword search:\n")

    query = "machine learning engineer"

    # Keyword search (simple contains check)
    print(f"Query: '{query}'\n")
    print("=" * 80)
    print("KEYWORD SEARCH (traditional)")
    print("=" * 80)

    keyword_results = []
    query_words = set(query.lower().split())

    for name, text in zip(job_names, job_texts):
        text_lower = text.lower()
        # Count matching keywords
        matches = sum(1 for word in query_words if word in text_lower)
        if matches > 0:
            keyword_results.append((name, matches, text[:150]))

    keyword_results.sort(key=lambda x: x[1], reverse=True)

    if keyword_results:
        for rank, (name, matches, preview) in enumerate(keyword_results[:3], 1):
            print(f"\n{rank}. {name}")
            print(f"   Keyword matches: {matches}/{len(query_words)}")
            print(f"   {preview.replace(chr(10), ' ')}...")
    else:
        print("\n   No results found!")

    # Semantic search
    print(f"\n{'=' * 80}")
    print("SEMANTIC SEARCH (embedding-based)")
    print("=" * 80)

    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
    top_indices = np.argsort(similarities)[::-1][:3]

    for rank, idx in enumerate(top_indices, 1):
        print(f"\n{rank}. {job_names[idx]}")
        print(f"   Semantic similarity: {similarities[idx]:.4f}")
        print(f"   {job_texts[idx][:150].replace(chr(10), ' ')}...")

    print(f"\n{'=' * 80}\n")
    print("💡 Key Differences:")
    print("   ✗ Keyword search: Needs exact word matches")
    print("   ✓ Semantic search: Understands synonyms and related concepts")
    print("   ✓ Semantic search: Finds 'software engineer' when searching 'developer'")
    print("   ✓ Semantic search: Understands 'ML' relates to 'machine learning'\n")


def lesson_4_ranking_and_thresholds():
    """Lesson 4: Ranking strategies and similarity thresholds"""
    print_section("Lesson 4: Ranking and Similarity Thresholds")

    job_ads = load_job_ads()
    job_texts = list(job_ads.values())
    job_names = list(job_ads.keys())

    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(job_texts)

    print("Understanding similarity scores and setting thresholds:\n")

    query = "senior software developer"
    print(f"Query: '{query}'\n")

    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Show all results with scores
    results_with_scores = list(zip(job_names, similarities))
    results_with_scores.sort(key=lambda x: x[1], reverse=True)

    print("All documents ranked by similarity:")
    print("-" * 80)

    table_data = []
    for rank, (name, score) in enumerate(results_with_scores, 1):
        # Determine relevance
        if score > 0.5:
            relevance = "🟢 Highly Relevant"
        elif score > 0.3:
            relevance = "🟡 Somewhat Relevant"
        else:
            relevance = "🔴 Not Relevant"

        table_data.append([rank, name, f"{score:.4f}", relevance])

    print(tabulate(table_data, headers=['Rank', 'Document', 'Similarity', 'Relevance'],
                   tablefmt='grid'))

    print("\n💡 Similarity Score Guidelines:")
    print("   > 0.7  → Very strong match (almost identical meaning)")
    print("   0.5-0.7 → Good match (clearly relevant)")
    print("   0.3-0.5 → Moderate match (somewhat related)")
    print("   < 0.3  → Weak match (might not be relevant)\n")

    print("💡 Best Practices:")
    print("   ✓ Set minimum threshold (e.g., 0.3) to filter noise")
    print("   ✓ Return top-k results (e.g., top 5)")
    print("   ✓ Show scores to users so they can judge relevance")
    print("   ✓ Log low scores - might indicate missing content!\n")


def lesson_5_advanced_search_features():
    """Lesson 5: Advanced search features"""
    print_section("Lesson 5: Advanced Search Features")

    job_ads = load_job_ads()
    job_texts = list(job_ads.values())
    job_names = list(job_ads.keys())

    model = SentenceTransformer('all-MiniLM-L6-v2')
    doc_embeddings = model.encode(job_texts)

    print("Advanced features for production search engines:\n")

    # 1. Multi-query search
    print("1. MULTI-QUERY SEARCH")
    print("   Search with multiple related queries and combine results\n")

    queries = [
        "Python programming",
        "software development",
        "web applications"
    ]

    print(f"   Queries: {queries}\n")

    # Encode all queries
    query_embeddings = model.encode(queries)

    # Average strategy
    avg_query_embedding = np.mean(query_embeddings, axis=0)
    similarities_avg = cosine_similarity(
        [avg_query_embedding], doc_embeddings)[0]

    print("   Results using AVERAGED queries:")
    top_idx = np.argsort(similarities_avg)[::-1][0]
    print(
        f"   Top result: {job_names[top_idx]} (similarity: {similarities_avg[top_idx]:.4f})\n")

    # 2. Re-ranking
    print("2. RE-RANKING")
    print("   Initial retrieval → Apply additional filters/scoring\n")

    query = "developer position"
    query_embedding = model.encode(query)
    similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

    # Get initial results
    initial_results = [(i, sim) for i, sim in enumerate(similarities)]
    initial_results.sort(key=lambda x: x[1], reverse=True)

    print("   Initial top 3:")
    for rank, (idx, sim) in enumerate(initial_results[:3], 1):
        print(f"     {rank}. {job_names[idx]} (sim: {sim:.4f})")

    # Re-rank by preferring longer descriptions (more info)
    reranked = [(idx, sim * (1 + len(job_texts[idx])/10000))
                for idx, sim in initial_results]
    reranked.sort(key=lambda x: x[1], reverse=True)

    print("\n   After re-ranking (boosting longer descriptions):")
    for rank, (idx, score) in enumerate(reranked[:3], 1):
        print(f"     {rank}. {job_names[idx]} (score: {score:.4f})")

    print("\n3. QUERY EXPANSION")
    print("   Automatically expand user query with related terms\n")

    original_query = "ML"
    expansions = ["ML", "machine learning",
                  "artificial intelligence", "data science"]

    print(f"   Original: '{original_query}'")
    print(f"   Expanded: {expansions}")
    print("   → This improves recall!\n")

    print("💡 Production Tips:")
    print("   ✓ Use multi-query for complex information needs")
    print("   ✓ Re-rank with business logic (recency, popularity, etc.)")
    print("   ✓ Expand queries to improve recall")
    print("   ✓ Cache embeddings - don't recompute for every search!")
    print("   ✓ Use approximate nearest neighbor (ANN) for scale\n")


def lesson_6_performance_optimization():
    """Lesson 6: Performance and scaling"""
    print_section("Lesson 6: Performance Optimization")

    print("Making search fast and scalable:\n")

    job_ads = load_job_ads()
    job_texts = list(job_ads.values())

    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Timing embedding creation
    print("1. EMBEDDING CREATION PERFORMANCE")

    start = time.time()
    embeddings_single = [model.encode(text) for text in job_texts]
    time_single = time.time() - start

    start = time.time()
    embeddings_batch = model.encode(job_texts)
    time_batch = time.time() - start

    print(f"   One-by-one: {time_single:.3f}s")
    print(f"   Batch: {time_batch:.3f}s")
    print(f"   Speedup: {time_single/time_batch:.1f}x faster!\n")

    print("   💡 Always batch encode when possible!\n")

    # Timing search
    print("2. SEARCH PERFORMANCE")

    query = "software engineer"
    query_embedding = model.encode(query)

    # Brute force
    start = time.time()
    similarities = cosine_similarity([query_embedding], embeddings_batch)[0]
    top_5 = np.argsort(similarities)[::-1][:5]
    time_search = time.time() - start

    print(f"   Brute force search: {time_search*1000:.2f}ms")
    print(f"   → Fine for small datasets (< 10,000 documents)")
    print(f"   → For millions of documents, use vector databases!\n")

    print("3. SCALING TO PRODUCTION")
    print("   ")
    print("   For large-scale applications:")
    print("   ✓ Use vector databases (ChromaDB, Pinecone, Weaviate)")
    print("   ✓ Approximate Nearest Neighbor (ANN) algorithms")
    print("   ✓ Cache frequently queried embeddings")
    print("   ✓ Use GPU for embedding creation if available")
    print("   ✓ Implement pagination for results")
    print("   ✓ Consider quantization to reduce memory\n")

    print("Next module: We'll learn about vector databases for scale!\n")


def run_module():
    """Run all lessons in Module 3"""
    print(f"\n{Fore.GREEN}{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "MODULE 3: VECTOR SIMILARITY AND SEMANTIC SEARCH".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}{Style.RESET_ALL}\n")

    print("This module teaches you how to build semantic search engines.\n")

    # Run lessons
    lesson_1_distance_metrics()
    lesson_2_building_search_engine()
    lesson_3_semantic_vs_keyword()
    lesson_4_ranking_and_thresholds()
    lesson_5_advanced_search_features()
    lesson_6_performance_optimization()

    print(f"\n{Fore.GREEN}{'='*80}")
    print("MODULE 3 COMPLETE! ✓")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    print("📚 What you learned:")
    print("  ✓ Different distance metrics (cosine, euclidean, dot product)")
    print("  ✓ How to build a semantic search engine from scratch")
    print("  ✓ Semantic search vs keyword search")
    print("  ✓ Ranking strategies and similarity thresholds")
    print("  ✓ Advanced features: multi-query, re-ranking, query expansion")
    print("  ✓ Performance optimization and scaling considerations\n")


if __name__ == "__main__":
    run_module()
