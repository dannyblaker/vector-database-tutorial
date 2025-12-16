"""
Module 2: Text Embeddings
=========================

This module teaches you how to convert text into vectors (embeddings) that
capture semantic meaning.

Learning Objectives:
- Understand what text embeddings are
- Learn different methods to create embeddings
- Use pre-trained transformer models
- Process real job ad data into embeddings
- Understand embedding dimensions and properties
"""

import numpy as np
from sentence_transformers import SentenceTransformer
from colorama import Fore, Style
import os
import time
from pathlib import Path


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


def lesson_1_what_are_embeddings():
    """Lesson 1: What are text embeddings?"""
    print_section("Lesson 1: What Are Text Embeddings?")

    print("Text embeddings are vector representations of text that capture semantic meaning.")
    print("Similar texts have similar embeddings (high cosine similarity).\n")

    print("Why do we need embeddings?")
    print("  ❌ Computers can't understand 'cat' and 'dog' are related")
    print("  ✓ With embeddings, 'cat' and 'dog' have similar vectors!")
    print("  ✓ We can do math with words: king - man + woman ≈ queen\n")

    examples = [
        "I love machine learning",
        "I enjoy artificial intelligence",
        "The weather is nice today"
    ]

    print("Example sentences:")
    for i, ex in enumerate(examples, 1):
        print(f"  {i}. '{ex}'")

    print("\nSentences 1 and 2 are semantically similar (both about AI/ML)")
    print("Sentence 3 is different (about weather)")
    print("\nEmbeddings will reflect this similarity! Let's see how...\n")

    return examples


def lesson_2_simple_embeddings():
    """Lesson 2: Simple embedding methods (bag of words, TF-IDF)"""
    print_section("Lesson 2: Simple Embedding Methods")

    print("Before neural networks, we used simpler methods:\n")

    # Bag of Words example
    print("1. Bag of Words (Count-based)")
    print("   Each dimension = one unique word in vocabulary\n")

    sentences = [
        "I love cats",
        "I love dogs",
        "Cats and dogs"
    ]

    # Build vocabulary
    vocab = sorted(set(' '.join(sentences).lower().split()))
    print(f"   Vocabulary: {vocab}")
    print(f"   Vocabulary size: {len(vocab)}\n")

    # Create bag of words vectors
    print("   Vector representations:")
    for sent in sentences:
        words = sent.lower().split()
        vector = [words.count(word) for word in vocab]
        print(f"   '{sent}' → {vector}")

    print("\n   ⚠️  Limitations:")
    print("      - Ignores word order: 'dog bites man' = 'man bites dog'")
    print("      - Ignores semantics: 'cat' and 'kitten' are unrelated")
    print("      - High dimensional (one dimension per word)\n")

    print("2. Modern Solution: Neural Embeddings")
    print("   - Pre-trained on massive text corpora")
    print("   - Capture semantic relationships")
    print("   - Fixed dimensions (e.g., 384, 768)")
    print("   - Understand context!\n")


def lesson_3_sentence_transformers():
    """Lesson 3: Using Sentence Transformers"""
    print_section("Lesson 3: Modern Embeddings with Sentence Transformers")

    print("We'll use the Sentence-BERT (SBERT) model.")
    print("This is a state-of-the-art model for generating sentence embeddings.\n")

    print("Loading model: all-MiniLM-L6-v2")
    print("  - Size: ~80 MB")
    print("  - Embedding dimensions: 384")
    print("  - Fast and efficient")
    print("  - Trained on 1B+ sentence pairs\n")

    start_time = time.time()
    model = SentenceTransformer('all-MiniLM-L6-v2')
    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f} seconds\n")

    # Encode simple sentences
    sentences = [
        "I love machine learning",
        "I enjoy artificial intelligence",
        "The weather is nice today"
    ]

    print("Encoding sentences...")
    embeddings = model.encode(sentences)

    print(f"\nShape of embeddings: {embeddings.shape}")
    print(f"  - {embeddings.shape[0]} sentences")
    print(f"  - {embeddings.shape[1]} dimensions each\n")

    print("First sentence embedding (first 10 dimensions):")
    print(f"  {embeddings[0][:10]}\n")

    # Calculate similarities
    from sklearn.metrics.pairwise import cosine_similarity

    similarities = cosine_similarity(embeddings)

    print("Cosine similarities between sentences:")
    for i in range(len(sentences)):
        for j in range(i+1, len(sentences)):
            sim = similarities[i][j]
            print(f"  Sentence {i+1} ↔ Sentence {j+1}: {sim:.4f}")
            print(f"    '{sentences[i]}'")
            print(f"    '{sentences[j]}'")
            print()

    print("💡 Notice: Sentences 1 and 2 have HIGH similarity (both about AI/ML)")
    print("   Sentences with weather have LOW similarity to AI/ML sentences!\n")

    return model, embeddings


def lesson_4_embedding_job_ads():
    """Lesson 4: Creating embeddings for our job ads"""
    print_section("Lesson 4: Embedding Real Job Advertisements")

    # Load job ads
    job_ads = load_job_ads()
    print(f"Loaded {len(job_ads)} job advertisements\n")

    # Show preview
    for i, (name, content) in enumerate(list(job_ads.items())[:2], 1):
        preview = content[:200].replace('\n', ' ')
        print(f"{i}. {name}:")
        print(f"   {preview}...\n")

    # Load model
    print("Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Create embeddings
    print("\nCreating embeddings for all job ads...")
    job_texts = list(job_ads.values())
    job_names = list(job_ads.keys())

    start_time = time.time()
    embeddings = model.encode(job_texts, show_progress_bar=True)
    encode_time = time.time() - start_time

    print(
        f"\n✓ Created {len(embeddings)} embeddings in {encode_time:.2f} seconds")
    print(f"  Shape: {embeddings.shape}\n")

    # Calculate pairwise similarities
    from sklearn.metrics.pairwise import cosine_similarity
    similarities = cosine_similarity(embeddings)

    print("Most similar job ad pairs:")
    # Get top pairs
    pairs = []
    for i in range(len(similarities)):
        for j in range(i+1, len(similarities)):
            pairs.append((i, j, similarities[i][j]))

    pairs.sort(key=lambda x: x[2], reverse=True)

    for i, j, sim in pairs[:3]:
        print(f"\n  {job_names[i]} ↔ {job_names[j]}")
        print(f"  Similarity: {sim:.4f}")

    print("\n💡 High similarity means these jobs require similar skills/experience!\n")

    return model, embeddings, job_names, job_texts


def lesson_5_embedding_properties():
    """Lesson 5: Properties and characteristics of embeddings"""
    print_section("Lesson 5: Understanding Embedding Properties")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Let's explore how embeddings capture different aspects:\n")

    # Semantic similarity
    print("1. SEMANTIC SIMILARITY")
    pairs = [
        ("software developer", "programmer"),
        ("doctor", "physician"),
        ("car", "automobile"),
        ("happy", "joyful"),
    ]

    for word1, word2 in pairs:
        emb1 = model.encode(word1)
        emb2 = model.encode(word2)
        sim = np.dot(emb1, emb2) / \
            (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"   '{word1}' ↔ '{word2}': {sim:.4f}")

    # Context matters
    print("\n2. CONTEXT MATTERS")
    sentences = [
        "The bank of the river",
        "The bank gave me a loan",
        "I need to bank some money"
    ]

    embeddings = model.encode(sentences)
    print(f"   Same word 'bank' in different contexts:")
    for i, sent in enumerate(sentences):
        print(f"   {i+1}. '{sent}'")

    from sklearn.metrics.pairwise import cosine_similarity
    sims = cosine_similarity(embeddings)
    print(
        f"\n   Similarity between sentence 1 (river) and 2 (loan): {sims[0][1]:.4f}")
    print(
        f"   Similarity between sentence 2 (loan) and 3 (money): {sims[1][2]:.4f}")
    print("   → Financial contexts are more similar!\n")

    # Length invariance
    print("3. LENGTH INVARIANCE")
    texts = [
        "AI",
        "Artificial Intelligence",
        "Artificial Intelligence is transforming technology"
    ]

    embeddings = model.encode(texts)
    print("   All these have the same dimensionality:")
    for text, emb in zip(texts, embeddings):
        print(f"   '{text[:50]}...' → shape {emb.shape}")

    print("\n💡 Key Insights:")
    print("   ✓ Embeddings capture semantic meaning, not just word matching")
    print("   ✓ Context matters - same word in different contexts → different embeddings")
    print("   ✓ Length doesn't matter - short and long texts → same dimensions\n")


def lesson_6_choosing_embedding_models():
    """Lesson 6: Choosing the right embedding model"""
    print_section("Lesson 6: Choosing Embedding Models")

    print("Different models for different needs:\n")

    models_info = [
        {
            "name": "all-MiniLM-L6-v2",
            "dimensions": 384,
            "speed": "Very Fast",
            "quality": "Good",
            "use_case": "General purpose, speed matters"
        },
        {
            "name": "all-mpnet-base-v2",
            "dimensions": 768,
            "speed": "Medium",
            "quality": "Excellent",
            "use_case": "Best quality, some speed tradeoff"
        },
        {
            "name": "multi-qa-MiniLM-L6-cos-v1",
            "dimensions": 384,
            "speed": "Fast",
            "quality": "Good",
            "use_case": "Question-answering tasks"
        },
        {
            "name": "paraphrase-multilingual-MiniLM-L12-v2",
            "dimensions": 384,
            "speed": "Fast",
            "quality": "Good",
            "use_case": "Multilingual applications"
        }
    ]

    for model_info in models_info:
        print(f"📦 {model_info['name']}")
        print(f"   Dimensions: {model_info['dimensions']}")
        print(f"   Speed: {model_info['speed']}")
        print(f"   Quality: {model_info['quality']}")
        print(f"   Best for: {model_info['use_case']}\n")

    print("💡 Trade-offs:")
    print("   - More dimensions = Better quality BUT slower & more storage")
    print("   - Smaller models = Faster BUT may miss nuances")
    print("   - Choose based on your specific needs!\n")

    print("For this tutorial, we use all-MiniLM-L6-v2 (fast + good quality).\n")


def run_module():
    """Run all lessons in Module 2"""
    print(f"\n{Fore.GREEN}{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "MODULE 2: TEXT EMBEDDINGS".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}{Style.RESET_ALL}\n")

    print("This module teaches you how to convert text into semantic vectors.\n")

    # Run lessons
    lesson_1_what_are_embeddings()
    lesson_2_simple_embeddings()
    lesson_3_sentence_transformers()
    lesson_4_embedding_job_ads()
    lesson_5_embedding_properties()
    lesson_6_choosing_embedding_models()

    print(f"\n{Fore.GREEN}{'='*80}")
    print("MODULE 2 COMPLETE! ✓")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    print("📚 What you learned:")
    print("  ✓ Text embeddings represent semantic meaning as vectors")
    print("  ✓ Modern models like Sentence-BERT are much better than bag-of-words")
    print("  ✓ How to use SentenceTransformer to encode text")
    print("  ✓ Embeddings capture context and semantic relationships")
    print("  ✓ How to choose the right embedding model for your needs\n")


if __name__ == "__main__":
    run_module()
