"""
Module 4: Vector Databases
==========================

This module introduces vector databases - specialized databases designed for
storing and querying high-dimensional vectors efficiently.

Learning Objectives:
- Understand why vector databases are needed
- Learn ChromaDB basics
- Store and query embeddings
- Work with collections
- Understand persistence and data management
- Compare vector databases to traditional databases
"""

import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
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


def lesson_1_why_vector_databases():
    """Lesson 1: Why do we need vector databases?"""
    print_section("Lesson 1: Why Vector Databases?")

    print("So far, we've stored embeddings in memory (numpy arrays).")
    print("This works for small datasets, but what about production?\n")

    print("❌ PROBLEMS WITH IN-MEMORY STORAGE:")
    print("   • Lost on restart - no persistence")
    print("   • Limited by RAM - can't scale to millions of vectors")
    print("   • Slow brute-force search - O(n) complexity")
    print("   • No concurrent access - can't share across services")
    print("   • No metadata filtering - can't filter by date, category, etc.\n")

    print("✓ VECTOR DATABASE SOLUTIONS:")
    print("   • Persistent storage on disk")
    print("   • Efficient indexing (HNSW, IVF, etc.)")
    print("   • Fast approximate nearest neighbor (ANN) search")
    print("   • Metadata filtering and hybrid search")
    print("   • Scalable to billions of vectors")
    print("   • Built-in embedding functions")
    print("   • Multi-user support with ACID properties\n")

    print("📊 POPULAR VECTOR DATABASES:")

    databases = [
        ["ChromaDB", "Python-native, embedded",
            "Development & Production", "Apache 2.0"],
        ["Pinecone", "Cloud-native, managed", "Production at scale", "Commercial"],
        ["Weaviate", "GraphQL API, ML-native", "Production", "BSD 3-Clause"],
        ["Milvus", "High performance, distributed",
            "Large-scale production", "Apache 2.0"],
        ["Qdrant", "Rust-based, fast", "Production", "Apache 2.0"],
        ["FAISS", "Facebook, library not DB", "Research & prototyping", "MIT"],
    ]

    print(tabulate(databases,
                   headers=['Database', 'Description', 'Best For', 'License'],
                   tablefmt='grid'))

    print("\n💡 For this tutorial, we use ChromaDB because:")
    print("   ✓ Easy to get started (no separate server)")
    print("   ✓ Pythonic API")
    print("   ✓ Excellent for learning and prototyping")
    print("   ✓ Can scale to production\n")


def lesson_2_chromadb_basics():
    """Lesson 2: ChromaDB basics"""
    print_section("Lesson 2: ChromaDB Basics")

    print("Let's start with ChromaDB!\n")

    # Initialize client
    print("Step 1: Initialize ChromaDB client")
    print("  → Creating persistent client...")

    client = chromadb.PersistentClient(path="./chroma_db")

    print("  ✓ Client created with persistent storage at ./chroma_db\n")

    # Load embedding model
    print("  → Loading embedding model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    print("  ✓ Model loaded\n")

    # Create a collection
    print("Step 2: Create a collection")
    print("  Collections are like tables in traditional databases\n")

    # Delete if exists (for clean demo)
    try:
        client.delete_collection("demo_collection")
    except:
        pass

    collection = client.create_collection(
        name="demo_collection",
        metadata={"description": "Demo collection for learning"}
    )

    print(f"  ✓ Created collection: {collection.name}")
    print(f"  ✓ Collection metadata: {collection.metadata}\n")

    # Add some simple documents
    print("Step 3: Add documents")

    documents = [
        "The cat sits on the mat",
        "The dog plays in the park",
        "Machine learning is fascinating",
        "Python is a programming language"
    ]

    ids = [f"doc_{i}" for i in range(len(documents))]

    print(f"  → Adding {len(documents)} documents...")
    print("  → Computing embeddings...")

    # Compute embeddings using our model
    embeddings = model.encode(documents).tolist()

    collection.add(
        documents=documents,
        embeddings=embeddings,
        ids=ids,
        metadatas=[{"source": "demo", "index": i}
                   for i in range(len(documents))]
    )

    print(f"  ✓ Added {len(documents)} documents\n")

    # Query the collection
    print("Step 4: Query the collection")

    query = "What does the animal do?"
    print(f"  Query: '{query}'\n")

    # Compute query embedding
    query_embedding = model.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=2
    )

    print("  Results:")
    for i, (doc, distance) in enumerate(zip(results['documents'][0],
                                            results['distances'][0]), 1):
        print(f"    {i}. '{doc}'")
        print(f"       Distance: {distance:.4f}\n")

    print("💡 Key Concepts:")
    print("   • Client: Interface to ChromaDB")
    print("   • Collection: Container for related documents")
    print("   • Documents: Your text data")
    print("   • IDs: Unique identifiers for each document")
    print("   • Metadata: Additional information about documents")
    print("   • Embeddings: Created automatically by default!\n")

    return client, collection


def lesson_3_adding_job_ads():
    """Lesson 3: Adding our job ads to ChromaDB"""
    print_section("Lesson 3: Storing Job Advertisements")

    # Load job ads
    job_ads = load_job_ads()
    print(f"Loaded {len(job_ads)} job advertisements\n")

    # Initialize ChromaDB
    print("Setting up ChromaDB...")
    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection if any
    try:
        client.delete_collection("job_ads")
    except:
        pass

    # Create collection with custom embedding function
    print("Creating collection with sentence-transformers embedding...\n")

    # Load sentence transformer model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    collection = client.create_collection(
        name="job_ads",
        metadata={"description": "Job advertisements corpus"}
    )

    # Prepare data
    ids = list(job_ads.keys())
    documents = list(job_ads.values())

    # Create embeddings
    print("Creating embeddings for all job ads...")
    embeddings = model.encode(documents, show_progress_bar=True)
    print()

    # Extract some metadata
    metadatas = []
    for i, (name, text) in enumerate(zip(ids, documents)):
        # Simple metadata extraction
        word_count = len(text.split())
        char_count = len(text)

        metadatas.append({
            "doc_id": name,
            "word_count": word_count,
            "char_count": char_count,
            "index": i
        })

    # Add to collection
    print("Adding to ChromaDB collection...")
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings.tolist(),
        metadatas=metadatas
    )

    print(f"✓ Successfully added {len(ids)} job ads to collection\n")

    # Verify
    print("Verifying collection:")
    count = collection.count()
    print(f"  Total documents: {count}")

    # Show collection info
    sample = collection.peek(limit=2)
    print(f"\n  Sample document IDs: {sample['ids'][:2]}")
    print(f"  Embedding dimensions: {len(sample['embeddings'][0])}\n")

    print("💡 Data is now persisted! Even if we restart, it's still there.\n")

    return client, collection


def lesson_4_querying_the_database():
    """Lesson 4: Querying the vector database"""
    print_section("Lesson 4: Querying the Vector Database")

    # Connect to existing collection
    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("job_ads")

    model = SentenceTransformer('all-MiniLM-L6-v2')

    print(f"Connected to collection with {collection.count()} documents\n")

    # Different types of queries
    print("1. BASIC SEMANTIC SEARCH")

    queries = [
        "Python developer",
        "web development position",
        "remote work opportunity"
    ]

    for query in queries:
        print(f"\n  Query: '{query}'")
        print("  " + "-" * 70)

        # Create query embedding
        query_embedding = model.encode(query)

        results = collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=2
        )

        for i, (doc_id, distance) in enumerate(zip(results['ids'][0],
                                                   results['distances'][0]), 1):
            metadata = results['metadatas'][0][i-1]
            preview = results['documents'][0][i-1][:100].replace('\n', ' ')

            print(f"  {i}. {doc_id}")
            print(f"     Distance: {distance:.4f}")
            print(f"     Words: {metadata['word_count']}")
            print(f"     Preview: {preview}...")
        print()

    # Where filter
    print("\n2. FILTERING BY METADATA")
    print("  Find documents with more than 100 words\n")

    # Compute query embedding
    query_embedding = model.encode(["software developer"])[0].tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"word_count": {"$gt": 100}}
    )

    print(f"  Found {len(results['ids'][0])} results:")
    for i, (doc_id, metadata) in enumerate(zip(results['ids'][0],
                                               results['metadatas'][0]), 1):
        print(f"    {i}. {doc_id} ({metadata['word_count']} words)")

    print("\n3. GET SPECIFIC DOCUMENTS")
    print("  Retrieve by ID without search\n")

    specific_doc = collection.get(
        ids=["job_ad_1"],
        include=["metadatas", "documents"]
    )

    print(f"  Retrieved: {specific_doc['ids'][0]}")
    print(f"  Metadata: {specific_doc['metadatas'][0]}")
    print(f"  Content preview: {specific_doc['documents'][0][:100]}...\n")

    print("💡 ChromaDB Query Features:")
    print("   ✓ Semantic search with embeddings")
    print("   ✓ Metadata filtering with where clauses")
    print("   ✓ Direct document retrieval by ID")
    print("   ✓ Combining semantic search + filters")
    print("   ✓ Automatic distance calculation\n")


def lesson_5_updating_and_deleting():
    """Lesson 5: Updating and deleting documents"""
    print_section("Lesson 5: Updating and Deleting Documents")

    client = chromadb.PersistentClient(path="./chroma_db")

    # Create a test collection
    try:
        client.delete_collection("test_updates")
    except:
        pass

    collection = client.create_collection("test_updates")

    # Add initial documents
    print("Adding initial documents...")
    collection.add(
        ids=["doc1", "doc2", "doc3"],
        documents=[
            "First document",
            "Second document",
            "Third document"
        ],
        metadatas=[
            {"status": "draft"},
            {"status": "draft"},
            {"status": "draft"}
        ]
    )
    print(f"✓ Added 3 documents\n")

    # Update a document
    print("1. UPDATING DOCUMENTS")
    print("  Updating doc2's content and metadata...\n")

    collection.update(
        ids=["doc2"],
        documents=["Second document - UPDATED!"],
        metadatas=[{"status": "published", "updated": True}]
    )

    updated = collection.get(ids=["doc2"])
    print(f"  ✓ Updated document:")
    print(f"    Content: {updated['documents'][0]}")
    print(f"    Metadata: {updated['metadatas'][0]}\n")

    # Upsert (update or insert)
    print("2. UPSERT")
    print("  Upsert doc4 (will create since it doesn't exist)...\n")

    collection.upsert(
        ids=["doc4"],
        documents=["Fourth document - upserted"],
        metadatas=[{"status": "published"}]
    )

    print(f"  ✓ Total documents now: {collection.count()}\n")

    # Delete documents
    print("3. DELETING DOCUMENTS")
    print("  Deleting doc3...\n")

    collection.delete(ids=["doc3"])

    print(f"  ✓ Total documents now: {collection.count()}")

    remaining = collection.get()
    print(f"  Remaining IDs: {remaining['ids']}\n")

    print("💡 CRUD Operations:")
    print("   ✓ Create: add()")
    print("   ✓ Read: get() or query()")
    print("   ✓ Update: update() or upsert()")
    print("   ✓ Delete: delete()\n")


def lesson_6_collections_and_management():
    """Lesson 6: Managing collections"""
    print_section("Lesson 6: Collection Management")

    client = chromadb.PersistentClient(path="./chroma_db")

    print("1. LISTING COLLECTIONS\n")

    collections = client.list_collections()
    print("  Existing collections:")

    table_data = []
    for col in collections:
        count = col.count()
        table_data.append([col.name, count, col.metadata])

    print(tabulate(table_data,
                   headers=['Name', 'Documents', 'Metadata'],
                   tablefmt='grid'))

    print("\n2. COLLECTION METADATA")

    # Get job_ads collection
    job_collection = client.get_collection("job_ads")
    print(f"\n  Collection: {job_collection.name}")
    print(f"  Metadata: {job_collection.metadata}")
    print(f"  Document count: {job_collection.count()}\n")

    print("3. MODIFYING COLLECTION")

    # Modify metadata
    job_collection.modify(
        metadata={"description": "Job advertisements corpus",
                  "last_updated": "2025-12-16"}
    )

    print("  ✓ Updated collection metadata")
    print(f"  New metadata: {job_collection.metadata}\n")

    print("4. CREATING MULTIPLE COLLECTIONS")
    print("  Collections allow organizing different types of data\n")

    examples = [
        ("user_profiles", "User profile embeddings"),
        ("products", "Product descriptions"),
        ("support_docs", "Customer support documentation")
    ]

    for name, description in examples:
        print(f"  • {name}: {description}")

    print("\n💡 Best Practices:")
    print("   ✓ Use separate collections for different document types")
    print("   ✓ Add descriptive metadata to collections")
    print("   ✓ Regular backups of the data directory")
    print("   ✓ Monitor collection sizes")
    print("   ✓ Clean up test collections\n")


def lesson_7_performance_and_scaling():
    """Lesson 7: Performance considerations"""
    print_section("Lesson 7: Performance and Scaling")

    client = chromadb.PersistentClient(path="./chroma_db")
    collection = client.get_collection("job_ads")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Understanding vector database performance:\n")

    # Query performance
    print("1. QUERY PERFORMANCE")

    query = "software engineer position"
    query_embedding = model.encode(query)

    # Time the query
    start = time.time()
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=5
    )
    query_time = time.time() - start

    print(f"  Query time: {query_time*1000:.2f}ms")
    print(f"  Documents searched: {collection.count()}")
    print(f"  Results returned: {len(results['ids'][0])}\n")

    print("2. INDEXING STRATEGIES")
    print("  ChromaDB uses HNSW (Hierarchical Navigable Small World) index")
    print("  • Approximate nearest neighbor (ANN)")
    print("  • Trade-off: Speed vs Accuracy")
    print("  • Fast queries even with millions of vectors")
    print("  • Automatically managed by ChromaDB\n")

    print("3. BATCH OPERATIONS")
    print("  Always add/update documents in batches when possible")
    print("  • Faster than one-by-one")
    print("  • More efficient use of resources")
    print("  • Better for bulk data loads\n")

    print("4. SCALING CONSIDERATIONS")

    scale_info = [
        ["< 1M vectors", "Single instance", "Embedded ChromaDB", "Fast"],
        ["1M - 10M vectors", "Single server", "ChromaDB server", "Good"],
        ["10M+ vectors", "Distributed", "Milvus/Weaviate", "Excellent"],
        ["100M+ vectors", "Cloud-managed", "Pinecone/Vespa", "Excellent"]
    ]

    print(tabulate(scale_info,
                   headers=['Scale', 'Architecture',
                            'Solution', 'Performance'],
                   tablefmt='grid'))

    print("\n💡 Optimization Tips:")
    print("   ✓ Batch operations when possible")
    print("   ✓ Use appropriate n_results (don't fetch more than needed)")
    print("   ✓ Add indexes on frequently filtered metadata fields")
    print("   ✓ Monitor disk space (embeddings take storage)")
    print("   ✓ Use smaller embedding models if speed critical")
    print("   ✓ Consider approximate search for massive scale\n")


def run_module():
    """Run all lessons in Module 4"""
    print(f"\n{Fore.GREEN}{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "MODULE 4: VECTOR DATABASES".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}{Style.RESET_ALL}\n")

    print("This module teaches you about vector databases and ChromaDB.\n")

    # Run lessons
    lesson_1_why_vector_databases()
    lesson_2_chromadb_basics()
    lesson_3_adding_job_ads()
    lesson_4_querying_the_database()
    lesson_5_updating_and_deleting()
    lesson_6_collections_and_management()
    lesson_7_performance_and_scaling()

    print(f"\n{Fore.GREEN}{'='*80}")
    print("MODULE 4 COMPLETE! ✓")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    print("📚 What you learned:")
    print("  ✓ Why vector databases are essential for production")
    print("  ✓ ChromaDB basics: clients, collections, and documents")
    print("  ✓ Adding and querying embeddings")
    print("  ✓ Metadata filtering and hybrid search")
    print("  ✓ CRUD operations on vector data")
    print("  ✓ Collection management and organization")
    print("  ✓ Performance optimization and scaling strategies\n")


if __name__ == "__main__":
    run_module()
