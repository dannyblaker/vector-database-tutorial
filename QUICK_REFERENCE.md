# Vector Database Tutorial - Quick Reference

## Project Structure
```
cloned_repo/
├── main.py                          # Main tutorial runner (start here)
├── module1_vectors_basics.py        # Vectors fundamentals
├── module2_text_embeddings.py       # Text to vectors
├── module3_similarity_search.py     # Semantic search
├── module4_vector_databases.py      # ChromaDB basics
├── module5_advanced_techniques.py   # Advanced patterns
├── module6_rag_patterns.py          # RAG for LLMs
├── example_corpus/                  # Your job ad data
├── outputs/                         # Generated visualizations
├── chroma_db/                       # Vector database storage
└── requirements.txt                 # Python dependencies
```

## Commands Cheat Sheet

```bash
# Docker Commands (Recommended)
MODULE=1 docker compose up           # Run Module 1
MODULE=2 docker compose up           # Run Module 2
MODULE=3 docker compose up           # Run Module 3
MODULE=4 docker compose up           # Run Module 4
MODULE=5 docker compose up           # Run Module 5
MODULE=6 docker compose up           # Run Module 6
MODULE=ALL docker compose up         # Run all modules
docker compose build                 # Rebuild container

# Alternative Docker Commands
docker compose run --rm vector-tutorial python module1_vectors_basics.py
docker compose run --rm vector-tutorial python main.py  # Interactive

# Local Commands  
python main.py                       # Interactive menu
python main.py --all                 # Run all modules
python main.py 1                     # Run module 1
python module1_vectors_basics.py     # Direct execution
```

## Key Concepts by Module

### Module 1: Vectors
- Vectors are lists of numbers
- Dot product measures alignment
- Cosine similarity: -1 to 1
- Normalization creates unit vectors

### Module 2: Embeddings
- Text → semantic vectors
- Sentence Transformers (SBERT)
- Common dimensions: 384, 768, 1536
- Context matters!

### Module 3: Semantic Search
- Cosine similarity > keyword matching
- Ranking by relevance score
- Distance metrics: cosine, euclidean
- Threshold: >0.5 = relevant

### Module 4: Vector Databases
- ChromaDB for persistence
- Collections = tables
- CRUD: add, get, query, delete
- Metadata filtering

### Module 5: Advanced
- Hybrid search: semantic + keyword
- Re-ranking with business logic
- Multi-collection architecture
- Query optimization

### Module 6: RAG
- Retrieval → Augment → Generate
- Chunk size: 200-500 tokens
- Top-k retrieval (k=3-5)
- Context window management

## Code Snippets

### Create Embeddings
```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(["Hello world", "Python tutorial"])
```

### ChromaDB Basic Usage
```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("my_collection")

collection.add(
    ids=["doc1", "doc2"],
    documents=["First doc", "Second doc"]
)

results = collection.query(
    query_texts=["search query"],
    n_results=3
)
```

### Cosine Similarity
```python
import numpy as np

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
```

### Semantic Search
```python
# 1. Embed documents
doc_embeddings = model.encode(documents)

# 2. Embed query
query_embedding = model.encode(query)

# 3. Calculate similarities
from sklearn.metrics.pairwise import cosine_similarity
similarities = cosine_similarity([query_embedding], doc_embeddings)[0]

# 4. Get top results
top_indices = np.argsort(similarities)[::-1][:5]
```

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Model downloading | First run downloads ~80MB model (one-time) |
| Out of memory | Reduce batch size or use smaller model |
| Slow search | Use metadata filters, reduce n_results |
| ChromaDB locked | Delete chroma_db/ folder and restart |
| Import errors | Run `pip install -r requirements.txt` |

## Best Practices

✓ **DO:**
- Normalize vectors before similarity
- Use batching for multiple embeddings
- Set similarity thresholds (>0.3)
- Add metadata for filtering
- Cache frequent query embeddings
- Monitor retrieval quality

✗ **DON'T:**
- Use keyword search alone
- Forget to normalize
- Retrieve too many results
- Ignore metadata filtering
- Skip evaluation
- Hardcode model names

## Model Selection Guide

| Model | Dimensions | Speed | Quality | Use Case |
|-------|-----------|-------|---------|----------|
| all-MiniLM-L6-v2 | 384 | Fast | Good | General purpose |
| all-mpnet-base-v2 | 768 | Medium | Excellent | Best quality |
| multi-qa-MiniLM-L6 | 384 | Fast | Good | Q&A tasks |
| paraphrase-multilingual | 384 | Fast | Good | Multilingual |

## Similarity Score Guidelines

| Score | Interpretation | Action |
|-------|---------------|--------|
| > 0.7 | Very similar | Strong match |
| 0.5-0.7 | Similar | Good match |
| 0.3-0.5 | Somewhat related | Review manually |
| < 0.3 | Different | Probably not relevant |

## RAG Architecture

```
User Query
    ↓
Embed Query
    ↓
Vector DB Search → Top-k Documents
    ↓
Build Prompt: Query + Retrieved Docs
    ↓
LLM (GPT-4, Claude, etc.)
    ↓
Generated Answer
```

## Useful Resources

- ChromaDB Docs: https://docs.trychroma.com/
- Sentence Transformers: https://www.sbert.net/
- MTEB Leaderboard: https://huggingface.co/spaces/mteb/leaderboard
- RAG Papers: https://arxiv.org/abs/2005.11401

## Vocabulary

- **Vector**: List of numbers representing data
- **Embedding**: Dense vector representation
- **Semantic**: Related to meaning
- **Cosine Similarity**: Angle-based similarity (-1 to 1)
- **Collection**: Group of documents in vector DB
- **Chunk**: Segment of a larger document
- **RAG**: Retrieval Augmented Generation
- **Top-k**: Top k most similar results
- **ANN**: Approximate Nearest Neighbor
- **HNSW**: Hierarchical Navigable Small World (index type)

---

Keep this reference handy as you work through the tutorial!
