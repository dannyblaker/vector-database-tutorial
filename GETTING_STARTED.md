# Getting Started with the Vector Database Tutorial

## Quick Start

You have two options to run the tutorial:

### Option 1: Docker (Recommended)
```bash
# Run a specific module using environment variable
MODULE=1 docker compose up      # Start with Module 1 (recommended)
MODULE=2 docker compose up      # Run Module 2
MODULE=ALL docker compose up    # Run all modules

# Or run interactively (requires terminal interaction)
docker compose run --rm vector-tutorial python main.py

# Or run a specific module directly
docker compose run --rm vector-tutorial python module1_vectors_basics.py
```

### Option 2: Local Python Environment
```bash
# Install dependencies
pip install -r requirements.txt

# Run the tutorial
python main.py
```

## What's Included

This tutorial contains **6 progressive modules** that will take you from zero understanding to advanced knowledge of vectors, embeddings, vector databases, and RAG patterns.

### Module Structure

1. **Module 1: Introduction to Vectors** (Beginner)
   - Vector fundamentals and operations
   - Magnitude, normalization, and cosine similarity
   - High-dimensional vectors in ML/NLP

2. **Module 2: Text Embeddings** (Beginner)
   - Converting text to semantic vectors
   - Using Sentence Transformers
   - Working with your job ads corpus

3. **Module 3: Vector Similarity and Semantic Search** (Intermediate)
   - Distance metrics comparison
   - Building a semantic search engine
   - Ranking and optimization

4. **Module 4: Vector Databases** (Intermediate)
   - ChromaDB fundamentals
   - CRUD operations
   - Persistence and scaling

5. **Module 5: Advanced Vector Database Techniques** (Advanced)
   - Complex filtering and hybrid search
   - Re-ranking strategies
   - Multi-collection architecture

6. **Module 6: Retrieval Augmented Generation (RAG)** (Advanced)
   - RAG architecture and patterns
   - Document chunking strategies
   - Building production RAG systems

## Example Usage

### Interactive Mode
```bash
python main.py
```
This will show you a menu where you can choose which module to run.

### Run All Modules
```bash
python main.py --all
```
This runs the complete tutorial from beginning to end (3-4 hours).

### Run Specific Module
```bash
python main.py 1    # Run Module 1
python main.py 2    # Run Module 2
# etc.
```

### Direct Module Execution
```bash
python module1_vectors_basics.py
python module2_text_embeddings.py
# etc.
```

## Your Data

The tutorial uses the job advertisements you provided in `example_corpus/`:
- job_ad_1.txt through job_ad_6.txt
- Real-world examples demonstrate semantic search
- You can add more files to expand the corpus

## Generated Files

As you progress through the tutorial, files will be created:

- **`outputs/`**: Visualizations and charts
  - Vector diagrams
  - Similarity matrices
  
- **`chroma_db/`**: Persistent vector database
  - Stores embeddings across sessions
  - Can be deleted to start fresh

## Tips for Learning

1. **Go Sequential**: Start with Module 1 and work your way up
2. **Take Your Time**: Each module has interactive lessons - read carefully
3. **Experiment**: Modify the code, try different queries, explore!
4. **Use Your Data**: Add your own documents to the corpus
5. **Run Multiple Times**: Each run reinforces the concepts

## Key Concepts You'll Master

- **Vectors**: Mathematical representation of data
- **Embeddings**: Semantic vector representations of text
- **Cosine Similarity**: Measuring semantic similarity
- **Vector Databases**: Efficient storage and retrieval
- **Semantic Search**: Finding meaning, not keywords
- **RAG**: Combining retrieval with LLMs

## Technical Requirements

- **Python**: 3.11+
- **Memory**: 4GB+ RAM recommended
- **Storage**: ~2GB for models and data
- **Time**: 3-4 hours for complete tutorial

## Troubleshooting

### Models downloading on first run
The first time you run modules 2-6, Sentence Transformers will download a ~80MB model. This is normal and happens once.

### Docker issues
```bash
# Rebuild if you get errors
docker compose build --no-cache

# Clean up and start fresh
docker compose down
docker compose up
```

### Permission issues
```bash
# Make sure directories are writable
chmod -R 755 outputs chroma_db
```

## Next Steps After Completion

Once you finish the tutorial, you'll be ready to:

1. Build your own RAG applications
2. Create production search systems
3. Work with any vector database (Pinecone, Weaviate, etc.)
4. Fine-tune embedding models
5. Deploy LLM-powered applications

## Support

All code is heavily commented and self-explanatory. Each module builds on previous concepts, so if something is unclear, review earlier modules.

---

**Happy Learning! 🚀**

You're about to embark on a journey into modern AI and machine learning. By the end, you'll understand the technology powering ChatGPT's retrieval, Perplexity's search, and countless other AI applications.
