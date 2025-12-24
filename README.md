# Vector Database Tutorial

A comprehensive, hands-on tutorial that takes you from zero to advanced understanding of vectors, embeddings, vector databases, and Retrieval Augmented Generation (RAG) patterns for NLP and LLM applications.

[![A Danny Blaker project badge](https://github.com/dannyblaker/dannyblaker.github.io/blob/main/danny_blaker_project_badge.svg)](https://github.com/dannyblaker/)

## 🚀 Quick Start

```bash
# Clone and navigate to the repository
cd cloned_repo

# Run with Docker Compose - specific module (recommended)
MODULE=1 docker compose up      # Module 1: Vector Basics
MODULE=2 docker compose up      # Module 2: Text Embeddings
MODULE=3 docker compose up      # Module 3: Similarity Search
MODULE=4 docker compose up      # Module 4: Vector Databases
MODULE=5 docker compose up      # Module 5: Advanced Techniques
MODULE=6 docker compose up      # Module 6: RAG Patterns
MODULE=ALL docker compose up    # Run all modules sequentially

# Or run locally (requires Python 3.11+)
pip install -r requirements.txt
python main.py
```

## 📚 What You'll Learn

This tutorial is structured into 6 progressive modules:

### Module 1: Introduction to Vectors (Beginner)
- What are vectors and why they matter
- Basic vector operations (addition, multiplication, dot product)
- Vector magnitude and normalization
- Introduction to cosine similarity
- Working with high-dimensional vectors

### Module 2: Text Embeddings (Beginner)
- Understanding text embeddings
- From bag-of-words to neural embeddings
- Using Sentence Transformers
- Creating embeddings for real documents
- Embedding properties and characteristics
- Choosing the right embedding model

### Module 3: Vector Similarity and Semantic Search (Intermediate)
- Distance metrics (cosine, euclidean, dot product)
- Building a semantic search engine from scratch
- Semantic search vs keyword search
- Ranking strategies and similarity thresholds
- Advanced features: multi-query, re-ranking, query expansion
- Performance optimization

### Module 4: Vector Databases (Intermediate)
- Why vector databases are essential
- ChromaDB fundamentals
- Storing and querying embeddings
- CRUD operations
- Collections and data management
- Performance and scaling considerations

### Module 5: Advanced Vector Database Techniques (Advanced)
- Complex metadata filtering
- Hybrid search (semantic + keyword + filters)
- Re-ranking strategies
- Multi-collection architecture
- Query optimization techniques
- Production-ready patterns

### Module 6: Retrieval Augmented Generation (RAG) (Advanced)
- RAG architecture and motivation
- Document chunking strategies
- Building a complete RAG system
- Advanced patterns: query expansion, HyDE, parent document retrieval
- Common challenges and solutions
- Evaluation metrics and monitoring

## 🎯 Learning Path

```
Zero Knowledge → Beginner → Intermediate → Advanced
     ↓              ↓            ↓             ↓
  Module 1     Modules 1-2   Modules 1-4   All Modules
```

**Estimated Time:**
- Complete tutorial: 3-4 hours
- Individual module: 20-40 minutes

## 🛠️ Technology Stack

- **Python 3.11+**: Programming language
- **NumPy**: Vector operations
- **Sentence Transformers**: Text embeddings
- **ChromaDB**: Vector database
- **scikit-learn**: ML utilities
- **Docker**: Containerization

## 📁 Project Structure

```
cloned_repo/
├── main.py                           # Main tutorial runner
├── module1_vectors_basics.py         # Module 1: Vector fundamentals
├── module2_text_embeddings.py        # Module 2: Text embeddings
├── module3_similarity_search.py      # Module 3: Semantic search
├── module4_vector_databases.py       # Module 4: Vector databases
├── module5_advanced_techniques.py    # Module 5: Advanced techniques
├── module6_rag_patterns.py           # Module 6: RAG patterns
├── example_corpus/                   # Sample job ad documents
│   ├── job_ad_1.txt
│   ├── job_ad_2.txt
│   └── ...
├── outputs/                          # Generated visualizations
├── chroma_db/                        # Persistent vector database
├── requirements.txt                  # Python dependencies
├── Dockerfile                        # Docker configuration
├── docker-compose.yml                # Docker Compose setup
└── README.md                         # This file
```

## 🐳 Docker Usage

### Run Specific Module (Recommended)
```bash
MODULE=1 docker compose up       # Module 1: Vector Basics
MODULE=2 docker compose up       # Module 2: Text Embeddings
MODULE=3 docker compose up       # Module 3: Similarity Search
MODULE=4 docker compose up       # Module 4: Vector Databases
MODULE=5 docker compose up       # Module 5: Advanced Techniques
MODULE=6 docker compose up       # Module 6: RAG Patterns
MODULE=ALL docker compose up     # Run all modules sequentially
```

### Alternative: Direct Module Execution
```bash
docker compose run --rm vector-tutorial python module1_vectors_basics.py
```

### Interactive Mode (requires terminal interaction)
```bash
docker compose run --rm vector-tutorial python main.py
```

## 💻 Local Usage (Without Docker)

### Prerequisites
- Python 3.11 or higher
- pip

### Installation
```bash
# Install dependencies
pip install -r requirements.txt

# Run the tutorial
python main.py
```

### Command Line Options
```bash
# Interactive mode (default)
python main.py

# Run all modules
python main.py --all

# Run specific module
python main.py 1   # Module 1
python main.py 2   # Module 2
# ... etc

# Run individual module directly
python module1_vectors_basics.py
```

## 📊 Example Outputs

The tutorial generates visualizations and outputs in the `outputs/` directory:
- Vector visualizations
- Similarity matrices
- Search result rankings
- Performance comparisons

## 🎓 Learning Features

- **Interactive**: Press Enter to proceed through lessons
- **Hands-on**: Work with real job advertisement data
- **Progressive**: Builds knowledge step-by-step
- **Practical**: Production-ready patterns and best practices
- **Visual**: Includes diagrams and visualizations
- **Code Examples**: Complete, runnable code for every concept

## 🔑 Key Concepts Covered

### Vectors & Embeddings
- Vector spaces and dimensionality
- Semantic meaning representation
- Transformer-based embeddings
- Embedding model selection

### Similarity & Search
- Cosine similarity
- Euclidean distance
- Semantic vs keyword search
- Ranking algorithms

### Vector Databases
- CRUD operations
- Metadata filtering
- Hybrid search
- Indexing strategies
- Scalability patterns

### RAG Patterns
- Document chunking
- Context retrieval
- Prompt engineering
- Query expansion
- Evaluation metrics

## 🚀 Next Steps After Completion

1. **Build Your Own RAG App**: Use the patterns learned to build a Q&A system
2. **Explore Other Vector DBs**: Try Pinecone, Weaviate, or Milvus
3. **Fine-tune Embeddings**: Learn to fine-tune models on your domain
4. **Production Deployment**: Scale your vector database for production
5. **Integrate LLMs**: Connect to GPT-4, Claude, or open-source LLMs

## 📚 Additional Resources

- [ChromaDB Documentation](https://docs.trychroma.com/)
- [Sentence Transformers](https://www.sbert.net/)
- [RAG Papers and Research](https://arxiv.org/abs/2005.11401)

## 🤝 Contributing

This is a learning project. Feel free to:
- Add more example documents
- Create additional modules
- Improve explanations
- Add more visualizations
- Suggest improvements

## 📝 License

This tutorial is provided for educational purposes.

## ⚠️ Note

This tutorial uses:
- Pre-trained models that will be downloaded on first run (~80MB)
- Local storage for ChromaDB (persistent across runs)
- The example corpus provided (job advertisements)

## 🎉 Credits

Built with:
- Sentence Transformers by UKPLab
- ChromaDB by Chroma
- Python open-source ecosystem

## 💡 Tips

1. **Go at your own pace**: Each module is self-contained
2. **Experiment**: Modify the code and see what happens
3. **Use your own data**: Replace the job ads with your own documents
4. **Ask questions**: The code is heavily commented
5. **Build projects**: Apply what you learn to real problems

---

**Happy Learning! 🚀**

Start your journey into the world of vector databases and modern AI applications!
