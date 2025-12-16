#!/usr/bin/env python3
"""
Quick test script to verify all dependencies are working
"""

print("Testing Vector Database Tutorial Dependencies...")
print("=" * 60)

try:
    import numpy as np
    print(f"✓ NumPy {np.__version__}")
except ImportError as e:
    print(f"✗ NumPy failed: {e}")

try:
    import sentence_transformers
    print(f"✓ Sentence Transformers")
except ImportError as e:
    print(f"✗ Sentence Transformers failed: {e}")

try:
    import chromadb
    print(f"✓ ChromaDB")
except ImportError as e:
    print(f"✗ ChromaDB failed: {e}")

try:
    import matplotlib
    print(f"✓ Matplotlib")
except ImportError as e:
    print(f"✗ Matplotlib failed: {e}")

try:
    import pandas
    print(f"✓ Pandas")
except ImportError as e:
    print(f"✗ Pandas failed: {e}")

try:
    from colorama import Fore, Style
    print(f"✓ Colorama")
except ImportError as e:
    print(f"✗ Colorama failed: {e}")

print("=" * 60)
print("All dependencies loaded successfully!")
print("\nYou can now run:")
print("  python main.py          - Interactive mode")
print("  python main.py --all    - Run all modules")
print("  python main.py 1        - Run specific module")
