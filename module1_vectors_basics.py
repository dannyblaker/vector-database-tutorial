"""
Module 1: Introduction to Vectors
==================================

This module introduces the fundamental concepts of vectors in the context of
machine learning and natural language processing.

Learning Objectives:
- Understand what vectors are and why they matter
- Learn basic vector operations
- Explore vector properties and dimensions
- See how vectors represent information
"""

import numpy as np
import matplotlib.pyplot as plt
from colorama import Fore, Style
import os


def print_section(title):
    """Helper function to print section headers"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print(f"{title}")
    print(f"{'='*80}{Style.RESET_ALL}\n")


def lesson_1_what_are_vectors():
    """Lesson 1: What are vectors?"""
    print_section("Lesson 1: What Are Vectors?")

    print("A vector is simply a list of numbers. In NLP and ML, vectors are used to")
    print("represent text, images, and other data in a way computers can understand.\n")

    # Simple vector
    print("Example 1: A simple 3-dimensional vector")
    vector = np.array([1, 2, 3])
    print(f"Vector: {vector}")
    print(f"Dimensions: {vector.shape[0]}")
    print(f"Type: {type(vector)}\n")

    # Larger vector
    print("Example 2: A 5-dimensional vector (could represent a simple word embedding)")
    word_vector = np.array([0.2, -0.5, 0.8, -0.1, 0.3])
    print(f"Word vector: {word_vector}")
    print(f"Dimensions: {word_vector.shape[0]}\n")

    print("💡 Key Insight: In modern NLP, words are often represented as vectors with")
    print("   hundreds or thousands of dimensions! Common sizes: 384, 768, 1536.\n")

    return vector, word_vector


def lesson_2_vector_operations():
    """Lesson 2: Basic vector operations"""
    print_section("Lesson 2: Basic Vector Operations")

    # Create sample vectors
    v1 = np.array([2, 3, 1])
    v2 = np.array([1, 2, 3])

    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}\n")

    # Addition
    print("1. Vector Addition:")
    result = v1 + v2
    print(f"   {v1} + {v2} = {result}\n")

    # Subtraction
    print("2. Vector Subtraction:")
    result = v1 - v2
    print(f"   {v1} - {v2} = {result}\n")

    # Scalar multiplication
    print("3. Scalar Multiplication:")
    scalar = 3
    result = scalar * v1
    print(f"   {scalar} × {v1} = {result}\n")

    # Dot product
    print("4. Dot Product (Very Important!):")
    dot_product = np.dot(v1, v2)
    print(f"   {v1} · {v2} = {dot_product}")
    print(f"   Calculation: (2×1) + (3×2) + (1×3) = {dot_product}\n")

    print("💡 Key Insight: The dot product measures how 'aligned' two vectors are.")
    print("   Larger values = more similar direction\n")

    return v1, v2, dot_product


def lesson_3_vector_magnitude_and_normalization():
    """Lesson 3: Vector magnitude and normalization"""
    print_section("Lesson 3: Vector Magnitude and Normalization")

    vector = np.array([3, 4])

    print(f"Original vector: {vector}\n")

    # Magnitude
    print("1. Magnitude (length) of a vector:")
    magnitude = np.linalg.norm(vector)
    print(f"   ||{vector}|| = {magnitude}")
    print(f"   Calculation: √(3² + 4²) = √25 = {magnitude}\n")

    # Normalization
    print("2. Normalization (creating a unit vector):")
    normalized = vector / magnitude
    print(f"   Normalized vector: {normalized}")
    print(f"   New magnitude: {np.linalg.norm(normalized):.6f}")
    print(f"   (Should be 1.0)\n")

    print("💡 Key Insight: Normalizing vectors is crucial in ML because it allows")
    print("   us to compare vectors based on direction, not magnitude.\n")

    return vector, normalized


def lesson_4_cosine_similarity_intro():
    """Lesson 4: Introduction to Cosine Similarity"""
    print_section("Lesson 4: Introduction to Cosine Similarity")

    print("Cosine similarity measures the angle between two vectors.")
    print("Range: -1 (opposite) to 1 (identical)\n")

    # Example vectors
    v1 = np.array([1, 1, 0])
    v2 = np.array([1, 0, 0])
    v3 = np.array([0, 1, 1])

    def cosine_similarity(a, b):
        """Calculate cosine similarity between two vectors"""
        dot_product = np.dot(a, b)
        magnitude_a = np.linalg.norm(a)
        magnitude_b = np.linalg.norm(b)
        return dot_product / (magnitude_a * magnitude_b)

    print(f"Vector 1: {v1}")
    print(f"Vector 2: {v2}")
    print(f"Vector 3: {v3}\n")

    sim_12 = cosine_similarity(v1, v2)
    sim_13 = cosine_similarity(v1, v3)
    sim_23 = cosine_similarity(v2, v3)

    print(f"Cosine similarity (v1, v2): {sim_12:.4f}")
    print(f"Cosine similarity (v1, v3): {sim_13:.4f}")
    print(f"Cosine similarity (v2, v3): {sim_23:.4f}\n")

    print("💡 Key Insight: This is THE fundamental concept for semantic search!")
    print("   Similar documents have vectors with high cosine similarity.\n")

    return v1, v2, v3, sim_12, sim_13, sim_23


def lesson_5_high_dimensional_vectors():
    """Lesson 5: Working with high-dimensional vectors"""
    print_section("Lesson 5: High-Dimensional Vectors")

    print("Modern embeddings use hundreds or thousands of dimensions.\n")

    # Create high-dimensional vectors
    dim_384 = np.random.randn(384)  # Common BERT embedding size
    dim_768 = np.random.randn(768)  # Common for larger models
    dim_1536 = np.random.randn(1536)  # OpenAI embedding size

    print(f"384-dimensional vector (e.g., MiniLM): shape {dim_384.shape}")
    print(f"First 10 values: {dim_384[:10]}\n")

    print(f"768-dimensional vector (e.g., BERT-base): shape {dim_768.shape}")
    print(f"First 10 values: {dim_768[:10]}\n")

    print(f"1536-dimensional vector (e.g., OpenAI): shape {dim_1536.shape}")
    print(f"First 10 values: {dim_1536[:10]}\n")

    # Demonstrate that operations work the same way
    print("These high-dimensional vectors work exactly like 2D or 3D vectors!")
    magnitude = np.linalg.norm(dim_384)
    print(f"Magnitude of 384-dim vector: {magnitude:.4f}")

    normalized = dim_384 / magnitude
    print(f"After normalization: {np.linalg.norm(normalized):.6f}\n")

    print("💡 Key Insight: Higher dimensions allow capturing more nuanced meanings")
    print("   and relationships between words and documents.\n")

    return dim_384, dim_768, dim_1536


def visualize_2d_vectors():
    """Bonus: Visualize 2D vectors"""
    print_section("Bonus: Visualizing Vectors in 2D")

    # Create output directory
    os.makedirs("outputs", exist_ok=True)

    # Create some 2D vectors
    vectors = {
        'v1': np.array([3, 2]),
        'v2': np.array([1, 3]),
        'v3': np.array([-2, 1]),
        'v4': np.array([2, -2])
    }

    plt.figure(figsize=(10, 10))
    plt.axhline(y=0, color='k', linewidth=0.5)
    plt.axvline(x=0, color='k', linewidth=0.5)
    plt.grid(True, alpha=0.3)

    colors = ['red', 'blue', 'green', 'purple']
    for (name, vec), color in zip(vectors.items(), colors):
        plt.arrow(0, 0, vec[0], vec[1], head_width=0.3, head_length=0.3,
                  fc=color, ec=color, linewidth=2, label=name)
        plt.text(vec[0]+0.3, vec[1]+0.3, name, fontsize=12, fontweight='bold')

    plt.xlim(-4, 4)
    plt.ylim(-4, 4)
    plt.xlabel('Dimension 1', fontsize=12)
    plt.ylabel('Dimension 2', fontsize=12)
    plt.title('2D Vector Visualization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.savefig('outputs/module1_vectors_2d.png', dpi=150, bbox_inches='tight')
    print("✓ Saved visualization to outputs/module1_vectors_2d.png\n")
    plt.close()


def run_module():
    """Run all lessons in Module 1"""
    print(f"\n{Fore.GREEN}{'#'*80}")
    print("#" + " "*78 + "#")
    print("#" + "MODULE 1: INTRODUCTION TO VECTORS".center(78) + "#")
    print("#" + " "*78 + "#")
    print(f"{'#'*80}{Style.RESET_ALL}\n")

    print("Welcome! This module will teach you the fundamentals of vectors.\n")

    # Run lessons
    lesson_1_what_are_vectors()
    lesson_2_vector_operations()
    lesson_3_vector_magnitude_and_normalization()
    lesson_4_cosine_similarity_intro()
    lesson_5_high_dimensional_vectors()
    visualize_2d_vectors()

    print(f"\n{Fore.GREEN}{'='*80}")
    print("MODULE 1 COMPLETE! ✓")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    print("📚 What you learned:")
    print("  ✓ Vectors are lists of numbers used to represent data")
    print("  ✓ Basic vector operations: addition, multiplication, dot product")
    print("  ✓ Magnitude and normalization")
    print("  ✓ Cosine similarity - the foundation of semantic search")
    print("  ✓ High-dimensional vectors in real-world NLP\n")


if __name__ == "__main__":
    run_module()
