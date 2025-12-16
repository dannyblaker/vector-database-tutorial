"""
Vector Database Tutorial - Main Runner
======================================

This is the main entry point for the vector database tutorial.
It runs all modules sequentially or allows you to choose specific modules.
"""

import module6_rag_patterns
import module5_advanced_techniques
import module4_vector_databases
import module3_similarity_search
import module2_text_embeddings
import module1_vectors_basics
import sys
from colorama import init, Fore, Style
import os

# Initialize colorama
init()

# Import all modules


def print_banner():
    """Print welcome banner"""
    banner = f"""
{Fore.GREEN}{'='*80}
{'='*80}

    🎓 VECTOR DATABASE TUTORIAL 🎓
    
    From Zero to Advanced: Learn Vectors, Embeddings, Vector DBs, and RAG
    
{'='*80}
{'='*80}{Style.RESET_ALL}
"""
    print(banner)


def print_menu():
    """Print module selection menu"""
    print(f"\n{Fore.CYAN}{'='*80}")
    print("AVAILABLE MODULES")
    print(f"{'='*80}{Style.RESET_ALL}\n")

    modules = [
        ("Module 1", "Introduction to Vectors", "Beginner"),
        ("Module 2", "Text Embeddings", "Beginner"),
        ("Module 3", "Vector Similarity and Semantic Search", "Intermediate"),
        ("Module 4", "Vector Databases (ChromaDB)", "Intermediate"),
        ("Module 5", "Advanced Vector DB Techniques", "Advanced"),
        ("Module 6", "Retrieval Augmented Generation (RAG)", "Advanced"),
    ]

    for i, (name, description, level) in enumerate(modules, 1):
        level_color = Fore.GREEN if level == "Beginner" else Fore.YELLOW if level == "Intermediate" else Fore.RED
        print(f"  {i}. {Fore.CYAN}{name}{Style.RESET_ALL}: {description}")
        print(f"     Level: {level_color}{level}{Style.RESET_ALL}\n")

    print(f"  {Fore.GREEN}A{Style.RESET_ALL}. Run ALL modules (complete tutorial)")
    print(f"  {Fore.RED}Q{Style.RESET_ALL}. Quit\n")


def run_module(module_number):
    """Run a specific module"""
    modules = {
        1: ("Module 1: Introduction to Vectors", module1_vectors_basics.run_module),
        2: ("Module 2: Text Embeddings", module2_text_embeddings.run_module),
        3: ("Module 3: Similarity and Search", module3_similarity_search.run_module),
        4: ("Module 4: Vector Databases", module4_vector_databases.run_module),
        5: ("Module 5: Advanced Techniques", module5_advanced_techniques.run_module),
        6: ("Module 6: RAG Patterns", module6_rag_patterns.run_module),
    }

    if module_number in modules:
        title, func = modules[module_number]
        try:
            func()
            return True
        except KeyboardInterrupt:
            print(
                f"\n\n{Fore.YELLOW}Module interrupted by user.{Style.RESET_ALL}")
            return False
        except Exception as e:
            print(f"\n\n{Fore.RED}Error running module: {e}{Style.RESET_ALL}")
            return False
    else:
        print(f"{Fore.RED}Invalid module number!{Style.RESET_ALL}")
        return False


def run_all_modules():
    """Run all modules in sequence"""
    print(f"\n{Fore.GREEN}{'='*80}", flush=True)
    print("RUNNING ALL MODULES", flush=True)
    print(f"{'='*80}{Style.RESET_ALL}\n", flush=True)

    print("This will take you through the complete tutorial from beginner to advanced.", flush=True)
    print("You can press Ctrl+C at any time to stop.\n", flush=True)

    for i in range(1, 7):
        print(f"{Fore.CYAN}Starting Module {i}...{Style.RESET_ALL}\n", flush=True)
        success = run_module(i)
        if not success:
            print(
                f"\n{Fore.YELLOW}Tutorial stopped at Module {i}.{Style.RESET_ALL}")
            return

        if i < 6:
            print(f"\n{Fore.GREEN}Module {i} complete!{Style.RESET_ALL}\n")

    print(f"\n{Fore.GREEN}{'='*80}")
    print("🎉 CONGRATULATIONS! 🎉")
    print(f"{'='*80}{Style.RESET_ALL}\n")
    print("You've completed the entire Vector Database Tutorial!")
    print("You now have a basic understanding of:")
    print("  ✓ Vector fundamentals")
    print("  ✓ Text embeddings")
    print("  ✓ Semantic search")
    print("  ✓ Vector databases")
    print("  ✓ Advanced querying techniques")
    print("  ✓ RAG patterns for LLM applications\n")


def interactive_mode():
    """Run in interactive mode with menu"""
    while True:
        print_menu()
        choice = input(
            f"{Fore.CYAN}Select a module (1-6, A, or Q): {Style.RESET_ALL}").strip().upper()

        if choice == 'Q':
            print(
                f"\n{Fore.GREEN}Thanks for learning! Goodbye! 👋{Style.RESET_ALL}\n")
            break
        elif choice == 'A':
            run_all_modules()
        elif choice.isdigit() and 1 <= int(choice) <= 6:
            run_module(int(choice))
            input(f"\n{Fore.CYAN}Press Enter to return to menu...{Style.RESET_ALL}")
        else:
            print(f"{Fore.RED}Invalid choice! Please try again.{Style.RESET_ALL}")


def main():
    """Main entry point"""
    # Create necessary directories
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    # Check for environment variable (for Docker)
    module_env = os.environ.get('MODULE', '').strip()

    if module_env:
        print_banner()

        if module_env.upper() == 'ALL':
            print(
                f"\n{Fore.GREEN}Running all modules (from MODULE environment variable){Style.RESET_ALL}\n")
            run_all_modules()
            return
        elif module_env.isdigit() and 1 <= int(module_env) <= 6:
            module_num = int(module_env)
            print(
                f"\n{Fore.GREEN}Running Module {module_num} (from MODULE environment variable){Style.RESET_ALL}\n")
            run_module(module_num)
            return
        else:
            print(
                f"{Fore.RED}Invalid MODULE value: '{module_env}'. Use 1-6 or ALL{Style.RESET_ALL}")
            print("Falling back to interactive mode...\n")

    print_banner()

    # Check for command line arguments
    if len(sys.argv) > 1:
        arg = sys.argv[1]

        if arg == "--all":
            run_all_modules()
        elif arg.isdigit() and 1 <= int(arg) <= 6:
            run_module(int(arg))
        elif arg == "--help":
            print("Usage:")
            print("  python main.py           - Interactive mode")
            print("  python main.py --all     - Run all modules")
            print("  python main.py <1-6>     - Run specific module")
            print("  python main.py --help    - Show this help")
            print()
        else:
            print(
                f"{Fore.RED}Invalid argument. Use --help for usage.{Style.RESET_ALL}")
    else:
        # Interactive mode
        print("Welcome to the Vector Database Tutorial!")
        print("This interactive tutorial will teach you everything about vectors,")
        print("embeddings, vector databases, and RAG patterns.\n")

        print(
            f"{Fore.YELLOW}💡 Tip: You can run modules individually or all at once.{Style.RESET_ALL}\n")

        interactive_mode()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(
            f"\n\n{Fore.YELLOW}Tutorial interrupted. Run again anytime!{Style.RESET_ALL}\n")
        sys.exit(0)
