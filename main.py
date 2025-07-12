# main.py

import argparse
import sys
import os
import shutil # Still useful for manual cleanup instructions

# Add the project root to the Python path to allow importing from 'Models'
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

# Import RAG models from the 'Models' package
# Ensure NormalRAG.py and HybridRAG.py exist in the Models/ directory
from Models.NormalRAG import NormalRAG
from Models.HybridRAG import HybridRAG

def get_rag_model_instance(rag_type: str, config: dict):
    """
    Returns an instance of the specified RAG model based on the model type.
    """
    rag_type_lower = rag_type.lower()
    if rag_type_lower == "normalrag":
        print("Creating NormalRAG instance...")
        return NormalRAG(
            llm_name=config.get("llm_name"),
            api_key=config.get("api_key"),
            model_name=config.get("llm_model_name"),
            embed_model_name=config.get("embed_model_name"),
            instruction_prompt=config.get("instruction_prompt_normal"),
            vectorstore_path=config.get("vectorstore_path_normal")
        )
    elif rag_type_lower == "hybridrag":
        print("Creating HybridRAG instance...")
        return HybridRAG(
            llm_name=config.get("llm_name"),
            api_key=config.get("api_key"),
            model_name=config.get("llm_model_name"),
            embed_model_name=config.get("embed_model_name"),
            instruction_prompt=config.get("instruction_prompt_hybrid"),
            vectorstore_path=config.get("vectorstore_path_hybrid")
        )
    else:
        raise ValueError(f"Unsupported RAG type specified: '{rag_type}'. Available types: 'NormalRAG', 'HybridRAG', or 'else'.")

# --- Utility functions (not called automatically in main, for manual use/reference) ---
def setup_dummy_data(base_dataset_folder="Datasets", data_name_subfolder="default"):
    """Creates dummy data for demonstration purposes in a specific subfolder."""
    target_folder = os.path.join(base_dataset_folder, data_name_subfolder)
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"Created dummy data folder: '{target_folder}'.")

    dummy_files = {
        "doc1.txt": "The capital of France is Paris. Paris is known for its Eiffel Tower and Louvre Museum.",
        "doc2.txt": "The Amazon rainforest is the largest tropical rainforest in the world, spanning multiple countries in South America.",
        "doc3.txt": "Artificial intelligence (AI) is intelligence demonstrated by machines, unlike the natural intelligence displayed by humans and animals.",
        "doc4.txt": "Python is a popular programming language, widely used in web development, data science, machine learning, and AI.",
        "doc5.txt": "The planet Mars is known as the Red Planet and is the fourth planet from the Sun."
    }

    if data_name_subfolder == "medical":
        dummy_files["med_doc1.txt"] = "Aspirin is a common medication used to reduce pain, fever, and inflammation. It is a nonsteroidal anti-inflammatory drug (NSAID)."
        dummy_files["med_doc2.txt"] = "Diabetes mellitus is a metabolic disease that causes high blood sugar. Insulin is a hormone that moves sugar from the blood into your cells for storage or energy."

    for fname, content in dummy_files.items():
        fpath = os.path.join(target_folder, fname)
        if not os.path.exists(fpath):
            with open(fpath, "w") as f:
                f.write(content)
    print(f"Dummy files populated in '{target_folder}'.")

def cleanup_data(base_dataset_folder="Datasets", vectorstore_paths=None):
    """Cleans up dummy data and vector stores."""
    if os.path.exists(base_dataset_folder):
        shutil.rmtree(base_dataset_folder)
        print(f"Cleaned up '{base_dataset_folder}' folder.")
    if vectorstore_paths:
        for path in vectorstore_paths:
            if os.path.exists(path):
                shutil.rmtree(path)
                print(f"Cleaned up '{path}' folder.")
# --- End Utility functions ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run a selected Retrieval Augmented Generation (RAG) model."
    )
    # Essential parameters from your list
    parser.add_argument(
        "--data-name",
        type=str,
        default="",
        choices=["all", "medical", ""],
        help="Specify the dataset to use ('all', 'medical', or leave empty for default 'default' folder)."
    )
    parser.add_argument(
        "--emb-model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the embedding model to use (e.g., 'sentence-transformers/all-MiniLM-L6-v2')."
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="gpt-3.5-turbo", # Default to an OpenAI model
        help="Specific LLM model name (e.g., 'gpt-3.5-turbo', 'gemini-pro')."
    )
    parser.add_argument(
        "--llm-provider", # Added to distinguish provider from model name
        type=str,
        default="openai", # Default provider
        choices=["openai", "gemini", "else"],
        help="LLM provider ('openai', 'gemini', 'else' for others)."
    )
    parser.add_argument(
        "--RAG-type",
        type=str,
        default="NormalRAG",
        choices=["NormalRAG", "HybridRAG", "else"],
        help="Type of RAG system ('NormalRAG', 'HybridRAG', 'else' for custom)."
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="",
        choices=["all", ""], # Empty string for no specific metrics
        help="Metric calculation mode ('all' for all metrics, or leave empty)."
    )
    parser.add_argument(
        "--prompt-based",
        type=lambda x: (str(x).lower() == 'true'),
        default=False,
        help="Enable or disable prompt-based mode (True/False)."
    )

    # Core operational parameters (kept as necessary for the RAG system to run)
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The input query string for the RAG model to process."
    )
    parser.add_argument(
        "--api_key",
        type=str,
        help="API key for the selected LLM. Overrides environment variables."
    )

    args = parser.parse_args()

    # Determine API key priority: command line > environment variables
    api_key_to_use = args.api_key
    if not api_key_to_use:
        if args.llm_provider.lower() == "openai":
            api_key_to_use = os.getenv("OPENAI_API_KEY")
        elif args.llm_provider.lower() == "gemini":
            api_key_to_use = os.getenv("GOOGLE_API_KEY")

    if not api_key_to_use:
        print(f"Error: API key is not provided for {args.llm_provider}. Please provide it via --api_key or as an environment variable (e.g., OPENAI_API_KEY for OpenAI, GOOGLE_API_KEY for Gemini).")
        parser.print_help()
        sys.exit(1)

    # Determine the dataset folder based on --data-name
    base_dataset_folder = "Datasets"
    data_name_subfolder = args.data_name if args.data_name else "default"
    current_dataset_folder = os.path.join(base_dataset_folder, data_name_subfolder)

    # Define configuration for the RAG models
    rag_config = {
        "llm_name": args.llm_provider, # Passes as llm_name to RAG class
        "api_key": api_key_to_use,
        "llm_model_name": args.llm_model, # Passes as model_name to RAG class
        "embed_model_name": args.emb_model,
        "instruction_prompt_normal": "You are a helpful assistant. Use the provided context to answer the question accurately. If the answer is not in the context, state that you don't have enough information.",
        "instruction_prompt_hybrid": "You are a sophisticated hybrid assistant. Use the provided context to answer the question accurately, potentially synthesizing information from diverse retrieval methods.",
        # Vector store paths are now tied to data_name for distinct storage
        "vectorstore_path_normal": f"./faiss_index_normal_{data_name_subfolder}",
        "vectorstore_path_hybrid": f"./faiss_index_hybrid_{data_name_subfolder}"
    }

    rag_instance = None
    try:
        rag_instance = get_rag_model_instance(args.RAG_type, rag_config)

        # The RAG class's __init__ will attempt to load its vector store automatically
        # and load_documents will create/update it if necessary.
        print(f"Loading documents from folder: {current_dataset_folder}")
        rag_instance.load_documents(current_dataset_folder)

        print(f"\n--- Running RAG System ---")
        print(f"RAG Type: {args.RAG_type}")
        print(f"LLM Provider: {args.llm_provider}, Model: {args.llm_model}")
        print(f"Embedding Model: {args.emb_model}")
        print(f"Data Name: {data_name_subfolder} (from folder: {current_dataset_folder})")
        print(f"Metrics Mode: {args.metrics if args.metrics else 'disabled'}")
        print(f"Prompt-Based Mode: {args.prompt_based}")
        print(f"Query: '{args.query}'")

        # Ask the question
        rag_instance.ask_question(args.query)

        # Placeholder for metrics calculation or prompt-based logic
        if args.metrics == "all":
            print("\n--- Calculating all metrics (placeholder) ---")
            # Implement your metric calculation logic here, potentially
            # using data from rag_instance or external evaluation sets.
            pass
        if args.prompt_based:
            print("\n--- Running in prompt-based mode (placeholder) ---")
            # Implement logic specific to prompt-based mode here,
            # perhaps modifying how the query or context is handled before calling ask_question.
            pass

    except ValueError as e:
        print(f"Configuration Error: {e}")
        parser.print_help()
        sys.exit(1)
    except ImportError as e:
        print(f"Dependency Error: {e}")
        print("Please ensure all required Langchain packages (langchain-community, langchain-text-splitters, langchain-openai, langchain-google-genai), 'faiss-cpu', 'sentence-transformers', and 'PyMuPDF' are installed.")
        print("Try: pip install langchain-community langchain-text-splitters langchain-openai langchain-google-genai 'faiss-cpu' sentence-transformers PyMuPDF")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        sys.exit(1)