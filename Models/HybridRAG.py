# Models/HybridRAG.py

import os
from typing import List, Dict, Optional

# Langchain imports (ensure these are consistent with your NormalRAG's imports)
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI, ChatOpenAI # For OpenAI LLM
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini LLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class HybridRAG:
    """
    A Hybrid Retrieval Augmented Generation (RAG) system.

    This class extends the basic RAG functionality, intending to incorporate
    multiple retrieval strategies (e.g., semantic + keyword search) for
    more robust context gathering. For this initial setup, its core
    functionality will resemble NormalRAG, but it's designed for expansion.
    """

    def __init__(self,
                 llm_name: str = "openai",
                 api_key: str = "",
                 model_name: str = "gpt-3.5-turbo",
                 base_url: str = "",
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 instruction_prompt: str = "You are a sophisticated hybrid assistant. Use the provided context to answer the question accurately, potentially synthesizing information from diverse retrieval methods.",
                 vectorstore_path: str = "./faiss_hybrid_index", # Distinct path for hybrid
                 # Add any specific parameters for hybrid strategies here, e.g.:
                 # keyword_retriever_config: Optional[Dict] = None
                ):
        """
        Initializes the HybridRAG system with specified LLM and embedding configurations.
        """
        self.llm_name = llm_name
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url
        self.embed_model_name = embed_model_name
        self.instruction_prompt = instruction_prompt
        self.vectorstore_path = vectorstore_path
        self.vectorstore: Optional[FAISS] = None
        self.retrieval_qa_chain = None

        # Initialize embedding model
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)
            print(f"HybridRAG: Embedding model '{self.embed_model_name}' initialized.")
        except Exception as e:
            print(f"HybridRAG: Error initializing embedding model: {e}")
            self.embeddings = None

        # Initialize LLM
        self._initialize_llm()

        # Attempt to load existing vector store or initialize empty one
        self.load_vectorstore()

        # Placeholder for hybrid-specific components, e.g., a keyword retriever
        # self.keyword_retriever = self._setup_keyword_retriever(keyword_retriever_config)

    def _initialize_llm(self):
        """
        Internal method to initialize the appropriate Langchain LLM for HybridRAG.
        """
        if self.api_key == "":
            print(f"HybridRAG Warning: API key not provided for {self.llm_name}. LLM may not function.")

        try:
            if self.llm_name == "openai":
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    base_url=self.base_url if self.base_url else None,
                    temperature=0.0
                )
                print(f"HybridRAG: OpenAI LLM '{self.model_name}' initialized.")
            elif self.llm_name == "gemini":
                self.llm = ChatGoogleGenerativeAI(
                    api_key=self.api_key,
                    model=self.model_name,
                    temperature=0.0
                )
                print(f"HybridRAG: Gemini LLM '{self.model_name}' initialized.")
            else:
                self.llm = None
                print(f"HybridRAG: Unsupported LLM name: {self.llm_name}. Choose 'openai' or 'gemini'.")
        except Exception as e:
            self.llm = None
            print(f"HybridRAG: Error initializing LLM '{self.llm_name}': {e}")

    def load_documents(self, dataset_folder: str = "Datasets"):
        """
        Loads documents for HybridRAG from a specified folder, chunks them, and creates/updates the vector store.
        """
        if not os.path.exists(dataset_folder):
            print(f"HybridRAG Error: Dataset folder '{dataset_folder}' not found.")
            return

        if self.embeddings is None:
            print("HybridRAG Error: Embedding model not initialized. Cannot load documents.")
            return

        print(f"HybridRAG: Loading documents from '{dataset_folder}'...")
        try:
            loader = DirectoryLoader(dataset_folder, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            print(f"HybridRAG: Found {len(documents)} documents.")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"HybridRAG: Split into {len(chunks)} chunks.")

            if self.vectorstore:
                self.vectorstore.add_documents(chunks)
                print("HybridRAG: Documents added to existing vector store.")
            else:
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                print("HybridRAG: New FAISS vector store created from documents.")

            self.save_vectorstore()
            self._setup_retrieval_qa_chain()
        except Exception as e:
            print(f"HybridRAG: Error loading documents: {e}")

    def add_document(self, document_content: str, document_metadata: Optional[Dict] = None):
        """
        Adds a single document (text string) to the HybridRAG's FAISS vector database.
        """
        if self.embeddings is None:
            print("HybridRAG Error: Embedding model not initialized. Cannot add document.")
            return
        if self.vectorstore is None:
            print("HybridRAG Error: Vector store not initialized. Load documents first or create an empty one.")
            return

        try:
            from langchain_core.documents import Document
            doc = Document(page_content=document_content, metadata=document_metadata or {})
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents([doc])
            self.vectorstore.add_documents(chunks)
            print(f"HybridRAG: Document added to vector store. Total chunks now: {len(self.vectorstore.docstore._dict)}")
            self.save_vectorstore()
            self._setup_retrieval_qa_chain()
        except Exception as e:
            print(f"HybridRAG: Error adding document: {e}")

    def _hybrid_retrieve_context(self, question: str, top_k: int = 4) -> List[str]:
        """
        Placeholder for advanced hybrid retrieval logic.
        This could combine semantic search with keyword search, or other methods.
        For now, it just uses semantic search like NormalRAG.
        """
        if self.vectorstore is None:
            print("HybridRAG Error: Vector store not loaded or created. Cannot retrieve context.")
            return []

        try:
            # Semantic search
            semantic_docs = self.vectorstore.similarity_search(question, k=top_k)
            semantic_context = [doc.page_content for doc in semantic_docs]
            print(f"HybridRAG: Retrieved {len(semantic_context)} semantic document chunks.")

            # --- Placeholder for Keyword Search or other methods ---
            # For a true hybrid, you'd integrate another retrieval method here,
            # e.g., using BM25 or a custom keyword index.
            # Example:
            # keyword_context = self.keyword_retriever.retrieve(question)
            # combined_context = list(set(semantic_context + keyword_context)) # Remove duplicates

            return semantic_context # For now, just return semantic
        except Exception as e:
            print(f"HybridRAG: Error during hybrid context retrieval: {e}")
            return []

    def load_vectorstore(self):
        """
        Loads a previously saved FAISS vector store for HybridRAG.
        """
        if self.embeddings is None:
            print("HybridRAG Error: Embedding model not initialized. Cannot load vector store.")
            return

        if os.path.exists(self.vectorstore_path):
            try:
                self.vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"HybridRAG: Vector store loaded from '{self.vectorstore_path}'.")
                self._setup_retrieval_qa_chain()
            except Exception as e:
                print(f"HybridRAG Error loading vector store from '{self.vectorstore_path}': {e}")
                print("HybridRAG: Initializing an empty vector store instead.")
                self.vectorstore = None
        else:
            print(f"HybridRAG: No existing vector store found at '{self.vectorstore_path}'.")

    def save_vectorstore(self):
        """
        Saves the current FAISS vector store for HybridRAG.
        """
        if self.vectorstore:
            try:
                os.makedirs(self.vectorstore_path, exist_ok=True)
                self.vectorstore.save_local(self.vectorstore_path)
                print(f"HybridRAG: Vector store saved to '{self.vectorstore_path}'.")
            except Exception as e:
                print(f"HybridRAG Error saving vector store to '{self.vectorstore_path}': {e}")
        else:
            print("HybridRAG: No vector store to save.")

    def _setup_retrieval_qa_chain(self):
        """
        Sets up the RetrievalQA chain for HybridRAG question answering.
        """
        if self.llm is None:
            print("HybridRAG: LLM not initialized. Cannot set up RetrievalQA chain.")
            return
        if self.vectorstore is None:
            print("HybridRAG: Vector store not initialized. Cannot set up RetrievalQA chain.")
            return

        try:
            prompt_template = PromptTemplate(
                template=self.instruction_prompt + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
                input_variables=["context", "question"]
            )

            # For a true hybrid system, you might create a custom retriever here
            # that uses _hybrid_retrieve_context, rather than just as_retriever()
            retriever = self.vectorstore.as_retriever()

            self.retrieval_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=retriever, # This retriever will call _hybrid_retrieve_context if integrated
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            print("HybridRAG: RetrievalQA chain setup complete.")
        except Exception as e:
            print(f"HybridRAG Error setting up RetrievalQA chain: {e}")
            self.retrieval_qa_chain = None

    def ask_question(self, question: str) -> str:
        """
        Asks a question to the HybridRAG system.
        """
        if self.retrieval_qa_chain is None:
            print("HybridRAG: RetrievalQA chain not setup. Please ensure LLM and vector store are initialized.")
            return "Unable to answer the question."

        try:
            print(f"HybridRAG: Asking question: '{question}'...")
            result = self.retrieval_qa_chain.invoke({"query": question})
            answer = result.get("result", "No answer found.")
            source_documents = result.get("source_documents", [])

            print("\n--- HybridRAG Answer ---")
            print(answer)
            if source_documents:
                print("\n--- HybridRAG Source Documents ---")
                for i, doc in enumerate(source_documents):
                    print(f"Document {i+1}:")
                    print(f"  Content: {doc.page_content[:200]}...")
                    print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            return answer
        except Exception as e:
            print(f"HybridRAG Error asking question: {e}")
            return "An error occurred while processing your question in HybridRAG."