import os
from typing import List, Dict, Optional

# Langchain imports
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAI, ChatOpenAI # For OpenAI LLM
from langchain_google_genai import ChatGoogleGenerativeAI # For Gemini LLM
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA

class NormalRAG:
    """
    A Retrieval Augmented Generation (RAG) system designed for flexible LLM integration
    and efficient document management using FAISS vector store.

    This class provides functionalities to load, add, retrieve, save, and load documents
    and their embeddings, enabling context-aware LLM responses.
    """

    def __init__(self,
                 llm_name: str = "openai",  # "openai" or "gemini"
                 api_key: str = "",
                 model_name: str = "gpt-3.5-turbo",  # e.g., "gpt-3.5-turbo" or "models/gemini-pro"
                 base_url: str = "", # Optional base URL for custom LLM endpoints
                 embed_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
                 instruction_prompt: str = "You are a helpful assistant. Use the provided context to answer the question accurately. If the answer is not in the context, state that you don't have enough information.",
                 vectorstore_path: str = "./faiss_index"):
        """
        Initializes the NormalRAG system with specified LLM and embedding configurations.

        Args:
            llm_name (str): The name of the LLM provider ("openai" or "gemini").
            api_key (str): API key for the chosen LLM service.
            model_name (str): Specific model identifier for the LLM.
            base_url (str): Optional base URL for custom LLM endpoints.
            embed_model_name (str): Name of the embedding model to use.
            instruction_prompt (str): System-level prompt for the LLM.
            vectorstore_path (str): File path for saving/loading the FAISS vector store.
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
        # For HuggingFaceEmbeddings, ensure 'sentence-transformers' is installed:
        # pip install sentence-transformers
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.embed_model_name)
            print(f"Embedding model '{self.embed_model_name}' initialized.")
        except Exception as e:
            print(f"Error initializing embedding model: {e}")
            self.embeddings = None

        # Initialize LLM based on llm_name
        self._initialize_llm()

        # Attempt to load existing vector store or initialize empty one
        self.load_vectorstore()

    def _initialize_llm(self):
        """
        Internal method to initialize the appropriate Langchain LLM based on self.llm_name.
        """
        if self.api_key == "":
            print(f"Warning: API key not provided for {self.llm_name}. LLM may not function.")

        try:
            if self.llm_name == "openai":
                # For OpenAI, you might use ChatOpenAI for chat models or OpenAI for completion models
                self.llm = ChatOpenAI(
                    api_key=self.api_key,
                    model_name=self.model_name,
                    base_url=self.base_url if self.base_url else None,
                    temperature=0.0 # Set temperature for more deterministic answers
                )
                print(f"OpenAI LLM '{self.model_name}' initialized.")
            elif self.llm_name == "gemini":
                # For Google Gemini, ensure GOOGLE_API_KEY is set or passed
                # You might need to set os.environ["GOOGLE_API_KEY"] = self.api_key
                self.llm = ChatGoogleGenerativeAI(
                    api_key=self.api_key,
                    model=self.model_name, # e.g., "models/gemini-pro"
                    temperature=0.0 # Set temperature for more deterministic answers
                )
                print(f"Gemini LLM '{self.model_name}' initialized.")
            # Add more LLM integrations here (e.g., Llama, HuggingFace local models)
            # elif self.llm_name == "llama":
            #     # Example for a local Llama model via Ollama
            #     from langchain_community.llms import Ollama
            #     self.llm = Ollama(model=self.model_name)
            #     print(f"Ollama LLM '{self.model_name}' initialized.")
            else:
                self.llm = None
                print(f"Unsupported LLM name: {self.llm_name}. Please choose 'openai' or 'gemini'.")
        except Exception as e:
            self.llm = None
            print(f"Error initializing LLM '{self.llm_name}': {e}")

    def load_documents(self, dataset_folder: str = "Datasets"):
        """
        Loads documents from a specified folder, chunks them, and creates/updates the vector store.

        Args:
            dataset_folder (str): The path to the folder containing documents.
        """
        if not os.path.exists(dataset_folder):
            print(f"Error: Dataset folder '{dataset_folder}' not found. Please create it and add documents.")
            return

        if self.embeddings is None:
            print("Error: Embedding model not initialized. Cannot load documents.")
            return

        print(f"Loading documents from '{dataset_folder}'...")
        try:
            # Use DirectoryLoader to load all text files from the folder
            # You can extend this to handle other file types like PDF, CSV etc.
            # For example: from langchain_community.document_loaders import PyPDFLoader
            # loader = PyPDFLoader("path/to/your/pdf.pdf")
            loader = DirectoryLoader(dataset_folder, glob="**/*.txt", loader_cls=TextLoader)
            documents = loader.load()
            print(f"Found {len(documents)} documents.")

            # Split documents into smaller chunks for better retrieval
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents(documents)
            print(f"Split into {len(chunks)} chunks.")

            # Create or update the FAISS vector store
            if self.vectorstore:
                # If vectorstore already exists, add new documents to it
                self.vectorstore.add_documents(chunks)
                print("Documents added to existing vector store.")
            else:
                # Otherwise, create a new one
                self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
                print("New FAISS vector store created from documents.")

            self.save_vectorstore() # Save after loading
            self._setup_retrieval_qa_chain() # Setup/update the chain
        except Exception as e:
            print(f"Error loading documents: {e}")

    def add_document(self, document_content: str, document_metadata: Optional[Dict] = None):
        """
        Adds a single document (text string) to the FAISS vector database.

        Args:
            document_content (str): The text content of the document to add.
            document_metadata (Optional[Dict]): Optional metadata for the document.
        """
        if self.embeddings is None:
            print("Error: Embedding model not initialized. Cannot add document.")
            return
        if self.vectorstore is None:
            print("Error: Vector store not initialized. Load documents first or create an empty one.")
            return

        try:
            # Create a Langchain Document object
            from langchain_core.documents import Document
            doc = Document(page_content=document_content, metadata=document_metadata or {})

            # Split the new document into chunks
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
                is_separator_regex=False,
            )
            chunks = text_splitter.split_documents([doc])

            # Add the chunks to the existing vector store
            self.vectorstore.add_documents(chunks)
            print(f"Document added to vector store. Total chunks now: {len(self.vectorstore.docstore._dict)}")
            self.save_vectorstore() # Save after adding
            self._setup_retrieval_qa_chain() # Setup/update the chain
        except Exception as e:
            print(f"Error adding document: {e}")

    def retrieve_context(self, question: str, top_k: int = 4) -> List[str]:
        """
        Retrieves the top_k most relevant document chunks for a given question.

        Args:
            question (str): The question to retrieve context for.
            top_k (int): The number of top relevant documents to retrieve.

        Returns:
            List[str]: A list of strings, where each string is a retrieved document chunk.
        """
        if self.vectorstore is None:
            print("Error: Vector store not loaded or created. Cannot retrieve context.")
            return []

        try:
            # Perform similarity search
            docs = self.vectorstore.similarity_search(question, k=top_k)
            context = [doc.page_content for doc in docs]
            print(f"Retrieved {len(context)} relevant document chunks.")
            return context
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []

    def load_vectorstore(self):
        """
        Loads a previously saved FAISS vector store from the specified path.
        """
        if self.embeddings is None:
            print("Error: Embedding model not initialized. Cannot load vector store.")
            return

        if os.path.exists(self.vectorstore_path):
            try:
                self.vectorstore = FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)
                print(f"Vector store loaded from '{self.vectorstore_path}'.")
                self._setup_retrieval_qa_chain() # Setup/update the chain after loading
            except Exception as e:
                print(f"Error loading vector store from '{self.vectorstore_path}': {e}")
                print("Initializing an empty vector store instead.")
                # If loading fails, ensure vectorstore is None or re-initialize if needed
                self.vectorstore = None # Or FAISS.from_documents([], self.embeddings)
        else:
            print(f"No existing vector store found at '{self.vectorstore_path}'.")
            # self.vectorstore = FAISS.from_documents([], self.embeddings) # Optionally initialize empty
            # print("An empty vector store has been initialized.")


    def save_vectorstore(self):
        """
        Saves the current FAISS vector store to the specified path.
        """
        if self.vectorstore:
            try:
                # Ensure the directory exists
                os.makedirs(self.vectorstore_path, exist_ok=True)
                self.vectorstore.save_local(self.vectorstore_path)
                print(f"Vector store saved to '{self.vectorstore_path}'.")
            except Exception as e:
                print(f"Error saving vector store to '{self.vectorstore_path}': {e}")
        else:
            print("No vector store to save.")

    def _setup_retrieval_qa_chain(self):
        """
        Sets up the RetrievalQA chain for question answering.
        This should be called after the vectorstore and LLM are ready.
        """
        if self.llm is None:
            print("LLM not initialized. Cannot set up RetrievalQA chain.")
            return
        if self.vectorstore is None:
            print("Vector store not initialized. Cannot set up RetrievalQA chain.")
            return

        try:
            # Create a prompt template for the RAG chain
            # The context will be inserted here by the retriever
            prompt_template = PromptTemplate(
                template=self.instruction_prompt + "\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:",
                input_variables=["context", "question"]
            )

            # Create a retriever from the vector store
            retriever = self.vectorstore.as_retriever()

            # Create the RetrievalQA chain
            self.retrieval_qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff", # "stuff" combines all retrieved documents into one prompt
                retriever=retriever,
                return_source_documents=True, # Optionally return the source documents
                chain_type_kwargs={"prompt": prompt_template}
            )
            print("RetrievalQA chain setup complete.")
        except Exception as e:
            print(f"Error setting up RetrievalQA chain: {e}")
            self.retrieval_qa_chain = None

    def ask_question(self, question: str) -> str:
        """
        Asks a question to the RAG system, retrieving context and generating an answer.

        Args:
            question (str): The question to ask.

        Returns:
            str: The answer generated by the LLM, augmented with retrieved context.
        """
        if self.retrieval_qa_chain is None:
            print("RetrievalQA chain not setup. Please ensure LLM and vector store are initialized.")
            return "Unable to answer the question."

        try:
            result = self.retrieval_qa_chain.invoke({"query": question})
            answer = result.get("result", "No answer found.")
            source_documents = result.get("source_documents", [])

            print("\n--- Answer ---")
            print(answer)
            if source_documents:
                print("\n--- Source Documents ---")
                for i, doc in enumerate(source_documents):
                    print(f"Document {i+1}:")
                    print(f"  Content: {doc.page_content[:200]}...") # Print first 200 chars
                    print(f"  Source: {doc.metadata.get('source', 'N/A')}")
            return answer
        except Exception as e:
            print(f"Error asking question: {e}")
            return "An error occurred while processing your question."

# --- Example Usage ---
if __name__ == "__main__":
    # --- Setup for demonstration ---
    # Create a dummy 'Datasets' folder and some dummy text files
    if not os.path.exists("Datasets"):
        os.makedirs("Datasets")
    with open("Datasets/doc1.txt", "w") as f:
        f.write("The capital of France is Paris. Paris is known for its Eiffel Tower.")
    with open("Datasets/doc2.txt", "w") as f:
        f.write("The Amazon rainforest is the largest tropical rainforest in the world.")
    with open("Datasets/doc3.txt", "w") as f:
        f.write("Artificial intelligence (AI) is intelligence demonstrated by machines.")
    with open("Datasets/doc4.txt", "w") as f:
        f.write("Python is a popular programming language for AI and data science.")
    print("Dummy 'Datasets' folder and files created for demonstration.")

    # --- Instantiate NormalRAG ---
    # IMPORTANT: Replace "YOUR_OPENAI_API_KEY" or "YOUR_GEMINI_API_KEY" with your actual API key
    # For OpenAI:
    # rag_system = NormalRAG(llm_name="openai", api_key="YOUR_OPENAI_API_KEY", model_name="gpt-3.5-turbo")
    # For Gemini:
    rag_system = NormalRAG(llm_name="gemini", api_key="YOUR_GEMINI_API_KEY", model_name="gemini-pro")

    # --- Load documents ---
    print("\n--- Loading documents ---")
    rag_system.load_documents(dataset_folder="Datasets")

    # --- Add a new document ---
    print("\n--- Adding a new document ---")
    rag_system.add_document(
        document_content="The moon is Earth's only natural satellite.",
        document_metadata={"source": "Astronomy Facts"}
    )

    # --- Retrieve context for a question ---
    print("\n--- Retrieving context ---")
    question1 = "What is the capital of France?"
    context1 = rag_system.retrieve_context(question1)
    print(f"Context for '{question1}': {context1}")

    question2 = "What is AI?"
    context2 = rag_system.retrieve_context(question2, top_k=2)
    print(f"Context for '{question2}': {context2}")

    # --- Ask a question using the RAG chain ---
    print("\n--- Asking questions with RAG chain ---")
    rag_system.ask_question("Tell me about the Eiffel Tower.")
    rag_system.ask_question("What is the largest rainforest?")
    rag_system.ask_question("Which programming language is popular for AI?")
    rag_system.ask_question("What is Earth's natural satellite?")
    rag_system.ask_question("What is the highest mountain in the world?") # Question not in context

    # --- Clean up dummy files (optional) ---
    import shutil
    if os.path.exists("Datasets"):
        shutil.rmtree("Datasets")
        print("\nCleaned up 'Datasets' folder.")
    if os.path.exists("./faiss_index"):
        shutil.rmtree("./faiss_index")
        print("Cleaned up 'faiss_index' folder.")
