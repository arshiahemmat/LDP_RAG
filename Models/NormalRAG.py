import os
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import HumanMessage, SystemMessage


class NormalRAG:
    def __init__(self, llm_name="openai", api_key=None, model_name=None, base_url=None):
        self.llm_name = llm_name
        self.api_key = api_key
        self.model_name = model_name
        self.base_url = base_url or "https://api.avalai.ir/v1"

        if self.llm_name == "openai":
            if not self.api_key:
                raise ValueError("OpenAI API key is required for llm_name='openai'")
            self.embeddings = OpenAIEmbeddings(api_key=self.api_key, base_url=self.base_url)
            self.llm = ChatOpenAI(api_key=self.api_key, model_name=self.model_name, base_url=self.base_url)
        elif self.llm_name == "gemini":
            if not self.api_key:
                raise ValueError("Gemini API key is required for llm_name='gemini'")
            self.embeddings = GoogleGenerativeAIEmbeddings(google_api_key=self.api_key)
            self.llm = ChatGoogleGenerativeAI(google_api_key=self.api_key, model=self.model_name)
        else:
            raise ValueError("Invalid llm_name. Choose 'openai' or 'gemini'.")

        self.vectorstore = None
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        self.vector_store_path = "RAG_Database"

    def load_documents(self, dataset_folder):
        if not os.path.exists(dataset_folder):
            print(f"Dataset folder '{dataset_folder}' not found.")
            return

        documents = []
        for filename in os.listdir(dataset_folder):
            folder_path = os.path.join(dataset_folder, filename)
            if os.path.isdir(folder_path):
                for i in range(1, 11):
                    filepath = os.path.join(folder_path, f'{i}.docx')
                    if os.path.exists(filepath):
                        loader = Docx2txtLoader(filepath)
                        documents.extend(loader.load())

        if documents:
            split_documents = self.text_splitter.split_documents(documents)
            if self.vectorstore is None:
                self.vectorstore = FAISS.from_documents(split_documents, self.embeddings)
            else:
                self.vectorstore.add_documents(split_documents)
            print(f"Loaded {len(documents)} documents from '{dataset_folder}'.")
        else:
            print(f"No valid .docx files found in '{dataset_folder}'.")

    def retrieve_context(self, question, top_k=4):
        if self.vectorstore is None:
            print("No documents loaded yet. Load documents first.")
            return []
        docs = self.vectorstore.similarity_search(question, k=top_k)
        return [doc.page_content for doc in docs]

    def generate(self, question, top_k=4, system_prompt="You are a helpful assistant that answers using only the provided context.", include_context=False):
        context_chunks = self.retrieve_context(question, top_k=top_k)
        if not context_chunks:
            raise ValueError("No context retrieved. Load documents or vector store first.")

        context_block = "\n\n".join(context_chunks)
        user_prompt = (
            f"Context:\n{context_block}\n\n"
            f"Question: {question}\n\n"
            "Answer in a concise and complete manner:"
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]
        answer = self.llm.invoke(messages).content.strip()

        return (answer, context_chunks) if include_context else answer

    def save_vector_store(self, index_name="faiss_index"):
        if self.vectorstore is None:
            print("No vector store to save.")
            return
        full_path = os.path.join(self.vector_store_path, index_name)
        os.makedirs(self.vector_store_path, exist_ok=True)
        self.vectorstore.save_local(full_path)
        print(f"Vector store saved to '{full_path}'.")

    def load_vector_store(self, index_name="faiss_index"):
        full_path = os.path.join(self.vector_store_path, index_name)
        if not os.path.exists(full_path):
            print(f"Vector store not found at '{full_path}'.")
            return
        self.vectorstore = FAISS.load_local(full_path, self.embeddings, allow_dangerous_deserialization=True)
        print(f"Vector store loaded from '{full_path}'."
