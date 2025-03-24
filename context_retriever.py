
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
import os


class ContextRetriever:
    def __init__(self, context_file_path: str, vector_store_serialized_path: str):

        with open("commedia.txt", "r") as f:
            full_context = f.read()

        context_entries = full_context.split("\n\n")
        context_docs = [Document(page_content=entry)
                        for entry in context_entries if len(entry.split("\n")) > 2]

        embeddings = MistralAIEmbeddings(model="mistral-embed",)
        if os.path.exists(vector_store_serialized_path):
            self.vector_store = FAISS.load_local(
                vector_store_serialized_path, embeddings, allow_dangerous_deserialization=True)
        else:
            self.vector_store = FAISS.from_documents(context_docs, embeddings)
            self.vector_store.save_local(vector_store_serialized_path)

    def __call__(self, query: str, k: int = 5) -> list[Document]:
        return self.vector_store.similarity_search(query, k=k)
