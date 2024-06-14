import os

OPENAI_API_KEY = ""
os.environ["openai_api_key"] = OPENAI_API_KEY

from langchain_community.document_loaders import ReadTheDocsLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone

INDEX_NAME = "langchain-doc-index"
PINECONE_API_KEY = ""


pc = Pinecone(api_key=PINECONE_API_KEY)


def ingest_docs():
    loader = ReadTheDocsLoader("langchain-docs/langchain.readthedocs.io/en/latest")

    raw_documents = loader.load()
    print(f"loaded {len(raw_documents)} documents")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100, separators=["\n\n", "\n", " ", ""]
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Splitted into {len(documents)} chunks")

    for doc in documents:
        new_url = doc.metadata["source"]
        new_url = new_url.replace("langchain-docs", "https:/")
        doc.metadata.update({"source": new_url})

    print(f"Going to add {len(documents)} to Pinecone")

    embeddings = OpenAIEmbeddings()
    PineconeVectorStore.from_documents(documents, embeddings, index_name=INDEX_NAME)
    print("****Loading to vectorstore done ***")


if __name__ == "__main__":
    ingest_docs()
