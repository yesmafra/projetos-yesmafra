import chromadb
from sentence_transformers import SentenceTransformer
from load_docs import load_documents


def create_chroma_db():
    # Inicializa o modelo de embeddings
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # Inicializa o ChromaDB local (nova API)
    client = chromadb.PersistentClient(path="chroma_db")

    # Cria ou pega a coleção existente
    collection = client.get_or_create_collection(name="docs")

    # Carrega documentos da pasta data/docs
    docs = load_documents()

    # Processa cada documento
    for idx, doc in enumerate(docs):
        text = doc["content"]
        source = doc["source"]

        # Gera embedding
        embedding = embedder.encode(text).tolist()

        # Adiciona ao ChromaDB
        collection.add(
            documents=[text],
            embeddings=[embedding],
            ids=[f"doc_{idx}"],
            metadatas=[{"source": source}]
        )

        print(f"📚 Documento '{source}' adicionado ao ChromaDB.")

    print("\n✅ Banco vetorial criado com sucesso!")
    # Persistência já é automática, mas deixamos explícito
    # client.persist()


if __name__ == "__main__":
    create_chroma_db()
