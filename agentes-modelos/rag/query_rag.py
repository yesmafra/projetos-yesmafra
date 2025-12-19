import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI


# Cliente do LM Studio (LLM local)
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")


def search_chroma(query, top_k=3):
    """Busca documentos semelhantes no banco vetorial."""

    # Conecta ao banco vetorial atualizado
    chroma_client = chromadb.PersistentClient(path="chroma_db")

    collection = chroma_client.get_collection("docs")

    # Cria embedding da pergunta
    query_embedding = embedder.encode(query).tolist()

    # Busca documentos relevantes
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )

    return results


def ask_rag(query):
    """Executa pipeline completo do RAG."""

    # 1. Recupera contexto
    results = search_chroma(query)

    contexts = results["documents"][0]
    sources = results["metadatas"][0]

    # Monta contexto legível
    context_text = ""
    for i, chunk in enumerate(contexts):
        context_text += f"\n[Documento {i+1} - {sources[i]['source']}]\n{chunk}\n"

    # 2. Monta prompt para o LLaMA
    prompt = f"""
Você responde SOMENTE com base no contexto abaixo.
Se a informação não estiver no contexto, diga exatamente:
"Não encontrei essa informação nos documentos."

### CONTEXTO ###
{context_text}

### PERGUNTA ###
{query}

### RESPOSTA ###
"""

    # 3. Gera resposta com o LLM local
    response = client.chat.completions.create(
        model="meta-llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    pergunta = input("Digite sua pergunta: ")
    resposta = ask_rag(pergunta)
    print("\n📌 RESPOSTA DO RAG:\n")
    print(resposta)
