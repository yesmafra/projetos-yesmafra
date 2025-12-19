import streamlit as st
import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# --- Configurações iniciais ---
st.set_page_config(page_title="Meu RAG Local", layout="wide")

# Cliente do LM Studio
client = OpenAI(
    base_url="http://localhost:1234/v1",
    api_key="lm-studio"
)

# Modelo de embeddings
embedder = SentenceTransformer("all-MiniLM-L6-v2")


# --- Função de busca no ChromaDB ---
def search_chroma(query, top_k=3):
    chroma_client = chromadb.PersistentClient(path="chroma_db")
    collection = chroma_client.get_collection("docs")

    embedding = embedder.encode(query).tolist()

    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )

    return results


# --- Função principal de RAG ---
def ask_rag(query):
    results = search_chroma(query)

    contexts = results["documents"][0]
    sources = results["metadatas"][0]

    context_text = ""
    for i, chunk in enumerate(contexts):
        context_text += f"\n[Documento {i+1} - {sources[i]['source']}]\n{chunk}\n"

    prompt = f"""
Responda SOMENTE com base no contexto abaixo.
Se a informação não estiver no contexto, diga: "Não encontrei essa informação nos documentos."

### CONTEXTO ###
{context_text}

### PERGUNTA ###
{query}

### RESPOSTA ###
"""

    response = client.chat.completions.create(
        model="meta-llama-3-8b-instruct",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content


# --- Interface Streamlit ---
st.title("🧠 Meu RAG Local (LLAMA + ChromaDB)")

pergunta = st.text_input("Digite sua pergunta:", "")

if st.button("Enviar"):
    if pergunta.strip() == "":
        st.warning("Digite uma pergunta!")
    else:
        with st.spinner("Consultando documentos e gerando resposta..."):
            resposta = ask_rag(pergunta)

        st.markdown("### 📌 Resposta:")
        st.write(resposta)
