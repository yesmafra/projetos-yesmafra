import os
from pypdf import PdfReader

def load_documents(folder_path="data/docs"):
    documents = []

    # Lista arquivos da pasta
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Se for PDF
        if filename.lower().endswith(".pdf"):
            print(f"🔍 Lendo PDF: {filename}")
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            documents.append({"content": text, "source": filename})

        # Se for TXT
        elif filename.lower().endswith(".txt"):
            print(f"🔍 Lendo TXT: {filename}")
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            documents.append({"content": text, "source": filename})

    return documents
