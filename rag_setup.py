import os
import sys
from dotenv import load_dotenv  # type: ignore

load_dotenv()

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader  # type: ignore
from langchain_text_splitters import RecursiveCharacterTextSplitter  # type: ignore
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI  # type: ignore
from langchain_community.vectorstores import FAISS  # type: ignore
from langchain_classic.chains import create_retrieval_chain  # type: ignore
from langchain_classic.chains.combine_documents import create_stuff_documents_chain  # type: ignore
from langchain_core.prompts import ChatPromptTemplate  # type: ignore

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    print("Error: GEMINI_API_KEY environment variable not set.")
    sys.exit(1)

DB_DIR = "faiss_index"
DATA_DIR = "knowledge_base"


def ingest_data():
    print(f"Loading documents from {DATA_DIR}...")
    if not os.path.exists(DATA_DIR):
        print(f"Directory {DATA_DIR} not found. Creating it!")
        os.makedirs(DATA_DIR)
        return False

    txt_loader = DirectoryLoader(DATA_DIR, glob="**/*.txt", loader_cls=TextLoader)
    pdf_loader = DirectoryLoader(DATA_DIR, glob="**/*.pdf", loader_cls=PyPDFLoader)

    print("Loading specialized TXT and PDF documents...")
    documents = txt_loader.load() + pdf_loader.load()

    if not documents:
        print(f"No text or PDF documents found in {DATA_DIR}.")
        return False

    print(f"Loaded {len(documents)} documents. Splitting text...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    print(f"Created {len(texts)} text chunks. Generating embeddings...")
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    vectorstore = FAISS.from_documents(texts, embeddings)
    vectorstore.save_local(DB_DIR)
    print(f"Successfully created and saved FAISS index to {DB_DIR}/")
    return True


def chat_with_rag():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")

    if not os.path.exists(DB_DIR):
        print(f"Vector Database not found at {DB_DIR}. Run ingestion first.")
        return

    print("Loading vector database...")
    vectorstore = FAISS.load_local(DB_DIR, embeddings, allow_dangerous_deserialization=True)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    print("Initializing Gemini...")
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0)

    system_prompt = (
        "You are an expert, multi-platinum music producer, songwriter, and creative co-writer.\n"
        "Use the following pieces of retrieved context to answer the user's question.\n"
        "If you don't know the answer based on the context, you can use your general knowledge, "
        "but prioritize the retrieved information.\n\n"
        "Context: {context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    print("\nRAG Co-writer initialized! (Type 'quit' or 'exit' to stop)")
    print("-" * 60)

    while True:
        try:
            user_input = input("\nYou: ")
            if user_input.lower() in ['quit', 'exit']:
                break

            if not user_input.strip():
                continue

            response = rag_chain.invoke({"input": user_input})
            print(f"\nCo-writer (RAG Enhanced):\n{response['answer']}")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest_data()
    else:
        print("Usage:")
        print("  python rag_setup.py ingest  -> To ingest files from /knowledge_base")
        print("  python rag_setup.py chat    -> To chat with the RAG model")

        if len(sys.argv) > 1 and sys.argv[1] == "chat":
            chat_with_rag()
