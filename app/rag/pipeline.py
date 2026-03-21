from langchain_core.documents import Document
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.config import get_settings
from app.rag.chunker import load_and_chunk, load_document, chunk_document
from app.rag.vectorstore import add_documents, clear
from app.rag.retriever import retrieve
from app.rag.vectorstore import add_documents, clear, delete_by_source


PROMPT = ChatPromptTemplate.from_template("""
You are a precise question-answering assistant.
Answer using ONLY the context provided below.
If the answer is not in the context, say "I don't have that information."
Do not make up facts.

Context:
{context}

Question: {question}
""")


def get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )




def ingest(file_path: str, original_filename: str = None, clear_existing: bool = False) -> dict:
    if clear_existing:
        clear()

    source  = original_filename or file_path.split("/")[-1]

    # delete existing chunks for this source before re-inserting
    deleted = delete_by_source(source)
    if deleted > 0:
        print(f"[pipeline] replaced {deleted} existing chunks for '{source}'")

    chunks = load_and_chunk(file_path, source)
    ids    = add_documents(chunks)

    return {
        "source":       source,
        "chunks_added": len(ids),
        "replaced":     deleted > 0,
    }


def query(question: str) -> dict:
    """
    Retrieve relevant chunks then generate an answer with Groq.
    Returns answer + sources used.
    """
    # 1. Retrieve
    docs = retrieve(question)

    if not docs:
        return {
            "answer":  "No relevant documents found.",
            "sources": [],
        }

    # 2. Build context string
    context = _build_context(docs)

    # what LCEL does under the hood
    # formatted   = PROMPT.invoke({"context": context, "question": question})
    # llm_output  = get_llm().invoke(formatted)
    # answer      = StrOutputParser().invoke(llm_output)


    # 3. Run LCEL chain: prompt | llm | output parser
    chain  = PROMPT | get_llm() | StrOutputParser()
    answer = chain.invoke({"context": context, "question": question})

    return {
        "answer":  answer,
        "sources": list({
            d.metadata.get("source") + str(d.metadata.get("chunk_index")): {
                "source":      d.metadata.get("source"),
                "chunk":       d.metadata.get("chunk_index"),
                "page_number": d.metadata.get("page_number"),
                "extraction":  d.metadata.get("extraction"),
                "content":     d.page_content[:200],
            }
            for d in docs
        }.values()),
}



def _build_context(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        parts.append(f"[{i}] {source}\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)