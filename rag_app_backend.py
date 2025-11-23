from dotenv import load_dotenv
load_dotenv()
import os
import json
import requests
from uuid import uuid4

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
# from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
# embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
embeddings = GoogleGenerativeAIEmbeddings(model="text-embedding-004")
embedding_dim = len(embeddings.embed_query("hello"))

API_URL = os.getenv("API_URL")
API_KEY = os.getenv("API_KEY")

index = faiss.IndexFlatL2(embedding_dim)
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

from langchain_core.documents import Document

docs = [

    # --------- METFORMIN ----------
    Document(
        page_content=(
            "Metformin reduces hepatic gluconeogenesis primarily through "
            "AMP-activated protein kinase (AMPK) activation. It improves "
            "insulin sensitivity, increases peripheral glucose uptake, and "
            "reduces intestinal glucose absorption. Metformin is first-line "
            "therapy for type 2 diabetes due to its efficacy, safety, and "
            "cardiovascular benefits."
        ),
        metadata={"source": "metformin", "topic": "diabetes"}
    ),

    Document(
        page_content=(
            "Non-glycemic benefits of Metformin include improved cardiovascular "
            "outcomes, reduced inflammation, and potential anti-neoplastic effects "
            "via mTOR pathway modulation. Studies suggest Metformin may reduce risk "
            "of certain cancers and delay aging-related metabolic decline."
        ),
        metadata={"source": "metformin", "topic": "benefits"}
    ),

    # --------- ATORVASTATIN + EZETIMIBE ----------
    Document(
        page_content=(
            "Atorvastatin inhibits HMG-CoA reductase in the liver, reducing "
            "endogenous cholesterol synthesis. Ezetimibe blocks intestinal "
            "cholesterol absorption by inhibiting the NPC1L1 transporter. "
            "Combining Atorvastatin with Ezetimibe produces additive LDL-C reduction "
            "and is clinically beneficial when monotherapy is insufficient."
        ),
        metadata={"source": "lipid", "topic": "statin-ezetimibe"}
    ),

    Document(
        page_content=(
            "Clinical trials have shown that adding Ezetimibe to statin therapy "
            "results in additional 15–25% reduction in LDL cholesterol. This combination "
            "is especially useful in patients with familial hypercholesterolemia or those "
            "unable to tolerate high-intensity statins."
        ),
        metadata={"source": "lipid", "topic": "cholesterol"}
    ),

    # --------- MRNA VACCINES ----------
    Document(
        page_content=(
            "mRNA vaccines enable rapid vaccine development due to their modular design. "
            "They deliver mRNA encoding viral antigens inside lipid nanoparticles (LNPs). "
            "Once inside host cells, mRNA is translated to produce the antigen, triggering "
            "strong antibody and T-cell responses. The technology allows fast updates for "
            "emerging variants such as BQ.1 and XBB.1.5."
        ),
        metadata={"source": "mrna", "topic": "vaccines"}
    ),

    Document(
        page_content=(
            "Research on next-generation mRNA vaccines includes self-amplifying mRNA, "
            "thermostable formulations, and mucosal delivery routes. These advancements "
            "aim to improve global distribution, reduce refrigeration needs, and enhance "
            "immune protection across populations."
        ),
        metadata={"source": "mrna", "topic": "research"}
    ),

    # --------- AI IN PHARMA R&D ----------
    Document(
        page_content=(
            "Artificial intelligence (AI) accelerates pharmaceutical R&D by improving "
            "target identification, structure-based drug design, and protein–ligand "
            "affinity prediction. Deep learning models can screen billions of compounds "
            "in silico, significantly reducing early discovery timelines."
        ),
        metadata={"source": "ai-pharma", "topic": "research"}
    ),

    Document(
        page_content=(
            "Challenges in AI-driven drug discovery include model interpretability, "
            "dataset bias, reproducibility, and regulatory approval for AI-designed molecules. "
            "Despite benefits in docking, screening, and lead optimization, validation in "
            "wet-lab settings remains essential."
        ),
        metadata={"source": "ai-pharma", "topic": "challenges"}
    ),

]

vector_store.add_documents(docs)

retriever = vector_store.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 1}
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

RAG_PROMPT = """
You are a helpful AI assistant.

Use ONLY the following context to answer the question.

Context:
{context}

Question:
{question}

Answer:
"""

def call_finetuned_llm(prompt: str) -> str:
    # Safety guard – don’t ever send an empty prompt to the model
    if not prompt.strip():
        return "Error: empty prompt was generated before calling the LLM."

    headers = {"Content-Type": "application/json"}
    if API_KEY:
        headers["x-api-key"] = API_KEY  # if you later protect the API with an API key

    payload = {"inputs": prompt}

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    try:
        resp.raise_for_status()
    except Exception as e:
        return f"HTTP error from API: {e} | body={resp.text}"

    outer = resp.json()

    # Handle both: proxy-wrapped {"statusCode":200,"body":"..."}
    # and direct {"result":[...]} styles, just in case.
    if isinstance(outer, dict) and "statusCode" in outer and "body" in outer:
        body_str = outer["body"]
        if isinstance(body_str, str):
            try:
                inner = json.loads(body_str)
            except json.JSONDecodeError:
                return f"Unexpected 'body' from API: {body_str}"
        else:
            inner = body_str
    else:
        inner = outer

    # inner should be {"result": [...]}
    result = inner.get("result", inner)

    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict) and "generated_text" in first:
            return first["generated_text"]
        return str(first)

    return str(result)
    
def generate_answer(question: str):

    # Retrieve
    docs = retriever.invoke(question)
    context = format_docs(docs)

    # Build prompt
    final_prompt = RAG_PROMPT.format(
        context=context,
        question=question
    )

    # LLM call
    answer = call_finetuned_llm(final_prompt)

    return {
        "question": question,
        "context": context,
        "answer": answer
    }