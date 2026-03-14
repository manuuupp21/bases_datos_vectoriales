from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import qdrant_client
import os
from dotenv import load_dotenv

# Configura api key Google
load_dotenv()

app = FastAPI(title="Lanzarote Tourist Guide RAG API")

# Inicializamos las conexiones globales al arrancar
qdrant_url = "http://localhost:6333"
collection_name = "lanzarote_guide"

embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/e5-base",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

client = qdrant_client.QdrantClient(url=qdrant_url)
vectorstore = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=qdrant_url,
)

# Configuramos el LLM (el "cerebro" que formulará la respuesta)
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)

# Prompt que le da la personalidad al bot
system_prompt = (
    "Eres un guía turístico experto en la isla de Lanzarote."
    "Usa los siguientes fragmentos de contexto recuperados para responder a la pregunta del usuario."
    "Si no sabes la respuesta o no está en el contexto, di simplemente que no tienes esa información."
    "Mantén una respuesta concisa, amable y útil.\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# Creamos la cadena RAG
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Modelos de datos para FastAPI
class QueryRequest(BaseModel):
    pregunta: str

class QueryResponse(BaseModel):
    respuesta: str

@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        # Ejecutamos el RAG
        response = rag_chain.invoke({"input": request.pregunta})
        return QueryResponse(respuesta=response["answer"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
