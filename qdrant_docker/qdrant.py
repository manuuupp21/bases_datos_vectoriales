import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Cargar el documento PDF de Lanzarote
logging.info("Cargando el documento PDF de Lanzarote...")
loader = PyPDFLoader("lanzarote_guia.pdf")
document_pages = loader.load()
logging.info(f"Documento PDF cargado con {len(document_pages)} páginas.")

# 2. Convertir a un formato parseable y dividir el texto (Ajustado para guía turística)
logging.info("Dividiendo el texto en chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
docs = text_splitter.split_documents(document_pages)
logging.info(f"Texto dividido en {len(docs)} chunks.")

# 3. Cargar el modelo de embeddings
logging.info("Cargando el modelo de embeddings...")
model_name = "intfloat/e5-base"
model_kwargs = {'device': 'cpu'}
encode_kwargs = {'normalize_embeddings': True, 'show_progress':True}
embeddings = HuggingFaceEmbeddings(
    model_name=model_name,
    model_kwargs=model_kwargs,
    encode_kwargs=encode_kwargs
)
embeddings_dimensions = 768
logging.info("Modelo de embeddings cargado.")

# 4. Guardar en la base de datos Qdrant
logging.info("Conectando a la base de datos Qdrant...")
url_database = "http://localhost:6333"
collection_name_index = "lanzarote_guide"

client = QdrantClient(url=url_database)
logging.info("Conexión establecida con Qdrant.")

try:
    logging.info(f"Verificando si la colección '{collection_name_index}' ya existe.")
    client.get_collection(collection_name_index)
    logging.info(f"La colección existe, procediendo a eliminarla para recargar.")
    client.delete_collection(collection_name_index)
except Exception as e:
    pass

logging.info(f"Creando la colección '{collection_name_index}'...")
client.create_collection(
    collection_name=collection_name_index,
    vectors_config=VectorParams(size=embeddings_dimensions, distance=Distance.COSINE)
)

logging.info("Transformando documentos a embeddings...")
transformed_docs = embeddings.embed_documents([doc.page_content for doc in docs])

logging.info("Insertando documentos en Qdrant...")
to_load_documents = []
for idx, (content, embedding_content) in enumerate(zip(docs, transformed_docs)):
    to_load_documents.append(
        PointStruct(
            id=idx,
            vector=embedding_content,
            payload= {'content': content.page_content, 'page': content.metadata.get('page', 0)}
        )
    )

client.upsert(
    collection_name=collection_name_index,
    wait=True,
    points=to_load_documents
)
logging.info(f"Se han añadido {len(docs)} documentos a '{collection_name_index}'.")

# Búsqueda de prueba
query = "¿Qué puedo visitar en el municipio de Haría?"
logging.info(f"Prueba de similitud: '{query}'")
found_docs = client.query_points(
    collection_name=collection_name_index,
    query= embeddings.embed_query(query),
    with_payload=True,
    limit=3
).points

for i, doc in enumerate(found_docs):
    print(f"\n--- Resultado {i+1} ---")
    print(doc.payload['content'])