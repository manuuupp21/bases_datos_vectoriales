import logging
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant

# Configuración del logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 1. Cargar el documento PDF
logging.info("Cargando el documento PDF...")
loader = PyPDFLoader("RGPD_ES.pdf")
document_pages = loader.load()
logging.info(f"Documento PDF cargado con {len(document_pages)} páginas.")

# 2. Convertir a un formato parseable y dividir el texto
logging.info("Dividiendo el texto en chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=300)
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
collection_name_index = "qdrant_index_class"

client = QdrantClient(url=url_database)
logging.info("Conexión establecida con Qdrant.")

try:
    logging.info(f"Verificando si la colección '{collection_name_index}' ya existe.")
    client.get_collection(collection_name_index)
    logging.info(f"La colección '{collection_name_index}' existe, procediendo a eliminarla.")
    client.delete_collection(collection_name_index)
    logging.info(f"Colección '{collection_name_index}' eliminada.")
except Exception as e:
    logging.info(f"La colección '{collection_name_index}' no existe o no se pudo eliminar: {e}")
    pass

logging.info(f"Creando la colección '{collection_name_index}'...")
client.create_collection(
    collection_name=collection_name_index,
    vectors_config=VectorParams(size=embeddings_dimensions, distance=Distance.COSINE)
)
logging.info(f"Colección '{collection_name_index}' creada.")

logging.info("Transformando documentos a embeddings...")
transformed_docs = embeddings.embed_documents([doc.page_content for doc in docs])
logging.info("Documentos transformados a embeddings.")

logging.info("Preparando documentos para cargar en Qdrant...")
to_load_documents = []
for idx, (content, embedding_content) in enumerate(zip(docs, transformed_docs)):
    to_load_documents.append(
        PointStruct(
            id=idx,
            vector=embedding_content,
            payload= {'content': content.page_content}
        )
    )
logging.info("Documentos preparados para la carga.")

logging.info("Insertando documentos en Qdrant...")
operation_info = client.upsert(
    collection_name=collection_name_index,
    wait=True,
    points=to_load_documents
)
logging.info(f"Operación de inserción completada: {operation_info}")
logging.info(f"Se han añadido {len(docs)} documentos a la colección '{collection_name_index}' en Qdrant.")

# Ejemplo de búsqueda de similitud
query = "¿Cuáles son los derechos del interesado?"
logging.info(f"Realizando búsqueda de similitud para la consulta: '{query}'")
found_docs = client.query_points(
    collection_name=collection_name_index,
    query= embeddings.embed_query(query),
    with_payload=True,
    limit=5
    ).points
logging.info(f"Búsqueda completada. Se encontraron {len(found_docs)} documentos.")

print(f"\nBúsqueda de similitud para: '{query}'")
for i, doc in enumerate(found_docs):
    print(f"\n--- Documento {i+1} ---")
    print(doc)

logging.info(f"Eliminando la colección '{collection_name_index}' para limpiar.")
client.delete_collection(collection_name_index)
logging.info("Proceso finalizado.")