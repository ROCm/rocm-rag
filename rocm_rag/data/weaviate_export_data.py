import weaviate
from weaviate.connect import ConnectionParams
from rocm_rag import config
import json
import os


connection_params = ConnectionParams.from_url(
    url=f"{config.ROCM_RAG_WEAVIATE_URL}:{config.ROCM_RAG_WEAVIATE_PORT}",
    grpc_port=50051,  
)
weaviate_client = weaviate.WeaviateClient(
    connection_params=connection_params
)
weaviate_client.connect()

weaviate_classname = config.ROCM_RAG_WEAVIATE_CLASSNAME
collections = weaviate_client.collections.get(config.ROCM_RAG_WEAVIATE_CLASSNAME) 

os.makedirs("/rag-workspace/rocm-rag/backup", exist_ok=True)
with open(f"/rag-workspace/rocm-rag/backup/rocm_rag_backup_data.json", "w") as f:
    for obj in collections.iterator(include_vector=True):  # ask for vectors explicitly
        record = {
            "uuid": str(obj.uuid),
            "properties": obj.properties,
            "vector": obj.vector,
        }
        f.write(json.dumps(record) + "\n")

weaviate_client.close()
