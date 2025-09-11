from weaviate.classes.backup import BackupLocation
from weaviate.classes.backup import BackupLocation
import weaviate
from weaviate.connect import ConnectionParams
from rocm_rag import config
import json
from rocm_rag import config



# initialize vector store
connection_params = ConnectionParams.from_url(
    url=f"{config.ROCM_RAG_WEAVIATE_URL}:{config.ROCM_RAG_WEAVIATE_PORT}",
    grpc_port=50051,  # Default gRPC port
)
weaviate_client = weaviate.WeaviateClient(
    connection_params=connection_params
)
weaviate_client.connect()

backup_file_path = "/rag-workspace/rocm-rag/backup/rocm_rag_backup_data.json"

with weaviate_client.batch.fixed_size(batch_size=100) as batch:

    with open(backup_file_path, 'r') as f:
        for line in f:
            record = json.loads(line)
           
            batch.add_object(
                collection=config.ROCM_RAG_WEAVIATE_CLASSNAME,
                uuid=record["uuid"],
                properties=record["properties"],
                vector=record.get("vector", None),  # Optional vector field
            )



weaviate_client.close()
