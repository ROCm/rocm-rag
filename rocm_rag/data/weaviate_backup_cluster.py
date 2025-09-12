from weaviate.classes.backup import BackupLocation
import weaviate
from weaviate.connect import ConnectionParams
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

weaviate_classname = config.ROCM_RAG_WEAVIATE_CLASSNAME

print(f"Using Weaviate class: {weaviate_classname}")


backup_path = config.ROCM_RAG_WORKSPACE / "rocm-rag" / "backup"

result = weaviate_client.backup.create(
    backup_id="rocm_blogs_backup",
    backend="filesystem",
    include_collections=[weaviate_classname],
    wait_for_completion=True,
    backup_location=BackupLocation.FileSystem(path=str(backup_path)),  # Optional, requires Weaviate 1.27.2 / 1.28.0 or above and Python client 4.10.3 or above
)

print(result)

weaviate_client.close()
