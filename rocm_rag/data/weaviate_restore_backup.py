from weaviate.classes.backup import BackupLocation
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


result = weaviate_client.backup.restore(
    backup_id="rocm_blogs_backup",
    backend="filesystem",
    wait_for_completion=True,
    backup_location=BackupLocation.FileSystem(path=str(config.ROCM_RAG_WORKSPACE / "rocm-rag" / "backup")),  # Required if a non-default location was used at creation
)

print(result)
