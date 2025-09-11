from weaviate.classes.backup import BackupLocation
import weaviate
from weaviate.connect import ConnectionParams
from rocm_rag import config


# initialize vector store
connection_params = ConnectionParams.from_url(
    url=f"{config.DEFAULT_WEAVIATE_URL}:{config.DEFAULT_WEAVIATE_PORT}",
    grpc_port=50051,  # Default gRPC port
)
weaviate_client = weaviate.WeaviateClient(
    connection_params=connection_params
)
weaviate_client.connect()

# weaviate_classname = config.DEFAULT_WEAVIATE_CLASSNAME
weaviate_classname = "default"

print(f"Using Weaviate class: {weaviate_classname}")


backup_path = "/scratch/users/linsun12/weaviate_backup"

result = weaviate_client.backup.create(
    backup_id="rocm_blogs_backup",
    backend="filesystem",
    include_collections=[weaviate_classname],
    wait_for_completion=True,
    backup_location=BackupLocation.FileSystem(path=backup_path),  # Optional, requires Weaviate 1.27.2 / 1.28.0 or above and Python client 4.10.3 or above
)

print(result)

weaviate_client.close()
