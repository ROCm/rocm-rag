# haystack WeaviateDocumentStore requires _original_id field during retrieval.
# this field is used to store the original document ID from haystack Document object
# if extraction is through haystack, this field is automaticaly populated
# if extraction is through langgraph, this field is not mandatory
# to ensure compatibility, we manually add this field to the weaviate vector db if it does not exist

import weaviate
from weaviate.connect import ConnectionParams
from rocm_rag import config
from weaviate.classes.query import Filter

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

collections = weaviate_client.collections.get(weaviate_classname)

count = 0
for obj in collections.iterator():
    props = obj.properties
    uuid = obj.uuid

    # Only update if missing
    if props.get("_original_id") is None:
        print(f"Adding _original_id to object with UUID: {uuid}")
        props["_original_id"] = uuid
        collections.data.update(uuid=uuid, properties=props)
        count += 1

print(f"Updated {count} objects with missing _original_id")
weaviate_client.close()
