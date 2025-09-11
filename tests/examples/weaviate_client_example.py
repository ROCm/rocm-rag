# test_weaviate_connection.py

import pytest
import weaviate
from weaviate.connect import ConnectionParams
from weaviate.classes.query import Filter
from rocm_rag import config

@pytest.fixture(scope="module")
def weaviate_client():
    """Fixture to connect and disconnect the Weaviate client."""
    connection_params = ConnectionParams.from_url(
        url=f"{config.ROCM_RAG_WEAVIATE_URL}:{config.ROCM_RAG_WEAVIATE_PORT}",
        grpc_port=50051,  # Default gRPC port
    )
    client = weaviate.WeaviateClient(connection_params=connection_params)
    client.connect()
    yield client
    client.close()


def test_fetch_example_url(weaviate_client):
    """Ensure documents can be retrieved from Weaviate by example URL."""
    classname = config.ROCM_RAG_WEAVIATE_CLASSNAME
    example_url = config.ROCM_RAG_START_URLS[0]

    collections = weaviate_client.collections.get(classname)

    response = collections.query.fetch_objects(
        filters=Filter.by_property("url").equal(example_url),
        limit=100,
    )

    # Validate response structure
    assert hasattr(response, "objects"), "Response missing 'objects'"
    assert isinstance(response.objects, list), "Response objects should be a list"

    # Validate at least one object
    assert response.objects, f"No objects found for URL: {example_url}"

    # Check each object
    for obj in response.objects:
        assert hasattr(obj, "properties"), "Object missing 'properties'"
        props = obj.properties
        assert "url" in props, "Object missing 'url' property"
        assert "content" in props, "Object missing 'content' property"
        assert props["url"] == example_url, f"Object URL {props['url']} does not match queried URL {example_url}"
        assert isinstance(props["content"], str), "Content must be a string"
        assert props["content"].strip(), "Content should not be empty"
