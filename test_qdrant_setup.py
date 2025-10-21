#!/usr/bin/env python3
"""
Simple test script to verify Qdrant setup
"""

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Initialize client
print("Connecting to Qdrant...")
client = QdrantClient(url="localhost", port=6333)

# Check server health
print(f"Server info: {client.get_collections()}")

# Create a simple test collection
print("\nCreating test collection...")
collection_name = "test_collection"

try:
    # Delete if exists
    collections = client.get_collections().collections
    if any(c.name == collection_name for c in collections):
        client.delete_collection(collection_name)
        print(f"Deleted existing collection '{collection_name}'")

    # Create new collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": VectorParams(size=128, distance=Distance.COSINE)
        }
    )
    print(f"✅ Created collection '{collection_name}'")

    # Verify
    info = client.get_collection(collection_name)
    print(f"Collection info: {info.vectors_count} vectors")
    print(f"Status: {info.status}")

    print("\n✅ Qdrant setup is working correctly!")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
