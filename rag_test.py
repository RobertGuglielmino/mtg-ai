
    
import json
import weaviate
from weaviate.classes.init import Auth
import os
from weaviate.classes.config import Configure, Property, DataType



client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=os.environ["WEAVIATE_API"],
)

client = weaviate.connect_to_weaviate_cloud(
    cluster_url=os.environ.get("WEAVIATE_URL"),
    auth_credentials=os.environ["WEAVIATE_API"],
)

collection_name = "MTGOfficialRules2"

chunks = client.collections.get(collection_name)
response = chunks.generate.near_text(
    query="history of git",
    limit=3,
    grouped_task="Summarize the key information here in bullet points"
)

print(response.generated)