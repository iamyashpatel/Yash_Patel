import os
from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from sentence_transformers import SentenceTransformer


MILVUS_HOST = "localhost"
MILVUS_PORT = "19530"
COLLECTION_NAME = "client_policy"
FOLDER_PATH = "Client_paragraphs"  
EMBEDDING_DIM = 1024  


connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)


if COLLECTION_NAME in utility.list_collections():
    collection = Collection(COLLECTION_NAME)  
else:
    
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="group", dtype=DataType.VARCHAR, max_length=255),  
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=5000),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=EMBEDDING_DIM),
        ],
        description="Client Policy Collection"
    )

    collection = Collection(COLLECTION_NAME, schema)
    print(f"✅ Created collection '{COLLECTION_NAME}' in Milvus!")


model = SentenceTransformer("BAAI/bge-m3")


text_data, embedding_data, group_data = [], [], []

for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".txt"):  
        file_path = os.path.join(FOLDER_PATH, filename)
        group_name = filename.replace(".txt", "")  

        with open(file_path, "r", encoding="utf-8") as f:
            paragraphs = [line.strip() for line in f.readlines() if line.strip()] 

        embeddings = model.encode(paragraphs, convert_to_tensor=False).tolist()

        
        text_data.extend(paragraphs)
        embedding_data.extend(embeddings)
        group_data.extend([group_name] * len(paragraphs))  


insert_data = [group_data, text_data, embedding_data]  
collection.insert(insert_data)

index_params = {"index_type": "IVF_FLAT", "metric_type": "COSINE", "params": {"nlist": 100}}
collection.create_index("embedding", index_params)

print(f"✅ Stored {len(text_data)} paragraphs from multiple files in Milvus!")
