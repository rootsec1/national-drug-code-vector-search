import os
import torch

# ChromaDB
CHROMADB_HOST = os.environ.get("CHROMADB_HOST", "localhost")
CHROMADB_PORT = os.environ.get("CHROMADB_PORT", 8000)

# Tensorflow
TF_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BATCH_SIZE = 1000
COLLECTION_NAME = "ndc"
