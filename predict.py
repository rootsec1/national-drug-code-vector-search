from constants import *
from train import *

import json


def main():
    # Load the trained model
    similarity_model = init_sentence_transformer_model(TF_DEVICE)

    # Load the vector database client
    vector_db_client = init_vector_db_client()

    # Get the collection
    collection = get_ndc_vector_db_collection(vector_db_client)

    # Query the collection
    query = "Albuterol"
    query_embedding = embed_sentence(query, similarity_model, TF_DEVICE)

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=5
    )
    print(json.dumps(results))


if __name__ == "__main__":
    main()
