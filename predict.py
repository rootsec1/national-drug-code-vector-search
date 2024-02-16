from constants import *
from train import *
from sheet2api import Sheet2APIClient


def export_search_results_to_sheet(results, user_query):
    print("Exporting first result to sheet")
    client = Sheet2APIClient(
        api_url="https://sheet2api.com/v1/9jLJI7TjaBFT/ss_rl_testing"
    )
    dict_results = dict(results)
    main_result = dict_results["metadatas"][0][0]
    summary = main_result["SUMMARY"]
    product_id = main_result["PRODUCTID"]
    client.create_row(
        sheet='Sheet1',
        row={
            "query": user_query,
            "result_summary": summary,
            "result_product_id": product_id,
            "score": 0
        }
    )
    print("Export completed")


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
    export_search_results_to_sheet(results, query)


if __name__ == "__main__":
    main()
