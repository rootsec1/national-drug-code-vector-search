import chromadb
import pandas as pd

from sentence_transformers import SentenceTransformer
from constants import *
from util import *


def init_sentence_transformer_model(ml_device_target):
    similarity_model = SentenceTransformer(
        "all-mpnet-base-v2"
    ).to(ml_device_target)
    return similarity_model


def init_vector_db_client():
    chromadb_client = chromadb.HttpClient(
        host=CHROMADB_HOST,
        port=CHROMADB_PORT
    )
    chromadb_client.heartbeat()
    return chromadb_client


def get_ndc_vector_db_collection(vector_db_client):
    collection = vector_db_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return collection


def load_and_clean_data(product_csv_path, package_csv_path):
    product_df = pd.read_csv(product_csv_path)
    package_df = pd.read_csv(package_csv_path)
    merged_df = product_df.merge(
        package_df[[
            "PRODUCTID",
            "PACKAGEDESCRIPTION"
        ]],
        on="PRODUCTID"
    )
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df[[
        "PRODUCTID",
        "PROPRIETARYNAME",
        "ACTIVE_INGRED_UNIT",
        "ACTIVE_NUMERATOR_STRENGTH",
        "PACKAGEDESCRIPTION",
        "DOSAGEFORMNAME"
    ]]
    merged_df = merged_df.rename(columns={"DOSAGEFORMNAME": "FORM"})

    # Apply extract_medicine_info to the PACKAGEDESCRIPTION column and create new columns to store the extracted information
    medicine_info = merged_df["PACKAGEDESCRIPTION"].apply(
        parse_ndc_package_description
    )
    # Drop the PACKAGEDESCRIPTION column
    merged_df = merged_df.drop(columns=["PACKAGEDESCRIPTION"])

    merged_df["UNIT"] = medicine_info.apply(lambda x: x["UNIT"])
    merged_df["PROPRIETARYNAME"] = merged_df["PROPRIETARYNAME"].apply(
        clean_proprietary_name
    )

    # Display the dataframe
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.dropna()

    # Create a new row for each form that is separated by a comma and active_ingred_unit, ACTIVE_NUMERATOR_STRENGTH by a semi-colon
    merged_df = merged_df.assign(
        FORM=merged_df["FORM"].str.split(", ")
    ).explode("FORM").reset_index(drop=True)
    merged_df = merged_df.assign(
        ACTIVE_INGRED_UNIT=merged_df["ACTIVE_INGRED_UNIT"].str.split(
            "; "
        )
    ).explode("ACTIVE_INGRED_UNIT").reset_index(drop=True)
    merged_df = merged_df.assign(
        ACTIVE_NUMERATOR_STRENGTH=merged_df["ACTIVE_NUMERATOR_STRENGTH"].str.split(
            "; "
        )
    ).explode("ACTIVE_NUMERATOR_STRENGTH").reset_index(drop=True)

    # Group by every column except PRODUCTNDC and pick the first PRODUCTNDC in order to remove redundant rows
    merged_df = merged_df.groupby([
        "PROPRIETARYNAME",
        "ACTIVE_INGRED_UNIT",
        "FORM",
        "UNIT",
        "ACTIVE_NUMERATOR_STRENGTH"
    ]).first().reset_index()

    # Create a new column called SUMMARY that combines the PROPRIETARYNAME, DOSAGEFORMNAME, DOSAGE, and ACTIVE_INGRED_UNIT
    merged_df["SUMMARY"] = merged_df.apply(generate_summary, axis=1)
    merged_df = merged_df.drop_duplicates()
    merged_df = merged_df.dropna()
    return merged_df


def generate_and_store_embeddings(df, vector_db_collection, similarity_model, ml_device_target):
    training_df = df.copy(deep=True)
    print(f"Shape of items to be added: {training_df.shape}")
    print(f"Collection count (current): {vector_db_collection.count()}")
    for i in range(0, len(training_df), BATCH_SIZE):
        print(
            f"Processing batch {(i+1)//BATCH_SIZE + 1} out of {len(training_df)//BATCH_SIZE + 1}"
        )
        batch_df = training_df[i:i+BATCH_SIZE]
        store_sentence_embeddings(
            batch_df,
            vector_db_collection,
            similarity_model,
            ml_device_target
        )

    print(f"Collection count (updated): {vector_db_collection.count()}")


def main():
    print("[+] Initiating training pipeline...")
    vector_db_client = init_vector_db_client()
    similarity_model = init_sentence_transformer_model(TF_DEVICE)
    df = load_and_clean_data(
        "data/ndc/product.csv",
        "data/ndc/package.csv"
    )
    generate_and_store_embeddings(
        df,
        vector_db_client,
        similarity_model,
        TF_DEVICE
    )
    print("[+] (Re)Training pipeline complete!")


if __name__ == '__main__':
    main()
