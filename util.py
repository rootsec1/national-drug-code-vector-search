from tqdm import tqdm
from collections import Counter

from chromadb.errors import DuplicateIDError
import re


def parse_ndc_package_description(package_description):
    package_description = str(package_description).lower().strip()
    dosage_pattern = r'(\d+(\.\d+)?)\s*(ml|mg|g|capsule|tablet)'
    quantity_pattern = r'(\d+)\s*(vial|syringe|capsule|tablet|bottle|pack|ampule|tube|container|bag)'

    dosage_matches = re.findall(
        dosage_pattern, package_description, re.IGNORECASE)
    dosages = [match[0] for match in dosage_matches]
    units = [match[2] for match in dosage_matches]
    units.sort()
    units = ", ".join(units)

    # Extract quantity
    quantity_matches = re.findall(
        quantity_pattern, package_description, re.IGNORECASE)
    quantities = [match[0] for match in quantity_matches]
    quantities.sort()
    quantities = ", ".join(quantities)

    if len(dosages) == 1:
        dosages = dosages[0]
    if len(quantities) == 1:
        quantities = quantities[0]

    dosages = str(dosages).strip()
    units = str(units).strip()
    quantities = str(quantities).strip()

    combined_info = {
        "DOSAGE": dosages,
        "UNIT": units,
        "QUANTITY": quantities,
    }

    return combined_info


def clean_proprietary_name(prop_name):
    prop_name = str(prop_name).strip()
    pattern = "[^a-zA-Z0-9 ()%]"
    cleaned_string = re.sub(pattern, "", prop_name)
    if cleaned_string.count("(") == 1:
        cleaned_string = cleaned_string.replace("(", "").strip()
    if cleaned_string.count(")") == 1:
        cleaned_string = cleaned_string.replace(")", "").strip()

    # check if first word is a number or ends with a percentage symbol, if yes and the number starts with 0, remove the 0
    word_split = cleaned_string.split(" ")
    if len(word_split) > 0:
        first_word = word_split[0]
        if first_word.endswith("%"):
            first_word_new = first_word.replace("%", "").strip()
            if first_word_new.startswith("0"):
                first_word_new = first_word_new[1:]
                first_word_new = first_word_new + "%"
                cleaned_string = cleaned_string.replace(
                    first_word, first_word_new)
    return cleaned_string


def remove_redundant_words(sentence):
    words = sentence.split()
    seen = set()
    unique_words = [
        word for word in words if not (
            word in seen or seen.add(word)
        )
    ]
    return ' '.join(unique_words).strip()


def generate_summary(row):
    form = row["FORM"].lower().strip()
    unit = row["UNIT"].lower().strip()

    if form == unit:
        if len(unit) == 2:
            summary = f"{row['PROPRIETARYNAME']} {row['ACTIVE_NUMERATOR_STRENGTH']}{unit} {row['ACTIVE_INGRED_UNIT']}".strip(
            ).lower()
        else:
            summary = f"{row['PROPRIETARYNAME']} {row['ACTIVE_NUMERATOR_STRENGTH']} {unit} {row['ACTIVE_INGRED_UNIT']}".strip(
            ).lower()
    else:
        summary = f"{row['PROPRIETARYNAME']} {form} {row['ACTIVE_NUMERATOR_STRENGTH']} {unit} {row['ACTIVE_INGRED_UNIT']}".strip(
        ).lower()
    summary = remove_redundant_words(summary)
    return summary


def convert_tensor_to_float_list(tensor):
    tensor_list = list(tensor)
    return [float(x) for x in tensor_list]


def embed_sentence(sentence, similarity_model, ml_device_target):
    sentence = sentence.lower().strip()
    sentence_embeddings = similarity_model.encode(
        sentence,
        convert_to_tensor=True
    ).to(ml_device_target)
    sentence_embeddings = convert_tensor_to_float_list(sentence_embeddings)
    return sentence_embeddings


def store_sentence_embeddings(df, vector_db_collection, similarity_model, ml_device_target):
    document_list = []
    embedding_list = []
    metadata_list = []
    id_list = []
    id_index_hm = {}

    for index, row in tqdm(df.iterrows(), total=df.shape[0]):
        summary = row["SUMMARY"]
        metadata = dict(row)

        summary_embedding = embed_sentence(
            summary,
            similarity_model,
            ml_device_target
        )

        document_list.append(summary)
        embedding_list.append(summary_embedding)
        metadata_list.append(metadata)
        # To capture duplicates since duplicates are only detected by ID and not by body in ChromaDB
        id = str(hash(frozenset(Counter(summary).items())))
        id_list.append(id)
        id_index_hm[id] = index

    # Do not add more than 83333 documents at once (Max batch size for add method is 83333)

    try:
        vector_db_collection.add(
            documents=document_list,
            embeddings=embedding_list,
            metadatas=metadata_list,
            ids=id_list
        )
    except DuplicateIDError as e:
        print("Encountered duplicates")
        print(error_message)
        error_message = e.message()
        error_second_half = error_message.split("IDs: ")[1]
        duplicate_id_set = set(error_second_half.split(", "))

        new_document_list = []
        new_embedding_list = []
        new_metadata_list = []
        new_id_list = []

        for id, index in id_index_hm.items():
            if id in duplicate_id_set:
                continue

            summary = document_list[index]
            summary_embedding = embedding_list[index]
            metadata = metadata_list[index]

            new_document_list.append(summary)
            new_embedding_list.append(summary_embedding)
            new_metadata_list.append(metadata)
            new_id_list.append(id)

            vector_db_collection.add(
                documents=new_document_list,
                embeddings=new_embedding_list,
                metadatas=new_metadata_list,
                ids=new_id_list
            )

        print(
            f"Inserting unique entries only (size: {new_document_list.length})..."
        )
