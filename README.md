# national-drug-code-vector-search

Usage: `python train.py`

## Steps
1. Clone repository
2. Create virtual environment and install dependencies
3. Run chromadb server using: `chroma run --path <PATH_TO_DATA_DIR>`
4. Next, run the `train.py` file using: `python train.py` (This will read the data files from `./data`, generate a unique 1 liner summary for each drug, extract word embeddings from this summary and store them in `chromadb`

## How to query chromadb through the python client?
```
# Query the collection
query = "Albuterol Sulfate SOLUTION 0.63 mg/3mL"
query_embedding = embed_sentence(query)

results = collection.query(
    query_embeddings=query_embedding,
    n_results=5
)
json.dumps(results)
```
