## Data Construction
This project uses a small sample subset generated from the SciSciNet v2 dataset: 
https://springernature.figshare.com/collections/_/6076908

Source dataset:
- SciSciNet v2
- https://huggingface.co/datasets/Northwestern-CSSI/sciscinet-v2

The dataset is filtered to University of Michigan affiliated papers only.

### Construction workflow

1. Stream SciSciNet parquet files using HuggingFace `datasets`
2. Filter `paper_author_affiliation` rows by the University of Michigan institution id
3. Collect associated paper ids and author ids
4. Build filtered tables:
   - papers.csv
   - authors.csv
   - paper_author.csv
   - paper_references.csv
   - fields.csv
5. Keep only references and metadata associated with the sampled papers

### Commands
conda activate scisci
uvicorn main:app --reload --port 8000
python -m http.server 3000. (frontend)