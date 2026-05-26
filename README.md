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

## System Architecture

The backend is implemented as a LangGraph multi-agent pipeline:

1. Filter Agent
   - Interprets the user question
   - Selects and calls the appropriate data query tool

2. Analysis Agent
   - Interprets queried JSON data
   - Generates natural-language statistical summaries

3. Visualization Agent
   - Produces Vega-Lite interactive chart specifications
   - Adds tooltip and filtering interactions

The FastAPI backend exposes:
- POST /chat
- GET /health

## Setup
```bash
conda activate scisci

pip install -r requirements.txt

uvicorn main:app --reload --port 8000
```

## Frontend

The frontend is implemented separately using Vue.js and Vega-Lite.

Frontend repo:
<https://github.com/AdelineXinyi/project2_frontend>