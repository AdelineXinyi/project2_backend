from pathlib import Path
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

REPO_ID = "Northwestern-CSSI/sciscinet-v2"

OUT_DIR = Path("./data")
OUT_DIR.mkdir(exist_ok=True)

UMICH_ID = "I27837315"

MAX_PAPERS = 100

SCAN_PAA = 5_000_000
SCAN_PAPERS = 5_000_000
SCAN_AUTHORS = 2_000_000
SCAN_REFS = 2_000_000
SCAN_PAPERFIELDS = 5_000_000

START_YEAR = 1900
END_YEAR = 2024

FILES = {
    "paa": "sciscinet_paper_author_affiliation.parquet",
    "papers": "sciscinet_papers.parquet",
    "authors": "sciscinet_authors.parquet",
    "refs": "sciscinet_paperrefs.parquet",
    "paperfields": "sciscinet_paperfields.parquet",
    "fields": "sciscinet_fields.parquet",
}


def stream_file(file_name):
    url = f"hf://datasets/{REPO_ID}/{file_name}"
    return load_dataset(
        "parquet",
        data_files={"train": url},
        split="train",
        streaming=True,
    )


def get_umich_paper_author_rows():
    print("\n=== Step 1: Find UMich Ann Arbor paper-author rows ===")

    rows = []
    seen_papers = set()

    for i, row in enumerate(tqdm(stream_file(FILES["paa"]), total=SCAN_PAA)):
        if i >= SCAN_PAA:
            break

        if str(row.get("institutionid")) == UMICH_ID:
            paper_id = str(row.get("paperid"))
            author_id = str(row.get("authorid"))

            rows.append({
                "paper_id": paper_id,
                "author_id": author_id,
            })
            seen_papers.add(paper_id)

            if len(seen_papers) >= MAX_PAPERS * 3:
                break

    if not rows:
        raise RuntimeError("No UMich Ann Arbor rows found. Increase SCAN_PAA.")

    df = pd.DataFrame(rows).drop_duplicates()

    print("Paper-author rows:", len(df))
    print("Unique papers:", df["paper_id"].nunique())
    print("Unique authors:", df["author_id"].nunique())

    return df


def build_papers(target_paper_ids):
    print("\n=== Step 2: Build papers.csv ===")

    rows = []

    for i, row in enumerate(tqdm(stream_file(FILES["papers"]), total=SCAN_PAPERS)):
        if i >= SCAN_PAPERS:
            break

        paper_id = str(row.get("paperid"))

        if paper_id not in target_paper_ids:
            continue

        try:
            year = int(row.get("year"))
        except Exception:
            continue

        if not (START_YEAR <= year <= END_YEAR):
            continue

        try:
            cited = int(row.get("citation_count") or row.get("cited_by_count") or 0)
        except Exception:
            cited = 0

        try:
            patent_count = int(row.get("patent_count") or 0)
        except Exception:
            patent_count = 0

        rows.append({
            "paper_id": paper_id,
            "title": f"Paper {paper_id}",
            "year": year,
            "cited_by_count": cited,
            "field": "Unknown",
            "patent_count": patent_count,
        })

        if len(rows) >= MAX_PAPERS:
            break

    if not rows:
        raise RuntimeError("No papers found. Increase SCAN_PAPERS.")

    papers = pd.DataFrame(rows).drop_duplicates("paper_id")
    papers.to_csv(OUT_DIR / "papers.csv", index=False)

    print("Saved data/papers.csv:", len(papers))
    return papers


def build_paper_author(pa_df, final_paper_ids):
    print("\n=== Step 3: Build paper_author.csv ===")

    out = pa_df[pa_df["paper_id"].isin(final_paper_ids)].drop_duplicates()
    out.to_csv(OUT_DIR / "paper_author.csv", index=False)

    print("Saved data/paper_author.csv:", len(out))
    return out


def build_authors(author_ids):
    print("\n=== Step 4: Build authors.csv ===")

    rows = []
    found = set()

    for i, row in enumerate(tqdm(stream_file(FILES["authors"]), total=SCAN_AUTHORS)):
        if i >= SCAN_AUTHORS:
            break

        author_id = str(row.get("authorid"))

        if author_id in author_ids:
            rows.append({
                "author_id": author_id,
                "name": str(row.get("display_name") or author_id),
                "institution": "University of Michigan–Ann Arbor",
            })
            found.add(author_id)

            if len(found) >= len(author_ids):
                break

    for author_id in author_ids - found:
        rows.append({
            "author_id": author_id,
            "name": author_id,
            "institution": "University of Michigan–Ann Arbor",
        })

    authors = pd.DataFrame(rows).drop_duplicates("author_id")
    authors.to_csv(OUT_DIR / "authors.csv", index=False)

    print("Saved data/authors.csv:", len(authors))
    return authors


def load_field_names():
    print("\n=== Step 5: Load field names ===")

    field_map = {}

    for row in tqdm(stream_file(FILES["fields"])):
        field_id = str(row.get("fieldid"))
        field_map[field_id] = str(row.get("display_name") or "Unknown")

    print("Loaded fields:", len(field_map))
    return field_map


def build_paper_fields(papers, field_map):
    print("\n=== Step 6: Attach fields and build fields.csv ===")

    final_paper_ids = set(papers["paper_id"].astype(str))
    paper_to_field = {}

    for i, row in enumerate(tqdm(stream_file(FILES["paperfields"]), total=SCAN_PAPERFIELDS)):
        if i >= SCAN_PAPERFIELDS:
            break

        paper_id = str(row.get("paperid"))

        if paper_id not in final_paper_ids:
            continue

        field_id = str(row.get("fieldid"))

        if field_id not in field_map:
            continue

        if paper_id not in paper_to_field:
            paper_to_field[paper_id] = field_map[field_id]

        if len(paper_to_field) >= len(final_paper_ids):
            break

    papers["field"] = papers["paper_id"].map(paper_to_field).fillna("Unknown")
    papers.to_csv(OUT_DIR / "papers.csv", index=False)

    fields = papers["field"].fillna("Unknown").value_counts().reset_index()
    fields.columns = ["field_name", "paper_count"]
    fields.to_csv(OUT_DIR / "fields.csv", index=False)

    print("Updated data/papers.csv with fields")
    print("Saved data/fields.csv:", len(fields))

    return papers, fields


def build_refs(final_paper_ids):
    print("\n=== Step 7: Build paper_references.csv ===")

    rows = []

    for i, row in enumerate(tqdm(stream_file(FILES["refs"]), total=SCAN_REFS)):
        if i >= SCAN_REFS:
            break

        citing = str(row.get("citing_paperid"))
        cited = str(row.get("cited_paperid"))

        if citing in final_paper_ids and cited in final_paper_ids:
            rows.append({
                "src_paper_id": citing,
                "tgt_paper_id": cited,
            })

    refs = pd.DataFrame(rows, columns=["src_paper_id", "tgt_paper_id"])
    refs = refs.drop_duplicates()
    refs.to_csv(OUT_DIR / "paper_references.csv", index=False)

    print("Saved data/paper_references.csv:", len(refs))
    return refs


def main():
    print("Using institution: University of Michigan–Ann Arbor")
    print("Institution ID:", UMICH_ID)
    print("Max papers:", MAX_PAPERS)

    pa_all = get_umich_paper_author_rows()
    target_paper_ids = set(pa_all["paper_id"].astype(str))

    papers = build_papers(target_paper_ids)
    final_paper_ids = set(papers["paper_id"].astype(str))

    pa = build_paper_author(pa_all, final_paper_ids)
    author_ids = set(pa["author_id"].astype(str))

    build_authors(author_ids)

    field_map = load_field_names()
    build_paper_fields(papers, field_map)

    build_refs(final_paper_ids)

    print("\n=== Done ===")
    print("Output files:")
    print("  data/papers.csv")
    print("  data/authors.csv")
    print("  data/paper_author.csv")
    print("  data/paper_references.csv")
    print("  data/fields.csv")


if __name__ == "__main__":
    main()