from datasets import load_dataset

REPO_ID = "Northwestern-CSSI/sciscinet-v2"

FILES = [
    "sciscinet_affiliations.parquet",
    "sciscinet_paper_author_affiliation.parquet",
    "sciscinet_papers.parquet",
    "sciscinet_authors.parquet",
    "sciscinet_paperrefs.parquet",
    "sciscinet_paperfields.parquet",
    "sciscinet_fields.parquet",
]


def load_first_row(file_name):
    print("\n" + "=" * 80)
    print(f"Testing: {file_name}")

    ds = load_dataset(
        "parquet",
        data_files={
            "train": f"hf://datasets/{REPO_ID}/{file_name}"
        },
        split="train",
        streaming=True,
    )

    row = next(iter(ds))

    print("\nColumns:")
    for col in row.keys():
        print(f"  - {col}")

    print("\nFirst row:")
    print(row)

    return row


def find_col(row, candidates):
    lower = {k.lower(): k for k in row.keys()}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def main():
    rows = {}

    for f in FILES:
        try:
            rows[f] = load_first_row(f)
        except Exception as e:
            print(f"\nFAILED loading {f}")
            print(type(e).__name__, e)

    print("\n" + "=" * 80)
    print("Column inference check")

    checks = {
        "sciscinet_affiliations.parquet": {
            "affiliation_id": ["AffiliationID", "affiliation_id", "id"],
            "affiliation_name": ["Affiliation_Name", "AffiliationName", "display_name", "name"],
        },
        "sciscinet_paper_author_affiliation.parquet": {
            "paper_id": ["PaperID", "paper_id"],
            "author_id": ["AuthorID", "author_id"],
            "affiliation_id": ["AffiliationID", "affiliation_id"],
        },
        "sciscinet_papers.parquet": {
            "paper_id": ["PaperID", "paper_id"],
            "title": ["PaperTitle", "Title", "title", "display_name"],
            "year": ["Year", "year", "PublicationYear", "publication_year"],
            "citation_count": ["CitationCount", "Citation_Count", "cited_by_count"],
            "doctype": ["DocType", "type"],
        },
        "sciscinet_authors.parquet": {
            "author_id": ["AuthorID", "author_id"],
            "author_name": ["Author_Name", "AuthorName", "name", "display_name"],
        },
        "sciscinet_paperrefs.parquet": {
            "citing_paper": ["Citing_PaperID", "CitingPaperID", "src_paper_id", "paper_id"],
            "cited_paper": ["Cited_PaperID", "CitedPaperID", "tgt_paper_id", "reference_paper_id"],
        },
        "sciscinet_paperfields.parquet": {
            "paper_id": ["PaperID", "paper_id"],
            "field_id": ["FieldID", "field_id"],
        },
        "sciscinet_fields.parquet": {
            "field_id": ["FieldID", "field_id"],
            "field_name": ["Field_Name", "FieldName", "field_name", "name"],
            "field_type": ["Field_Type", "FieldType", "field_type", "type"],
        },
    }

    for file_name, expected in checks.items():
        print("\n" + "-" * 80)
        print(file_name)

        row = rows.get(file_name)
        if row is None:
            print("  skipped because file failed to load")
            continue

        for logical_name, candidates in expected.items():
            found = find_col(row, candidates)
            print(f"  {logical_name}: {found}")

    print("\nDone.")


if __name__ == "__main__":
    main()