"""
agents/tools.py
---------------
Data query tools that the LangGraph agents can call.
Each tool reads from the local CSV files in ./data/
"""

import pandas as pd
import json
from pathlib import Path
from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent.parent / "data"

# ── Load data once at startup ─────────────────────────────────────────────────
papers_df     = pd.read_csv(DATA_DIR / "papers.csv")
authors_df    = pd.read_csv(DATA_DIR / "authors.csv")
paper_auth_df = pd.read_csv(DATA_DIR / "paper_author.csv")
refs_df       = pd.read_csv(DATA_DIR / "paper_references.csv")
fields_df     = pd.read_csv(DATA_DIR / "fields.csv")


@tool
def get_papers_by_year() -> str:
    """Return the count of papers published each year."""
    result = (
        papers_df.groupby("year")
        .size()
        .reset_index(name="count")
        .sort_values("year")
    )
    return result.to_json(orient="records")


@tool
def get_papers_by_field(top_n: int = 15) -> str:
    """Return the top N fields by paper count.
    
    Args:
        top_n: how many fields to return (default 15)
    """
    result = (
        papers_df["field"]
        .value_counts()
        .head(top_n)
        .reset_index()
    )
    result.columns = ["field", "count"]
    return result.to_json(orient="records")


@tool
def get_citation_stats() -> str:
    """Return summary statistics about paper citations."""
    stats = {
        "total_papers": len(papers_df),
        "total_citation_edges": len(refs_df),
        "avg_citations": round(papers_df["cited_by_count"].mean(), 2),
        "max_citations": int(papers_df["cited_by_count"].max()),
        "most_cited_paper": papers_df.loc[
            papers_df["cited_by_count"].idxmax(), "title"
        ],
    }
    return json.dumps(stats)


@tool
def get_top_authors(top_n: int = 10) -> str:
    """Return the most prolific authors by paper count.
    
    Args:
        top_n: how many authors to return (default 10)
    """
    counts = (
        paper_auth_df.groupby("author_id")
        .size()
        .reset_index(name="paper_count")
    )
    merged = counts.merge(authors_df, on="author_id")
    result = (
        merged[["name", "paper_count"]]
        .sort_values("paper_count", ascending=False)
        .head(top_n)
    )
    return result.to_json(orient="records")


@tool
def get_collaboration_network(min_papers: int = 3) -> str:
    """Return author collaboration network edges.
    Authors who co-authored the same paper are connected.
    Only includes authors with at least min_papers papers.

    Args:
        min_papers: minimum papers an author must have to be included
    """
    # filter to prolific authors only (keep graph manageable)
    author_counts = (
        paper_auth_df.groupby("author_id")
        .size()
        .reset_index(name="count")
    )
    active = set(
        author_counts[author_counts["count"] >= min_papers]["author_id"]
    )

    filtered = paper_auth_df[paper_auth_df["author_id"].isin(active)]

    # self-join on paper_id to get co-author pairs
    collab = filtered.merge(filtered, on="paper_id")
    collab = collab[collab["author_id_x"] < collab["author_id_y"]]  # deduplicate
    collab = (
        collab.groupby(["author_id_x", "author_id_y"])
        .size()
        .reset_index(name="weight")
    )

    # attach names
    id_to_name = authors_df.set_index("author_id")["name"].to_dict()
    collab["source"] = collab["author_id_x"].map(id_to_name)
    collab["target"] = collab["author_id_y"].map(id_to_name)

    nodes = [
        {"id": aid, "name": id_to_name.get(aid, aid)}
        for aid in active
    ]
    edges = collab[["source", "target", "weight"]].to_dict(orient="records")

    return json.dumps({"nodes": nodes[:200], "edges": edges[:500]})


@tool
def get_citation_network(max_nodes: int = 100) -> str:
    """Return paper citation network (nodes = papers, edges = citations).

    Args:
        max_nodes: maximum number of paper nodes to include
    """
    # pick most-cited papers as nodes
    top_papers = (
        papers_df.nlargest(max_nodes, "cited_by_count")[
            ["paper_id", "title", "year", "cited_by_count", "field"]
        ]
    )
    node_ids = set(top_papers["paper_id"])

    edges = refs_df[
        refs_df["src_paper_id"].isin(node_ids) &
        refs_df["tgt_paper_id"].isin(node_ids)
    ].to_dict(orient="records")

    nodes = top_papers.to_dict(orient="records")
    return json.dumps({"nodes": nodes, "edges": edges})


# expose all tools as a list for LangGraph
ALL_TOOLS = [
    get_papers_by_year,
    get_papers_by_field,
    get_citation_stats,
    get_top_authors,
    get_collaboration_network,
    get_citation_network,
]