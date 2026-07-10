import re
import difflib
from pathlib import Path
import os
import requests

ZENODO_API = "https://zenodo.org/api/records"
COMMUNITY = "pdexplorer"
_DATA_DIR = Path(__file__).parent.parent / "data"
ZENODO_DEST_DIR = _DATA_DIR / "zenodo_spreadsheets"


_SUFFIX_RE = re.compile(r"[\s\-]*solutions?\s+assessment\s+spreadsheet.*$", re.IGNORECASE)


def _normalize(title):
    return _SUFFIX_RE.sub("", title).strip().lower()


def _list_community_records(community=COMMUNITY):
    records = []
    page = 1
    while True:
        response = requests.get(
            ZENODO_API,
            params={"communities": community, "size": 25, "sort": "mostrecent", "page": page},
        )
        response.raise_for_status()
        hits = response.json()["hits"]["hits"]
        if not hits:
            break
        records.extend(hits)
        page += 1
    return records

SOLUTIONS = [x["metadata"]["title"].split("- S")[0] for x in _list_community_records()]

# def download_solution_spreadsheet(solution_name, dest_dir=ZENODO_DEST_DIR, community=COMMUNITY):
#     """Download the latest-version assessment spreadsheet for solution_name from
#     the pdexplorer Zenodo. Returns the local xlsx path."""
#     records = _list_community_records(community)
#     target = _normalize(solution_name)

#     matches = [r for r in records if _normalize(r["metadata"]["title"]) == target]
#     if not matches:
#         titles = [r["metadata"]["title"] for r in records]
#         close = difflib.get_close_matches(solution_name, titles, n=3, cutoff=0.5)
#         raise FileNotFoundError(
#             f"No record found for {solution_name!r} in community {community!r}. "
#             f"Closest titles: {close}"
#         )
#     if len(matches) > 1:
#         raise ValueError(
#             f"Multiple records match {solution_name!r}: "
#             f"{[m['metadata']['title'] for m in matches]}"
#         )

#     record = matches[0]
#     xlsx_files = [f for f in record.get("files", []) if f["key"].lower().endswith(".xlsx")]
#     if not xlsx_files:
#         raise FileNotFoundError(
#             f"No .xlsx attached to record {record['id']} ({record['metadata']['title']})"
#         )
#     file_info = xlsx_files[0]

#     dest_dir = Path(dest_dir)
#     dest_dir.mkdir(parents=True, exist_ok=True)
#     dest_path = dest_dir / file_info["key"]

#     response = requests.get(file_info["links"]["self"])
#     response.raise_for_status()
#     dest_path.write_bytes(response.content)
#     return dest_path

def download_solution_spreadsheet(solution_name, dest_dir=ZENODO_DEST_DIR, community=COMMUNITY, overwrite=False):
    """Download the latest-version assessment spreadsheet for solution_name from
    the pdexplorer Zenodo. Returns the local xlsx path.

    If a matching .xlsx already exists in dest_dir, skips the Zenodo lookup and
    download entirely unless overwrite=True.
    """
    dest_dir = Path(dest_dir)
    target = _normalize(solution_name)

    if not overwrite:
        existing = {
            f: _normalize(f.stem) for f in dest_dir.glob("*.xlsx")
        } if dest_dir.exists() else {}
        for f, stem_norm in existing.items():
            if stem_norm == target:
                return f

    # Not found locally (or overwrite requested) — hit Zenodo.
    records = _list_community_records(community)
    matches = [r for r in records if _normalize(r["metadata"]["title"]) == target]
    if not matches:
        titles = [r["metadata"]["title"] for r in records]
        close = difflib.get_close_matches(solution_name, titles, n=3, cutoff=0.5)
        raise FileNotFoundError(
            f"No record found for {solution_name!r} in community {community!r}. "
            f"Closest titles: {close}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Multiple records match {solution_name!r}: "
            f"{[m['metadata']['title'] for m in matches]}"
        )
    record = matches[0]
    xlsx_files = [f for f in record.get("files", []) if f["key"].lower().endswith(".xlsx")]
    if not xlsx_files:
        raise FileNotFoundError(
            f"No .xlsx attached to record {record['id']} ({record['metadata']['title']})"
        )
    file_info = xlsx_files[0]
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / file_info["key"]

    if dest_path.exists() and not overwrite:
        return dest_path

    response = requests.get(file_info["links"]["self"])
    response.raise_for_status()
    dest_path.write_bytes(response.content)
    return dest_path