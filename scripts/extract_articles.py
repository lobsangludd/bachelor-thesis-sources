import json
import sys
import tarfile
from pathlib import Path

PERIODICAL_NAME = "Kmetijske in rokodelske novice"

rows = []
with tarfile.open(sys.argv[1], "r|gz") as tar:
    for member in tar:
        if not member.isfile() or not member.name.endswith(".json"):
            continue

        raw = tar.extractfile(member).read()

        if PERIODICAL_NAME.encode() not in raw:
            continue

        doc = json.loads(raw)

        if doc["periodical_name"] != PERIODICAL_NAME:
            continue

        pages = [page["text_csmtised_splitfixed"].strip() for page in doc["pages"]]

        rows.append({
            "urn": doc["dlib_url"].rstrip("/").rsplit("/", 1)[1],
            "title": doc["title"],
            "date": doc["date"],
            "issue": doc["issue_number"],
            "text": "\n".join(text for text in pages if text),
        })

rows.sort(key=lambda r: (r["date"], r["urn"]))

with open(Path(__file__).parent.parent / "articles.json", "w", encoding="utf-8") as fh:
    json.dump(rows, fh, ensure_ascii=False, indent=1)
    
print(len(rows), "articles")
