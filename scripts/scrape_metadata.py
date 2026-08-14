"""
Download all DRP metadata JSON files from the TACC Corral archive.

Scans the Digital Porous Media Portal archive index for project IDs matching
the pattern DRP-### and downloads each project's metadata JSON file.

Usage:
    python scripts/scrape_metadata.py
    python scripts/scrape_metadata.py --output data/metadata/

Output directory is gitignored; re-run to refresh metadata.
"""

import argparse
import os
import re
import ssl
import urllib.request


ARCHIVE_URL = "https://web.corral.tacc.utexas.edu/digitalporousmedia/archive/"


def scrape(output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    context = ssl._create_unverified_context()

    with urllib.request.urlopen(ARCHIVE_URL, context=context) as req:
        index_html = req.read().decode("utf-8")

    # \d+ (not \d{1,3}) so project numbers >= 1000 (e.g. DRP-1112) parse correctly
    # instead of being silently truncated to their first 3 digits.
    all_ids = sorted(set(re.findall(r"DRP-\d+(?:v\d+)?", index_html)))

    # The archive keeps every published version as its own directory
    # (DRP-1112, DRP-1112v2, ..., DRP-1112v5). Group by base project number and
    # keep only the highest version per project — a bare "DRP-N" with no "vN"
    # suffix is version 1.
    latest_by_project: dict[str, tuple[int, str]] = {}
    for drp_id in all_ids:
        m = re.match(r"(DRP-\d+)(?:v(\d+))?$", drp_id)
        if not m:
            continue
        base_id, version_str = m.group(1), m.group(2)
        version = int(version_str) if version_str else 1
        current = latest_by_project.get(base_id)
        if current is None or version > current[0]:
            latest_by_project[base_id] = (version, drp_id)

    project_ids = sorted(latest_by_project, key=lambda b: int(b.split("-")[1]))
    print(f"Found {len(project_ids)} projects (latest version of each). Downloading metadata...")

    success, failed = 0, []
    for base_id in project_ids:
        _version, dir_id = latest_by_project[base_id]
        url = f"{ARCHIVE_URL}{dir_id}/{dir_id}_metadata.json"
        # Always save under the base (unversioned) name so re-running the scraper
        # after a new version is published overwrites the same file rather than
        # leaving older-version files behind.
        dest = os.path.join(output_dir, f"{base_id}.json")
        try:
            with urllib.request.urlopen(url, context=context) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                text = resp.read().decode(charset)
            with open(dest, "w") as f:
                f.write(text)
            success += 1
        except Exception as e:
            failed.append((dir_id, str(e)))

    print(f"Downloaded {success}/{len(project_ids)} metadata files to {output_dir}/")
    if failed:
        print(f"Failed ({len(failed)}): {[drp for drp, _ in failed]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape DPM portal metadata from TACC Corral.")
    parser.add_argument("--output", default="data/metadata/", help="Output directory (default: data/metadata/)")
    args = parser.parse_args()
    scrape(args.output)
