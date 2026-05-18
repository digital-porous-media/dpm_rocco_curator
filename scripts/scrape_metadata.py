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

    project_ids = sorted(set(re.findall(r"DRP-\d{1,3}", index_html)))
    print(f"Found {len(project_ids)} projects. Downloading metadata...")

    success, failed = 0, []
    for drp_id in project_ids:
        url = f"{ARCHIVE_URL}{drp_id}/{drp_id}_metadata.json"
        dest = os.path.join(output_dir, f"{drp_id}.json")
        try:
            with urllib.request.urlopen(url, context=context) as resp:
                charset = resp.headers.get_content_charset() or "utf-8"
                text = resp.read().decode(charset)
            with open(dest, "w") as f:
                f.write(text)
            success += 1
        except Exception as e:
            failed.append((drp_id, str(e)))

    print(f"Downloaded {success}/{len(project_ids)} metadata files to {output_dir}/")
    if failed:
        print(f"Failed ({len(failed)}): {[drp for drp, _ in failed]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scrape DPM portal metadata from TACC Corral.")
    parser.add_argument("--output", default="data/metadata/", help="Output directory (default: data/metadata/)")
    args = parser.parse_args()
    scrape(args.output)
