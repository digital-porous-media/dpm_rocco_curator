"""
Load DRP metadata JSON files into Neo4j as a property graph.

Ported from CurationTools/JsonToNeo4jwKeywords.ipynb. Supports both a full
rebuild (clear + reload) and an incremental upsert (merge existing datasets,
refresh sub-nodes, preserve embeddings on unchanged nodes).

Embeddings and LLM keywords are NOT generated here; run
scripts/build_dataset_vector_index.py after loading to populate
descriptionEmbedding on each Dataset node.

Usage:
    # Full rebuild (default)
    python scripts/load_graph.py --mode rebuild

    # Incremental upsert (preserves descriptionEmbedding on unchanged nodes)
    python scripts/load_graph.py --mode upsert

    # Single file (useful for testing)
    python scripts/load_graph.py --file data/metadata/DRP-1.json

    # Skip vector index creation (e.g. before embeddings are ready)
    python scripts/load_graph.py --mode rebuild --skip-index
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()


# ---------------------------------------------------------------------------
# Helper functions (ported from notebook)
# ---------------------------------------------------------------------------

def _find_root(nodes: list[dict]) -> int:
    for i, node in enumerate(nodes):
        if node.get("id") == "NODE_ROOT":
            return i
    raise ValueError("Root node (NODE_ROOT) not found in metadata")


def _get_dataset_number(nodes: list[dict], root_key: int) -> int:
    project_id = nodes[root_key]["value"]["projectId"]
    nums = re.findall(r"\d{1,3}", project_id)
    return int(nums[0])


def _specify_root(metadata: dict, root_key: int) -> str:
    """Append projectId to NODE_ROOT identifier to make it globally unique."""
    project_id = metadata["nodes"][root_key]["value"]["projectId"]
    new_id = f"NODE_ROOT+{project_id}"
    for link in metadata["links"]:
        if link["source"] == "NODE_ROOT":
            link["source"] = new_id
    metadata["nodes"][root_key]["id"] = new_id
    return new_id


def _author_names(nodes: list[dict], root_key: int) -> str:
    authors = nodes[root_key]["value"].get("authors", [])
    return ", ".join(
        f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        for p in authors
    )


def _voxel_dim(node: dict) -> str | None:
    v = node["value"]
    x, y, z, dim = v.get("voxelX"), v.get("voxelY"), v.get("voxelZ"), v.get("voxelUnits")
    if x is None and y is None and z is None and dim is None:
        return None
    if dim:
        return f"X, Y, Z units (in {dim}s): {x}, {y}, {z}"
    return f"X, Y, Z units: {x}, {y}, {z}. Unit type not provided"


def _get_file_types(node: dict) -> list[str]:
    file_objs = node.get("value", {}).get("fileObjs", [])
    project_id = node.get("value", {}).get("projectId", "")
    types: set[str] = set()
    for f in file_objs:
        try:
            full_type = re.findall(r"(?:\.[A-Za-z]+\d?)+$", f.get("path", ""))[0]
            cleaned = re.sub(r"^\.", "", full_type)
            cleaned = re.sub(r"\.thumb\.jpg", "", cleaned)
            cleaned = re.sub(r"npy_[A-Za-z]{7}\.", "", cleaned)
            cleaned = re.sub(r"\.histogram\.(?:csv|jpg)", "", cleaned)
            cleaned = re.sub(r"(?:\.gif)|(?:\.jpg)", "", cleaned)
            if cleaned in ("tar.bz2", "tar.gz", "zip"):
                types.add("archive")
            elif cleaned in ("fre", "txt"):
                types.add("text")
            elif cleaned in ("ubc", "ubs", "raw"):
                types.add("binary")
            else:
                types.add(cleaned.lower())
        except IndexError:
            if project_id in ("drp.project.published.DRP-44", "drp.project.published.DRP-5"):
                types.add("binary")
    return list(types)


# ---------------------------------------------------------------------------
# GraphLoader
# ---------------------------------------------------------------------------

class GraphLoader:
    """
    Loads DRP metadata JSON files into a Neo4j property graph.

    Parameters
    ----------
    uri:      NEO4J_URI from .env (default bolt://localhost:7687)
    user:     NEO4J_USER from .env (default neo4j)
    password: NEO4J_PASSWORD from .env
    """

    def __init__(
        self,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ):
        self._uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self._user = user or os.getenv("NEO4J_USER", "neo4j")
        self._password = password or os.getenv("NEO4J_PASSWORD", "")
        self._driver = GraphDatabase.driver(self._uri, auth=(self._user, self._password))
        self._driver.verify_connectivity()

    def close(self):
        self._driver.close()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_directory(self, folder: str | Path, mode: str = "rebuild", skip_index: bool = False) -> None:
        """Load all JSON files in *folder*.

        mode='rebuild' clears the graph first.
        mode='upsert'  merges each dataset, preserving descriptionEmbedding.
        """
        folder = Path(folder)
        files = sorted(p for p in folder.iterdir() if p.suffix == ".json")
        if not files:
            print(f"No JSON files found in {folder}")
            return

        if mode == "rebuild":
            print(f"Clearing graph before rebuild...")
            self.clear()

        success, failed = 0, []
        for i, path in enumerate(files, 1):
            try:
                self.load_file(path, mode=mode)
                success += 1
                print(f"  [{i}/{len(files)}] {path.name} ✓")
            except Exception as exc:
                failed.append((path.name, str(exc)))
                print(f"  [{i}/{len(files)}] {path.name} ✗  {exc}")

        print(f"\nLoaded {success}/{len(files)} files.")
        if failed:
            print(f"Failed: {[name for name, _ in failed]}")

        if not skip_index:
            self.create_vector_index()

    def load_file(self, path: str | Path, mode: str = "upsert") -> None:
        """Load a single metadata JSON file."""
        path = Path(path)
        with open(path) as f:
            metadata = json.load(f)

        root_key = _find_root(metadata["nodes"])
        dataset_number = _get_dataset_number(metadata["nodes"], root_key)
        root_id = _specify_root(metadata, root_key)

        with self._driver.session() as session:
            if mode == "upsert":
                session.execute_write(self._upsert_dataset_node, metadata, root_key, root_id, dataset_number)
                # Wipe existing sub-nodes so we can safely recreate them
                session.execute_write(self._delete_sub_nodes, root_id)
            else:
                session.execute_write(self._create_dataset_node, metadata, root_key, root_id, dataset_number)

            self._load_sub_nodes(session, metadata, root_key, root_id, dataset_number)

    def clear(self) -> None:
        with self._driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")

    def create_vector_index(self) -> None:
        with self._driver.session() as session:
            session.run("""
                CREATE VECTOR INDEX `datasetDescription` IF NOT EXISTS
                FOR (n:Dataset) ON (n.descriptionEmbedding)
            """)
        print("Vector index 'datasetDescription' ready.")

    # ------------------------------------------------------------------
    # Internal: Dataset node
    # ------------------------------------------------------------------

    @staticmethod
    def _create_dataset_node(tx, metadata, root_key, root_id, dataset_number):
        v = metadata["nodes"][root_key]["value"]
        authors = _author_names(metadata["nodes"], root_key)
        tx.run("""
            CREATE (d:Dataset {identifier: $identifier})
            SET d.title            = $title,
                d.description      = $description,
                d.doi              = $doi,
                d.authors          = $authors,
                d.license          = $license,
                d.publicationDate  = $pub_date,
                d.datasetNumber    = $dataset_number
        """,
            identifier=root_id, title=v.get("title"), description=v.get("description"),
            doi=v.get("doi"), authors=authors, license=v.get("license"),
            pub_date=v.get("publicationDate"), dataset_number=dataset_number,
        )

    @staticmethod
    def _upsert_dataset_node(tx, metadata, root_key, root_id, dataset_number):
        v = metadata["nodes"][root_key]["value"]
        authors = _author_names(metadata["nodes"], root_key)
        # MERGE on identifier so descriptionEmbedding / llmKeywords are not overwritten
        tx.run("""
            MERGE (d:Dataset {identifier: $identifier})
            SET d.title            = $title,
                d.description      = $description,
                d.doi              = $doi,
                d.authors          = $authors,
                d.license          = $license,
                d.publicationDate  = $pub_date,
                d.datasetNumber    = $dataset_number
        """,
            identifier=root_id, title=v.get("title"), description=v.get("description"),
            doi=v.get("doi"), authors=authors, license=v.get("license"),
            pub_date=v.get("publicationDate"), dataset_number=dataset_number,
        )

    @staticmethod
    def _delete_sub_nodes(tx, root_id):
        tx.run("""
            MATCH (child)-[:PART_OF]->(d:Dataset {identifier: $root_id})
            DETACH DELETE child
        """, root_id=root_id)

    # ------------------------------------------------------------------
    # Internal: sub-nodes and relationships
    # ------------------------------------------------------------------

    def _load_sub_nodes(self, session, metadata, root_key, root_id, dataset_number):
        nodes = metadata["nodes"]
        v_root = nodes[root_key]["value"]

        for i, pub in enumerate(v_root.get("relatedPublications", [])):
            session.execute_write(self._create_related_publication, pub, root_id, dataset_number)

        for sw in v_root.get("relatedSoftware", []):
            session.execute_write(self._create_related_software, sw, root_id, dataset_number)

        for rd in v_root.get("relatedDatasets", []):
            session.execute_write(self._create_related_dataset, rd, root_id, dataset_number)

        for node in nodes:
            dtype = node.get("value", {}).get("dataType")
            if dtype == "sample":
                session.execute_write(self._create_sample, node, root_id, dataset_number)
            elif dtype == "digital_dataset":
                session.execute_write(self._create_digital_dataset, node, root_id, dataset_number)
            elif dtype == "analysis_data":
                session.execute_write(self._create_analysis_dataset, node, root_id, dataset_number)

        for link in metadata["links"]:
            session.execute_write(self._establish_connection, link["source"], link["target"], root_id)

    @staticmethod
    def _create_related_publication(tx, pub, root_id, dataset_number):
        tx.run("""
            CREATE (rp:RelatedPublication {title: $title})
            SET rp.authors         = $authors,
                rp.abstract        = $abstract,
                rp.link            = $link,
                rp.publicationDate = $pub_date,
                rp.datasetNumber   = $dataset_number
            WITH rp
            MATCH (d:Dataset {identifier: $root_id})
            CREATE (d)<-[:PART_OF]-(rp)
        """,
            title=pub.get("publicationTitle"), authors=pub.get("publicationAuthor"),
            abstract=pub.get("publicationDescription"), link=pub.get("publicationLink"),
            pub_date=pub.get("publicationDateOfPublication"), dataset_number=dataset_number,
            root_id=root_id,
        )

    @staticmethod
    def _create_related_software(tx, sw, root_id, dataset_number):
        tx.run("""
            CREATE (rs:RelatedSoftware {title: $title})
            SET rs.description   = $description,
                rs.link          = $link,
                rs.datasetNumber = $dataset_number
            WITH rs
            MATCH (d:Dataset {identifier: $root_id})
            CREATE (d)<-[:PART_OF]-(rs)
        """,
            title=sw.get("softwareTitle"), description=sw.get("softwareDescription"),
            link=sw.get("softwareLink"), dataset_number=dataset_number, root_id=root_id,
        )

    @staticmethod
    def _create_related_dataset(tx, rd, root_id, dataset_number):
        tx.run("""
            CREATE (rd:RelatedDataset {title: $title})
            SET rd.description   = $description,
                rd.link          = $link,
                rd.datasetNumber = $dataset_number
            WITH rd
            MATCH (d:Dataset {identifier: $root_id})
            CREATE (d)<-[:PART_OF]-(rd)
        """,
            title=rd.get("datasetTitle"), description=rd.get("datasetDescription"),
            link=rd.get("datasetLink"), dataset_number=dataset_number, root_id=root_id,
        )

    @staticmethod
    def _create_sample(tx, node, root_id, dataset_number):
        v = node["value"]
        tx.run("""
            CREATE (s:Sample {title: $name})
            SET s.identifier          = $identifier,
                s.location            = $location,
                s.porousMediaType     = $media_type,
                s.porosity            = $porosity,
                s.source              = $source,
                s.description         = $description,
                s.geographicOrigin    = $geo_origin,
                s.grainSizeAvg        = $gs_avg,
                s.grainSizeMin        = $gs_min,
                s.grainSizeMax        = $gs_max,
                s.grainSizeUnits      = $gs_units,
                s.collectionMethod    = $collection,
                s.onshoreOffshore     = $onshore,
                s.depth               = $depth,
                s.waterDepth          = $water_depth,
                s.procedure           = $procedure,
                s.equipment           = $equipment,
                s.algorithmDescription= $algo_desc,
                s.datasetNumber       = $dataset_number
            WITH s
            MATCH (d:Dataset {identifier: $root_id})
            MERGE (s)-[:PART_OF]->(d)
        """,
            identifier=node["id"], name=v.get("name"), location=v.get("geographicalLocation"),
            media_type=v.get("porousMediaType"), porosity=v.get("porosity"),
            source=v.get("source"), description=v.get("description"),
            geo_origin=v.get("geographicOrigin"), gs_avg=v.get("grainSizeAvg"),
            gs_min=v.get("grainSizeMin"), gs_max=v.get("grainSizeMax"),
            gs_units=v.get("grainSizeUnits"), collection=v.get("collectionMethod"),
            onshore=v.get("onshoreOffshore"), depth=v.get("depth"),
            water_depth=v.get("waterDepth"), procedure=v.get("procedure"),
            equipment=v.get("equipment"), algo_desc=v.get("algorithmDescription"),
            dataset_number=dataset_number, root_id=root_id,
        )

    @staticmethod
    def _create_digital_dataset(tx, node, root_id, dataset_number):
        v = node["value"]
        file_types = _get_file_types(node)
        voxels = _voxel_dim(node)
        tx.run("""
            CREATE (dd:DigitalDataset {title: $name})
            SET dd.identifier                = $identifier,
                dd.voxelDimensions           = $voxels,
                dd.description               = $description,
                dd.segmented                 = $segmented,
                dd.imagingCenter             = $imaging_center,
                dd.imagingEquipmentAndModel  = $imaging_eq,
                dd.imageFormat               = $img_format,
                dd.imageDimensions           = $img_dims,
                dd.imageByteOrder            = $byte_order,
                dd.dimensionality            = $dimensionality,
                dd.referencedSample         = $ref_sample,
                dd.numberOfFiles             = $n_files,
                dd.fileTypes                 = $file_types,
                dd.datasetNumber             = $dataset_number
            WITH dd
            MATCH (d:Dataset {identifier: $root_id})
            MERGE (dd)-[:PART_OF]->(d)
        """,
            identifier=node["id"], name=node.get("label"), voxels=voxels,
            description=v.get("description"), segmented=v.get("isSegmented"),
            imaging_center=v.get("imagingCenter"), imaging_eq=v.get("imagingEquipmentAndModel"),
            img_format=v.get("imageFormat"), img_dims=v.get("imageDimensions"),
            byte_order=v.get("imageByteOrder"), dimensionality=v.get("dimensionality"),
            ref_sample=v.get("sample"),
            n_files=len(v.get("fileObjs", [])), file_types=file_types,
            dataset_number=dataset_number, root_id=root_id,
        )

    @staticmethod
    def _create_analysis_dataset(tx, node, root_id, dataset_number):
        v = node["value"]
        file_types = _get_file_types(node)
        tx.run("""
            CREATE (ad:AnalysisDataset {title: $name})
            SET ad.identifier               = $identifier,
                ad.segmented                = $segmented,
                ad.description              = $description,
                ad.type                     = $type,
                ad.referencedDigitalDataset = $ref_dig,
                ad.referencedSample         = $ref_sample,
                ad.numberOfFiles            = $n_files,
                ad.fileTypes                = $file_types,
                ad.datasetNumber            = $dataset_number
            WITH ad
            MATCH (d:Dataset {identifier: $root_id})
            MERGE (ad)-[:PART_OF]->(d)
        """,
            identifier=node["id"], name=node.get("label"),
            segmented=v.get("isSegmented"), description=v.get("description"),
            type=v.get("datasetType"), ref_dig=v.get("digitalDataset"),
            ref_sample=v.get("sample"), n_files=len(v.get("fileObjs", [])),
            file_types=file_types, dataset_number=dataset_number, root_id=root_id,
        )

    @staticmethod
    def _establish_connection(tx, source, target, root_id):
        if source == root_id:
            tx.run("""
                MATCH (s {identifier: $source})
                MATCH (t {identifier: $target})
                MERGE (s)<-[:PART_OF]-(t)
            """, source=source, target=target)
        else:
            tx.run("""
                MATCH (s {identifier: $source})
                MATCH (t {identifier: $target})
                MERGE (s)<-[:INPUT_FOR]-(t)
            """, source=source, target=target)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Load DRP metadata JSON files into Neo4j.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode", choices=["rebuild", "upsert"], default="rebuild",
                        help="rebuild: clear then reload all. upsert: merge, preserve embeddings. (default: rebuild)")
    parser.add_argument("--folder", default="data/metadata/",
                        help="Directory of metadata JSON files (default: data/metadata/)")
    parser.add_argument("--file", default=None,
                        help="Load a single JSON file instead of a directory")
    parser.add_argument("--skip-index", action="store_true",
                        help="Skip creating the vector index (useful before embeddings are ready)")
    args = parser.parse_args()

    loader = GraphLoader()
    try:
        if args.file:
            print(f"Loading single file: {args.file}  [mode={args.mode}]")
            loader.load_file(args.file, mode=args.mode)
            if not args.skip_index:
                loader.create_vector_index()
        else:
            print(f"Loading directory: {args.folder}  [mode={args.mode}]")
            loader.load_directory(args.folder, mode=args.mode, skip_index=args.skip_index)
    finally:
        loader.close()


if __name__ == "__main__":
    main()
