"""
Verify that the Neo4j instance supports vector indexes before intern development begins.

Checks:
  1. Neo4j version (must be 5.11+ for vector index support)
  2. Creates a small test vector index and confirms it appears in SHOW INDEXES
  3. Cleans up the test index

Usage:
    python scripts/check_neo4j_vector_support.py

Environment variables (same as rest of project):
    NEO4J_URI       — default bolt://localhost:7687
    NEO4J_USER      — default neo4j
    NEO4J_PASSWORD  — required
"""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

REQUIRED_MAJOR = 5
REQUIRED_MINOR = 11  # vector index GA release
TEST_INDEX_NAME = "_rocco_vec_support_check"
EMBEDDING_DIM = 1536  # OpenAI text-embedding-3-small / ada-002


def _connect():
    try:
        from neo4j import GraphDatabase
    except ImportError:
        print("ERROR: neo4j driver not installed. Run: pip install neo4j")
        sys.exit(1)

    uri = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    user = os.getenv("NEO4J_USER", "neo4j")
    password = os.getenv("NEO4J_PASSWORD", "")

    print(f"Connecting to {uri} as {user} ...")
    driver = GraphDatabase.driver(uri, auth=(user, password))
    driver.verify_connectivity()
    print("  Connected.\n")
    return driver


def check_version(session) -> bool:
    rows = session.run("CALL dbms.components() YIELD name, versions, edition").data()
    for row in rows:
        if row["name"] == "Neo4j Kernel":
            version_str = row["versions"][0]
            edition = row["edition"]
            print(f"  Neo4j version : {version_str} ({edition})")
            major, minor, *_ = (int(x) for x in version_str.split("."))
            ok = (major, minor) >= (REQUIRED_MAJOR, REQUIRED_MINOR)
            status = "PASS" if ok else "FAIL"
            print(
                f"  Version check : {status}  (need {REQUIRED_MAJOR}.{REQUIRED_MINOR}+)\n"
            )
            return ok
    print("  WARN: Could not determine Neo4j version from dbms.components()\n")
    return False


def check_vector_index(session) -> bool:
    # Create a temporary vector index on a dummy label
    try:
        session.run(
            f"""
            CREATE VECTOR INDEX {TEST_INDEX_NAME} IF NOT EXISTS
            FOR (n:_RoccoTestNode) ON (n.embedding)
            OPTIONS {{
                indexConfig: {{
                    `vector.dimensions`: {EMBEDDING_DIM},
                    `vector.similarity_function`: 'cosine'
                }}
            }}
            """
        )
    except Exception as e:
        print(f"  FAIL: Could not create vector index: {e}\n")
        return False

    # Confirm it appears
    indexes = session.run(
        "SHOW INDEXES WHERE type = 'VECTOR' AND name = $name",
        name=TEST_INDEX_NAME,
    ).data()

    if indexes:
        state = indexes[0].get("state", "unknown")
        print(f"  Vector index created successfully (state: {state})")
        print(f"  Index creation : PASS\n")
        created = True
    else:
        print("  FAIL: Vector index was not found after creation attempt.\n")
        created = False

    # Clean up
    session.run(f"DROP INDEX {TEST_INDEX_NAME} IF EXISTS")
    print(f"  Test index '{TEST_INDEX_NAME}' dropped (cleanup complete).\n")
    return created


def main():
    print("=" * 55)
    print("  Neo4j Vector Index Support Check")
    print("=" * 55 + "\n")

    driver = _connect()
    passed = []

    with driver.session() as session:
        print("[ 1 ] Version check")
        passed.append(check_version(session))

        print("[ 2 ] Vector index smoke test")
        passed.append(check_vector_index(session))

    driver.close()

    print("=" * 55)
    if all(passed):
        print("  ALL CHECKS PASSED")
    else:
        print("  ONE OR MORE CHECKS FAILED — review output above.")
        print("  Users should use USE_NEO4J=false until resolved.")
    print("=" * 55)
    sys.exit(0 if all(passed) else 1)


if __name__ == "__main__":
    main()
