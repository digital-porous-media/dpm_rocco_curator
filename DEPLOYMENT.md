# Deploying Rocco on the TACC VM

This document describes how Rocco is currently deployed on the TACC-hosted VM, for collaborators who need to update, restart, or debug the running instance.

**No secrets live in this file.** Actual hostnames, ports, and credentials live in the VM's
environment file (see [Secrets & the environment file](#secrets--the-environment-file)
below), never in git.

## Overview

| Component | Location | Managed by |
|---|---|---|
| Working clone (for editing/pulling) | `/home/<username>/dpm_rocco_curator` | Whoever's deploying |
| Deployed copy (what actually runs) | `/opt/dpm-rocco` | `rsync` from the working clone |
| Rocco app | systemd service | `systemctl` |
| Neo4j | systemd service | `systemctl` |
| Secrets | `/etc/dpm_rocco/app.env` | Manually maintained on the VM, outside git |

Each collaborator keeps their own clone under their own `/home/<username>` for pulling and
testing changes, then syncs a known-good state to the shared deployment path.

## Deploy / Update Workflow

1. **On the VM, in your personal clone**, pull the latest changes:

   ```bash
   cd /home/<username>/dpm_rocco_curator
   git pull origin main
   ```

2. **Sync to the deployment path.** `rsync` (not `cp`/`git clone` directly into `/opt`) keeps
   the deployed copy in sync without disturbing file permissions or unrelated files already
   there (e.g. logs, local `.env` overrides):

   ```bash
   sudo rsync -av --exclude='.venv' --exclude='.git' --exclude='.env' \
     /home/<username>/dpm_rocco_curator/ /opt/dpm-rocco/
   ```

3. **Install any new/updated dependencies** (if `pyproject.toml` changed):

   ```bash
   cd /opt/dpm-rocco
   source .venv/bin/activate
   python -m pip install -e ".[graph]"   # or whichever extras this deployment uses
   ```

4. **Restart the services:**

   ```bash
   sudo systemctl restart dpm-rocco
   sudo systemctl status dpm-rocco    # confirm it came back up cleanly
   ```

   Only restart `neo4j` if the graph itself needs a config change — it doesn't need to
   bounce on every app deploy:

   ```bash
   sudo systemctl restart neo4j
   ```

3. **Tail logs** to confirm the new version is actually serving:

   ```bash
   journalctl -u dpm-rocco -f
   ```

## Updating the Neo4j Graph (New or Changed Datasets)

See `docs/developer_guide/architecture.rst`'s
"Maintenance" section for more information about these scripts.

Run from `/opt/dpm-rocco` with the venv activated. These scripts read `NEO4J_URI`/`NEO4J_USER`/`NEO4J_PASSWORD` from the environment, but `systemd`'s `EnvironmentFile=` only applies to the `dpm-rocco` service itself — export the same file into your shell first:

```bash
cd /opt/dpm-rocco
source .venv/bin/activate
set -a; source /etc/dpm_rocco/app.env; set +a
```

1. **Fetch/refresh source metadata:**

   ```bash
   python scripts/scrape_metadata.py
   ```

2. **Load into Neo4j:**

   ```bash
   # Incremental — merges new/changed datasets, preserves embeddings on unchanged
   # nodes. Use this for adding a handful of new datasets.
   python scripts/load_graph.py --mode upsert

   # Full rebuild — clears and reloads everything. Only after a schema change.
   python scripts/load_graph.py --mode rebuild
   ```

3. **Re-embed:**

   ```bash
   # Whole graph — needed after --mode rebuild, or after changing the embedding
   # model / text-assembly logic
   python scripts/build_dataset_vector_index.py

   # Or just patch the dataset(s) added via --mode upsert
   python scripts/reembed_single_dataset.py --doi 10.xxxx/xxxx
   ```

4. **Verify:**

   ```bash
   python scripts/audit_schema.py --neo4j --verify
   ```

No service restart is needed for this one — `GraphStore` queries Neo4j live on every request,
it doesn't cache dataset data in the running process.

## Pulling Portal Documentation Updates (dpm_docs)

`search_portal_docs` reads from `data/portal_docs/docs/`, a synced copy of the
[dpm_docs](https://github.com/digital-porous-media/dpm_docs) repo — not a live fetch per query.

```bash
cd /opt/dpm_rocco
source .venv/bin/activate

# Check whether the local copy is behind dpm_docs' current HEAD, without fetching
python scripts/sync_dpm_docs.py --check

# Fetch and overwrite data/portal_docs/docs/ with the latest dpm_docs content
python scripts/sync_dpm_docs.py
```

**Restart required.** Unlike the Neo4j graph, the portal-docs heading tree is parsed once per
process and cached in memory (`portal_docs_tree.get_portal_docs_tree()`) — a sync with no
restart leaves the running service answering from the old snapshot:

```bash
sudo systemctl restart dpm-rocco
```

`--check` compares against `data/portal_docs/_sync_meta.json`'s last-synced commit SHA.

## Registries to Keep Updated

A few YAML files back the assistant and are loaded once per process, same caching behavior (and
same restart requirement) as the portal-docs tree above:

- **`data/tutorials.yaml`** — the Community Data tutorial notebook list. Update this whenever a notebook is added, renamed, or moved in the [dpm_teach](https://github.com/digital-porous-media/dpm_teach) repo/Community Data JupyterHub. The assistant treats notebook paths with the same strictness as dataset DOIs.
- **`data/domain_workflows.yaml`** — the curated DRP workflow/best-practices library behind `get_workflow_guidance`/`get_educational_context`. Update when a new method-level workflow should be surfaced, or when an entry's `example_datasets` needs adjustment.
- **`.env.example`** — not loaded by the running service, but keep it in sync with any new environment variable introduced in `src/`; it's the reference for `/etc/dpm_rocco/app.env`.

Restart after editing any of the cached files above:

```bash
sudo systemctl restart dpm-rocco
```

## systemd Services

Both Rocco and Neo4j run as systemd services so they survive VM reboots and restart automatically on failure. Unit file for the app (adjust paths/user for your actual VM setup — this is not committed as a real unit file anywhere in this repo):

```ini
# /etc/systemd/system/dpm-rocco.service
[Unit]
Description=Rocco Streamlit app
After=network.target neo4j.service

[Service]
Type=simple
WorkingDirectory=/opt/dpm_rocco
EnvironmentFile=/etc/dpm_rocco/app.env
ExecStart=/opt/dpm_rocco/.venv/bin/streamlit run rocco_ui.py --server.port 8501 --server.address 0.0.0.0
Restart=on-failure
User=dpm-rocco

[Install]
WantedBy=multi-user.target
```

Neo4j's service is whatever the standard Neo4j Debian/RPM package installs
(`neo4j.service`) — no custom unit needed there, just standard `systemctl
enable/start/stop/restart neo4j`.

## Secrets & the Environment File

All API keys and connection secrets live in `/etc/dpm_rocco/app.env` on the VM, loaded into the `dpm-rocco` service via `EnvironmentFile=`.

Environment variables are documented in [`.env.example`](.env.example) (`LLM_API_KEY`, `NEO4J_PASSWORD`, `SEMANTIC_SCHOLAR_API_KEY`, etc.).

**TODO:** `/etc/dpm_rocco/app.env` currently holds one collaborator's personal LLM API key rather than a dedicated credential. This should be replaced with a **service account.**

- File permissions on `/etc/dpm_rocco/app.env` should be locked down (`chmod 600`, owned by the service user) regardless of whose key is in it.

## Rollback

Since `/opt/dpm-rocco` is a plain `rsync` target (not its own git clone), the simplest rollback is to `git checkout <previous-tag-or-commit>` in the personal working clone, then re-run the sync + restart steps above.
