from __future__ import annotations

import argparse
import os
from pathlib import Path

from mrta_data.ppo_instances import (
    build_dataset_manifest,
    select_phase3_smokeset,
    write_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit frozen PPO Excel platform instances")
    parser.add_argument("--ppo-root", default=os.environ.get("MRTA_PPO_ROOT"))
    parser.add_argument(
        "--manifest",
        default="data/manifests/PPO_DATASET_MANIFEST_V1.json",
    )
    parser.add_argument(
        "--smokeset",
        default="data/manifests/PPO_PHASE3_SMOKESET_V1.json",
    )
    args = parser.parse_args()
    if not args.ppo_root:
        parser.error("--ppo-root or MRTA_PPO_ROOT is required")
    manifest = build_dataset_manifest(args.ppo_root)
    smokeset = select_phase3_smokeset(manifest)
    write_json(Path(args.manifest), manifest)
    write_json(Path(args.smokeset), smokeset)
    summary = manifest["validation_summary"]
    print(f"dataset_manifest_hash={manifest['dataset_manifest_hash']}")
    print(f"excel_file_count={manifest['inventory_summary']['excel_file_count']}")
    print(f"platform_instance_sheet_count={manifest['inventory_summary']['platform_instance_sheet_count']}")
    print(f"valid_unique_instance_count={summary['valid_unique_instance_count']}")
    for entry in smokeset["instances"]:
        print(f"{entry['tier']}={entry['instance_id']} N={entry['actual_weld_count']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
