"""Optional adapters for external, frozen MRTA datasets."""

from .ppo_instances import (
    DATASET_MANIFEST_ID,
    PHASE3_SMOKESET_ID,
    build_dataset_manifest,
    discover_ppo_instances,
    inspect_ppo_workbook,
    load_ppo_platform_instance,
    select_phase3_smokeset,
    to_parent_welds,
    validate_ppo_platform_instance,
)

__all__ = (
    "DATASET_MANIFEST_ID",
    "PHASE3_SMOKESET_ID",
    "build_dataset_manifest",
    "discover_ppo_instances",
    "inspect_ppo_workbook",
    "load_ppo_platform_instance",
    "select_phase3_smokeset",
    "to_parent_welds",
    "validate_ppo_platform_instance",
)
