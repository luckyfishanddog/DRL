"""Optional adapters for external, frozen MRTA datasets."""

from .ppo_instances import (
    DATASET_MANIFEST_ID,
    PHASE3_SMOKESET_ID,
    PHASE3_SMOKESET_V2_ID,
    INIT_BOOTSTRAP_DEVSET_ID,
    build_dataset_manifest,
    discover_ppo_instances,
    inspect_ppo_workbook,
    load_ppo_platform_instance,
    main_experimental_range_summary,
    select_init_bootstrap_devset,
    select_phase3_smokeset,
    select_phase3_smokeset_v2,
    to_parent_welds,
    validate_ppo_platform_instance,
)

__all__ = (
    "DATASET_MANIFEST_ID",
    "PHASE3_SMOKESET_ID",
    "PHASE3_SMOKESET_V2_ID",
    "INIT_BOOTSTRAP_DEVSET_ID",
    "build_dataset_manifest",
    "discover_ppo_instances",
    "inspect_ppo_workbook",
    "load_ppo_platform_instance",
    "main_experimental_range_summary",
    "select_init_bootstrap_devset",
    "select_phase3_smokeset",
    "select_phase3_smokeset_v2",
    "to_parent_welds",
    "validate_ppo_platform_instance",
)
