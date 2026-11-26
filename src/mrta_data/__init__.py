"""Optional adapters for external, frozen MRTA datasets."""

from .platform_instances import (
    DATASET_MANIFEST_ID,
    build_dataset_manifest,
    discover_platform_instances,
    inspect_platform_workbook,
    load_platform_instance,
    main_experimental_range_summary,
    to_parent_welds,
    validate_platform_instance,
)

__all__ = (
    "DATASET_MANIFEST_ID",
    "build_dataset_manifest",
    "discover_platform_instances",
    "inspect_platform_workbook",
    "load_platform_instance",
    "main_experimental_range_summary",
    "to_parent_welds",
    "validate_platform_instance",
)
