"""Public identity shared by SHIMP publishing tools."""

DEFAULT_HF_DATASET_ID = "gregb/shimp"
DEFAULT_HF_PRETTY_NAME = "SHIMP (Shopify Hierarchy Inference from Merchant Products)"


def resolve_hf_repo_id(dataset_id: str, hf_repo_id: str | None = None) -> str:
    """Resolve a repository while allowing legacy short dataset-card labels."""

    return hf_repo_id or (dataset_id if "/" in dataset_id else DEFAULT_HF_DATASET_ID)
