import glob
import os
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def zip_artifacts() -> str:
    """Zip up the artifacts."""
    file_suffix = os.environ.get("ARTIFACTS_FILE_SUFFIX")
    if not file_suffix:
        raise ValueError("ARTIFACTS_FILE_SUFFIX is not set")
    file_name = f"{REPO_ROOT}/test-reports-{file_suffix}.zip"

    with zipfile.ZipFile(file_name, "w") as f:
        for file in glob.glob(f"{REPO_ROOT}/test/**/*.xml", recursive=True):
            f.write(file, os.path.relpath(file, REPO_ROOT))
        for file in glob.glob(f"{REPO_ROOT}/test/**/*.csv", recursive=True):
            f.write(file, os.path.relpath(file, REPO_ROOT))

    return file_name


def upload_to_s3_artifacts(file_name: str) -> None:
    """Upload the file to S3."""
    workflow_id = os.environ.get("GITHUB_RUN_ID")
    if not workflow_id:
        raise ValueError("GITHUB_RUN_ID is not set")
    import boto3  # type: ignore[import]

    S3_RESOURCE = boto3.client("s3")
    S3_RESOURCE.upload_file(
        file_name,
        "gha-artifacts",
        f"cattest_deleteme/pytorch/pytorch/{workflow_id}/{Path(file_name).name}",
    )


def zip_and_upload_artifacts() -> None:
    file_name = zip_artifacts()
    upload_to_s3_artifacts(file_name)
