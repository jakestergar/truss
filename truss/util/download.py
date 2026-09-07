import os
import shutil
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from truss.base.truss_config import ExternalData

B10CP_EXECUTABLE_NAME = "b10cp"
BLOB_DOWNLOAD_TIMEOUT_SECS = 600  # 10 minutes
B10CP_PATH_TRUSS_ENV_VAR_NAME = "B10CP_PATH_TRUSS"
B10CP_MAX_CONCURRENCY_ENV_VAR_NAME = "TRUSS_B10CP_MAX_CONCURRENCY"
B10CP_DEFAULT_MAX_CONCURRENCY = 8


def download_external_data(external_data: Optional[ExternalData], data_dir: Path):
    if external_data is None:
        return
    data_dir.mkdir(exist_ok=True)
    b10cp_path = _b10cp_path()

    # ensure parent directories exist
    for item in external_data.items:
        path = data_dir / item.local_data_path
        if data_dir not in path.parents:
            raise ValueError(
                "Local data path of external data cannot point to outside data directory"
            )
        path.parent.mkdir(exist_ok=True, parents=True)

    if b10cp_path is not None:
        print("b10cp found, using it to download external data")
        _download_external_data_using_b10cp(b10cp_path, data_dir, external_data)
        return

    # slow path
    _download_external_data_using_requests(data_dir, external_data)


def _b10cp_path() -> Optional[str]:
    return os.environ.get(B10CP_PATH_TRUSS_ENV_VAR_NAME)


def _b10cp_max_concurrency() -> int:
    raw_value = os.environ.get(B10CP_MAX_CONCURRENCY_ENV_VAR_NAME)
    if not raw_value:
        return B10CP_DEFAULT_MAX_CONCURRENCY
    max_concurrency = int(raw_value)
    if max_concurrency < 1:
        raise ValueError(
            f"`{B10CP_MAX_CONCURRENCY_ENV_VAR_NAME}` must be a positive integer, "
            f"got `{raw_value}`."
        )
    return max_concurrency


def _wait_for_b10cp(proc: "subprocess.Popen", url: str, download_to: Path) -> None:
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(
            f"Downloading `{url}` to `{download_to}` with b10cp failed with exit "
            f"code {returncode}."
        )


def _download_external_data_using_b10cp(
    b10cp_path: str, data_dir: Path, external_data: ExternalData
):
    max_concurrency = _b10cp_max_concurrency()
    in_flight: List[Tuple[subprocess.Popen, str, Path]] = []

    try:
        for item in external_data.items:
            while len(in_flight) >= max_concurrency:
                _wait_for_b10cp(*in_flight.pop(0))
            path = (data_dir / item.local_data_path).resolve()
            proc = _download_from_url_using_b10cp(b10cp_path, item.url, path)
            in_flight.append((proc, item.url, path))

        while in_flight:
            _wait_for_b10cp(*in_flight.pop(0))
    finally:
        for proc, _, _ in in_flight:
            proc.kill()


def _download_from_url_using_b10cp(b10cp_path: str, url: str, download_to: Path):
    return subprocess.Popen(
        [
            b10cp_path,
            "-source",
            url,  # Add quotes to work with any special characters.
            "-target",
            str(download_to),
        ]
    )


def _download_external_data_using_requests(data_dir: Path, external_data: ExternalData):
    for item in external_data.items:
        download_from_url_using_requests(
            item.url, (data_dir / item.local_data_path).resolve()
        )


def download_from_url_using_requests(URL: str, download_to: Path):
    # Streaming download to keep memory usage low
    resp = requests.get(
        URL, allow_redirects=True, stream=True, timeout=BLOB_DOWNLOAD_TIMEOUT_SECS
    )
    resp.raise_for_status()
    with download_to.open("wb") as file:
        shutil.copyfileobj(resp.raw, file)
