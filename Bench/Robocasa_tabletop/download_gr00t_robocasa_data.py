"""Download the released RoboCasa GR1 datasets from Hugging Face.

The NVIDIA repository is large. This downloader intentionally lists one selected
folder at a time instead of enumerating every file in the repository first.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, urljoin

import requests
from huggingface_hub import hf_hub_download


REPO_ID = "nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim"
REPO_TYPE = "dataset"
LOCAL_DIR = Path("playground/Datasets/nvidia/PhysicalAI-Robotics-GR00T-X-Embodiment-Sim")
DEFAULT_ENDPOINT = "https://huggingface.co"

FOLDERS = [
    "gr1_unified.PnPBottleToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCanToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPCupToDrawerClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPMilkToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPPotatoToMicrowaveClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PnPWineToCabinetClose_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromCuttingboardToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlacematToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToBowlSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPanSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromPlateToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToCardboardboxSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPlateSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToPotSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredbasketSplitA_GR1ArmsAndWaistFourierHands_1000",
    "gr1_unified.PosttrainPnPNovelFromTrayToTieredshelfSplitA_GR1ArmsAndWaistFourierHands_1000",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--folder",
        action="append",
        choices=FOLDERS,
        help="Download one selected folder. Repeat the option to select multiple folders.",
    )
    parser.add_argument("--workers", type=int, default=8, help="Concurrent download workers (default: 8).")
    parser.add_argument("--list-retries", type=int, default=5, help="Retries for each folder listing (default: 5).")
    parser.add_argument("--download-retries", type=int, default=8, help="Retries for each file download (default: 8).")
    parser.add_argument("--connect-timeout", type=float, default=15.0, help="Listing connection timeout in seconds.")
    parser.add_argument("--read-timeout", type=float, default=60.0, help="Listing read timeout in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="List selected files without downloading them.")
    return parser.parse_args()


def hub_headers() -> dict[str, str]:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
    return {"Authorization": f"Bearer {token}"} if token else {}


def get_with_retry(
    session: requests.Session,
    url: str,
    *,
    params: dict[str, str] | None,
    headers: dict[str, str],
    timeout: tuple[float, float],
    max_retries: int,
) -> requests.Response:
    for attempt in range(1, max_retries + 1):
        try:
            response = session.get(url, params=params, headers=headers, timeout=timeout)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            if attempt == max_retries:
                raise RuntimeError(f"Listing request failed after {max_retries} attempts: {url}") from exc
            delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 1.0))
            print(f"  listing request failed ({exc}); retrying in {delay:.1f}s [{attempt}/{max_retries}]")
            time.sleep(delay)
    raise AssertionError("unreachable")


def list_folder_files(
    session: requests.Session,
    folder: str,
    *,
    timeout: tuple[float, float],
    max_retries: int,
) -> list[str]:
    endpoint = os.getenv("HF_ENDPOINT", DEFAULT_ENDPOINT).rstrip("/")
    encoded_folder = quote(folder, safe="")
    url = f"{endpoint}/api/datasets/{REPO_ID}/tree/main/{encoded_folder}"
    params: dict[str, str] | None = {"recursive": "true", "expand": "false"}
    files: list[str] = []
    page = 0

    while url:
        response = get_with_retry(
            session,
            url,
            params=params,
            headers=hub_headers(),
            timeout=timeout,
            max_retries=max_retries,
        )
        page += 1
        entries = response.json()
        files.extend(entry["path"] for entry in entries if entry.get("type") == "file")
        print(f"  page {page}: {len(entries)} entries ({len(files)} files so far)")
        url = response.links.get("next", {}).get("url")
        if url:
            url = urljoin(response.url, url)
        params = None

    return files


def list_target_files(
    folders: Iterable[str],
    *,
    timeout: tuple[float, float],
    max_retries: int,
) -> list[str]:
    folders = list(folders)
    target_files: list[str] = []

    with requests.Session() as session:
        for index, folder in enumerate(folders, start=1):
            print(f"Listing folder {index}/{len(folders)}: {folder}")
            folder_files = list_folder_files(session, folder, timeout=timeout, max_retries=max_retries)
            if not folder_files:
                raise RuntimeError(f"No files found in requested folder: {folder}")
            target_files.extend(folder_files)

    return target_files


def download_with_retry(filename: str, max_retries: int) -> bool:
    for attempt in range(1, max_retries + 1):
        try:
            hf_hub_download(repo_id=REPO_ID, repo_type=REPO_TYPE, filename=filename, local_dir=LOCAL_DIR)
            return True
        except Exception as exc:
            if attempt == max_retries:
                print(f"Giving up after {max_retries} download attempts: {filename} ({exc})")
                return False
            delay = min(30.0, (2 ** (attempt - 1)) + random.uniform(0.0, 1.0))
            print(f"  download failed: {filename} ({exc}); retrying in {delay:.1f}s [{attempt}/{max_retries}]")
            time.sleep(delay)
    raise AssertionError("unreachable")


def main() -> int:
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    if args.list_retries < 1 or args.download_retries < 1:
        raise ValueError("Retry counts must be at least 1")
    if args.connect_timeout <= 0 or args.read_timeout <= 0:
        raise ValueError("Timeouts must be positive")

    selected_folders = args.folder or FOLDERS
    target_files = list_target_files(
        selected_folders,
        timeout=(args.connect_timeout, args.read_timeout),
        max_retries=args.list_retries,
    )
    print(f"Found {len(target_files)} files across {len(selected_folders)} selected folders.")

    if args.dry_run:
        print("Dry run complete; no files were downloaded.")
        return 0

    LOCAL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading with {args.workers} workers into {LOCAL_DIR}.")
    failed_files: list[str] = []

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_with_retry, filename, args.download_retries): filename for filename in target_files
        }
        for index, future in enumerate(as_completed(futures), start=1):
            filename = futures[future]
            try:
                if future.result():
                    print(f"[{index}/{len(target_files)}] Downloaded: {filename}")
                else:
                    failed_files.append(filename)
            except Exception as exc:
                print(f"Unexpected download failure for {filename}: {exc}")
                failed_files.append(filename)

    if failed_files:
        print(f"{len(failed_files)} files failed to download:")
        for filename in failed_files:
            print(f"  - {filename}")
        return 1

    print("All requested files downloaded successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
