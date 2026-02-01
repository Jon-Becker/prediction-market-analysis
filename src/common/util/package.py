import subprocess
import sys
from pathlib import Path


def package_data(data_dir: Path = Path("data"), output_path: Path = Path("data.tar.zst")) -> bool:
    """Package the data directory into a zstd-compressed tar archive.

    Args:
        data_dir: Path to the data directory to compress.
        output_path: Path for the output archive.

    Returns:
        True if successful, False otherwise.
    """
    if not data_dir.exists():
        print(f"Error: Data directory '{data_dir}' does not exist.")
        return False

    print(f"Packaging {data_dir} -> {output_path}")
    result = subprocess.run(
        ["tar", "--zstd", "-cf", str(output_path), str(data_dir)],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False

    print(f"Successfully created {output_path}")
    return True
