import concurrent.futures
import subprocess
import sys
import zipfile
from pathlib import Path


def generate_data_directory():
    data_zip = Path("data.zip")
    data_dir = Path("data")

    if data_zip.exists() and not data_dir.exists():
        print("Extracting data.zip...")
        with zipfile.ZipFile(data_zip, "r") as zf:
            zf.extractall(".")
        print("Extraction complete.")


def cleanup_data_directory():
    data_dir = Path("data")
    if data_dir.exists():
        print("Cleaning up data directory...")
        for item in data_dir.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                import shutil

                shutil.rmtree(item)
        data_dir.rmdir()
        print("Cleanup complete.")


def run_analysis():
    generate_data_directory()

    analysis_dir = Path("research/analysis")
    scripts = sorted(analysis_dir.glob("*.py"))
    print(f"Running {len(scripts)} analysis scripts...")

    def run_script(script):
        result = subprocess.run(
            [sys.executable, str(script)], capture_output=True, text=True
        )
        return script.name, result

    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(run_script, s): s for s in scripts}
        for future in concurrent.futures.as_completed(futures):
            name, result = future.result()
            if result.returncode != 0:
                print(f"[FAIL] {name}\n{result.stderr}")
            else:
                print(f"[OK] {name}")

    print(f"\nAll {len(scripts)} analysis scripts complete.")
    cleanup_data_directory()
