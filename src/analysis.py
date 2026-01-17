import concurrent.futures
import glob
import os
import subprocess
import sys
import zipfile
from pathlib import Path


def reassemble_data_zip():
    """Reassemble data.zip from chunks if it doesn't exist."""
    if os.path.exists("data.zip"):
        return

    chunks = sorted(glob.glob("data.zip.*"))
    if not chunks:
        print("No data.zip chunks found.")
        return

    print("Reassembling data.zip from chunks...")
    with open("data.zip", "wb") as output:
        for chunk in chunks:
            with open(chunk, "rb") as f:
                output.write(f.read())
    print("data.zip reassembled.")


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

    # delete data.zip as well
    data_zip = Path("data.zip")
    if data_zip.exists():
        data_zip.unlink()
        print("data.zip deleted.")


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


def run_single_analysis(script_name: str):
    generate_data_directory()

    if not script_name.endswith(".py"):
        script_name = f"{script_name}.py"

    script = Path("research/analysis") / script_name
    if not script.exists():
        print(f"Script not found: {script}")
        sys.exit(1)

    print(f"Running {script_name}...")
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True)

    if result.returncode != 0:
        print(f"[FAIL] {script_name}\n{result.stderr}")
    else:
        print(f"[OK] {script_name}")
        if result.stdout:
            print(result.stdout)

    cleanup_data_directory()
