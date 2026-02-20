from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import questionary
import typer
from typing_extensions import Annotated

from src.common.analysis import Analysis
from src.common.indexer import Indexer
from src.common.util import package_data
from src.common.util.strings import snake_to_title

app = typer.Typer(help="Prediction Market Analysis CLI")


@app.command()
def analyze(
    name: Annotated[
        Optional[str],
        typer.Argument(help="Name of the analysis to run. If not provided, an interactive menu is shown."),
    ] = None,
):
    """
    Run analysis scripts.
    """
    analyses = Analysis.load()

    if not analyses:
        typer.echo("No analyses found in src/analysis/", err=True)
        return

    output_dir = Path("output")

    # If name provided, run that specific analysis
    if name:
        if name == "all":
            typer.echo("\nRunning all analyses...\n")
            for analysis_cls in analyses:
                instance = analysis_cls()
                typer.echo(f"Running: {instance.name}")
                saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
                for fmt, path in saved.items():
                    typer.echo(f"  {fmt}: {path}")
            typer.echo("\nAll analyses complete.")
            return

        # Find matching analysis
        for analysis_cls in analyses:
            instance = analysis_cls()
            if instance.name == name:
                typer.echo(f"\nRunning: {instance.name}\n")
                saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
                typer.echo("Saved files:")
                for fmt, path in saved.items():
                    typer.echo(f"  {fmt}: {path}")
                return

        # No match found
        typer.echo(f"Analysis '{name}' not found. Available analyses:", err=True)
        for analysis_cls in analyses:
            instance = analysis_cls()
            typer.echo(f"  - {instance.name}", err=True)
        raise typer.Exit(code=1)

    # Interactive menu mode
    # Map display names to analysis classes/commands
    choices = ["Run all analyses"]
    analysis_map = {}
    
    for analysis_cls in analyses:
        instance = analysis_cls()
        display_name = f"{snake_to_title(instance.name)}: {instance.description}"
        choices.append(display_name)
        analysis_map[display_name] = analysis_cls

    choices.append("Exit")

    choice = questionary.select(
        "Select an analysis to run:",
        choices=choices,
    ).ask()

    if choice is None or choice == "Exit":
        typer.echo("Exiting.")
        return

    if choice == "Run all analyses":
        # Run all analyses
        typer.echo("\nRunning all analyses...\n")
        for analysis_cls in analyses:
            instance = analysis_cls()
            typer.echo(f"Running: {instance.name}")
            saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
            for fmt, path in saved.items():
                typer.echo(f"  {fmt}: {path}")
        typer.echo("\nAll analyses complete.")
    else:
        # Run selected analysis
        analysis_cls = analysis_map[choice]
        instance = analysis_cls()
        typer.echo(f"\nRunning: {instance.name}\n")
        saved = instance.save(output_dir, formats=["png", "pdf", "csv", "json", "gif"])
        typer.echo("Saved files:")
        for fmt, path in saved.items():
            typer.echo(f"  {fmt}: {path}")


@app.command()
def index():
    """
    Run data collection indexers.
    """
    indexers = Indexer.load()

    if not indexers:
        typer.echo("No indexers found in src/indexers/", err=True)
        return

    # Build menu options
    choices = []
    indexer_map = {}
    
    for indexer_cls in indexers:
        instance = indexer_cls()
        display_name = f"{snake_to_title(instance.name)}: {instance.description}"
        choices.append(display_name)
        indexer_map[display_name] = indexer_cls
        
    choices.append("Exit")

    choice = questionary.select(
        "Select an indexer to run:",
        choices=choices,
    ).ask()

    if choice is None or choice == "Exit":
        typer.echo("Exiting.")
        return

    indexer_cls = indexer_map[choice]
    instance = indexer_cls()
    typer.echo(f"\nRunning: {instance.name}\n")
    instance.run()
    typer.echo("\nIndexer complete.")


@app.command()
def package():
    """
    Package the data directory into a zstd-compressed tar archive.
    """
    success = package_data()
    if not success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
