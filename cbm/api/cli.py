"""Command-line interface for charting-by-machines."""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

app = typer.Typer(
    name="cbm",
    help="Charting by Machines: ML-based portfolio selection",
    add_completion=False,
)
console = Console()


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration YAML file"
    ),
    architecture: str = typer.Option(
        "cnn_lstm", "--architecture", "-a", 
        help="Model architecture (fnn, cnn, lstm, cnn_lstm)"
    ),
    output: Optional[Path] = typer.Option(
        None, "--output", "-o", help="Output path for trained model"
    ),
    universe: str = typer.Option(
        "sp500", "--universe", "-u", help="Stock universe to use"
    ),
):
    """Train an ML model for return forecasting."""
    from cbm import PortfolioEngine, CBMConfig
    
    console.print("[bold blue]Charting by Machines - Training[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Load configuration
        task = progress.add_task("Loading configuration...", total=None)
        
        if config:
            cfg = CBMConfig.from_yaml(str(config))
        else:
            cfg = CBMConfig()
        
        progress.update(task, description="Initializing engine...")
        engine = PortfolioEngine(config=cfg)
        
        # Load data
        progress.update(task, description="Loading data...")
        engine.load_data(universe=universe)
        
        # Prepare features
        progress.update(task, description="Preparing features...")
        engine.prepare_features()
        
        # Train model
        progress.update(task, description="Training model...")
        model_id = engine.train_model(architecture=architecture)
        
        progress.update(task, completed=True, description="Training complete!")
    
    console.print(f"\n[green]✓[/green] Model trained: [bold]{model_id}[/bold]")
    
    if output:
        engine.save_model(model_id, str(output))
        console.print(f"[green]✓[/green] Model saved to: {output}")


@app.command()
def backtest(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to configuration YAML file"
    ),
    universe: str = typer.Option(
        "sp500", "--universe", "-u", help="Stock universe to use"
    ),
    start_date: str = typer.Option(
        "2010-01-01", "--start", help="Start date (YYYY-MM-DD)"
    ),
    end_date: str = typer.Option(
        "2023-12-31", "--end", help="End date (YYYY-MM-DD)"
    ),
    n_portfolios: int = typer.Option(
        10, "--portfolios", "-n", help="Number of portfolios"
    ),
):
    """Run a full backtest pipeline."""
    from cbm import PortfolioEngine, CBMConfig
    
    console.print("[bold blue]Charting by Machines - Backtest[/bold blue]")
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Initializing...", total=None)
        
        if config:
            cfg = CBMConfig.from_yaml(str(config))
        else:
            cfg = CBMConfig()
        
        cfg.data.start_date = start_date
        cfg.data.end_date = end_date
        cfg.portfolio.n_portfolios = n_portfolios
        
        engine = PortfolioEngine(config=cfg)
        
        progress.update(task, description="Running backtest...")
        result = engine.run_backtest(universe=universe)
        
        progress.update(task, completed=True, description="Backtest complete!")
    
    # Display results
    console.print("\n[bold]Backtest Results[/bold]\n")
    
    # Create performance table
    table = Table(title="Portfolio Performance")
    table.add_column("Portfolio", style="cyan")
    table.add_column("Mean Return", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("t-stat", justify="right")
    
    for name, metrics in result.performance.items():
        table.add_row(
            name,
            f"{metrics.mean_return:.2%}",
            f"{metrics.std_dev:.2%}",
            f"{metrics.sharpe_ratio:.2f}",
            f"{metrics.t_statistic:.2f}",
        )
    
    console.print(table)
    
    # Long-short summary
    if "long_short" in result.performance:
        ls = result.performance["long_short"]
        console.print(f"\n[bold green]Long-Short Portfolio:[/bold green]")
        console.print(f"  Mean Return: {ls.mean_return:.2%}/month")
        console.print(f"  Sharpe Ratio: {ls.sharpe_ratio:.2f}")
        console.print(f"  t-statistic: {ls.t_statistic:.2f}")


@app.command()
def forecast(
    model_path: Path = typer.Argument(..., help="Path to trained model"),
    output: Path = typer.Option(
        "forecasts.parquet", "--output", "-o", help="Output file path"
    ),
    start_date: str = typer.Option(
        None, "--start", help="Start date for forecasts"
    ),
    end_date: str = typer.Option(
        None, "--end", help="End date for forecasts"
    ),
):
    """Generate forecasts using a trained model."""
    from cbm import PortfolioEngine
    
    console.print("[bold blue]Charting by Machines - Forecast[/bold blue]")
    
    engine = PortfolioEngine()
    model_id = engine.load_model(str(model_path))
    
    console.print(f"Loaded model: {model_id}")
    
    # Generate forecasts
    test_period = None
    if start_date and end_date:
        test_period = (start_date[:7], end_date[:7])
    
    forecasts = engine.forecast(model_id=model_id, test_period=test_period)
    
    # Save forecasts
    forecasts.values.to_parquet(str(output))
    console.print(f"[green]✓[/green] Forecasts saved to: {output}")


@app.command()
def list_models(
    model_dir: Path = typer.Option(
        "./models", "--dir", "-d", help="Model directory"
    ),
):
    """List all saved models."""
    from cbm.ml import ModelRegistry
    
    registry = ModelRegistry(path=str(model_dir))
    models = registry.list_models()
    
    if not models:
        console.print("[yellow]No models found[/yellow]")
        return
    
    table = Table(title="Saved Models")
    table.add_column("Model ID", style="cyan")
    table.add_column("Path")
    table.add_column("Metrics")
    
    for model in models:
        metrics_str = ", ".join(
            f"{k}={v:.4f}" for k, v in model.get("metrics", {}).items()
        )
        table.add_row(
            model["model_id"],
            model["path"],
            metrics_str[:50] + "..." if len(metrics_str) > 50 else metrics_str,
        )
    
    console.print(table)


@app.command()
def info():
    """Show package information."""
    from cbm import __version__
    
    console.print("[bold blue]Charting by Machines[/bold blue]")
    console.print(f"Version: {__version__}")
    console.print("\nML-based portfolio selection from historical price patterns")
    console.print("Based on Murray, Xia, and Xiao (2024)")
    console.print("\nCommands:")
    console.print("  train     - Train an ML model")
    console.print("  backtest  - Run a full backtest")
    console.print("  forecast  - Generate forecasts")
    console.print("  list-models - List saved models")


if __name__ == "__main__":
    app()
