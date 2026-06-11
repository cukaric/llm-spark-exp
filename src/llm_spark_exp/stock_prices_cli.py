"""Command-line entry point for the stock price agent."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import httpx
import typer
from rich.console import Console
from rich.table import Table

from llm_spark_exp.agents.stock_prices import (
    DEFAULT_NYSE_SYMBOLS,
    DEFAULT_OLLAMA_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_STOCK_PRICE_TABLE,
    OllamaStockPriceAnalyst,
    StockPriceAgent,
)

app = typer.Typer(help="Agentic stock price collection workflows.")
console = Console()


@app.command()
def collect(
    symbols: Annotated[
        list[str] | None,
        typer.Argument(help="NYSE/US ticker symbols. Defaults to IBM KO JPM WMT XOM."),
    ] = None,
    table_path: Annotated[
        Path,
        typer.Option("--table", "-t", help="CSV table to append quote rows to."),
    ] = DEFAULT_STOCK_PRICE_TABLE,
    request: Annotated[
        str | None,
        typer.Option(
            "--request",
            "-r",
            help="Natural-language request for the local Ollama planner.",
        ),
    ] = None,
    model: Annotated[
        str,
        typer.Option("--model", "-m", help="Ollama model used for natural-language planning."),
    ] = DEFAULT_OLLAMA_MODEL,
    ollama_url: Annotated[
        str,
        typer.Option("--ollama-url", help="Base URL for the local Ollama server."),
    ] = DEFAULT_OLLAMA_URL,
) -> None:
    """Fetch stock prices from Stooq and append them to a CSV table."""

    selected_symbols = symbols or list(DEFAULT_NYSE_SYMBOLS)
    try:
        analyst = OllamaStockPriceAnalyst(model=model, base_url=ollama_url) if request else None
        run = StockPriceAgent(planner=analyst, summarizer=analyst).run(
            selected_symbols,
            request=request,
            table_path=table_path,
        )
    except (httpx.HTTPError, ValueError) as error:
        console.print(f"[red]Stock price collection failed:[/red] {error}")
        raise typer.Exit(1) from error

    if run.plan is not None:
        console.print(f"[bold]LLM plan:[/bold] {', '.join(run.plan.symbols)}")
        if run.plan.rationale:
            console.print(run.plan.rationale)
    if run.skipped_symbols:
        console.print(
            f"[yellow]Skipped unsupported symbols:[/yellow] {', '.join(run.skipped_symbols)}"
        )

    output = Table(title=f"Added {len(run.quotes)} quote rows to {run.table_path}")
    output.add_column("Symbol")
    output.add_column("Price", justify="right")
    output.add_column("Previous", justify="right")
    output.add_column("Change", justify="right")
    output.add_column("%", justify="right")
    output.add_column("Quote date")
    output.add_column("Quote time")
    output.add_column("Source")

    for observation in run.observations:
        quote = observation.quote
        output.add_row(
            quote.symbol,
            f"{quote.price} {quote.currency}",
            "" if observation.previous_price is None else str(observation.previous_price),
            "" if observation.price_change is None else str(observation.price_change),
            "" if observation.percent_change is None else f"{observation.percent_change}%",
            quote.quote_date,
            quote.quote_time,
            quote.source,
        )

    console.print(output)
    if run.summary:
        console.print(f"[bold]Summary:[/bold] {run.summary}")


if __name__ == "__main__":
    app()
