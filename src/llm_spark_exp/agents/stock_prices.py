"""Agentic stock price collection workflow."""

from __future__ import annotations

import csv
import json
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Protocol

import httpx

from llm_spark_exp.paths import DATA_DIR

DEFAULT_NYSE_SYMBOLS = ("IBM", "KO", "JPM", "WMT", "XOM")
DEFAULT_OLLAMA_MODEL = "gemma4:e4b"
DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_STOCK_PRICE_TABLE = DATA_DIR / "processed" / "stock_prices.csv"

STOCK_PRICE_COLUMNS = (
    "fetched_at_utc",
    "symbol",
    "market_symbol",
    "price",
    "currency",
    "quote_date",
    "quote_time",
    "open",
    "high",
    "low",
    "volume",
    "previous_price",
    "price_change",
    "percent_change",
    "run_summary",
    "source",
    "source_url",
)


@dataclass(frozen=True)
class StockQuote:
    """A normalized quote row ready to persist."""

    fetched_at_utc: datetime
    symbol: str
    market_symbol: str
    price: Decimal
    currency: str
    quote_date: str
    quote_time: str
    open: Decimal | None
    high: Decimal | None
    low: Decimal | None
    volume: int | None
    source: str
    source_url: str

    def as_row(self) -> dict[str, str]:
        return {
            "fetched_at_utc": self.fetched_at_utc.isoformat(),
            "symbol": self.symbol,
            "market_symbol": self.market_symbol,
            "price": str(self.price),
            "currency": self.currency,
            "quote_date": self.quote_date,
            "quote_time": self.quote_time,
            "open": "" if self.open is None else str(self.open),
            "high": "" if self.high is None else str(self.high),
            "low": "" if self.low is None else str(self.low),
            "volume": "" if self.volume is None else str(self.volume),
            "previous_price": "",
            "price_change": "",
            "percent_change": "",
            "run_summary": "",
            "source": self.source,
            "source_url": self.source_url,
        }


class StockQuoteSource(Protocol):
    name: str

    def fetch(self, symbols: Sequence[str]) -> list[StockQuote]:
        """Fetch normalized quotes for the requested symbols."""


class NoStockQuotesError(ValueError):
    """Raised when a quote source has no usable data for one or more symbols."""


@dataclass(frozen=True)
class StockPricePlan:
    """Symbols selected by a planning model."""

    symbols: tuple[str, ...]
    rationale: str


class StockPricePlanner(Protocol):
    def plan(self, request: str) -> StockPricePlan:
        """Choose ticker symbols for a natural-language request."""


class StockPriceSummarizer(Protocol):
    def summarize(
        self,
        *,
        request: str | None,
        plan: StockPricePlan | None,
        observations: Sequence[StockPriceObservation],
    ) -> str:
        """Summarize a stock price collection run."""


class OllamaStockPriceAnalyst:
    """Use a local Ollama model to plan tickers and summarize stock price runs."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_OLLAMA_MODEL,
        base_url: str = DEFAULT_OLLAMA_URL,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    def plan(self, request: str) -> StockPricePlan:
        cleaned_request = request.strip()
        if not cleaned_request:
            raise ValueError("A natural-language request is required for LLM planning.")

        prompt = build_stock_planning_prompt(cleaned_request)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0},
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

        response_text = response.json().get("response", "")
        return parse_stock_price_plan(response_text)

    def summarize(
        self,
        *,
        request: str | None,
        plan: StockPricePlan | None,
        observations: Sequence[StockPriceObservation],
    ) -> str:
        if not observations:
            return ""

        prompt = build_stock_summary_prompt(
            request=request,
            plan=plan,
            observations=observations,
        )
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()

        return response.json().get("response", "").strip()


class StooqQuoteSource:
    """Fetch delayed US stock quotes from Stooq's CSV endpoint."""

    name = "stooq"
    base_url = "https://stooq.com/q/l/"

    def __init__(self, *, timeout_seconds: float = 15.0) -> None:
        self.timeout_seconds = timeout_seconds

    def fetch(self, symbols: Sequence[str]) -> list[StockQuote]:
        market_symbols = [normalize_stooq_symbol(symbol) for symbol in symbols]
        requested_symbols = [symbol.strip().upper() for symbol in symbols]
        quotes: list[StockQuote] = []

        with httpx.Client(timeout=self.timeout_seconds, follow_redirects=True) as client:
            for symbol, market_symbol in zip(requested_symbols, market_symbols, strict=True):
                params = {"s": market_symbol, "f": "sd2t2ohlcv", "e": "csv"}
                fetched_at = datetime.now(UTC)
                response = client.get(self.base_url, params=params)
                response.raise_for_status()
                try:
                    quotes.extend(
                        parse_stooq_csv(
                            response.text,
                            requested_symbols=[symbol],
                            fetched_at_utc=fetched_at,
                            source_url=str(response.url),
                        )
                    )
                except NoStockQuotesError:
                    continue

        if not quotes:
            raise NoStockQuotesError(
                f"No stock quotes were returned for: {', '.join(requested_symbols)}"
            )

        return quotes


@dataclass(frozen=True)
class StockPriceObservation:
    quote: StockQuote
    previous_price: Decimal | None = None
    price_change: Decimal | None = None
    percent_change: Decimal | None = None

    def as_row(self, *, run_summary: str = "") -> dict[str, str]:
        row = self.quote.as_row()
        row.update(
            {
                "previous_price": decimal_to_string(self.previous_price),
                "price_change": decimal_to_string(self.price_change),
                "percent_change": decimal_to_string(self.percent_change),
                "run_summary": run_summary,
            }
        )
        return row


@dataclass(frozen=True)
class StockPriceRun:
    table_path: Path
    quotes: tuple[StockQuote, ...]
    observations: tuple[StockPriceObservation, ...]
    skipped_symbols: tuple[str, ...] = ()
    plan: StockPricePlan | None = None
    summary: str = ""


class StockPriceAgent:
    """Collect stock prices and append them to the project table."""

    def __init__(
        self,
        *,
        source: StockQuoteSource | None = None,
        planner: StockPricePlanner | None = None,
        summarizer: StockPriceSummarizer | None = None,
        default_symbols: Sequence[str] = DEFAULT_NYSE_SYMBOLS,
    ) -> None:
        self.source = source or StooqQuoteSource()
        self.planner = planner
        self.summarizer = summarizer
        self.default_symbols = tuple(default_symbols)

    def run(
        self,
        symbols: Sequence[str] | None = None,
        *,
        request: str | None = None,
        table_path: Path = DEFAULT_STOCK_PRICE_TABLE,
    ) -> StockPriceRun:
        plan: StockPricePlan | None = None
        if request is not None:
            if self.planner is None:
                raise ValueError(
                    "A planner is required when running from a natural-language request."
                )
            plan = self.planner.plan(request)
            selected_symbols = plan.symbols
        else:
            selected_symbols = tuple(symbols or self.default_symbols)

        if not selected_symbols:
            raise ValueError("At least one stock symbol is required.")

        quotes = self.source.fetch(selected_symbols)
        skipped_symbols = find_skipped_symbols(selected_symbols, quotes)
        previous_prices = load_latest_prices(table_path)
        observations = tuple(build_observations(quotes, previous_prices))
        summary = ""
        if self.summarizer is not None:
            summary = self.summarizer.summarize(
                request=request,
                plan=plan,
                observations=observations,
            )
        append_observations(table_path, observations, run_summary=summary)
        return StockPriceRun(
            table_path=table_path,
            quotes=tuple(quotes),
            observations=observations,
            skipped_symbols=skipped_symbols,
            plan=plan,
            summary=summary,
        )


def build_stock_planning_prompt(request: str) -> str:
    return f"""You choose US-listed stock ticker symbols for a stock price collection tool.

Return only valid JSON with this exact shape:
{{"symbols":["TICKER"],"rationale":"short reason"}}

Rules:
- Use common NYSE or NASDAQ tickers only.
- Return 1 to 15 symbols.
- Symbols must be uppercase and should not include exchange suffixes.
- The symbols array must contain ticker symbols only, never sector names or explanations.
- Prefer liquid, well-known stocks when the user asks for a sector or category.
- For uranium-related stocks, prefer supported examples like CCJ, UEC, UUUU, NXE, and URA.
- Do not invent ticker symbols.

User request: {request}
"""


def build_stock_summary_prompt(
    *,
    request: str | None,
    plan: StockPricePlan | None,
    observations: Sequence[StockPriceObservation],
) -> str:
    request_text = request or "No natural-language request was provided."
    plan_text = "No LLM plan was used."
    if plan is not None:
        plan_text = f"Symbols: {', '.join(plan.symbols)}. Rationale: {plan.rationale}"

    rows = "\n".join(
        (
            f"- {observation.quote.symbol}: price {observation.quote.price} "
            f"{observation.quote.currency}; previous {decimal_to_string(observation.previous_price) or 'n/a'}; "
            f"change {decimal_to_string(observation.price_change) or 'n/a'}; "
            f"percent {decimal_to_string(observation.percent_change) or 'n/a'}%"
        )
        for observation in observations
    )

    return f"""Summarize this stock price collection run in 2 short sentences.
Mention the largest move when previous prices are available. Do not give financial advice.

User request: {request_text}
Plan: {plan_text}
Rows:
{rows}
"""


def parse_stock_price_plan(response_text: str) -> StockPricePlan:
    payload = json.loads(extract_json_object(response_text))
    raw_symbols = payload.get("symbols")
    if not isinstance(raw_symbols, list):
        raise ValueError("LLM plan did not include a symbols list.")

    symbols = tuple(dedupe_symbols(parse_llm_symbols(raw_symbols)))
    if not symbols:
        raise ValueError("LLM plan did not choose any ticker symbols.")
    if len(symbols) > 15:
        raise ValueError("LLM plan returned too many ticker symbols.")

    rationale = payload.get("rationale", "")
    if not isinstance(rationale, str):
        rationale = ""

    return StockPricePlan(symbols=symbols, rationale=rationale.strip())


def extract_json_object(response_text: str) -> str:
    stripped = response_text.strip()
    if stripped.startswith("```"):
        stripped = stripped.strip("`")
        if stripped.startswith("json"):
            stripped = stripped[4:].strip()

    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("LLM response did not contain a JSON object.")
    return stripped[start : end + 1]


def normalize_llm_symbol(symbol: object) -> str:
    if not isinstance(symbol, str):
        raise ValueError(f"LLM returned a non-string ticker symbol: {symbol!r}")

    cleaned = symbol.strip().upper()
    if "." in cleaned:
        cleaned = cleaned.split(".", maxsplit=1)[0]
    if not re.fullmatch(r"[A-Z][A-Z0-9-]{0,5}", cleaned):
        raise ValueError(f"LLM returned an invalid ticker symbol: {symbol!r}")
    return cleaned


def parse_llm_symbols(raw_symbols: Sequence[object]) -> list[str]:
    symbols: list[str] = []
    for symbol in raw_symbols:
        try:
            symbols.append(normalize_llm_symbol(symbol))
        except ValueError:
            continue
    return symbols


def dedupe_symbols(symbols: Sequence[str]) -> list[str]:
    deduped_symbols: list[str] = []
    seen_symbols: set[str] = set()
    for symbol in symbols:
        if symbol in seen_symbols:
            continue
        deduped_symbols.append(symbol)
        seen_symbols.add(symbol)
    return deduped_symbols


def find_skipped_symbols(
    requested_symbols: Sequence[str],
    quotes: Sequence[StockQuote],
) -> tuple[str, ...]:
    quoted_symbols = {quote.symbol.strip().upper() for quote in quotes}
    skipped_symbols: list[str] = []
    for symbol in requested_symbols:
        normalized_symbol = symbol.strip().upper()
        if normalized_symbol and normalized_symbol not in quoted_symbols:
            skipped_symbols.append(normalized_symbol)
    return tuple(skipped_symbols)


def normalize_stooq_symbol(symbol: str) -> str:
    cleaned = symbol.strip().upper()
    if not cleaned:
        raise ValueError("Stock symbols cannot be blank.")
    if "." in cleaned:
        return cleaned
    return f"{cleaned}.US"


def parse_stooq_csv(
    csv_text: str,
    *,
    requested_symbols: Sequence[str],
    fetched_at_utc: datetime,
    source_url: str,
) -> list[StockQuote]:
    requested_by_market_symbol = {
        normalize_stooq_symbol(symbol): symbol.strip().upper() for symbol in requested_symbols
    }
    rows = csv.DictReader(csv_text.splitlines())
    if rows.fieldnames and "Symbol" not in rows.fieldnames:
        rows = csv.DictReader(
            csv_text.splitlines(),
            fieldnames=("Symbol", "Date", "Time", "Open", "High", "Low", "Close", "Volume"),
        )
    quotes: list[StockQuote] = []

    for row in rows:
        market_symbol = row["Symbol"].strip().upper()
        close_value = row["Close"].strip()
        if close_value == "N/D":
            continue

        quotes.append(
            StockQuote(
                fetched_at_utc=fetched_at_utc,
                symbol=requested_by_market_symbol.get(
                    market_symbol, market_symbol.removesuffix(".US")
                ),
                market_symbol=market_symbol,
                price=parse_decimal(close_value),
                currency="USD",
                quote_date=row["Date"].strip(),
                quote_time=row["Time"].strip(),
                open=parse_optional_decimal(row["Open"]),
                high=parse_optional_decimal(row["High"]),
                low=parse_optional_decimal(row["Low"]),
                volume=parse_optional_int(row["Volume"]),
                source="Stooq",
                source_url=source_url,
            )
        )

    if not quotes:
        raise NoStockQuotesError(
            "No stock quotes were returned. Check the symbols or the data source."
        )

    return quotes


def build_observations(
    quotes: Iterable[StockQuote],
    previous_prices: dict[str, Decimal],
) -> list[StockPriceObservation]:
    observations: list[StockPriceObservation] = []
    for quote in quotes:
        previous_price = previous_prices.get(quote.symbol)
        price_change = None
        percent_change = None
        if previous_price is not None:
            price_change = quote.price - previous_price
            if previous_price != 0:
                percent_change = (price_change / previous_price * Decimal("100")).quantize(
                    Decimal("0.01")
                )

        observations.append(
            StockPriceObservation(
                quote=quote,
                previous_price=previous_price,
                price_change=price_change,
                percent_change=percent_change,
            )
        )
    return observations


def append_quotes(table_path: Path, quotes: Iterable[StockQuote]) -> None:
    append_observations(
        table_path,
        build_observations(quotes, load_latest_prices(table_path)),
    )


def append_observations(
    table_path: Path,
    observations: Iterable[StockPriceObservation],
    *,
    run_summary: str = "",
) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_table_columns(table_path)
    write_header = not table_path.exists() or table_path.stat().st_size == 0

    with table_path.open("a", newline="", encoding="utf-8") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=STOCK_PRICE_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerows(
            observation.as_row(run_summary=run_summary) for observation in observations
        )


def load_latest_prices(table_path: Path) -> dict[str, Decimal]:
    if not table_path.exists() or table_path.stat().st_size == 0:
        return {}

    latest_prices: dict[str, Decimal] = {}
    with table_path.open(newline="", encoding="utf-8") as table_file:
        for row in csv.DictReader(table_file):
            symbol = row.get("symbol", "").strip().upper()
            price = row.get("price", "").strip()
            if not symbol or not price:
                continue
            latest_prices[symbol] = parse_decimal(price)
    return latest_prices


def ensure_table_columns(table_path: Path) -> None:
    if not table_path.exists() or table_path.stat().st_size == 0:
        return

    with table_path.open(newline="", encoding="utf-8") as table_file:
        reader = csv.DictReader(table_file)
        if reader.fieldnames == list(STOCK_PRICE_COLUMNS):
            return
        rows = list(reader)

    with table_path.open("w", newline="", encoding="utf-8") as table_file:
        writer = csv.DictWriter(table_file, fieldnames=STOCK_PRICE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in STOCK_PRICE_COLUMNS})


def parse_decimal(value: str) -> Decimal:
    try:
        return Decimal(value.strip())
    except InvalidOperation as error:
        raise ValueError(f"Could not parse decimal value: {value!r}") from error


def parse_optional_decimal(value: str) -> Decimal | None:
    cleaned = value.strip()
    if cleaned in {"", "N/D"}:
        return None
    return parse_decimal(cleaned)


def parse_optional_int(value: str) -> int | None:
    cleaned = value.strip()
    if cleaned in {"", "N/D"}:
        return None
    return int(cleaned)


def decimal_to_string(value: Decimal | None) -> str:
    if value is None:
        return ""
    return str(value)
