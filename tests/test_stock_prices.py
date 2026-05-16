from datetime import UTC, datetime
from decimal import Decimal

from llm_spark_exp.agents.stock_prices import (
    NoStockQuotesError,
    StockPriceAgent,
    StockPricePlan,
    StockQuote,
    append_quotes,
    build_observations,
    find_skipped_symbols,
    load_latest_prices,
    normalize_stooq_symbol,
    parse_stock_price_plan,
    parse_stooq_csv,
)


def test_normalize_stooq_symbol_adds_us_suffix() -> None:
    assert normalize_stooq_symbol("ibm") == "IBM.US"


def test_normalize_stooq_symbol_preserves_explicit_market_suffix() -> None:
    assert normalize_stooq_symbol("brk.b.us") == "BRK.B.US"


def test_parse_stooq_csv_returns_normalized_quotes() -> None:
    quote_time = datetime(2026, 5, 13, 10, 15, tzinfo=UTC)
    csv_text = "\n".join(
        [
            "Symbol,Date,Time,Open,High,Low,Close,Volume",
            "IBM.US,2026-05-12,22:00:10,282.49,287.75,281.28,283.04,4100000",
        ]
    )

    quotes = parse_stooq_csv(
        csv_text,
        requested_symbols=["IBM"],
        fetched_at_utc=quote_time,
        source_url="https://stooq.com/q/l/?s=ibm.us",
    )

    assert quotes[0].symbol == "IBM"
    assert quotes[0].price == Decimal("283.04")
    assert quotes[0].currency == "USD"
    assert quotes[0].volume == 4100000


def test_parse_stooq_csv_accepts_stooq_headerless_response() -> None:
    quotes = parse_stooq_csv(
        "IBM.US,2026-05-13,16:44:29,217.74,218.31,212.45,212.61,1542169",
        requested_symbols=["IBM"],
        fetched_at_utc=datetime(2026, 5, 13, 14, 44, tzinfo=UTC),
        source_url="https://stooq.com/q/l/?s=ibm.us",
    )

    assert quotes[0].symbol == "IBM"
    assert quotes[0].price == Decimal("212.61")
    assert quotes[0].quote_time == "16:44:29"


def test_parse_stooq_csv_raises_no_quotes_error_for_missing_quote() -> None:
    try:
        parse_stooq_csv(
            "UALF.US,N/D,N/D,N/D,N/D,N/D,N/D,N/D",
            requested_symbols=["UALF"],
            fetched_at_utc=datetime(2026, 5, 13, 14, 44, tzinfo=UTC),
            source_url="https://stooq.com/q/l/?s=ualf.us",
        )
    except NoStockQuotesError:
        pass
    else:
        raise AssertionError("Expected NoStockQuotesError")


def test_append_quotes_creates_table(tmp_path) -> None:
    table_path = tmp_path / "stock_prices.csv"
    quote = StockQuote(
        fetched_at_utc=datetime(2026, 5, 13, 10, 15, tzinfo=UTC),
        symbol="IBM",
        market_symbol="IBM.US",
        price=Decimal("283.04"),
        currency="USD",
        quote_date="2026-05-12",
        quote_time="22:00:10",
        open=Decimal("282.49"),
        high=Decimal("287.75"),
        low=Decimal("281.28"),
        volume=4100000,
        source="Stooq",
        source_url="https://stooq.com/q/l/?s=ibm.us",
    )

    append_quotes(table_path, [quote])

    rows = table_path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == (
        "fetched_at_utc,symbol,market_symbol,price,currency,quote_date,quote_time,open,high,"
        "low,volume,previous_price,price_change,percent_change,run_summary,source,source_url"
    )
    assert rows[1].endswith(",4100000,,,,,Stooq,https://stooq.com/q/l/?s=ibm.us")


def test_stock_price_agent_uses_default_symbols_and_appends(tmp_path) -> None:
    class FakeSource:
        name = "fake"

        def fetch(self, symbols):
            assert symbols == ("IBM", "KO")
            return [
                StockQuote(
                    fetched_at_utc=datetime(2026, 5, 13, 10, 15, tzinfo=UTC),
                    symbol="IBM",
                    market_symbol="IBM.US",
                    price=Decimal("283.04"),
                    currency="USD",
                    quote_date="2026-05-12",
                    quote_time="22:00:10",
                    open=None,
                    high=None,
                    low=None,
                    volume=None,
                    source="Fake",
                    source_url="https://example.test",
                )
            ]

    run = StockPriceAgent(source=FakeSource(), default_symbols=("IBM", "KO")).run(
        table_path=tmp_path / "prices.csv"
    )

    assert len(run.quotes) == 1
    assert len(run.observations) == 1
    assert run.table_path.exists()


def test_build_observations_adds_price_changes() -> None:
    quote = StockQuote(
        fetched_at_utc=datetime(2026, 5, 13, 10, 15, tzinfo=UTC),
        symbol="IBM",
        market_symbol="IBM.US",
        price=Decimal("110"),
        currency="USD",
        quote_date="2026-05-13",
        quote_time="16:00:00",
        open=None,
        high=None,
        low=None,
        volume=None,
        source="Fake",
        source_url="https://example.test",
    )

    observations = build_observations([quote], {"IBM": Decimal("100")})

    assert observations[0].previous_price == Decimal("100")
    assert observations[0].price_change == Decimal("10")
    assert observations[0].percent_change == Decimal("10.00")


def test_find_skipped_symbols_reports_requested_symbols_without_quotes() -> None:
    quote = StockQuote(
        fetched_at_utc=datetime(2026, 5, 13, 10, 15, tzinfo=UTC),
        symbol="JPM",
        market_symbol="JPM.US",
        price=Decimal("301.41"),
        currency="USD",
        quote_date="2026-05-13",
        quote_time="16:51:35",
        open=None,
        high=None,
        low=None,
        volume=None,
        source="Fake",
        source_url="https://example.test",
    )

    assert find_skipped_symbols(("JPM", "UALF"), [quote]) == ("UALF",)


def test_append_quotes_migrates_existing_table_and_tracks_previous_price(tmp_path) -> None:
    table_path = tmp_path / "prices.csv"
    table_path.write_text(
        "\n".join(
            [
                "fetched_at_utc,symbol,market_symbol,price,currency,quote_date,quote_time,open,high,low,volume,source,source_url",
                "2026-05-13T10:15:00+00:00,IBM,IBM.US,100,USD,2026-05-13,16:00:00,,,,,Fake,https://example.test",
            ]
        ),
        encoding="utf-8",
    )
    quote = StockQuote(
        fetched_at_utc=datetime(2026, 5, 13, 11, 15, tzinfo=UTC),
        symbol="IBM",
        market_symbol="IBM.US",
        price=Decimal("105"),
        currency="USD",
        quote_date="2026-05-13",
        quote_time="17:00:00",
        open=None,
        high=None,
        low=None,
        volume=None,
        source="Fake",
        source_url="https://example.test",
    )

    append_quotes(table_path, [quote])

    assert load_latest_prices(table_path)["IBM"] == Decimal("105")
    rows = table_path.read_text(encoding="utf-8").splitlines()
    assert rows[0].endswith(
        "previous_price,price_change,percent_change,run_summary,source,source_url"
    )
    assert ",100,5,5.00,,Fake," in rows[-1]


def test_parse_stock_price_plan_accepts_json_response() -> None:
    plan = parse_stock_price_plan(
        '{"symbols":["jpm","bac.us","WFC","big banks","JPM"],"rationale":"large US banks"}'
    )

    assert plan.symbols == ("JPM", "BAC", "WFC")
    assert plan.rationale == "large US banks"


def test_parse_stock_price_plan_accepts_fenced_json_response() -> None:
    plan = parse_stock_price_plan(
        """
        ```json
        {"symbols":["AAPL","MSFT"],"rationale":"large technology stocks"}
        ```
        """
    )

    assert plan.symbols == ("AAPL", "MSFT")


def test_stock_price_agent_uses_planner_for_natural_language_request(tmp_path) -> None:
    class FakePlanner:
        def plan(self, request):
            assert request == "check big bank stocks"
            return StockPricePlan(symbols=("JPM",), rationale="large US bank")

    class FakeSource:
        name = "fake"

        def fetch(self, symbols):
            assert symbols == ("JPM",)
            return [
                StockQuote(
                    fetched_at_utc=datetime(2026, 5, 13, 10, 15, tzinfo=UTC),
                    symbol="JPM",
                    market_symbol="JPM.US",
                    price=Decimal("301.41"),
                    currency="USD",
                    quote_date="2026-05-13",
                    quote_time="16:51:35",
                    open=None,
                    high=None,
                    low=None,
                    volume=None,
                    source="Fake",
                    source_url="https://example.test",
                )
            ]

    class FakeSummarizer:
        def summarize(self, *, request, plan, observations):
            assert request == "check big bank stocks"
            assert plan == StockPricePlan(symbols=("JPM",), rationale="large US bank")
            assert observations[0].quote.symbol == "JPM"
            return "JPM was collected."

    run = StockPriceAgent(
        source=FakeSource(),
        planner=FakePlanner(),
        summarizer=FakeSummarizer(),
    ).run(
        request="check big bank stocks",
        table_path=tmp_path / "prices.csv",
    )

    assert run.plan == StockPricePlan(symbols=("JPM",), rationale="large US bank")
    assert run.summary == "JPM was collected."
    assert run.quotes[0].symbol == "JPM"


def test_stock_price_agent_reports_symbols_that_source_skips(tmp_path) -> None:
    class FakePlanner:
        def plan(self, request):
            return StockPricePlan(symbols=("JPM", "UALF"), rationale="mixed request")

    class FakeSource:
        name = "fake"

        def fetch(self, symbols):
            assert symbols == ("JPM", "UALF")
            return [
                StockQuote(
                    fetched_at_utc=datetime(2026, 5, 13, 10, 15, tzinfo=UTC),
                    symbol="JPM",
                    market_symbol="JPM.US",
                    price=Decimal("301.41"),
                    currency="USD",
                    quote_date="2026-05-13",
                    quote_time="16:51:35",
                    open=None,
                    high=None,
                    low=None,
                    volume=None,
                    source="Fake",
                    source_url="https://example.test",
                )
            ]

    run = StockPriceAgent(source=FakeSource(), planner=FakePlanner()).run(
        request="check banks and uranium",
        table_path=tmp_path / "prices.csv",
    )

    assert run.skipped_symbols == ("UALF",)
    assert run.quotes[0].symbol == "JPM"
