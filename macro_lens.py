import os
from fredapi import Fred
import yfinance as yf
from dotenv import load_dotenv

from typing import Annotated, Optional, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta


load_dotenv(override=True)


class MacroState(TypedDict):
    # Meta
    current_date: str

    # Indicators
    macro_data: Optional[Dict[str, Any]]
    fetch_attempts: int
    data_requests: Optional[List[str]]          # reserved for Phase 2

    # Regime
    growth_direction: Optional[str]
    inflation_direction: Optional[str]
    regime: Optional[str]
    regime_confidence: Optional[str]
    regime_rationale: Optional[str]
    previous_regime: Optional[str]

    # Allocation
    tilts: Optional[Dict[str, str]]
    weights: Optional[Dict[str, float]]
    allocation_rationale: Optional[str]

    # Output
    report: Optional[str]
    messages: Annotated[List[Any], add_messages]


PRIMARY_SERIES = {
    "t10y2y":       "T10Y2Y",        # yield curve slope
    "bamlh0a0hym2": "BAMLH0A0HYM2",  # HY credit spread
    "t10yie":       "T10YIE",        # 10Y breakeven inflation
    "pcepilfe":     "PCEPILFE",      # core PCE (Fed's preferred inflation gauge)
    "indpro":       "INDPRO",        # industrial production
    "sahmrealtime": "SAHMREALTIME",  # Sahm rule recession indicator
    "unrate":       "UNRATE",        # unemployment rate
    "stlfsi4":      "STLFSI4",       # financial stress index
}

SECONDARY_SERIES = {
    "cfnai":        "CFNAI",         # Chicago Fed national activity index
    "permit":       "PERMIT",        # building permits (leading indicator)
    "umcsent":      "UMCSENT",       # U Michigan consumer sentiment
}


def fetch_fred_series(fred: Fred, series_id: str) -> dict:
    """Fetch the last observations of a FRED series efficiently."""
    try:
        start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
        data = fred.get_series(series_id, observation_start=start_date)
        data = data.dropna()

        if len(data) == 0:
            return {"error": "no data"}

        latest = float(data.iloc[-1])
        previous = float(data.iloc[-2]) if len(data) >= 2 else None
        mom_change = round(latest - previous, 4) if previous is not None else None

        return {
            "latest": round(latest, 4),
            "previous": round(previous, 4) if previous else None,
            "change": mom_change,
            "date": str(data.index[-1].date()),
        }
    except Exception as e:
        return {"error": str(e)}


def fetch_vix() -> dict:
    """Fetch latest VIX close from yFinance."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d")
        if hist.empty:
            return {"error": "no data"}
        latest = round(float(hist["Close"].iloc[-1]), 2)
        previous = round(float(hist["Close"].iloc[-2]), 2) if len(hist) >= 2 else None
        return {
            "latest": latest,
            "previous": previous,
            "change": round(latest - previous, 2) if previous else None,
            "date": str(hist.index[-1].date()),
        }
    except Exception as e:
        return {"error": str(e)}


def data_fetcher(state: MacroState) -> dict:
    fred = Fred(api_key=os.getenv("FRED_API_KEY"))
    fetch_attempts = state.get("fetch_attempts", 0)

    # On retry, also pull secondary series
    series_to_fetch = PRIMARY_SERIES.copy()
    if fetch_attempts > 0:
        series_to_fetch.update(SECONDARY_SERIES)

    macro_data = {}
    for key, series_id in series_to_fetch.items():
        macro_data[key] = fetch_fred_series(fred, series_id)

    macro_data["vix"] = fetch_vix()

    return {
        "macro_data": macro_data,
        "fetch_attempts": fetch_attempts + 1,
    }


if __name__ == "__main__":
    from datetime import datetime

    test_state: MacroState = {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "macro_data": None,
        "fetch_attempts": 0,
        "data_requests": None,
        "growth_direction": None,
        "inflation_direction": None,
        "regime": None,
        "regime_confidence": None,
        "regime_rationale": None,
        "previous_regime": None,
        "tilts": None,
        "weights": None,
        "allocation_rationale": None,
        "report": None,
        "messages": [],
    }

    print("Testing data fetcher...")
    result = data_fetcher(test_state)

    print(f"Fetch attempts: {result['fetch_attempts']}")
    print(f"Indicators fetched: {len(result['macro_data'])}")
    print("\nSample output:")
    for key, value in result["macro_data"].items():
        print(f"  {key}: {value}")
