import os
from fredapi import Fred
import yfinance as yf
from dotenv import load_dotenv

from typing import Annotated, Optional, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from enum import Enum


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

class GrowthDirection(str, Enum):
    rising = "rising"
    falling = "falling"


class InflationDirection(str, Enum):
    rising = "rising"
    falling = "falling"


class Confidence(str, Enum):
    high = "high"
    low = "low"


class RegimeOutput(BaseModel):
    growth_direction: GrowthDirection = Field(
        description="Whether growth is rising or falling relative to trend"
    )
    inflation_direction: InflationDirection = Field(
        description="Whether inflation is rising or falling relative to trend"
    )
    regime: str = Field(
        description="One of: 'High Growth / High Inflation', 'High Growth / Low Inflation', 'Low Growth / High Inflation', 'Low Growth / Low Inflation'"
    )
    regime_confidence: Confidence = Field(
        description="High if indicators clearly point to one quadrant, low if mixed signals"
    )
    regime_rationale: str = Field(
        description="2-3 sentence explanation of the classification citing specific indicators"
    )


def regime_classifier(state: MacroState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini")
    structured_llm = llm.with_structured_output(RegimeOutput)

    previous_regime = state.get("previous_regime")
    macro_data = state.get("macro_data", {})

    system_message = """You are a macroeconomic analyst classifying the current market regime 
using the Bridgewater 2x2 framework. You evaluate whether growth and inflation are 
rising or falling relative to trend, and classify the economy into one of four quadrants:

- High Growth / High Inflation
- High Growth / Low Inflation  
- Low Growth / High Inflation (Stagflation)
- Low Growth / Low Inflation

Be conservative about declaring regime changes. If prior regime was provided, 
require clear contradictory evidence across multiple indicators before switching."""

    indicator_text = ""
    for key, val in macro_data.items():
        if "error" not in val:
            indicator_text += f"\n{key.upper()}: latest={val['latest']}, previous={val['previous']}, change={val['change']}, as_of={val['date']}"
        else:
            indicator_text += f"\n{key.upper()}: unavailable"

    user_message = f"""Today is {state['current_date']}.

Macro indicators:
{indicator_text}

"""
    if previous_regime:
        user_message += f"Previous classified regime: {previous_regime}\n"
    else:
        user_message += "No previous regime on record (first run).\n"

    user_message += "\nClassify the current macro regime."

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]

    result: RegimeOutput = structured_llm.invoke(messages)

    return {
        "growth_direction": result.growth_direction.value,
        "inflation_direction": result.inflation_direction.value,
        "regime": result.regime,
        "regime_confidence": result.regime_confidence.value,
        "regime_rationale": result.regime_rationale,
        "previous_regime": result.regime,
        "messages": messages,
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

    print("Step 1: data fetcher...")
    fetcher_result = data_fetcher(test_state)
    test_state.update(fetcher_result)
    print(f"Fetched {len(test_state['macro_data'])} indicators")

    print("\nStep 2: regime classifier...")
    regime_result = regime_classifier(test_state)
    test_state.update(regime_result)

    print(f"Growth:     {test_state['growth_direction']}")
    print(f"Inflation:  {test_state['inflation_direction']}")
    print(f"Regime:     {test_state['regime']}")
    print(f"Confidence: {test_state['regime_confidence']}")
    print(f"Rationale:  {test_state['regime_rationale']}")
