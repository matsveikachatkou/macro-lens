import os
from fredapi import Fred
import yfinance as yf
import pandas as pd
from dotenv import load_dotenv

from typing import Annotated, Optional, Dict, Any, List
from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from datetime import datetime, timedelta

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from enum import Enum

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

import time
import uuid
import concurrent.futures
from tenacity import retry, wait_exponential, stop_after_attempt


load_dotenv(override=True)


class MacroState(TypedDict):
    # Meta
    current_date: str
    observation_date: Optional[str]             # v2: point-in-time date for backtest

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


def fetch_fred_series(fred: Fred, series_id: str, observation_date: Optional[str] = None) -> dict:
    try:
        def _fetch():
            if observation_date:
                start_date = (
                    datetime.strptime(observation_date, "%Y-%m-%d") - timedelta(days=365)
                ).strftime("%Y-%m-%d")
                return fred.get_series(
                    series_id,
                    observation_start=start_date,
                    observation_end=observation_date,
                    realtime_start=observation_date,
                    realtime_end=observation_date,
                )
            else:
                start_date = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
                return fred.get_series(series_id, observation_start=start_date)

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_fetch)
            data = future.result(timeout=15)

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
    except concurrent.futures.TimeoutError:
        return {"error": f"timeout fetching {series_id}"}
    except Exception as e:
        return {"error": str(e)}
        

def fetch_vix(observation_date: Optional[str] = None) -> dict:
    """Fetch VIX from yFinance. In backtest mode returns the closest prior close."""
    try:
        vix = yf.Ticker("^VIX")
        if observation_date:
            end_dt = datetime.strptime(observation_date, "%Y-%m-%d")
            start_dt = end_dt - timedelta(days=10)
            hist = vix.history(
                start=start_dt.strftime("%Y-%m-%d"),
                end=(end_dt + timedelta(days=1)).strftime("%Y-%m-%d"),
            )
        else:
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
    observation_date = state.get("observation_date")      # None in live mode

    series_to_fetch = PRIMARY_SERIES.copy()
    if fetch_attempts > 0:
        series_to_fetch.update(SECONDARY_SERIES)

    macro_data = {}
    for key, series_id in series_to_fetch.items():
        macro_data[key] = fetch_fred_series(fred, series_id, observation_date)

    macro_data["vix"] = fetch_vix(observation_date)

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


class RegimeType(str, Enum):
    high_high = "High Growth / High Inflation"
    high_low  = "High Growth / Low Inflation"
    low_high  = "Low Growth / High Inflation"
    low_low   = "Low Growth / Low Inflation"


class RegimeOutput(BaseModel):
    growth_direction: GrowthDirection = Field(
        description="Whether growth is rising or falling relative to trend"
    )
    inflation_direction: InflationDirection = Field(
        description="Whether inflation is rising or falling relative to trend"
    )
    regime: RegimeType = Field(
        description="One of the four Bridgewater 2x2 quadrants"
    )
    regime_confidence: Confidence = Field(
        description="High if indicators clearly point to one quadrant, low if mixed signals"
    )
    regime_rationale: str = Field(
        description="2-3 sentence explanation citing specific indicators"
    )


def regime_classifier(state: MacroState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
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
            "regime": result.regime.value,
            "regime_confidence": result.regime_confidence.value,
            "regime_rationale": result.regime_rationale,
            "messages": messages,
        }


def confidence_router(state: MacroState) -> str:
    confidence = state.get("regime_confidence")
    fetch_attempts = state.get("fetch_attempts", 0)

    if confidence == "high":
        return "allocation_generator"
    elif fetch_attempts < 2:
        return "data_fetcher"
    else:
        return "allocation_generator"


class TiltLevel(str, Enum):
    strong_underweight = "Strong Underweight"
    underweight        = "Underweight"
    neutral            = "Neutral"
    overweight         = "Overweight"
    strong_overweight  = "Strong Overweight"


class AllocationOutput(BaseModel):
    equities: TiltLevel = Field(description="Tilt for global equities")
    bonds: TiltLevel = Field(description="Tilt for nominal government bonds")
    inflation_linked: TiltLevel = Field(description="Tilt for inflation-linked bonds / TIPS")
    commodities: TiltLevel = Field(description="Tilt for commodities")
    gold: TiltLevel = Field(description="Tilt for gold")
    cash: TiltLevel = Field(description="Tilt for cash")
    rationale: str = Field(description="3-4 sentence rationale citing the regime and key indicators")


def allocation_generator(state: MacroState) -> dict:
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    structured_llm = llm.with_structured_output(AllocationOutput)

    system_message = """You are a senior multi-asset portfolio strategist.
Given a macroeconomic regime classification, you produce tactical tilts 
relative to a neutral benchmark for six asset classes.

Bridgewater 2x2 regime guidance:
- High Growth / High Inflation:  overweight equities and commodities, underweight bonds, neutral gold
- High Growth / Low Inflation:   strong overweight equities, overweight bonds, underweight commodities and gold
- Low Growth / High Inflation:   underweight equities and bonds, strong overweight commodities and gold, overweight cash
- Low Growth / Low Inflation:    overweight bonds, neutral equities, underweight commodities, neutral gold

Apply these as starting points but adjust based on the specific indicator readings provided."""

    user_message = f"""Today is {state['current_date']}.

Regime: {state['regime']}
Growth direction: {state['growth_direction']}
Inflation direction: {state['inflation_direction']}
Confidence: {state['regime_confidence']}
Rationale: {state['regime_rationale']}

Key indicators:
"""
    macro_data = state.get("macro_data", {})
    for key, val in macro_data.items():
        if "error" not in val:
            user_message += f"  {key.upper()}: latest={val['latest']}, change={val['change']}\n"

    user_message += "\nGenerate tactical tilts for each asset class."

    messages = [
        SystemMessage(content=system_message),
        HumanMessage(content=user_message),
    ]

    result: AllocationOutput = structured_llm.invoke(messages)

    tilts = {
        "equities":        result.equities.value,
        "bonds":           result.bonds.value,
        "inflation_linked": result.inflation_linked.value,
        "commodities":     result.commodities.value,
        "gold":            result.gold.value,
        "cash":            result.cash.value,
    }

    return {
        "tilts": tilts,
        "allocation_rationale": result.rationale,
        "messages": messages,
    }


BASELINE_WEIGHTS = {
    "equities":         0.35,
    "bonds":            0.30,
    "inflation_linked": 0.10,
    "commodities":      0.10,
    "gold":             0.05,
    "cash":             0.10,
}

TILT_MAP = {
    "Strong Underweight": -0.10,
    "Underweight":        -0.05,
    "Neutral":             0.00,
    "Overweight":         +0.05,
    "Strong Overweight":  +0.10,
}

ASSET_PROXIES = {
    "equities":         "SPY",
    "bonds":            "TLT",
    "inflation_linked": "TIP",  # TIPS, not LQD (corporate credit)
    "commodities":      "GSG",
    "gold":             "GLD",
    "cash":             "BIL",
}


def _fetch_price_history(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    """Monthly adjusted close prices for all tickers in a single yfinance call."""
    raw = yf.download(
        tickers,
        start=start,
        end=(datetime.strptime(end, "%Y-%m-%d") + timedelta(days=5)).strftime("%Y-%m-%d"),
        interval="1mo",
        auto_adjust=True,
        progress=False,
    )
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]] if "Close" in raw.columns else raw

    return prices.ffill().dropna(how="all")


def _monthly_dates(start: str = "2010-01-01", end: Optional[str] = None) -> List[str]:
    """Month-end dates (YYYY-MM-DD) from start to end inclusive."""
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d") if end else datetime.today()

    dates = []
    current = start_dt.replace(day=1)
    while current <= end_dt:
        if current.month == 12:
            last_day = current.replace(day=31)
        else:
            last_day = current.replace(month=current.month + 1, day=1) - timedelta(days=1)
        if last_day > end_dt:
            last_day = end_dt
        dates.append(last_day.strftime("%Y-%m-%d"))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)

    return dates


def _compute_performance(
    monthly_records: list,
    monthly_returns: pd.DataFrame,
) -> dict:
    """Apply weights[t] to returns[t+1] and compute equity curves + metrics."""

    portfolio_returns = []
    benchmark_returns = []
    perf_dates = []

    for i, record in enumerate(monthly_records[:-1]):
        next_date_str = monthly_records[i + 1]["date"]
        next_dt = pd.Timestamp(next_date_str)

        ret_idx = monthly_returns.index
        mask = (ret_idx.year == next_dt.year) & (ret_idx.month == next_dt.month)
        if not mask.any():
            continue

        ret_row = monthly_returns[mask].iloc[0]
        w = record["weights"]

        port_ret = 0.0
        for asset, ticker in ASSET_PROXIES.items():
            asset_w = w.get(asset, 0.0)
            if ticker in ret_row.index and not pd.isna(ret_row[ticker]):
                port_ret += asset_w * float(ret_row[ticker])

        spy_ret = float(ret_row["SPY"]) if "SPY" in ret_row.index and not pd.isna(ret_row["SPY"]) else 0.0
        tlt_ret = float(ret_row["TLT"]) if "TLT" in ret_row.index and not pd.isna(ret_row["TLT"]) else 0.0
        bench_ret = 0.60 * spy_ret + 0.40 * tlt_ret

        portfolio_returns.append(port_ret)
        benchmark_returns.append(bench_ret)
        perf_dates.append(next_dt)

    port_series = pd.Series(portfolio_returns, index=perf_dates)
    bench_series = pd.Series(benchmark_returns, index=perf_dates)

    equity_curve = (1 + port_series).cumprod() * 100
    benchmark_curve = (1 + bench_series).cumprod() * 100

    def max_drawdown(curve: pd.Series) -> float:
        rolling_max = curve.cummax()
        return float(((curve - rolling_max) / rolling_max).min())

    def sharpe(returns: pd.Series, rf: float = 0.02) -> float:
        ann_ret = (1 + returns.mean()) ** 12 - 1
        ann_vol = returns.std() * (12 ** 0.5)
        return (ann_ret - rf) / ann_vol if ann_vol > 0 else float("nan")

    n_months = len(port_series)
    n_years = n_months / 12 if n_months > 0 else 1
    total_ret_port  = float((equity_curve.iloc[-1] / 100) - 1) if n_months > 0 else 0.0
    total_ret_bench = float((benchmark_curve.iloc[-1] / 100) - 1) if n_months > 0 else 0.0

    return {
        "equity_curve":              equity_curve,
        "benchmark_curve":           benchmark_curve,
        "metrics": {
            "sharpe_portfolio":        sharpe(port_series),
            "sharpe_benchmark":        sharpe(bench_series),
            "max_drawdown_portfolio":  max_drawdown(equity_curve),
            "max_drawdown_benchmark":  max_drawdown(benchmark_curve),
            "total_return_portfolio":  total_ret_port,
            "total_return_benchmark":  total_ret_bench,
            "ann_return_portfolio":    float((1 + total_ret_port)  ** (1 / n_years) - 1),
            "ann_return_benchmark":    float((1 + total_ret_bench) ** (1 / n_years) - 1),
            "n_months":                n_months,
        },
    }


def run_backtest(
    start: str = "2010-01-01",
    end: Optional[str] = None,
    progress_callback=None,
) -> dict:
    """Run the full point-in-time backtest.
    weights[t] → returns[t+1], no look-ahead.
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    observation_dates = _monthly_dates(start, end)
    n_dates = len(observation_dates)

    # Fetch all price data upfront — one yfinance call for the whole range
    all_tickers = list(ASSET_PROXIES.values())
    prices = _fetch_price_history(all_tickers, start, end)
    monthly_returns = prices.pct_change().dropna()

    graph = build_graph()
    monthly_records = []
    previous_regime = None

    # THE FIX: Wrap the invoke call in a tenacity retry to survive rate limits
    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10), 
        stop=stop_after_attempt(3),
        reraise=True
    )
    def invoke_with_retry(state, config):
        return graph.invoke(state, config=config)

    for i, obs_date in enumerate(observation_dates):
        if progress_callback:
            progress_callback(i / n_dates, f"Processing {obs_date}  ({i+1}/{n_dates})")

        # THE FIX: Generate a NEW thread ID per month to prevent context bloat
        # We don't need LangGraph memory because previous_regime handles state
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        initial_state: MacroState = {
            "current_date": obs_date,
            "observation_date": obs_date,
            "macro_data": None,
            "fetch_attempts": 0,
            "data_requests": None,
            "growth_direction": None,
            "inflation_direction": None,
            "regime": None,
            "regime_confidence": None,
            "regime_rationale": None,
            "previous_regime": previous_regime,
            "tilts": None,
            "weights": None,
            "allocation_rationale": None,
            "report": None,
            "messages": [],
        }

        try:
            result = invoke_with_retry(initial_state, config)
        except Exception as e:
            print(f"  WARNING: {obs_date} failed permanently ({e}), using baseline weights")
            result = {
                "weights": BASELINE_WEIGHTS.copy(),
                "regime": previous_regime or "Unknown",
                "regime_confidence": "low",
            }

        weights = result.get("weights", BASELINE_WEIGHTS.copy())
        previous_regime = result.get("regime")

        monthly_records.append({
            "date": obs_date,
            "regime": result.get("regime", "Unknown"),
            "confidence": result.get("regime_confidence", "low"),
            "weights": weights.copy(),
        })

        # Base rate limit buffer
        time.sleep(1)

    if progress_callback:
        progress_callback(0.95, "Computing performance metrics...")
        
    result_perf = _compute_performance(monthly_records, monthly_returns)
    
    if progress_callback:
        progress_callback(1.0, "Done.")

    return {
        "equity_curve":    result_perf["equity_curve"],
        "benchmark_curve": result_perf["benchmark_curve"],
        "monthly_records": monthly_records,
        "metrics":         result_perf["metrics"],
    }


def weight_calculator(state: MacroState) -> dict:
    tilts = state.get("tilts", {})

    raw_weights = {}
    for asset, baseline in BASELINE_WEIGHTS.items():
        tilt_label = tilts.get(asset, "Neutral")
        adjustment = TILT_MAP.get(tilt_label, 0.0)
        raw_weights[asset] = baseline + adjustment

    # Clip to zero (no shorting)
    raw_weights = {k: max(0.0, v) for k, v in raw_weights.items()}

    # Normalise to exactly 100%
    total = sum(raw_weights.values())
    weights = {k: round(v / total, 4) for k, v in raw_weights.items()}

    # Force sum to exactly 1.0 by adjusting largest weight for rounding residual
    residual = round(1.0 - sum(weights.values()), 4)
    if residual != 0.0:
        largest = max(weights, key=weights.get)
        weights[largest] = round(weights[largest] + residual, 4)

    return {"weights": weights}


def reporter(state: MacroState) -> dict:
    weights = state.get("weights", {})
    tilts = state.get("tilts", {})
    
    # Format weights safely
    weights_text = "\n".join(
        f"  {asset.replace('_', ' ').title():<22} {weight:.1%}  ({tilts.get(asset, 'Neutral')})"
        for asset, weight in weights.items()
    )
    
    macro_data = state.get("macro_data", {})
    
    # Helper function to safely format indicators to avoid KeyErrors
    def safe_ind(name: str, key: str) -> str:
        data = macro_data.get(key, {})
        if "error" in data:
            return f"  {name:<25} [Fetch Error/Unavailable]"
        
        latest = data.get('latest')
        change = data.get('change')
        
        latest_str = f"{latest:>8.2f}" if latest is not None else "     N/A"
        change_str = f"{change:>+.2f}" if change is not None else " N/A"
        return f"  {name:<25} {latest_str}  (chg: {change_str})"

    report = f"""
╔══════════════════════════════════════════════════════╗
║           MACRO-LENS REGIME REPORT                   ║
║           {state['current_date']:<42} ║
╚══════════════════════════════════════════════════════╝

MACRO REGIME
────────────
Regime:      {state.get('regime', 'Unknown')}
Growth:      {str(state.get('growth_direction')).title()}
Inflation:   {str(state.get('inflation_direction')).title()}
Confidence:  {str(state.get('regime_confidence')).title()}

REGIME RATIONALE
────────────────
{state.get('regime_rationale', 'N/A')}

KEY INDICATORS
──────────────
{safe_ind('Yield curve (T10Y2Y)', 't10y2y')}
{safe_ind('HY spread (BAMLH)', 'bamlh0a0hym2')}
{safe_ind('Breakeven inf (T10YIE)', 't10yie')}
{safe_ind('Core PCE', 'pcepilfe')}
{safe_ind('Industrial production', 'indpro')}
{safe_ind('Sahm rule', 'sahmrealtime')}
{safe_ind('VIX', 'vix')}

TACTICAL ALLOCATION
───────────────────
{weights_text}

ALLOCATION RATIONALE
────────────────────
{state.get('allocation_rationale', 'N/A')}
""".strip()

    return {"report": report}


def build_graph():
    memory = MemorySaver()
    graph_builder = StateGraph(MacroState)

    # Add nodes
    graph_builder.add_node("data_fetcher", data_fetcher)
    graph_builder.add_node("regime_classifier", regime_classifier)
    graph_builder.add_node("allocation_generator", allocation_generator)
    graph_builder.add_node("weight_calculator", weight_calculator)
    graph_builder.add_node("reporter", reporter)

    # Linear edges
    graph_builder.add_edge(START, "data_fetcher")
    graph_builder.add_edge("data_fetcher", "regime_classifier")
    graph_builder.add_edge("allocation_generator", "weight_calculator")
    graph_builder.add_edge("weight_calculator", "reporter")
    graph_builder.add_edge("reporter", END)

    # Conditional edge: confidence router
    graph_builder.add_conditional_edges(
        "regime_classifier",
        confidence_router,
        {
            "data_fetcher": "data_fetcher",
            "allocation_generator": "allocation_generator",
        }
    )

    return graph_builder.compile(checkpointer=memory)


if __name__ == "__main__":

    graph = build_graph()
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}

    initial_state: MacroState = {
        "current_date": datetime.now().strftime("%Y-%m-%d"),
        "observation_date": None,          # v2: None = live mode
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

    print("Running macro-lens graph...")
    result = graph.invoke(initial_state, config=config)
    print(result["report"])