"""
CNOOC (00883.HK) Real-Time Factor Deviation Monitor
=====================================================
Tracks price deviation between CNOOC, Brent Crude (BZ=F), and Offshore RMB (CNH=X).
Computes 20-period rolling Beta, 2-sigma bands, Pearson correlation, and generates
Long/Lagging signals with FX risk alerts.
"""

import argparse
import time
import logging
from datetime import datetime, timedelta
from functools import wraps

import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("cnooc_monitor")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICKERS = {
    "stock": "0883.HK",
    "oil": "BZ=F",
    "fx": "CNH=X",
}

# HK market hours in HKT (UTC+8)
HK_OPEN_HOUR, HK_OPEN_MIN = 9, 30
HK_CLOSE_HOUR, HK_CLOSE_MIN = 16, 0

ROLLING_WINDOW = 20          # 20-period rolling beta
SIGMA_BAND = 2               # 2-sigma deviation band
OIL_THRESHOLD = 0.015        # 1.5% oil move
STOCK_LAG_THRESHOLD = 0.005  # 0.5% stock lag
FX_VOL_THRESHOLD = 0.003     # 0.3% CNH intraday move triggers FX alert

MAX_RETRIES = 3
RETRY_BASE_DELAY = 2  # seconds


# ---------------------------------------------------------------------------
# Simulated data fallback (for offline / sandbox environments)
# ---------------------------------------------------------------------------
def _generate_simulated_data(days: int = 5) -> pd.DataFrame:
    """Generate realistic simulated minute-bar data when Yahoo Finance is
    unreachable.  Prices follow correlated GBMs calibrated to recent levels."""
    logger.warning("Using SIMULATED data — Yahoo Finance unreachable")
    np.random.seed(42)

    # Build minute index for HK trading hours over `days` business days
    base = pd.Timestamp.now().normalize() - pd.tseries.offsets.BDay(days)
    timestamps = []
    for d in range(days * 2):  # iterate extra to cover weekends
        day = base + pd.tseries.offsets.BDay(d)
        if len(timestamps) and timestamps[-1].date() >= pd.Timestamp.now().date():
            break
        day_minutes = pd.date_range(
            day.replace(hour=HK_OPEN_HOUR, minute=HK_OPEN_MIN),
            day.replace(hour=HK_CLOSE_HOUR, minute=HK_CLOSE_MIN),
            freq="1min",
        )
        timestamps.extend(day_minutes)
        if len(set(t.date() for t in timestamps)) >= days:
            break
    idx = pd.DatetimeIndex(timestamps).tz_localize("Asia/Hong_Kong").tz_convert("UTC")

    n = len(idx)
    dt = 1 / (390 * 252)  # fraction of year per minute

    # Correlated Brownian motions: stock correlates with oil ~0.6
    corr_matrix = np.array([
        [1.0,  0.60, -0.15],
        [0.60, 1.0,  -0.10],
        [-0.15, -0.10, 1.0],
    ])
    L = np.linalg.cholesky(corr_matrix)
    Z = np.random.randn(n, 3)
    W = Z @ L.T

    # GBM parameters (annualized)
    params = {  # (S0, mu, sigma)
        "stock": (11.50, 0.08, 0.35),   # 00883.HK ~HKD 11.50
        "oil":   (74.50, 0.05, 0.30),   # Brent ~$74.50
        "fx":    (7.2600, 0.00, 0.05),   # USD/CNH ~7.26
    }

    data = {}
    for i, (label, (s0, mu, sigma)) in enumerate(params.items()):
        log_returns = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * W[:, i]
        prices = s0 * np.exp(np.cumsum(log_returns))
        data[label] = prices

    df = pd.DataFrame(data, index=idx)
    logger.info("Simulated data shape: %s (covering %d business days)", df.shape, days)
    return df


# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------
def retry(max_retries=MAX_RETRIES, base_delay=RETRY_BASE_DELAY):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(1, max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    if result is None or (isinstance(result, pd.DataFrame) and result.empty):
                        raise ValueError(f"{func.__name__} returned empty data")
                    return result
                except Exception as e:
                    delay = base_delay * (2 ** (attempt - 1))
                    logger.warning(
                        "Attempt %d/%d for %s failed: %s — retrying in %ds",
                        attempt, max_retries, func.__name__, e, delay,
                    )
                    if attempt == max_retries:
                        logger.error("All %d retries exhausted for %s", max_retries, func.__name__)
                        return pd.DataFrame()
                    time.sleep(delay)
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# 1. fetch_realtime_data
# ---------------------------------------------------------------------------
@retry()
def fetch_realtime_data(period: str = "5d", interval: str = "1m") -> pd.DataFrame:
    """Fetch minute-level data for CNOOC, Brent Crude, and CNH=X."""
    frames = {}
    for label, ticker in TICKERS.items():
        logger.info("Fetching %s (%s) ...", label, ticker)
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            raise ValueError(f"No data returned for {ticker}")
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        frames[label] = df["Close"].rename(label)

    merged = pd.concat(frames.values(), axis=1)
    merged.index = pd.to_datetime(merged.index)

    # Filter to overlapping HK trading hours only
    merged = _filter_hk_hours(merged)
    merged.dropna(how="all", inplace=True)
    logger.info("Merged data shape after HK-hours filter: %s", merged.shape)
    return merged


def _filter_hk_hours(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only rows that fall within HK trading hours (09:30–16:00 HKT).

    yfinance returns timestamps in the exchange's local timezone or UTC.
    We convert to Asia/Hong_Kong then filter.
    """
    if df.index.tz is None:
        idx = df.index.tz_localize("UTC")
    else:
        idx = df.index
    hkt = idx.tz_convert("Asia/Hong_Kong")
    mask = (
        (hkt.hour * 60 + hkt.minute >= HK_OPEN_HOUR * 60 + HK_OPEN_MIN)
        & (hkt.hour * 60 + hkt.minute <= HK_CLOSE_HOUR * 60 + HK_CLOSE_MIN)
    )
    return df.loc[mask]


# ---------------------------------------------------------------------------
# 2. calculate_correlation
# ---------------------------------------------------------------------------
def calculate_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """Align the three series, forward-fill gaps, compute Pearson correlation
    and 20-period rolling Beta (stock ~ oil)."""
    aligned = df.ffill().dropna()
    if aligned.shape[0] < ROLLING_WINDOW:
        logger.warning("Not enough data (%d rows) for rolling window %d", aligned.shape[0], ROLLING_WINDOW)
        return aligned

    # Returns
    ret = aligned.pct_change().dropna()

    # Pearson correlation matrix
    corr = ret.corr()
    logger.info("Pearson Correlation Matrix:\n%s", corr.to_string())

    # 20-period rolling OLS: stock_ret = alpha + beta * oil_ret
    rolling_beta = []
    rolling_pred = []
    rolling_resid_std = []
    for i in range(ROLLING_WINDOW, len(ret)):
        window = ret.iloc[i - ROLLING_WINDOW : i]
        y = window["stock"].values
        X = sm.add_constant(window["oil"].values)
        try:
            model = sm.OLS(y, X).fit()
            rolling_beta.append(model.params[1])
            # Predict current return
            x_cur = np.array([1.0, ret.iloc[i]["oil"]])
            pred_ret = model.predict(x_cur)[0]
            rolling_pred.append(pred_ret)
            rolling_resid_std.append(model.resid.std())
        except Exception:
            rolling_beta.append(np.nan)
            rolling_pred.append(np.nan)
            rolling_resid_std.append(np.nan)

    idx = ret.index[ROLLING_WINDOW:]
    result = ret.iloc[ROLLING_WINDOW:].copy()
    result["beta"] = rolling_beta
    result["pred_stock_ret"] = rolling_pred
    result["resid_std"] = rolling_resid_std
    result["upper_band"] = result["pred_stock_ret"] + SIGMA_BAND * result["resid_std"]
    result["lower_band"] = result["pred_stock_ret"] - SIGMA_BAND * result["resid_std"]
    result["deviation"] = result["stock"] - result["pred_stock_ret"]
    result["outside_band"] = (result["stock"] > result["upper_band"]) | (
        result["stock"] < result["lower_band"]
    )

    logger.info(
        "Rolling Beta (last 5):\n%s",
        result[["beta", "pred_stock_ret", "stock", "deviation", "outside_band"]].tail().to_string(),
    )
    return result


# ---------------------------------------------------------------------------
# 3. signal_generator
# ---------------------------------------------------------------------------
def signal_generator(ret_df: pd.DataFrame, raw_df: pd.DataFrame) -> pd.DataFrame:
    """Generate trading signals:
    - Long Signal (Lagging): oil up >1.5% but stock lags <0.5%
    - FX Risk Alert: CNH intraday move > 0.3%
    - 2-Sigma Deviation: stock return outside predicted band
    """
    signals = []

    # Compute rolling cumulative returns over a short window for threshold checks
    # Use 30-min rolling window (30 bars at 1m)
    lookback = 30
    if len(ret_df) < lookback:
        lookback = max(len(ret_df) // 2, 1)

    for i in range(lookback, len(ret_df)):
        ts = ret_df.index[i]
        window = ret_df.iloc[i - lookback : i]
        oil_cum = (1 + window["oil"]).prod() - 1
        stock_cum = (1 + window["stock"]).prod() - 1
        fx_cum = abs((1 + window["fx"]).prod() - 1)

        row_signals = []

        # Long signal: oil up >1.5% and stock lagging <0.5%
        if oil_cum > OIL_THRESHOLD and stock_cum < STOCK_LAG_THRESHOLD:
            row_signals.append("Long Signal (Lagging)")

        # FX risk alert
        if fx_cum > FX_VOL_THRESHOLD:
            row_signals.append(f"FX Risk Alert (CNH move {fx_cum:.2%})")

        # 2-sigma band deviation
        if "outside_band" in ret_df.columns and ret_df.iloc[i]["outside_band"]:
            direction = "Above" if ret_df.iloc[i]["deviation"] > 0 else "Below"
            row_signals.append(f"2σ Deviation ({direction})")

        if row_signals:
            signals.append(
                {
                    "timestamp": ts,
                    "oil_ret_30m": f"{oil_cum:.2%}",
                    "stock_ret_30m": f"{stock_cum:.2%}",
                    "fx_ret_30m": f"{fx_cum:.2%}",
                    "beta": ret_df.iloc[i].get("beta", np.nan),
                    "signal": " | ".join(row_signals),
                }
            )

    sig_df = pd.DataFrame(signals)
    if not sig_df.empty:
        sig_df.set_index("timestamp", inplace=True)
        logger.info("=== SIGNALS TRIGGERED ===")
        logger.info("\n%s", sig_df.to_string())
    else:
        logger.info("No signals triggered in current dataset.")
    return sig_df


# ---------------------------------------------------------------------------
# 4. Visualization
# ---------------------------------------------------------------------------
def build_dashboard(raw_df: pd.DataFrame, ret_df: pd.DataFrame, sig_df: pd.DataFrame):
    """Build interactive Plotly dashboard with 4 panels."""
    fig = make_subplots(
        rows=4, cols=1, shared_xaxes=True,
        subplot_titles=(
            "CNOOC (00883.HK) vs Brent Crude (normalized)",
            "Rolling 20-period Beta (Stock ~ Oil)",
            "Stock Return vs Predicted Band (±2σ)",
            "CNH=X (Offshore RMB)",
        ),
        vertical_spacing=0.06,
        row_heights=[0.3, 0.2, 0.3, 0.2],
    )

    # --- Panel 1: Normalized price overlay ---
    for col, color, name in [
        ("stock", "#EF553B", "00883.HK"),
        ("oil", "#636EFA", "Brent (BZ=F)"),
    ]:
        normed = raw_df[col] / raw_df[col].iloc[0] * 100
        fig.add_trace(go.Scatter(x=raw_df.index, y=normed, name=name, line=dict(color=color)), row=1, col=1)

    # --- Panel 2: Rolling Beta ---
    if "beta" in ret_df.columns:
        fig.add_trace(
            go.Scatter(x=ret_df.index, y=ret_df["beta"], name="Beta", line=dict(color="#00CC96")),
            row=2, col=1,
        )

    # --- Panel 3: Stock return vs predicted band ---
    if "pred_stock_ret" in ret_df.columns:
        fig.add_trace(
            go.Scatter(x=ret_df.index, y=ret_df["stock"], name="Actual Stock Ret", line=dict(color="#EF553B")),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ret_df.index, y=ret_df["pred_stock_ret"], name="Predicted", line=dict(color="#636EFA", dash="dash")),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ret_df.index, y=ret_df["upper_band"], name="+2σ", line=dict(color="#AB63FA", dash="dot"), showlegend=False),
            row=3, col=1,
        )
        fig.add_trace(
            go.Scatter(x=ret_df.index, y=ret_df["lower_band"], name="-2σ", line=dict(color="#AB63FA", dash="dot"),
                       fill="tonexty", fillcolor="rgba(171,99,250,0.1)", showlegend=False),
            row=3, col=1,
        )
        # Mark deviation points
        dev_points = ret_df[ret_df["outside_band"] == True]
        if not dev_points.empty:
            fig.add_trace(
                go.Scatter(x=dev_points.index, y=dev_points["stock"], mode="markers",
                           name="2σ Deviation", marker=dict(color="red", size=6, symbol="x")),
                row=3, col=1,
            )

    # --- Panel 4: CNH ---
    fig.add_trace(
        go.Scatter(x=raw_df.index, y=raw_df["fx"], name="CNH=X", line=dict(color="#FFA15A")),
        row=4, col=1,
    )

    # Signal annotations on Panel 1
    if not sig_df.empty:
        for ts, row in sig_df.iterrows():
            if "Long Signal" in row["signal"]:
                fig.add_vline(x=ts, line_dash="dash", line_color="green", opacity=0.5, row=1, col=1)

    fig.update_layout(
        height=1000, title_text="CNOOC 00883.HK — Factor Deviation Monitor",
        template="plotly_dark", legend=dict(orientation="h", y=-0.05),
    )

    output_path = "/home/user/FinRL-Tutorials/cnooc_dashboard.html"
    fig.write_html(output_path)
    logger.info("Dashboard saved to %s", output_path)
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CNOOC Factor Deviation Monitor")
    parser.add_argument("--simulate", action="store_true",
                        help="Use simulated data instead of Yahoo Finance")
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("CNOOC Factor Deviation Monitor — Starting")
    logger.info("=" * 60)

    # Step 1: Fetch data (with automatic fallback to simulation)
    if args.simulate:
        raw_df = _generate_simulated_data()
    else:
        raw_df = fetch_realtime_data(period="5d", interval="1m")
        if raw_df.empty:
            logger.warning("Live data unavailable — falling back to simulated data")
            raw_df = _generate_simulated_data()

    if raw_df.empty:
        logger.error("Failed to obtain data. Exiting.")
        return

    logger.info("Raw data sample (last 5 rows):\n%s", raw_df.tail().to_string())

    # Step 2: Correlation & rolling beta
    ret_df = calculate_correlation(raw_df)
    if ret_df.empty:
        logger.error("Correlation calculation failed. Exiting.")
        return

    # Step 3: Generate signals
    sig_df = signal_generator(ret_df, raw_df)

    # Step 4: Dashboard
    build_dashboard(raw_df, ret_df, sig_df)

    # Step 5: Summary output
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    if "beta" in ret_df.columns:
        latest_beta = ret_df["beta"].dropna().iloc[-1] if not ret_df["beta"].dropna().empty else "N/A"
        print(f"Latest Rolling Beta (Stock~Oil): {latest_beta:.4f}" if isinstance(latest_beta, float) else f"Latest Rolling Beta: {latest_beta}")
    print(f"Total data points (HK hours): {len(raw_df)}")
    print(f"Signals triggered: {len(sig_df)}")
    if not sig_df.empty:
        print(f"\nSignal log:\n{sig_df.to_string()}")
    print(f"\nDashboard: cnooc_dashboard.html")


if __name__ == "__main__":
    main()
