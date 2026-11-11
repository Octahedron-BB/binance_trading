"""Cross-market event-driven daily backtester and diagnostic report generator."""

import argparse
from dataclasses import dataclass, asdict
import datetime as dt
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from data_loader import (
    fetch_binance_daily,
    fetch_xaut_stitched_spot,
    fetch_yfinance_daily,
)
from strategy import StrategyConfig, add_indicators, generate_signal_series

REPORTS_DIR = Path(__file__).resolve().parent / "reports"


@dataclass(frozen=True)
class AssetSpec:
    """Mapping between a futures signal brain and a Binance spot execution pair."""

    name: str
    signal_ticker: str
    spot_symbol: str
    weight: float


@dataclass
class TradeRecord:
    """Completed or force-closed round-trip spot trade."""

    asset: str
    entry_signal_date: str
    entry_date: str
    entry_price: float
    exit_signal_date: str
    exit_date: str
    exit_price: float
    qty: float
    gross_cost_usdt: float
    net_proceeds_usdt: float
    pnl_usdt: float
    return_pct: float
    hold_days: int
    fees_and_slip_usdt: float
    forced_close: bool = False


def _prepare_signal_map(
    signal_df: pd.DataFrame,
    exec_dates: pd.DatetimeIndex,
    strategy_cfg: StrategyConfig,
) -> dict[pd.Timestamp, tuple[str, pd.Timestamp]]:
    """Map each futures signal bar D to the next available spot bar strictly > D."""
    enriched = add_indicators(signal_df, strategy_cfg)
    enriched["signal"] = generate_signal_series(
        enriched, signal_mode=strategy_cfg.signal_mode
    )

    exec_arr = exec_dates.sort_values().to_numpy()
    if len(exec_arr) == 0:
        return {}

    # Do not let old signals from weeks/months prior pile onto the first window bar.
    min_sig_date = pd.Timestamp(exec_arr[0]) - pd.Timedelta(days=1)
    mapped: dict[pd.Timestamp, tuple[str, pd.Timestamp]] = {}

    for row_date, sig in zip(enriched["date"], enriched["signal"]):
        if sig == "HOLD":
            continue
        sig_date = pd.Timestamp(row_date)
        if sig_date < min_sig_date:
            continue
        # Causal alignment: execute at the open of the first spot bar strictly after signal date.
        pos = np.searchsorted(exec_arr, sig_date.to_datetime64(), side="right")
        if pos >= len(exec_arr):
            continue
        exec_date = pd.Timestamp(exec_arr[pos])
        mapped[exec_date] = (sig, sig_date)

    return mapped


def _compute_metrics(
    equity: pd.Series,
    exposure: pd.Series,
    trades: list[TradeRecord],
    total_fees_usdt: float,
    initial_capital: float,
) -> dict:
    """Calculate comprehensive return, drawdown, and trade-level statistics."""
    rets = equity.pct_change().fillna(0.0)
    n_days = max(len(equity) - 1, 1)
    years = n_days / 365.0

    total_ret = (equity.iloc[-1] / initial_capital - 1.0) * 100.0
    cagr = ((equity.iloc[-1] / initial_capital) ** (1.0 / years) - 1.0) * 100.0 if years > 0 else 0.0
    ann_vol = float(rets.std(ddof=0) * np.sqrt(365.0) * 100.0)

    sharpe = (
        float((rets.mean() / rets.std(ddof=0)) * np.sqrt(365.0))
        if rets.std(ddof=0) > 0
        else 0.0
    )
    downside = rets[rets < 0]
    downside_dev = float(np.sqrt(np.mean(downside**2))) if len(downside) > 0 else 0.0
    sortino = float((rets.mean() / downside_dev) * np.sqrt(365.0)) if downside_dev > 0 else 0.0

    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    max_dd_pct = float(dd.min() * 100.0)
    trough_date = dd.idxmin()
    peak_date = equity.loc[:trough_date].idxmax()
    post_trough = equity.loc[trough_date:]
    recovered = post_trough[post_trough >= equity.loc[peak_date]]
    recovery_date = str(recovered.index[0].date()) if len(recovered) > 0 else "Unrecovered"

    underwater = equity < running_max
    max_uw_days = 0
    cur_uw_start = None
    for date_idx, is_uw in underwater.items():
        if is_uw and cur_uw_start is None:
            cur_uw_start = date_idx
        elif not is_uw and cur_uw_start is not None:
            max_uw_days = max(max_uw_days, (date_idx - cur_uw_start).days)
            cur_uw_start = None
    if cur_uw_start is not None:
        max_uw_days = max(max_uw_days, (equity.index[-1] - cur_uw_start).days)

    calmar = float(cagr / abs(max_dd_pct)) if max_dd_pct < 0 else 0.0

    closed = [t for t in trades if not t.forced_close]
    wins = [t for t in closed if t.pnl_usdt > 0]
    losses = [t for t in closed if t.pnl_usdt <= 0]

    gross_profit = sum(t.pnl_usdt for t in wins)
    gross_loss = abs(sum(t.pnl_usdt for t in losses))
    profit_factor = (
        round(gross_profit / gross_loss, 3)
        if gross_loss > 0
        else (999.0 if gross_profit > 0 else 0.0)
    )
    avg_win_pct = float(np.mean([t.return_pct for t in wins])) if wins else 0.0
    avg_loss_pct = float(np.mean([t.return_pct for t in losses])) if losses else 0.0
    payoff_ratio = (
        round(avg_win_pct / abs(avg_loss_pct), 3)
        if avg_loss_pct < 0
        else (999.0 if avg_win_pct > 0 else 0.0)
    )
    expectancy_pct = float(np.mean([t.return_pct for t in closed])) if closed else 0.0
    win_rate_pct = (len(wins) / len(closed) * 100.0) if closed else 0.0

    max_consec_losses = 0
    cur_streak = 0
    for t in closed:
        if t.pnl_usdt <= 0:
            cur_streak += 1
            max_consec_losses = max(max_consec_losses, cur_streak)
        else:
            cur_streak = 0

    yearly = {}
    for year, grp in equity.groupby(equity.index.year):
        prev_idx = equity.index.get_loc(grp.index[0]) - 1
        base = equity.iloc[prev_idx] if prev_idx >= 0 else grp.iloc[0]
        yearly[int(year)] = round((grp.iloc[-1] / base - 1.0) * 100.0, 2)

    return {
        "start_date": str(equity.index[0].date()),
        "end_date": str(equity.index[-1].date()),
        "days": int(n_days),
        "years": round(years, 2),
        "final_equity": round(float(equity.iloc[-1]), 2),
        "total_return_pct": round(total_ret, 2),
        "cagr_pct": round(cagr, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "calmar": round(calmar, 3),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "max_dd_peak": str(peak_date.date()),
        "max_dd_trough": str(trough_date.date()),
        "max_dd_recovery": recovery_date,
        "max_underwater_days": int(max_uw_days),
        "avg_exposure_pct": round(float(exposure.mean() * 100.0), 2),
        "closed_trades": len(closed),
        "forced_open_trades": len(trades) - len(closed),
        "win_rate_pct": round(win_rate_pct, 2),
        "profit_factor": profit_factor,
        "payoff_ratio": payoff_ratio,
        "avg_win_pct": round(avg_win_pct, 2),
        "avg_loss_pct": round(avg_loss_pct, 2),
        "expectancy_pct": round(expectancy_pct, 2),
        "avg_hold_days": round(float(np.mean([t.hold_days for t in closed])), 1) if closed else 0.0,
        "max_consec_losses": int(max_consec_losses),
        "total_fees_pct_of_init": round(total_fees_usdt / initial_capital * 100.0, 2),
        "yearly_returns_pct": yearly,
    }


def run_portfolio_backtest(
    assets: list[AssetSpec],
    signal_frames: dict[str, pd.DataFrame],
    spot_frames: dict[str, pd.DataFrame],
    strategy_cfg: StrategyConfig = StrategyConfig(),
    asset_configs: dict[str, StrategyConfig] | None = None,
    sizing_mode: Literal[
        "normalized", "legacy_bot", "total_equity_target", "buy_and_hold"
    ] = "normalized",
    initial_capital: float = 10_000.0,
    fee_bps: float = 10.0,
    slippage_bps: float = 5.0,
    deploy_ratio: float = 1.0,
    portfolio_ratio: float = 0.5,
    investment_ratio: float = 0.7,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    """Simulate causal cross-market spot execution across one or more asset legs."""
    fee_rate = fee_bps / 10_000.0
    slip_rate = slippage_bps / 10_000.0
    cfg_by_asset = {
        spec.name: (asset_configs.get(spec.name, strategy_cfg) if asset_configs else strategy_cfg)
        for spec in assets
    }

    min_dates = []
    max_dates = []
    for spec in assets:
        sp = spot_frames[spec.name]
        sig_ind = add_indicators(signal_frames[spec.name], cfg_by_asset[spec.name])
        min_dates.append(max(sp["date"].min(), sig_ind["date"].min() + pd.Timedelta(days=1)))
        max_dates.append(sp["date"].max())

    common_start = max(min_dates)
    common_end = min(max_dates)
    if start_date:
        common_start = max(common_start, pd.Timestamp(start_date))
    if end_date:
        common_end = min(common_end, pd.Timestamp(end_date))

    calendar = pd.date_range(common_start, common_end, freq="D")
    n_days = len(calendar)

    open_arr: dict[str, np.ndarray] = {}
    close_arr: dict[str, np.ndarray] = {}
    signal_maps: dict[str, dict[pd.Timestamp, tuple[str, pd.Timestamp]]] = {}
    for spec in assets:
        sp = spot_frames[spec.name].set_index("date").reindex(calendar).ffill().bfill()
        open_arr[spec.name] = sp["open"].to_numpy(dtype=float)
        close_arr[spec.name] = sp["close"].to_numpy(dtype=float)
        signal_maps[spec.name] = _prepare_signal_map(
            signal_frames[spec.name],
            calendar,
            cfg_by_asset[spec.name],
        )

    if sizing_mode == "normalized":
        sleeves_cash = {spec.name: initial_capital * spec.weight for spec in assets}
        shared_cash = 0.0
    else:
        sleeves_cash = {spec.name: 0.0 for spec in assets}
        shared_cash = initial_capital

    positions: dict[str, dict | None] = {spec.name: None for spec in assets}
    trades: list[TradeRecord] = []
    total_fees_usdt = 0.0

    equity_vals = np.empty(n_days, dtype=float)
    exposure_vals = np.empty(n_days, dtype=float)

    for idx, day in enumerate(calendar):
        if sizing_mode == "buy_and_hold":
            if idx == 0:
                for spec in assets:
                    open_px = open_arr[spec.name][idx]
                    alloc = initial_capital * deploy_ratio * spec.weight
                    shared_cash -= alloc
                    eff_px = open_px * (1.0 + slip_rate)
                    qty = (alloc * (1.0 - fee_rate)) / eff_px
                    fee_paid = alloc - qty * open_px
                    total_fees_usdt += max(fee_paid, 0.0)
                    positions[spec.name] = {
                        "qty": qty,
                        "entry_date": day,
                        "entry_signal_date": day,
                        "entry_price": eff_px,
                        "cost_usdt": alloc,
                        "entry_fee_slip": max(fee_paid, 0.0),
                    }
        else:
            day_sigs = {spec.name: signal_maps[spec.name].get(day) for spec in assets}

            # 1a. Execute exits first so cash is freed before sizing new entries on the same day.
            for spec in assets:
                sig_info = day_sigs[spec.name]
                pos = positions[spec.name]
                if not sig_info or sig_info[0] != "LONG_EXIT" or pos is None:
                    continue

                sig_date = sig_info[1]
                open_px = open_arr[spec.name][idx]
                eff_px = open_px * (1.0 - slip_rate)
                gross_val = pos["qty"] * eff_px
                proceeds = gross_val * (1.0 - fee_rate)
                exit_drag = pos["qty"] * open_px - proceeds
                total_fees_usdt += max(exit_drag, 0.0)

                if sizing_mode == "normalized":
                    sleeves_cash[spec.name] += proceeds
                else:
                    shared_cash += proceeds

                pnl = proceeds - pos["cost_usdt"]
                trades.append(
                    TradeRecord(
                        asset=spec.name,
                        entry_signal_date=str(pos["entry_signal_date"].date()),
                        entry_date=str(pos["entry_date"].date()),
                        entry_price=round(pos["entry_price"], 4),
                        exit_signal_date=str(sig_date.date()),
                        exit_date=str(day.date()),
                        exit_price=round(eff_px, 4),
                        qty=pos["qty"],
                        gross_cost_usdt=round(pos["cost_usdt"], 2),
                        net_proceeds_usdt=round(proceeds, 2),
                        pnl_usdt=round(pnl, 2),
                        return_pct=round((proceeds / pos["cost_usdt"] - 1.0) * 100.0, 2),
                        hold_days=(day - pos["entry_date"]).days,
                        fees_and_slip_usdt=round(pos["entry_fee_slip"] + max(exit_drag, 0.0), 2),
                        forced_close=False,
                    )
                )
                positions[spec.name] = None

            # Compute open equity once after exits for total_equity_target sizing.
            open_invested = sum(
                positions[s.name]["qty"] * open_arr[s.name][idx]
                for s in assets
                if positions[s.name] is not None
            )
            equity_at_open = (
                sum(sleeves_cash.values()) if sizing_mode == "normalized" else shared_cash
            ) + open_invested

            # 1b. Execute entries.
            for spec in assets:
                sig_info = day_sigs[spec.name]
                pos = positions[spec.name]
                if not sig_info or sig_info[0] != "LONG_ENTRY" or pos is not None:
                    continue

                sig_date = sig_info[1]
                open_px = open_arr[spec.name][idx]

                if sizing_mode == "normalized":
                    alloc = sleeves_cash[spec.name]
                    sleeves_cash[spec.name] = 0.0
                elif sizing_mode == "total_equity_target":
                    target_alloc = equity_at_open * deploy_ratio * spec.weight
                    alloc = min(shared_cash, target_alloc)
                    shared_cash -= alloc
                else:
                    # Reproduces bot.py lines 125 & 143: remaining USDT * portfolio_ratio * investment_ratio * weight
                    alloc = shared_cash * portfolio_ratio * investment_ratio * spec.weight
                    shared_cash -= alloc

                if alloc > 0:
                    eff_px = open_px * (1.0 + slip_rate)
                    qty = (alloc * (1.0 - fee_rate)) / eff_px
                    cost_drag = alloc * (fee_rate + slip_rate / (1.0 + slip_rate))
                    total_fees_usdt += cost_drag
                    positions[spec.name] = {
                        "qty": qty,
                        "entry_date": day,
                        "entry_signal_date": sig_date,
                        "entry_price": eff_px,
                        "cost_usdt": alloc,
                        "entry_fee_slip": cost_drag,
                    }

        # 2. Mark to market at today's Close
        invested_val = 0.0
        for spec in assets:
            pos = positions[spec.name]
            if pos is not None:
                invested_val += pos["qty"] * close_arr[spec.name][idx]

        cash_total = sum(sleeves_cash.values()) if sizing_mode == "normalized" else shared_cash
        eq = cash_total + invested_val
        equity_vals[idx] = eq
        exposure_vals[idx] = invested_val / eq if eq > 0 else 0.0

    final_day = calendar[-1]
    for spec in assets:
        pos = positions[spec.name]
        if pos is not None:
            close_px = close_arr[spec.name][-1]
            mtm_val = pos["qty"] * close_px
            trades.append(
                TradeRecord(
                    asset=spec.name,
                    entry_signal_date=str(pos["entry_signal_date"].date()),
                    entry_date=str(pos["entry_date"].date()),
                    entry_price=round(pos["entry_price"], 4),
                    exit_signal_date=str(final_day.date()),
                    exit_date=str(final_day.date()),
                    exit_price=round(close_px, 4),
                    qty=pos["qty"],
                    gross_cost_usdt=round(pos["cost_usdt"], 2),
                    net_proceeds_usdt=round(mtm_val, 2),
                    pnl_usdt=round(mtm_val - pos["cost_usdt"], 2),
                    return_pct=round((mtm_val / pos["cost_usdt"] - 1.0) * 100.0, 2),
                    hold_days=(final_day - pos["entry_date"]).days,
                    fees_and_slip_usdt=round(pos["entry_fee_slip"], 2),
                    forced_close=True,
                )
            )

    equity_series = pd.Series(equity_vals, index=calendar, name="equity")
    exposure_series = pd.Series(exposure_vals, index=calendar, name="exposure")
    metrics = _compute_metrics(equity_series, exposure_series, trades, total_fees_usdt, initial_capital)

    return {
        "metrics": metrics,
        "equity": equity_series,
        "exposure": exposure_series,
        "trades": trades,
    }


def _svg_polyline(series: pd.Series, width: int, height: int, y_min: float, y_max: float) -> str:
    n = len(series)
    if n < 2 or y_max <= y_min:
        return ""
    pts = []
    for i, val in enumerate(series.to_numpy()):
        x = (i / (n - 1)) * width
        y = height - ((val - y_min) / (y_max - y_min)) * height
        pts.append(f"{x:.1f},{y:.1f}")
    return " ".join(pts)


def _generate_html_report(
    output_path: Path,
    title: str,
    series_list: list[tuple[str, str, dict]],
    trades_source_label: str,
    trades: list[TradeRecord],
) -> None:
    """Build a zero-dependency standalone HTML visual tearsheet for arbitrary strategy series."""
    w, h = 920, 320
    y_min = float(min(res["equity"].min() for _, _, res in series_list) * 0.95)
    y_max = float(max(res["equity"].max() for _, _, res in series_list) * 1.05)

    eq_polylines = "\n".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="2.2" '
        f'points="{_svg_polyline(res["equity"], w, h, y_min, y_max)}" />'
        for _, color, res in series_list
    )
    legends = "\n".join(
        f'<span><span class="dot" style="background:{color}"></span>{label}</span>'
        for label, color, _ in series_list
    )

    dd_curves = [
        (label, color, (res["equity"] / res["equity"].cummax() - 1.0) * 100.0)
        for label, color, res in series_list
    ]
    dd_min = float(min(min(dd.min() for _, _, dd in dd_curves), -5.0))
    dd_polylines = "\n".join(
        f'<polyline fill="none" stroke="{color}" stroke-width="1.9" '
        f'points="{_svg_polyline(dd, w, 180, dd_min, 0.0)}" />'
        for _, color, dd in dd_curves
    )

    headers = "".join(f"<th>{label}</th>" for label, _, _ in series_list)
    metric_rows = [
        ("總報酬率", lambda m: f"<b>{m['total_return_pct']:+.2f}%</b>"),
        ("年化報酬率 (CAGR)", lambda m: f"<b>{m['cagr_pct']:+.2f}%</b>"),
        ("夏普比率 (Sharpe)", lambda m: f"{m['sharpe']:.2f}"),
        ("卡瑪比率 (Calmar)", lambda m: f"{m['calmar']:.2f}"),
        ("最大回撤 (MDD)", lambda m: f"{m['max_drawdown_pct']:.2f}%"),
        ("最長水下套牢天數", lambda m: f"{m['max_underwater_days']} 天"),
        ("平均資金曝險率", lambda m: f"{m['avg_exposure_pct']:.1f}%"),
        (
            "已平倉筆數 / 勝率 / PF",
            lambda m: f"{m['closed_trades']} 筆 / {m['win_rate_pct']:.1f}% / {m['profit_factor']:.2f}"
            if m["closed_trades"] > 0
            else "-",
        ),
    ]
    table_rows = "\n".join(
        f"<tr><td>{name}</td>"
        + "".join(f"<td>{fmt(res['metrics'])}</td>" for _, _, res in series_list)
        + "</tr>"
        for name, fmt in metric_rows
    )

    trade_rows = "\n".join(
        f"<tr><td>{t.asset}</td><td>{t.entry_date}</td><td>{t.exit_date}{' (Open)' if t.forced_close else ''}</td>"
        f"<td>{t.entry_price:,.2f}</td><td>{t.exit_price:,.2f}</td>"
        f"<td style='color:{'#16a34a' if t.return_pct >= 0 else '#dc2626'}'>{t.return_pct:+.2f}%</td>"
        f"<td>{t.hold_days}d</td></tr>"
        for t in trades
    )

    m0 = series_list[0][2]["metrics"]
    html = f"""<!DOCTYPE html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 32px; color: #1f2937; background: #f8fafc; }}
  .card {{ background: #fff; border-radius: 10px; padding: 20px; margin-bottom: 20px; box-shadow: 0 1px 3px rgba(0,0,0,0.08); }}
  h1, h2 {{ margin-top: 0; }}
  table {{ width: 100%; border-collapse: collapse; font-size: 14px; }}
  th, td {{ padding: 8px 10px; border-bottom: 1px solid #e5e7eb; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  th {{ background: #f1f5f9; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 18px; font-size: 13px; margin-bottom: 8px; }}
  .dot {{ display: inline-block; width: 12px; height: 4px; vertical-align: middle; margin-right: 6px; }}
</style>
</head>
<body>
  <div class="card">
    <h1>{title}</h1>
    <p>測試區間：{m0['start_date']} ~ {m0['end_date']}（共 {m0['years']} 年） | 成本設定：單邊 0.10% 手續費 + 0.05% 滑點（來回 0.30%） | 初始資金：10,000 USDT</p>
    <table>
      <tr><th>指標</th>{headers}</tr>
      {table_rows}
    </table>
  </div>
  <div class="card">
    <h2>淨值曲線對比 (USDT)</h2>
    <div class="legend">{legends}</div>
    <svg viewBox="0 0 {w} {h}" width="100%" height="300" style="background:#fafafa;border:1px solid #e2e8f0">
      {eq_polylines}
    </svg>
    <p style="font-size:12px;color:#64748b">Y 軸範圍：{y_min:,.0f} ~ {y_max:,.0f} USDT</p>
  </div>
  <div class="card">
    <h2>水下回撤曲線 (%)</h2>
    <div class="legend">{legends}</div>
    <svg viewBox="0 0 {w} 180" width="100%" height="180" style="background:#fafafa;border:1px solid #e2e8f0">
      {dd_polylines}
    </svg>
    <p style="font-size:12px;color:#64748b">Y 軸範圍：{dd_min:.1f}% ~ 0%</p>
  </div>
  <div class="card">
    <h2>完整交易明細（{trades_source_label}）</h2>
    <table>
      <tr><th>幣種</th><th>進場日</th><th>出場日</th><th>買入均價</th><th>賣出均價</th><th>單筆報酬</th><th>持倉天數</th></tr>
      {trade_rows}
    </table>
  </div>
</body>
</html>"""
    output_path.write_text(html, encoding="utf-8")


def _append_experiment_log(run_name: str, notes: str, metrics: dict) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = REPORTS_DIR / "experiment_log.csv"
    row = pd.DataFrame(
        [
            {
                "timestamp": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "run_name": run_name,
                "window": f"{metrics['start_date']}..{metrics['end_date']}",
                "total_return_pct": metrics["total_return_pct"],
                "cagr_pct": metrics["cagr_pct"],
                "sharpe": metrics["sharpe"],
                "calmar": metrics["calmar"],
                "max_dd_pct": metrics["max_drawdown_pct"],
                "win_rate_pct": metrics["win_rate_pct"],
                "profit_factor": metrics["profit_factor"],
                "closed_trades": metrics["closed_trades"],
                "notes": notes,
            }
        ]
    )
    if log_path.exists():
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)


def _load_dual_datasets(refresh_data: bool = False) -> dict:
    btc_fut = fetch_yfinance_daily("BTC=F", refresh=refresh_data)
    gc_fut = fetch_yfinance_daily("GC=F", refresh=refresh_data)
    btc_spot = fetch_binance_daily("BTC/USDT", refresh=refresh_data)
    xaut_stitched, xaut_bn_start = fetch_xaut_stitched_spot(refresh=refresh_data)
    xaut_bn_only = fetch_binance_daily("XAUT/USDT", refresh=refresh_data)
    paxg_spot = fetch_binance_daily("PAXG/USDT", refresh=refresh_data)
    return {
        "btc_fut": btc_fut,
        "gc_fut": gc_fut,
        "btc_spot": btc_spot,
        "xaut_stitched": xaut_stitched,
        "xaut_bn_start": xaut_bn_start,
        "xaut_bn_only": xaut_bn_only,
        "paxg_spot": paxg_spot,
    }


def run_baseline_suite(refresh_data: bool = False) -> dict:
    """Execute full multi-window baseline diagnostics for the current bot strategy."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    d = _load_dual_datasets(refresh_data=refresh_data)

    dual_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 0.5),
    ]
    sig_frames_dual = {"BTC/USDT": d["btc_fut"], "XAUT/USDT": d["gc_fut"]}
    spot_frames_stitched = {"BTC/USDT": d["btc_spot"], "XAUT/USDT": d["xaut_stitched"]}

    main_norm = run_portfolio_backtest(dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="normalized")
    main_legacy = run_portfolio_backtest(dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="legacy_bot")
    main_bh = run_portfolio_backtest(dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="buy_and_hold")

    paxg_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("PAXG/USDT", "GC=F", "PAXG/USDT", 0.5),
    ]
    paxg_norm = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        sizing_mode="normalized",
    )
    paxg_bh = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        sizing_mode="buy_and_hold",
    )

    spot_frames_bn_xaut = {"BTC/USDT": d["btc_spot"], "XAUT/USDT": d["xaut_bn_only"]}
    live_xaut_norm = run_portfolio_backtest(dual_assets, sig_frames_dual, spot_frames_bn_xaut, sizing_mode="normalized")
    live_xaut_bh = run_portfolio_backtest(dual_assets, sig_frames_dual, spot_frames_bn_xaut, sizing_mode="buy_and_hold")

    btc_only_assets = [AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 1.0)]
    btc_only_strat = run_portfolio_backtest(
        btc_only_assets, {"BTC/USDT": d["btc_fut"]}, {"BTC/USDT": d["btc_spot"]}, sizing_mode="normalized"
    )
    btc_only_bh = run_portfolio_backtest(
        btc_only_assets, {"BTC/USDT": d["btc_fut"]}, {"BTC/USDT": d["btc_spot"]}, sizing_mode="buy_and_hold"
    )

    gold_only_assets = [AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 1.0)]
    gold_only_strat = run_portfolio_backtest(
        gold_only_assets, {"XAUT/USDT": d["gc_fut"]}, {"XAUT/USDT": d["xaut_stitched"]}, sizing_mode="normalized"
    )
    gold_only_bh = run_portfolio_backtest(
        gold_only_assets, {"XAUT/USDT": d["gc_fut"]}, {"XAUT/USDT": d["xaut_stitched"]}, sizing_mode="buy_and_hold"
    )

    full_end = pd.Timestamp(main_norm["metrics"]["end_date"])
    oos_start = (full_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    is_end = (pd.Timestamp(oos_start) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    is_norm = run_portfolio_backtest(
        dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="normalized", end_date=is_end
    )
    is_bh = run_portfolio_backtest(
        dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="buy_and_hold", end_date=is_end
    )
    oos_norm = run_portfolio_backtest(
        dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="normalized", start_date=oos_start
    )
    oos_bh = run_portfolio_backtest(
        dual_assets, sig_frames_dual, spot_frames_stitched, sizing_mode="buy_and_hold", start_date=oos_start
    )

    sensitivity = []
    for fast in [20, 30, 40]:
        for slow in [150, 200, 250]:
            cfg = StrategyConfig(ema_fast=fast, ema_slow=slow)
            res = run_portfolio_backtest(
                dual_assets, sig_frames_dual, spot_frames_stitched, strategy_cfg=cfg, sizing_mode="normalized"
            )
            m = res["metrics"]
            sensitivity.append(
                {
                    "ema_fast": fast,
                    "ema_slow": slow,
                    "cagr_pct": m["cagr_pct"],
                    "sharpe": m["sharpe"],
                    "max_dd_pct": m["max_drawdown_pct"],
                    "calmar": m["calmar"],
                    "profit_factor": m["profit_factor"],
                    "closed_trades": m["closed_trades"],
                }
            )

    summary = {
        "xaut_binance_listing_date": str(d["xaut_bn_start"].date()),
        "main_dual_normalized": main_norm["metrics"],
        "main_dual_legacy_bot": main_legacy["metrics"],
        "main_dual_buy_and_hold": main_bh["metrics"],
        "paxg_crosscheck_normalized": paxg_norm["metrics"],
        "paxg_crosscheck_buy_and_hold": paxg_bh["metrics"],
        "binance_xaut_live_window_normalized": live_xaut_norm["metrics"],
        "binance_xaut_live_window_bh": live_xaut_bh["metrics"],
        "btc_only_full_strategy": btc_only_strat["metrics"],
        "btc_only_full_bh": btc_only_bh["metrics"],
        "gold_only_strategy": gold_only_strat["metrics"],
        "gold_only_bh": gold_only_bh["metrics"],
        "dev_split_strategy": is_norm["metrics"],
        "dev_split_bh": is_bh["metrics"],
        "last_12m_strategy": oos_norm["metrics"],
        "last_12m_bh": oos_bh["metrics"],
        "parameter_neighborhood": sensitivity,
        "trades": [asdict(t) for t in main_norm["trades"]],
    }

    (REPORTS_DIR / "baseline_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _generate_html_report(
        REPORTS_DIR / "baseline_report.html",
        "CME/COMEX 雙腦現貨策略 — 基準回測體檢報告",
        [
            ("策略本體 (50/50)", "#2563eb", main_norm),
            ("50/50 Buy & Hold", "#94a3b8", main_bh),
            ("現行 bot.py 資金配置", "#f59e0b", main_legacy),
        ],
        "策略本體 50/50",
        main_norm["trades"],
    )
    _append_experiment_log(
        "baseline_v1_current_bot",
        "Unaltered bot.py logic (EMA30/EMA200/MACD 12-26-9), normalized 50/50 sleeves",
        main_norm["metrics"],
    )
    return summary


def run_experiment_v2(refresh_data: bool = False) -> dict:
    """Evaluate 30% reserve + 35%/35% total-equity allocation & Gold EMA200-only rules."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    d = _load_dual_datasets(refresh_data=refresh_data)

    dual_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 0.5),
    ]
    sig_frames = {"BTC/USDT": d["btc_fut"], "XAUT/USDT": d["gc_fut"]}
    spot_frames = {"BTC/USDT": d["btc_spot"], "XAUT/USDT": d["xaut_stitched"]}

    # Reference: legacy bot.py sizing
    legacy_bot = run_portfolio_backtest(dual_assets, sig_frames, spot_frames, sizing_mode="legacy_bot")

    # Step 1: Fix sizing to 70% deploy (30% cash reserve, 35% BTC + 35% Gold of total equity), keep old signals
    sizing_only_70 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
    )

    # Step 2a: 70% deploy + Gold uses EMA200 regime (long whenever close > EMA200, exit < EMA200)
    cfg_gold_regime = {
        "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
        "XAUT/USDT": StrategyConfig(signal_mode="ema200_regime"),
    }
    v2_gold_ema200_regime = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        asset_configs=cfg_gold_regime,
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
    )

    # Step 2b: 70% deploy + Gold uses strict EMA200 crossover only (cross > EMA200, exit < EMA200)
    cfg_gold_cross = {
        "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
        "XAUT/USDT": StrategyConfig(signal_mode="ema200_cross"),
    }
    v2_gold_ema200_cross = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        asset_configs=cfg_gold_cross,
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
    )

    # Risk-matched benchmark: 35% BTC + 35% Gold + 30% Cash Buy & Hold
    bh_70 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        sizing_mode="buy_and_hold",
        deploy_ratio=0.70,
    )

    # Gold single-leg isolation (70% deploy vs 100% deploy? Use 100% single leg to compare signal quality cleanly!)
    gold_only = [AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 1.0)]
    gold_old = run_portfolio_backtest(
        gold_only,
        {"XAUT/USDT": d["gc_fut"]},
        {"XAUT/USDT": d["xaut_stitched"]},
        strategy_cfg=StrategyConfig(signal_mode="ema30_macd"),
        sizing_mode="normalized",
    )
    gold_ema200_regime = run_portfolio_backtest(
        gold_only,
        {"XAUT/USDT": d["gc_fut"]},
        {"XAUT/USDT": d["xaut_stitched"]},
        strategy_cfg=StrategyConfig(signal_mode="ema200_regime"),
        sizing_mode="normalized",
    )
    gold_ema200_cross = run_portfolio_backtest(
        gold_only,
        {"XAUT/USDT": d["gc_fut"]},
        {"XAUT/USDT": d["xaut_stitched"]},
        strategy_cfg=StrategyConfig(signal_mode="ema200_cross"),
        sizing_mode="normalized",
    )
    gold_bh = run_portfolio_backtest(
        gold_only,
        {"XAUT/USDT": d["gc_fut"]},
        {"XAUT/USDT": d["xaut_stitched"]},
        sizing_mode="buy_and_hold",
    )

    # Pure Binance PAXG cross-check (2020-08-28 ~ present)
    paxg_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("PAXG/USDT", "GC=F", "PAXG/USDT", 0.5),
    ]
    paxg_v2 = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        asset_configs={
            "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
            "PAXG/USDT": StrategyConfig(signal_mode="ema200_regime"),
        },
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
    )
    paxg_bh_70 = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        sizing_mode="buy_and_hold",
        deploy_ratio=0.70,
    )

    # Last 12 months OOS check
    full_end = pd.Timestamp(v2_gold_ema200_regime["metrics"]["end_date"])
    oos_start = (full_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    oos_v2 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        asset_configs=cfg_gold_regime,
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
        start_date=oos_start,
    )
    oos_step1 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
        start_date=oos_start,
    )
    oos_bh_70 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        sizing_mode="buy_and_hold",
        deploy_ratio=0.70,
        start_date=oos_start,
    )

    summary = {
        "legacy_bot": legacy_bot["metrics"],
        "step1_sizing_70pct_old_signals": sizing_only_70["metrics"],
        "step2_v2_gold_ema200_regime": v2_gold_ema200_regime["metrics"],
        "step2_v2_gold_ema200_cross": v2_gold_ema200_cross["metrics"],
        "benchmark_70pct_bh_30pct_cash": bh_70["metrics"],
        "gold_only_old_ema30_macd": gold_old["metrics"],
        "gold_only_ema200_regime": gold_ema200_regime["metrics"],
        "gold_only_ema200_cross": gold_ema200_cross["metrics"],
        "gold_only_bh": gold_bh["metrics"],
        "paxg_v2_crosscheck": paxg_v2["metrics"],
        "paxg_bh_70pct": paxg_bh_70["metrics"],
        "last_12m_step1_old_signals": oos_step1["metrics"],
        "last_12m_v2_gold_ema200": oos_v2["metrics"],
        "last_12m_bh_70pct": oos_bh_70["metrics"],
        "gold_trades_ema200_regime": [asdict(t) for t in gold_ema200_regime["trades"]],
    }

    (REPORTS_DIR / "exp_v2_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _generate_html_report(
        REPORTS_DIR / "exp_v2_report.html",
        "實驗 v2：30% 備用資金 + 總淨值五五分 + 黃金只用 EMA200 對比報告",
        [
            ("v2 (70%總淨值 + 黃金EMA200)", "#2563eb", v2_gold_ema200_regime),
            ("Step1 (70%總淨值 + 舊訊號)", "#10b981", sizing_only_70),
            ("基準 70% B&H (35/35+30現金)", "#94a3b8", bh_70),
            ("現行 bot.py 原碼資金", "#f59e0b", legacy_bot),
        ],
        "v2 (70%總淨值 + 黃金EMA200)",
        v2_gold_ema200_regime["trades"],
    )
    _append_experiment_log(
        "exp_v2_70pct_step1_old_signals",
        "30% reserve + 35%/35% total equity target, BTC+Gold both EMA30+MACD",
        sizing_only_70["metrics"],
    )
    _append_experiment_log(
        "exp_v2_70pct_gold_ema200_regime",
        "30% reserve + 35%/35% total equity target, BTC=EMA30+MACD, Gold=EMA200 regime",
        v2_gold_ema200_regime["metrics"],
    )
    return summary


def run_experiment_v3(refresh_data: bool = False) -> dict:
    """Evaluate Gold EMA200+MACD variants under 70% total-equity allocation."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    d = _load_dual_datasets(refresh_data=refresh_data)

    dual_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 0.5),
    ]
    sig_frames = {"BTC/USDT": d["btc_fut"], "XAUT/USDT": d["gc_fut"]}
    spot_frames = {"BTC/USDT": d["btc_spot"], "XAUT/USDT": d["xaut_stitched"]}

    def _run_dual(gold_mode: str) -> dict:
        return run_portfolio_backtest(
            dual_assets,
            sig_frames,
            spot_frames,
            asset_configs={
                "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
                "XAUT/USDT": StrategyConfig(signal_mode=gold_mode),
            },
            sizing_mode="total_equity_target",
            deploy_ratio=0.70,
        )

    old_70 = _run_dual("ema30_macd")
    ema200_only_70 = _run_dual("ema200_regime")
    ema200_macd_70 = _run_dual("ema200_macd")
    ema200_macd_both_exit_70 = _run_dual("ema200_macd_both_exit")
    bh_70 = run_portfolio_backtest(
        dual_assets, sig_frames, spot_frames, sizing_mode="buy_and_hold", deploy_ratio=0.70
    )

    gold_only = [AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 1.0)]

    def _run_gold(mode: str) -> dict:
        return run_portfolio_backtest(
            gold_only,
            {"XAUT/USDT": d["gc_fut"]},
            {"XAUT/USDT": d["xaut_stitched"]},
            strategy_cfg=StrategyConfig(signal_mode=mode),
            sizing_mode="normalized",
        )

    g_old = _run_gold("ema30_macd")
    g_ema200 = _run_gold("ema200_regime")
    g_ema200_macd = _run_gold("ema200_macd")
    g_ema200_macd_both = _run_gold("ema200_macd_both_exit")
    g_bh = run_portfolio_backtest(
        gold_only, {"XAUT/USDT": d["gc_fut"]}, {"XAUT/USDT": d["xaut_stitched"]}, sizing_mode="buy_and_hold"
    )

    # Pure Binance PAXG cross-check (2020-08-28 ~ present)
    paxg_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("PAXG/USDT", "GC=F", "PAXG/USDT", 0.5),
    ]
    paxg_ema200_macd = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        asset_configs={
            "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
            "PAXG/USDT": StrategyConfig(signal_mode="ema200_macd"),
        },
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
    )
    paxg_ema200_macd_both = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        asset_configs={
            "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
            "PAXG/USDT": StrategyConfig(signal_mode="ema200_macd_both_exit"),
        },
        sizing_mode="total_equity_target",
        deploy_ratio=0.70,
    )

    # Last 12M OOS slice
    full_end = pd.Timestamp(ema200_macd_70["metrics"]["end_date"])
    oos_start = (full_end - pd.DateOffset(years=1) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    def _run_oos(gold_mode: str) -> dict:
        return run_portfolio_backtest(
            dual_assets,
            sig_frames,
            spot_frames,
            asset_configs={
                "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
                "XAUT/USDT": StrategyConfig(signal_mode=gold_mode),
            },
            sizing_mode="total_equity_target",
            deploy_ratio=0.70,
            start_date=oos_start,
        )

    oos_ema200_macd = _run_oos("ema200_macd")
    oos_ema200_macd_both = _run_oos("ema200_macd_both_exit")

    summary = {
        "dual_70_old_ema30_macd": old_70["metrics"],
        "dual_70_gold_ema200_only": ema200_only_70["metrics"],
        "dual_70_gold_ema200_macd_strict": ema200_macd_70["metrics"],
        "dual_70_gold_ema200_macd_both_exit": ema200_macd_both_exit_70["metrics"],
        "dual_70_bh": bh_70["metrics"],
        "gold_only_old_ema30_macd": g_old["metrics"],
        "gold_only_ema200_only": g_ema200["metrics"],
        "gold_only_ema200_macd_strict": g_ema200_macd["metrics"],
        "gold_only_ema200_macd_both_exit": g_ema200_macd_both["metrics"],
        "gold_only_bh": g_bh["metrics"],
        "paxg_70_ema200_macd_strict": paxg_ema200_macd["metrics"],
        "paxg_70_ema200_macd_both_exit": paxg_ema200_macd_both["metrics"],
        "last_12m_ema200_macd_strict": oos_ema200_macd["metrics"],
        "last_12m_ema200_macd_both_exit": oos_ema200_macd_both["metrics"],
    }

    (REPORTS_DIR / "exp_v3_metrics.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _generate_html_report(
        REPORTS_DIR / "exp_v3_report.html",
        "實驗 v3：黃金改用 EMA200 + MACD 對比報告（70% 總淨值配置）",
        [
            ("v3a (黃金 EMA200+MACD 任弱出)", "#2563eb", ema200_macd_70),
            ("v3b (黃金 EMA200+MACD 雙弱才出)", "#7c3aed", ema200_macd_both_exit_70),
            ("v2 (黃金純 EMA200)", "#10b981", ema200_only_70),
            ("Step1 (舊規則 EMA30+MACD)", "#f59e0b", old_70),
            ("基準 70% B&H", "#94a3b8", bh_70),
        ],
        "v3a (黃金 EMA200+MACD 任弱出)",
        ema200_macd_70["trades"],
    )
    _append_experiment_log(
        "exp_v3a_70pct_gold_ema200_macd_strict",
        "30% reserve + 35%/35% total equity, Gold=EMA200+MACD (OR exit)",
        ema200_macd_70["metrics"],
    )
    _append_experiment_log(
        "exp_v3b_70pct_gold_ema200_macd_both_exit",
        "30% reserve + 35%/35% total equity, Gold=EMA200+MACD (AND exit)",
        ema200_macd_both_exit_70["metrics"],
    )
    return summary


def run_simulation_2025_1000u(refresh_data: bool = False) -> dict:
    """Simulate v3b starting 2025-01-01 with 1,000 USDT through latest closed bar."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    d = _load_dual_datasets(refresh_data=refresh_data)

    dual_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("XAUT/USDT", "GC=F", "XAUT/USDT", 0.5),
    ]
    sig_frames = {"BTC/USDT": d["btc_fut"], "XAUT/USDT": d["gc_fut"]}
    spot_frames = {"BTC/USDT": d["btc_spot"], "XAUT/USDT": d["xaut_stitched"]}
    cfg_v3b = {
        "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
        "XAUT/USDT": StrategyConfig(signal_mode="ema200_macd_both_exit"),
    }

    sim_v3b = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        asset_configs=cfg_v3b,
        sizing_mode="total_equity_target",
        initial_capital=1_000.0,
        deploy_ratio=0.70,
        start_date="2025-01-01",
    )
    bh_70 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        sizing_mode="buy_and_hold",
        initial_capital=1_000.0,
        deploy_ratio=0.70,
        start_date="2025-01-01",
    )
    bh_100 = run_portfolio_backtest(
        dual_assets,
        sig_frames,
        spot_frames,
        sizing_mode="buy_and_hold",
        initial_capital=1_000.0,
        deploy_ratio=1.0,
        start_date="2025-01-01",
    )
    paxg_assets = [
        AssetSpec("BTC/USDT", "BTC=F", "BTC/USDT", 0.5),
        AssetSpec("PAXG/USDT", "GC=F", "PAXG/USDT", 0.5),
    ]
    sim_paxg = run_portfolio_backtest(
        paxg_assets,
        {"BTC/USDT": d["btc_fut"], "PAXG/USDT": d["gc_fut"]},
        {"BTC/USDT": d["btc_spot"], "PAXG/USDT": d["paxg_spot"]},
        asset_configs={
            "BTC/USDT": StrategyConfig(signal_mode="ema30_macd"),
            "PAXG/USDT": StrategyConfig(signal_mode="ema200_macd_both_exit"),
        },
        sizing_mode="total_equity_target",
        initial_capital=1_000.0,
        deploy_ratio=0.70,
        start_date="2025-01-01",
    )

    summary = {
        "v3b_1000u": sim_v3b["metrics"],
        "bh_70pct_1000u": bh_70["metrics"],
        "bh_100pct_1000u": bh_100["metrics"],
        "paxg_crosscheck_1000u": sim_paxg["metrics"],
        "trades": [asdict(t) for t in sim_v3b["trades"]],
    }
    (REPORTS_DIR / "sim_2025_1000u.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _generate_html_report(
        REPORTS_DIR / "sim_2025_1000u.html",
        "v3b 實戰模擬：1,000 USDT 自 2025-01-01 起至今",
        [
            ("v3b 策略 (1,000 U)", "#2563eb", sim_v3b),
            ("70% B&H (35/35 + 30%現金)", "#94a3b8", bh_70),
            ("100% B&H (50/50 滿倉)", "#f59e0b", bh_100),
        ],
        "v3b 策略 (1,000 U)",
        sim_v3b["trades"],
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run cross-market strategy backtests.")
    parser.add_argument("--exp", choices=["baseline", "v2", "v3", "sim_2025"], default="baseline")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    if args.exp == "sim_2025":
        result = run_simulation_2025_1000u(refresh_data=args.refresh)
        printable = result
    elif args.exp == "v3":
        result = run_experiment_v3(refresh_data=args.refresh)
        printable = result
    elif args.exp == "v2":
        result = run_experiment_v2(refresh_data=args.refresh)
        printable = {k: v for k, v in result.items() if "trades" not in k}
    else:
        result = run_baseline_suite(refresh_data=args.refresh)
        printable = {k: v for k, v in result.items() if k != "trades"}
    print(json.dumps(printable, indent=2, ensure_ascii=False))
