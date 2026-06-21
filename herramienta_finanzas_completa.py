from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# HERRAMIENTA FINANCIERA - AVANCE MODULOS 1, 2 Y 3 - FINANZAS I
# Un solo archivo:
# - Limpieza de base Refinitiv
# - Modulo 1: riesgo, rentabilidad y analisis comparativo
# - Modulo 2: optimizacion de portafolios, perfiles y aversion al riesgo
# - Modulo 3: CAPM, beta OLS, SML y alfa de Jensen
#
# Dependencias:
# python -m pip install pandas numpy openpyxl streamlit plotly
#
# Ejecucion:
# streamlit run herramienta_finanzas_completa.py
# ============================================================


DEFAULT_DATA_PATHS = [
    Path("base_diaria.xlsx"),
    Path("datos") / "base_diaria.xlsx",
    Path(__file__).resolve().parent / "base_diaria.xlsx",
    Path(__file__).resolve().parent / "datos" / "base_diaria.xlsx",
    Path(r"C:\UPB\PROYECTO FINANZAS\base_diaria.xlsx"),
    Path(r"C:\UPB\QUINTO SEMESTRE\FINANZAS I\Proyecto_finanzas\datos\base_diaria.xlsx"),
]

BENCHMARK_TICKER = "SPX"
TRADING_DAYS = 252


RISK_PROFILES = {
    "Conservador": {
        "gamma": 9.0,
        "max_vol": 0.12,
        "description": "Prioriza preservacion de capital y baja volatilidad. Se apoya en alta aversion al riesgo y menor tolerancia a caidas.",
        "basis": "Adecuado cuando la capacidad de riesgo o el horizonte son bajos; en media-varianza penaliza con fuerza la varianza.",
    },
    "Moderado": {
        "gamma": 5.0,
        "max_vol": 0.18,
        "description": "Busca equilibrio entre crecimiento y estabilidad. Acepta volatilidad intermedia si mejora el retorno esperado.",
        "basis": "Perfil balanceado: combina tolerancia psicologica y capacidad financiera medias.",
    },
    "Crecimiento": {
        "gamma": 3.0,
        "max_vol": 0.25,
        "description": "Acepta fluctuaciones relevantes para capturar mayor retorno esperado de largo plazo.",
        "basis": "Perfil con mayor horizonte/capacidad de riesgo; la penalizacion por varianza es menor.",
    },
    "Agresivo": {
        "gamma": 1.5,
        "max_vol": 0.40,
        "description": "Maximiza crecimiento esperado y tolera caidas amplias en el camino.",
        "basis": "Perfil de baja aversion al riesgo: la funcion de utilidad permite mayor volatilidad por retorno adicional.",
    },
}


@dataclass(frozen=True)
class MarketData:
    prices: pd.DataFrame
    benchmark: pd.DataFrame
    risk_free: pd.DataFrame
    metadata: pd.DataFrame


def find_data_file() -> Path | None:
    for path in DEFAULT_DATA_PATHS:
        if path.exists():
            return path
    return None


def normalize_name(value: object) -> str:
    return str(value).strip().upper()


def load_asset_metadata(path_or_buffer) -> pd.DataFrame:
    raw = pd.read_excel(path_or_buffer, sheet_name="ACTIVOS", header=1, usecols="A:D")
    raw = raw.dropna(subset=["Ticker", "RIC"])
    raw.columns = ["Ticker", "RIC", "Grupo", "Tipo"]
    raw["Ticker"] = raw["Ticker"].map(normalize_name)
    raw["RIC"] = raw["RIC"].map(lambda x: str(x).strip())
    raw["Grupo"] = raw["Grupo"].fillna("Sin clasificar").astype(str).str.strip()
    raw["Tipo"] = raw["Tipo"].fillna("Sin clasificar").astype(str).str.strip()
    return raw.drop_duplicates("Ticker").reset_index(drop=True)


def read_refinitiv_prices(path_or_buffer, sheet_name: str) -> pd.DataFrame:
    raw = pd.read_excel(path_or_buffer, sheet_name=sheet_name, header=8)
    raw = raw.dropna(how="all").dropna(axis=1, how="all")
    raw = raw.rename(columns={raw.columns[0]: "Date"})
    raw["Date"] = pd.to_datetime(raw["Date"], errors="coerce", dayfirst=True)
    raw = raw.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date")
    raw = raw.set_index("Date")
    raw.columns = [str(c).strip() for c in raw.columns]
    for col in raw.columns:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")
    return raw.replace(0, np.nan).ffill().dropna(axis=1, how="all")


def load_prices(path_or_buffer) -> pd.DataFrame:
    metadata = load_asset_metadata(path_or_buffer)
    ric_to_ticker = dict(zip(metadata["RIC"], metadata["Ticker"]))
    prices = read_refinitiv_prices(path_or_buffer, "BASE FINAL")
    prices = prices.rename(columns=ric_to_ticker)
    prices.columns = [normalize_name(c) for c in prices.columns]
    return prices.loc[:, ~prices.columns.duplicated()]


def load_benchmark(path_or_buffer) -> pd.DataFrame:
    benchmark = read_refinitiv_prices(path_or_buffer, "BENCHMARK")
    first_col = benchmark.columns[0]
    return benchmark[[first_col]].rename(columns={first_col: BENCHMARK_TICKER})


def load_risk_free(path_or_buffer) -> pd.DataFrame:
    rf = pd.read_excel(
        path_or_buffer,
        sheet_name="T-BILL",
        skiprows=5,
        header=None,
        usecols="B:C",
        names=["Date", "rf_annual_pct"],
    )
    rf["Date"] = pd.to_datetime(rf["Date"], errors="coerce", dayfirst=True)
    rf["rf_annual_pct"] = pd.to_numeric(rf["rf_annual_pct"], errors="coerce")
    rf = rf.dropna(subset=["Date", "rf_annual_pct"])
    rf = rf.drop_duplicates("Date").sort_values("Date").set_index("Date")
    rf = rf[(rf["rf_annual_pct"] >= 0) & (rf["rf_annual_pct"] <= 25)]
    rf["rf_annual"] = rf["rf_annual_pct"] / 100
    rf["rf_daily"] = (1 + rf["rf_annual"]) ** (1 / TRADING_DAYS) - 1
    return rf


def load_market_data(path_or_buffer) -> MarketData:
    return MarketData(
        prices=load_prices(path_or_buffer),
        benchmark=load_benchmark(path_or_buffer),
        risk_free=load_risk_free(path_or_buffer),
        metadata=load_asset_metadata(path_or_buffer),
    )


def filter_dates(df: pd.DataFrame, start=None, end=None) -> pd.DataFrame:
    out = df.copy()
    if start is not None:
        out = out.loc[out.index >= pd.to_datetime(start)]
    if end is not None:
        out = out.loc[out.index <= pd.to_datetime(end)]
    return out


def resample_prices(prices: pd.DataFrame, frequency: str) -> tuple[pd.DataFrame, int]:
    frequency = frequency.lower()
    if frequency.startswith("di"):
        return prices.dropna(how="all"), 252
    if frequency.startswith("se"):
        return prices.resample("W-FRI").last().dropna(how="all"), 52
    if frequency.startswith("me"):
        return prices.resample("ME").last().dropna(how="all"), 12
    raise ValueError("Frecuencia invalida. Usa Diaria, Semanal o Mensual.")


def calculate_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan)


def align_risk_free(rf: pd.DataFrame, index: pd.Index, annual_factor: int) -> pd.Series:
    annual = rf["rf_annual"].reindex(index, method="ffill")
    return (1 + annual) ** (1 / annual_factor) - 1


def annualize_geometric(returns: pd.Series, annual_factor: int) -> float:
    r = returns.dropna()
    if r.empty:
        return np.nan
    growth = (1 + r).prod()
    if growth <= 0:
        return np.nan
    return growth ** (annual_factor / len(r)) - 1


def max_drawdown_from_prices(prices: pd.Series) -> float:
    p = prices.dropna()
    if p.empty:
        return np.nan
    wealth = p / p.iloc[0]
    return (wealth / wealth.cummax() - 1).min()


def safe_divide(num: float, den: float) -> float:
    if den == 0 or pd.isna(den):
        return np.nan
    return num / den


def metadata_lookup(metadata: pd.DataFrame, ticker: str, field: str) -> str:
    ticker = normalize_name(ticker)
    row = metadata.loc[metadata["Ticker"] == ticker]
    if row.empty:
        return "Benchmark" if ticker == BENCHMARK_TICKER else "Sin clasificar"
    return str(row.iloc[0][field])


# ============================================================
# MODULO 1
# ============================================================


def beta_against_market(asset_returns: pd.Series, market_returns: pd.Series) -> float:
    data = pd.concat([asset_returns, market_returns], axis=1).dropna()
    if len(data) < 5:
        return np.nan
    return safe_divide(data.iloc[:, 0].cov(data.iloc[:, 1]), data.iloc[:, 1].var(ddof=1))


def downside_deviation(excess_returns: pd.Series, annual_factor: int) -> float:
    downside = excess_returns.dropna()
    downside = downside[downside < 0]
    if len(downside) < 2:
        return np.nan
    return downside.std(ddof=1) * np.sqrt(annual_factor)


def omega_ratio(excess_returns: pd.Series, threshold: float = 0.0) -> float:
    x = excess_returns.dropna() - threshold
    return safe_divide(x[x > 0].sum(), -x[x < 0].sum())


def metrics_table(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    rf_period: pd.Series,
    benchmark: str,
    metadata: pd.DataFrame,
    annual_factor: int,
) -> pd.DataFrame:
    rows = []
    market_returns = returns[benchmark]
    for asset in returns.columns:
        r = returns[asset].dropna()
        aligned = pd.concat([returns[asset].rename("asset"), rf_period.rename("rf")], axis=1).dropna()
        excess = aligned["asset"] - aligned["rf"]

        ret_ann = annualize_geometric(returns[asset], annual_factor)
        mean_period = r.mean() if not r.empty else np.nan
        var_period = r.var(ddof=1) if len(r) > 1 else np.nan
        std_period = r.std(ddof=1) if len(r) > 1 else np.nan
        mean_ann = r.mean() * annual_factor if not r.empty else np.nan
        vol_ann = r.std(ddof=1) * np.sqrt(annual_factor) if len(r) > 1 else np.nan
        var_ann = var_period * annual_factor if not pd.isna(var_period) else np.nan
        excess_ann = excess.mean() * annual_factor if not excess.empty else np.nan
        excess_vol_ann = excess.std(ddof=1) * np.sqrt(annual_factor) if len(excess) > 1 else np.nan
        dd = max_drawdown_from_prices(prices[asset]) if asset in prices else np.nan
        var_95 = r.quantile(0.05) if not r.empty else np.nan
        cvar_95 = r[r <= var_95].mean() if not r.empty else np.nan

        rows.append(
            {
                "Activo": asset,
                "Tipo": metadata_lookup(metadata, asset, "Tipo"),
                "Grupo": metadata_lookup(metadata, asset, "Grupo"),
                "Retorno anualizado": ret_ann,
                "Media periodo": mean_period,
                "Varianza periodo": var_period,
                "Desviacion periodo": std_period,
                "Retorno medio anualizado": mean_ann,
                "Varianza anualizada": var_ann,
                "Volatilidad anualizada": vol_ann,
                "Sharpe Ratio": safe_divide(excess_ann, excess_vol_ann),
                "Sortino Ratio": safe_divide(excess_ann, downside_deviation(excess, annual_factor)),
                "Max Drawdown": dd,
                "Calmar Ratio": safe_divide(ret_ann, abs(dd)),
                "Beta": 1.0 if asset == benchmark else beta_against_market(returns[asset], market_returns),
                "VaR 95% periodo": var_95,
                "CVaR 95% periodo": cvar_95,
                "Omega Ratio": omega_ratio(excess),
                "Observaciones": int(r.count()),
            }
        )
    return pd.DataFrame(rows).sort_values("Activo").reset_index(drop=True)


def run_modulo1(data: MarketData, start=None, end=None, frequency: str = "Diaria", benchmark: str = BENCHMARK_TICKER):
    benchmark = benchmark.strip().upper() or BENCHMARK_TICKER
    prices = filter_dates(data.prices, start, end)
    bench = filter_dates(data.benchmark, start, end)
    combined_prices = prices.join(bench, how="inner")
    combined_prices, annual_factor = resample_prices(combined_prices, frequency)

    if benchmark not in combined_prices.columns:
        raise ValueError(f"No se encontro el benchmark {benchmark}. Columnas disponibles: {list(combined_prices.columns)}")

    returns = calculate_returns(combined_prices).dropna(how="all")
    rf_period = align_risk_free(data.risk_free, returns.index, annual_factor)
    indicators = metrics_table(combined_prices, returns, rf_period, benchmark, data.metadata, annual_factor)

    return {
        "prices": combined_prices,
        "returns": returns,
        "rf_period": rf_period,
        "indicators": indicators,
        "metadata": data.metadata,
        "annual_factor": annual_factor,
        "benchmark": benchmark,
    }


# ============================================================
# MODULO 2
# ============================================================


def project_to_simplex(values: np.ndarray) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n + 1)
    cond = u - cssv / ind > 0
    if not np.any(cond):
        return np.repeat(1 / n, n)
    theta = cssv[cond][-1] / ind[cond][-1]
    return np.maximum(v - theta, 0)


def portfolio_metrics(weights: np.ndarray, mu: pd.Series, cov: pd.DataFrame, rf_annual: float) -> tuple[float, float, float]:
    w = np.asarray(weights, dtype=float)
    ret = float(w @ mu.values)
    var = float(w.T @ cov.values @ w)
    vol = float(np.sqrt(max(var, 0)))
    sharpe = np.nan if vol <= 0 else (ret - rf_annual) / vol
    return ret, vol, sharpe


def optimize_min_variance(cov: pd.DataFrame, max_iter: int = 4000) -> np.ndarray:
    n = cov.shape[0]
    w = np.repeat(1 / n, n)
    sigma = cov.values
    step = 1 / (2 * max(np.linalg.norm(sigma, ord=2), 1e-8))
    for _ in range(max_iter):
        new_w = project_to_simplex(w - step * (2 * sigma @ w))
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
    return w


def optimize_max_sharpe(mu: pd.Series, cov: pd.DataFrame, rf_annual: float, max_iter: int = 5000) -> np.ndarray:
    n = len(mu)
    w = np.repeat(1 / n, n)
    sigma = cov.values
    excess_mu = mu.values - rf_annual
    step = 0.05

    for i in range(max_iter):
        port_var = max(float(w.T @ sigma @ w), 1e-12)
        port_vol = np.sqrt(port_var)
        port_excess = float(w @ excess_mu)
        grad = excess_mu / port_vol - (port_excess * (sigma @ w)) / (port_vol**3)
        new_w = project_to_simplex(w + step * grad)
        old_sharpe = portfolio_metrics(w, mu, cov, rf_annual)[2]
        new_sharpe = portfolio_metrics(new_w, mu, cov, rf_annual)[2]
        if pd.isna(new_sharpe) or new_sharpe < old_sharpe:
            step *= 0.5
            if step < 1e-8:
                break
            continue
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
        if i % 200 == 0:
            step = min(step * 1.05, 0.10)
    return w


def optimize_target_return(mu: pd.Series, cov: pd.DataFrame, target_return: float, max_iter: int = 3500) -> np.ndarray:
    n = len(mu)
    w = np.repeat(1 / n, n)
    sigma = cov.values
    mu_values = mu.values
    penalty = 200.0
    step = 1 / (2 * max(np.linalg.norm(sigma, ord=2), 1e-8) + penalty * max(float(mu_values @ mu_values), 1e-8))
    for _ in range(max_iter):
        gap = float(w @ mu_values - target_return)
        grad = 2 * sigma @ w + 2 * penalty * gap * mu_values
        new_w = project_to_simplex(w - step * grad)
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
    return w


def optimize_mean_variance_utility(mu: pd.Series, cov: pd.DataFrame, risk_aversion: float, max_iter: int = 5000) -> np.ndarray:
    n = len(mu)
    w = np.repeat(1 / n, n)
    sigma = cov.values
    mu_values = mu.values
    risk_aversion = max(float(risk_aversion), 1e-6)
    step = 1 / max(risk_aversion * np.linalg.norm(sigma, ord=2), 1e-8)
    step = min(step, 0.25)
    for _ in range(max_iter):
        grad = mu_values - risk_aversion * (sigma @ w)
        new_w = project_to_simplex(w + step * grad)
        old_utility = float(w @ mu_values - 0.5 * risk_aversion * (w.T @ sigma @ w))
        new_utility = float(new_w @ mu_values - 0.5 * risk_aversion * (new_w.T @ sigma @ new_w))
        if new_utility + 1e-12 < old_utility:
            step *= 0.5
            if step < 1e-9:
                break
            continue
        if np.linalg.norm(new_w - w) < 1e-10:
            break
        w = new_w
    return w


def concentration_index(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    return float(np.sum(w**2))


def simulate_portfolios(mu: pd.Series, cov: pd.DataFrame, rf_annual: float, n_portfolios: int = 12000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    assets = mu.index.tolist()
    rows = []
    for i in range(n_portfolios):
        w = rng.dirichlet(np.ones(len(assets)))
        ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf_annual)
        row = {"Portafolio": i + 1, "Retorno anualizado": ret, "Volatilidad anualizada": vol, "Sharpe Ratio": sharpe}
        row.update({f"Peso_{asset}": weight for asset, weight in zip(assets, w)})
        rows.append(row)
    return pd.DataFrame(rows)


def efficient_frontier_from_simulation(simulated: pd.DataFrame, points: int = 80) -> pd.DataFrame:
    df = simulated.dropna(subset=["Retorno anualizado", "Volatilidad anualizada"]).copy()
    bins = np.linspace(df["Volatilidad anualizada"].min(), df["Volatilidad anualizada"].max(), points + 1)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        bucket = df[(df["Volatilidad anualizada"] >= low) & (df["Volatilidad anualizada"] < high)]
        if not bucket.empty:
            rows.append(bucket.loc[bucket["Retorno anualizado"].idxmax()])
    frontier = pd.DataFrame(rows).sort_values("Volatilidad anualizada")
    if frontier.empty:
        return frontier
    frontier = frontier[frontier["Retorno anualizado"].cummax() <= frontier["Retorno anualizado"] + 1e-12]
    return frontier.reset_index(drop=True)


def efficient_frontier_optimized(mu: pd.Series, cov: pd.DataFrame, rf_annual: float, points: int = 60) -> pd.DataFrame:
    min_w = optimize_min_variance(cov)
    min_ret, min_vol, min_sharpe = portfolio_metrics(min_w, mu, cov, rf_annual)
    max_ret = float(mu.max())
    targets = np.linspace(min_ret, max_ret, points)
    rows = []
    for target in targets:
        w = optimize_target_return(mu, cov, target)
        ret, vol, sharpe = portfolio_metrics(w, mu, cov, rf_annual)
        rows.append(
            {
                "Retorno objetivo": target,
                "Retorno anualizado": ret,
                "Volatilidad anualizada": vol,
                "Sharpe Ratio": sharpe,
                **{f"Peso_{asset}": weight for asset, weight in zip(mu.index, w)},
            }
        )
    frontier = pd.DataFrame(rows).drop_duplicates(subset=["Volatilidad anualizada", "Retorno anualizado"])
    return frontier.sort_values("Volatilidad anualizada").reset_index(drop=True)


def run_modulo2(
    returns: pd.DataFrame,
    rf_period: pd.Series,
    annual_factor: int,
    assets: list[str],
    n_portfolios: int = 12000,
    risk_aversion: float = 5.0,
):
    selected = returns[assets].dropna(how="any")
    if selected.shape[1] < 2:
        raise ValueError("Selecciona al menos dos activos con datos suficientes.")
    if len(selected) < 30:
        raise ValueError("Se necesitan al menos 30 observaciones para optimizar.")

    rf_aligned = rf_period.reindex(selected.index, method="ffill").dropna()
    selected = selected.loc[rf_aligned.index]
    rf_annual = float((1 + rf_aligned.mean()) ** annual_factor - 1)
    mu = selected.mean() * annual_factor
    cov = selected.cov() * annual_factor
    corr = selected.corr()

    simulated = simulate_portfolios(mu, cov, rf_annual, n_portfolios=n_portfolios)
    min_w = optimize_min_variance(cov)
    tan_w = optimize_max_sharpe(mu, cov, rf_annual)
    utility_w = optimize_mean_variance_utility(mu, cov, risk_aversion)
    min_ret, min_vol, min_sharpe = portfolio_metrics(min_w, mu, cov, rf_annual)
    tan_ret, tan_vol, tan_sharpe = portfolio_metrics(tan_w, mu, cov, rf_annual)
    utility_ret, utility_vol, utility_sharpe = portfolio_metrics(utility_w, mu, cov, rf_annual)
    frontier = efficient_frontier_optimized(mu, cov, rf_annual)
    if frontier.empty:
        frontier = efficient_frontier_from_simulation(simulated)

    max_vol = max(simulated["Volatilidad anualizada"].max(), tan_vol, min_vol)
    cml_vol = np.linspace(0, max_vol * 1.05, 100)
    cml = pd.DataFrame({"Volatilidad anualizada": cml_vol, "Retorno CML": rf_annual + tan_sharpe * cml_vol})
    summary = pd.DataFrame(
        [
            {
                "Portafolio": "Minima varianza",
                "Retorno anualizado": min_ret,
                "Volatilidad anualizada": min_vol,
                "Varianza anualizada": min_vol**2,
                "Sharpe Ratio": min_sharpe,
                "Concentracion HHI": concentration_index(min_w),
                "Utilidad media-varianza": min_ret - 0.5 * risk_aversion * min_vol**2,
            },
            {
                "Portafolio": "Tangente max Sharpe",
                "Retorno anualizado": tan_ret,
                "Volatilidad anualizada": tan_vol,
                "Varianza anualizada": tan_vol**2,
                "Sharpe Ratio": tan_sharpe,
                "Concentracion HHI": concentration_index(tan_w),
                "Utilidad media-varianza": tan_ret - 0.5 * risk_aversion * tan_vol**2,
            },
            {
                "Portafolio": "Recomendado por aversion",
                "Retorno anualizado": utility_ret,
                "Volatilidad anualizada": utility_vol,
                "Varianza anualizada": utility_vol**2,
                "Sharpe Ratio": utility_sharpe,
                "Concentracion HHI": concentration_index(utility_w),
                "Utilidad media-varianza": utility_ret - 0.5 * risk_aversion * utility_vol**2,
            },
        ]
    )
    weights = pd.DataFrame(
        {
            "Activo": selected.columns,
            "Peso minima varianza": min_w,
            "Peso tangente max Sharpe": tan_w,
            "Peso recomendado aversion": utility_w,
        }
    )
    weights = weights.sort_values("Peso recomendado aversion", ascending=False)

    risk_decomposition = []
    for label, w in [
        ("Minima varianza", min_w),
        ("Tangente max Sharpe", tan_w),
        ("Recomendado por aversion", utility_w),
    ]:
        variance = float(w.T @ cov.values @ w)
        marginal = cov.values @ w
        contribution = w * marginal / variance if variance > 0 else np.repeat(np.nan, len(w))
        for asset, weight, contrib in zip(selected.columns, w, contribution):
            risk_decomposition.append({"Portafolio": label, "Activo": asset, "Peso": weight, "Contribucion al riesgo": contrib})
    risk_decomposition = pd.DataFrame(risk_decomposition)

    return {
        "returns": selected,
        "mu": mu,
        "cov": cov,
        "corr": corr,
        "rf_annual": rf_annual,
        "simulated": simulated,
        "frontier": frontier,
        "cml": cml,
        "summary": summary,
        "weights": weights,
        "risk_decomposition": risk_decomposition,
        "risk_aversion": risk_aversion,
    }



# ============================================================
# MODULO 3
# ============================================================


def ols_against_benchmark(asset_returns: pd.Series, market_returns: pd.Series, rf_period: pd.Series, annual_factor: int) -> dict:
    data = pd.concat(
        [
            asset_returns.rename("asset"),
            market_returns.rename("market"),
            rf_period.rename("rf"),
        ],
        axis=1,
    ).dropna()

    empty_result = {
        "Beta OLS": np.nan,
        "Alpha por periodo": np.nan,
        "R2": np.nan,
        "regression_data": data,
        "Retorno historico anualizado": np.nan,
        "Retorno benchmark anualizado": np.nan,
        "RF anualizada": np.nan,
        "Prima de mercado anual": np.nan,
        "Retorno CAPM anual": np.nan,
        "Alpha Jensen anual": np.nan,
        "Observaciones": int(len(data)),
    }
    if len(data) < 10:
        return empty_result

    x = data["market"] - data["rf"]
    y = data["asset"] - data["rf"]
    x_var = x.var(ddof=1)
    if pd.isna(x_var) or x_var == 0:
        return empty_result

    beta = y.cov(x) / x_var
    alpha_period = y.mean() - beta * x.mean()
    y_hat = alpha_period + beta * x
    sse = float(((y - y_hat) ** 2).sum())
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = np.nan if sst == 0 else 1 - sse / sst

    historical_return = annualize_geometric(data["asset"], annual_factor)
    benchmark_return = annualize_geometric(data["market"], annual_factor)
    rf_annual = float((1 + data["rf"].mean()) ** annual_factor - 1) if not data["rf"].dropna().empty else np.nan
    market_premium = benchmark_return - rf_annual if not (pd.isna(benchmark_return) or pd.isna(rf_annual)) else np.nan
    capm_return = rf_annual + beta * market_premium if not (pd.isna(rf_annual) or pd.isna(beta) or pd.isna(market_premium)) else np.nan
    alpha_jensen = historical_return - capm_return if not (pd.isna(historical_return) or pd.isna(capm_return)) else np.nan

    return {
        "Beta OLS": beta,
        "Alpha por periodo": alpha_period,
        "R2": r2,
        "regression_data": data,
        "Retorno historico anualizado": historical_return,
        "Retorno benchmark anualizado": benchmark_return,
        "RF anualizada": rf_annual,
        "Prima de mercado anual": market_premium,
        "Retorno CAPM anual": capm_return,
        "Alpha Jensen anual": alpha_jensen,
        "Observaciones": int(len(data)),
    }


def sml_position(alpha: float, tolerance: float = 0.0025) -> str:
    if pd.isna(alpha):
        return "Sin datos suficientes"
    if alpha > tolerance:
        return "Por encima de la SML"
    if alpha < -tolerance:
        return "Por debajo de la SML"
    return "Cerca de la SML"


def interpret_beta(beta: float) -> str:
    if pd.isna(beta):
        return "No hay observaciones suficientes para interpretar la beta."
    if beta < 0:
        return "Beta negativa: el activo muestra relacion inversa con el benchmark en la muestra."
    if abs(beta - 1) <= 0.10:
        return "Beta cercana a 1: sensibilidad similar a la del benchmark."
    if beta > 1:
        return "Beta mayor que 1: amplifica los movimientos del benchmark; sube y cae con mayor sensibilidad."
    return "Beta entre 0 y 1: se mueve en la misma direccion del benchmark, pero con menor sensibilidad."


def interpret_r2(r2: float) -> str:
    if pd.isna(r2):
        return "No hay R2 confiable por falta de datos o varianza insuficiente del benchmark."
    if r2 < 0.30:
        level = "bajo"
    elif r2 < 0.60:
        level = "medio"
    else:
        level = "alto"
    return f"R2 {level}: el benchmark explica aproximadamente {r2:.1%} de la variabilidad de los retornos excedentes del activo."


def interpret_jensen_alpha(alpha: float) -> str:
    if pd.isna(alpha):
        return "No hay alfa de Jensen suficiente para concluir."
    if alpha > 0.0025:
        return "Alfa de Jensen positivo: el activo supera el retorno exigido por CAPM para su beta historica."
    if alpha < -0.0025:
        return "Alfa de Jensen negativo: el activo rindio menos que lo exigido por CAPM para su beta historica."
    return "Alfa de Jensen cercano a cero: el activo esta alineado con el retorno esperado por CAPM."


def run_modulo3(
    returns: pd.DataFrame,
    rf_period: pd.Series,
    indicators: pd.DataFrame,
    benchmark: str,
    annual_factor: int,
    assets: list[str],
) -> dict:
    benchmark = benchmark.strip().upper() or BENCHMARK_TICKER
    if benchmark not in returns.columns:
        raise ValueError(f"No se encontro el benchmark {benchmark} en los retornos disponibles.")

    candidate_assets = [asset for asset in assets if asset in returns.columns and asset != benchmark]
    market_returns = returns[benchmark]
    rows = []
    regressions = {}

    for asset in candidate_assets:
        result = ols_against_benchmark(returns[asset], market_returns, rf_period, annual_factor)
        regressions[asset] = result
        ind = indicators.loc[indicators["Activo"] == asset]
        ind_row = ind.iloc[0] if not ind.empty else pd.Series(dtype=float)
        alpha_jensen = result["Alpha Jensen anual"]
        rows.append(
            {
                "Activo": asset,
                "Tipo": ind_row.get("Tipo", "Sin clasificar"),
                "Grupo": ind_row.get("Grupo", "Sin clasificar"),
                "Beta OLS": result["Beta OLS"],
                "R2": result["R2"],
                "Retorno esperado CAPM": result["Retorno CAPM anual"],
                "Retorno historico anualizado": result["Retorno historico anualizado"],
                "Alpha Jensen anual": alpha_jensen,
                "Posicion respecto a SML": sml_position(alpha_jensen),
                "Volatilidad anualizada": ind_row.get("Volatilidad anualizada", np.nan),
                "Sharpe Ratio": ind_row.get("Sharpe Ratio", np.nan),
                "Max Drawdown": ind_row.get("Max Drawdown", np.nan),
                "VaR 95%": ind_row.get("VaR 95% periodo", np.nan),
                "CVaR 95%": ind_row.get("CVaR 95% periodo", np.nan),
                "Observaciones": result["Observaciones"],
            }
        )

    capm_table = pd.DataFrame(rows)
    if not capm_table.empty:
        capm_table = capm_table.sort_values(["Posicion respecto a SML", "Alpha Jensen anual", "Activo"], ascending=[True, False, True]).reset_index(drop=True)

    valid_params = [reg for reg in regressions.values() if not pd.isna(reg["RF anualizada"])]
    first_valid = valid_params[0] if valid_params else {}
    params = pd.DataFrame(
        [
            {
                "Benchmark": benchmark,
                "RF anual": first_valid.get("RF anualizada", np.nan),
                "Retorno anual benchmark": first_valid.get("Retorno benchmark anualizado", np.nan),
                "Prima de mercado anual": first_valid.get("Prima de mercado anual", np.nan),
                "Annual factor": annual_factor,
                "Activos analizados": len(candidate_assets),
            }
        ]
    )
    return {"capm_table": capm_table, "regressions": regressions, "params": params}


# ============================================================
# APP STREAMLIT
# ============================================================


def pct_cols(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
    return out


def make_excel_download(result1, result2=None, result3=None) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        result1["prices"].to_excel(writer, sheet_name="Precios")
        result1["returns"].to_excel(writer, sheet_name="Retornos")
        result1["rf_period"].to_frame("rf_period").to_excel(writer, sheet_name="RF")
        result1["indicators"].to_excel(writer, sheet_name="Indicadores_M1", index=False)
        if result2:
            result2["corr"].to_excel(writer, sheet_name="Correlaciones")
            result2["cov"].to_excel(writer, sheet_name="Covarianzas")
            result2["summary"].to_excel(writer, sheet_name="Portafolios_Optimos", index=False)
            result2["weights"].to_excel(writer, sheet_name="Pesos_Optimos", index=False)
            result2["risk_decomposition"].to_excel(writer, sheet_name="Riesgo_Portafolio", index=False)
            result2["frontier"].to_excel(writer, sheet_name="Frontera", index=False)
        if result3:
            result3["capm_table"].to_excel(writer, sheet_name="CAPM_Modulo3", index=False)
            result3["params"].to_excel(writer, sheet_name="Parametros_CAPM", index=False)
    return buffer.getvalue()


def format_pct(value: float, digits: int = 2) -> str:
    """Devuelve porcentajes legibles para metricas y textos de interpretacion."""
    if pd.isna(value):
        return "Sin dato"
    return f"{value:.{digits}%}"


def format_num(value: float, digits: int = 3) -> str:
    """Devuelve numeros legibles y evita mostrar nan en la interfaz."""
    if pd.isna(value):
        return "Sin dato"
    return f"{value:.{digits}f}"


def run_app() -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    import streamlit as st

    st.set_page_config(page_title="PortfolioLab", page_icon="PL", layout="wide")

    # ------------------------------------------------------------------
    # UI THEME: todo el diseno queda concentrado aqui para mantener un
    # archivo unico y facil de publicar en Streamlit Cloud.
    # ------------------------------------------------------------------
    PRIMARY = "#2563eb"
    SECONDARY = "#0d9488"
    NAVY = "#0f172a"
    RED = "#dc2626"
    GREEN = "#16a34a"
    AMBER = "#f59e0b"
    MUTED = "#64748b"
    BORDER = "#e5e7eb"

    st.markdown(
        f"""
        <style>
        .stApp {{
            background: #f8fafc;
            color: #0f172a;
        }}
        [data-testid="stSidebar"] {{
            background: #ffffff;
            border-right: 1px solid {BORDER};
        }}
        .hero-dark {{
            width: 100%;
            padding: 30px 34px;
            border-radius: 18px;
            margin-bottom: 18px;
            background: linear-gradient(135deg, #0f172a 0%, #1e3a8a 100%);
            color: #ffffff;
            box-shadow: 0 16px 36px rgba(15, 23, 42, 0.18);
        }}
        .hero-dark h1 {{
            margin: 0;
            font-size: 38px;
            line-height: 1.05;
            font-weight: 800;
            letter-spacing: 0;
        }}
        .hero-dark p {{
            margin: 9px 0 0 0;
            color: #dbeafe;
            font-size: 17px;
        }}
        .kpi-card {{
            min-height: 132px;
            padding: 18px 18px 16px 18px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid {BORDER};
            box-shadow: 0 12px 26px rgba(15, 23, 42, 0.055);
        }}
        .kpi-emoji {{
            font-size: 23px;
            margin-bottom: 6px;
        }}
        .kpi-label {{
            color: {MUTED};
            font-size: 12px;
            text-transform: uppercase;
            font-weight: 700;
            letter-spacing: .04em;
        }}
        .kpi-value {{
            color: {NAVY};
            font-size: 26px;
            line-height: 1.15;
            font-weight: 800;
            margin-top: 5px;
        }}
        .kpi-help {{
            color: {MUTED};
            font-size: 12px;
            margin-top: 7px;
            line-height: 1.35;
        }}
        .analyst-note {{
            padding: 16px 18px;
            border-radius: 14px;
            background: #fffbeb;
            border: 1px solid #fde68a;
            border-left: 6px solid {AMBER};
            color: #78350f;
            line-height: 1.45;
        }}
        .profile-box {{
            padding: 20px 22px;
            border-radius: 16px;
            background: linear-gradient(135deg, #eff6ff 0%, #f0fdfa 100%);
            border: 1px solid #bfdbfe;
        }}
        .gamma-pill {{
            padding: 18px;
            border-radius: 16px;
            background: #ffffff;
            border: 1px solid #bfdbfe;
            text-align: center;
            box-shadow: 0 10px 24px rgba(37, 99, 235, 0.10);
        }}
        .gamma-pill b {{
            font-size: 32px;
            color: {PRIMARY};
        }}
        .interpretation-box {{
            padding: 18px;
            border-radius: 16px;
            background: #ecfdf5;
            border: 1px solid #bbf7d0;
            border-left: 6px solid {GREEN};
            color: #14532d;
            min-height: 430px;
        }}
        .section-subtitle {{
            color: {MUTED};
            font-size: 14px;
            margin-top: -8px;
            margin-bottom: 12px;
        }}
        div[data-testid="stDataFrame"] {{
            border-radius: 12px;
            overflow: hidden;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    def chart_title(title: str, subtitle: str) -> str:
        """Titulo HTML estandarizado para todos los graficos Plotly."""
        return f"<b>{title}</b><br><span style='font-size:12px;color:gray;'>{subtitle}</span>"

    def kpi_card(icon: str, label: str, value: str, help_text: str) -> None:
        """Card KPI personalizada. Evita st.metric para lograr el look corporativo pedido."""
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-emoji">{icon}</div>
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-help">{help_text}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def polish(fig, *, show_legend: bool = True):
        """Aplica identidad visual corporativa a los graficos Plotly."""
        fig.update_layout(
            template="plotly_white",
            margin=dict(l=20, r=20, t=72, b=38),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=NAVY, family="Arial"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0) if show_legend else dict(visible=False),
        )
        fig.update_xaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
        fig.update_yaxes(showgrid=True, gridcolor="#eef2f7", zeroline=False)
        return fig

    def rounded_bars(fig, radius: int = 8):
        """Redondea barras cuando la version de Plotly soporta marker_cornerradius."""
        try:
            fig.update_traces(marker_cornerradius=radius, selector=dict(type="bar"))
        except Exception:
            pass
        return fig

    # Carga de datos: primero intenta encontrar base_diaria.xlsx; si no existe,
    # permite subirla desde la barra lateral para que funcione en Streamlit Cloud.
    path = find_data_file()
    uploaded = None
    with st.sidebar:
        st.markdown("## PortfolioLab")
        page = st.radio("Navegacion", ["Cobertura", "Riesgo y Rentabilidad", "Optimizacion", "CAPM"], label_visibility="collapsed")
        if st.button("Actualizar analisis", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
        st.divider()
        uploaded = st.file_uploader("Archivo base_diaria.xlsx", type=["xlsx"])

    data_source = uploaded if uploaded is not None else path
    if data_source is None:
        st.info("Carga `base_diaria.xlsx` desde la barra lateral para iniciar el analisis.")
        st.stop()

    @st.cache_data(show_spinner="Cargando y limpiando datos de Refinitiv...")
    def cached_market_data(source_id, source):
        return load_market_data(source)

    source_id = uploaded.name if uploaded is not None else str(path)
    try:
        with st.spinner("Preparando precios, benchmark y tasa libre de riesgo..."):
            data = cached_market_data(source_id, data_source)
    except Exception as exc:
        st.error("No se pudo cargar la base. Revisa que tenga las hojas ACTIVOS, BASE FINAL, BENCHMARK y T-BILL.")
        st.exception(exc)
        st.stop()

    min_date = data.prices.index.min().date()
    max_date = data.prices.index.max().date()

    with st.sidebar:
        st.success("Archivo cargado")
        st.caption(str(data_source))
        start = st.date_input("Fecha inicial", value=min_date, min_value=min_date, max_value=max_date)
        end = st.date_input("Fecha final", value=max_date, min_value=min_date, max_value=max_date)
        frequency = st.selectbox("Frecuencia", ["Diaria", "Semanal", "Mensual"], index=0)
        benchmark = st.text_input("Benchmark", value=BENCHMARK_TICKER).strip().upper() or BENCHMARK_TICKER
        n_portfolios = st.slider("Portafolios simulados", 2000, 30000, 12000, step=1000)
        profile_name = st.selectbox("Perfil de riesgo", list(RISK_PROFILES.keys()), index=1)
        profile = RISK_PROFILES[profile_name]
        risk_aversion = st.slider(
            "Grado de aversion al riesgo",
            min_value=0.5,
            max_value=12.0,
            value=float(profile["gamma"]),
            step=0.5,
            help="Mayor aversion penaliza mas la varianza en la utilidad media-varianza.",
        )

    if start > end:
        st.error("La fecha inicial no puede ser posterior a la fecha final.")
        st.stop()

    try:
        with st.spinner("Calculando Modulo 1: riesgo y rentabilidad..."):
            result1 = run_modulo1(data, start=start, end=end, frequency=frequency, benchmark=benchmark)
    except Exception as exc:
        st.error("No se pudo procesar el Modulo 1 con los filtros seleccionados.")
        st.exception(exc)
        st.stop()

    prices = result1["prices"]
    returns = result1["returns"]
    indicators = result1["indicators"]
    rf_period = result1["rf_period"]
    annual_factor = result1["annual_factor"]
    asset_universe = [c for c in prices.columns if c != benchmark]
    default_assets = asset_universe[: min(10, len(asset_universe))]

    with st.sidebar:
        selected_assets = st.multiselect("Activos para analizar", asset_universe, default=default_assets)

    if not selected_assets:
        st.warning("Selecciona al menos un activo en la barra lateral.")
        st.stop()

    selected_with_benchmark = list(dict.fromkeys(selected_assets + [benchmark]))
    selected_indicators = indicators[indicators["Activo"].isin(selected_with_benchmark)].copy()
    focus_asset = st.sidebar.selectbox("Activo foco", selected_with_benchmark)

    # Calcula los modulos 2 y 3 una sola vez por ejecucion para que el boton de Excel
    # siempre exporte todos los resultados disponibles.
    result2 = None
    if len(selected_assets) >= 2:
        try:
            with st.spinner("Calculando frontera eficiente y portafolios optimos..."):
                result2 = run_modulo2(returns, rf_period, annual_factor, selected_assets, n_portfolios=n_portfolios, risk_aversion=risk_aversion)
        except Exception as exc:
            st.warning("El Modulo 2 no pudo calcularse con la seleccion actual.")
            st.exception(exc)

    result3 = None
    capm_assets = [asset for asset in selected_assets if asset != benchmark]
    if capm_assets:
        try:
            with st.spinner("Estimando CAPM, SML y regresiones OLS..."):
                result3 = run_modulo3(returns, rf_period, indicators, benchmark, annual_factor, capm_assets)
        except Exception as exc:
            st.warning("El Modulo 3 no pudo calcularse con la seleccion actual.")
            st.exception(exc)

    if page == "Cobertura":
        # FILA 0: banner hero.
        st.markdown(
            """
            <div class="hero-dark">
                <h1>PortfolioLab</h1>
                <p>Analiza, optimiza y entiende tu portafolio de inversion.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # FILA 1: tarjetas KPI HTML personalizadas.
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            kpi_card("📁", "Activos disponibles", f"{len(asset_universe)}", f"Universo listo para analisis y seleccion.")
        with c2:
            kpi_card("📊", "Observaciones", f"{len(prices):,}", f"Precios filtrados entre {start} y {end}.")
        with c3:
            kpi_card("🛡️", "Benchmark", benchmark, "Indice de referencia usado en comparaciones y CAPM.")
        with c4:
            kpi_card("🎯", "Activos seleccionados", f"{len(selected_assets)}", "Canasta actual para riesgo, optimizacion y CAPM.")

        # FILA 2: graficos de cobertura.
        selected_meta = data.metadata[data.metadata["Ticker"].isin(selected_assets)]
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                type_count = data.metadata["Tipo"].value_counts().reset_index()
                type_count.columns = ["Tipo", "Cantidad"]
                fig = px.bar(
                    type_count,
                    x="Tipo",
                    y="Cantidad",
                    title=chart_title("Distribucion por tipo", "Cantidad de instrumentos disponibles por clase."),
                    color_discrete_sequence=[PRIMARY],
                )
                fig.update_traces(marker_color=PRIMARY)
                fig.update_yaxes(title="Cantidad")
                fig.update_xaxes(title="")
                st.plotly_chart(polish(rounded_bars(fig), show_legend=False), use_container_width=True)
        with col_b:
            with st.container(border=True):
                group_count = selected_meta["Grupo"].value_counts().reset_index()
                group_count.columns = ["Grupo", "Cantidad"]
                group_count = group_count.sort_values("Cantidad", ascending=True)
                fig = px.bar(
                    group_count,
                    x="Cantidad",
                    y="Grupo",
                    orientation="h",
                    title=chart_title("Distribucion por grupo", "Diversificacion sectorial/geografica de la seleccion actual."),
                    color_discrete_sequence=[SECONDARY],
                )
                fig.update_traces(marker_color=SECONDARY)
                fig.update_xaxes(title="Cantidad")
                fig.update_yaxes(title="")
                st.plotly_chart(polish(rounded_bars(fig), show_legend=False), use_container_width=True)

        # FILA 3: guia del analista.
        st.markdown(
            """
            <div class="analyst-note">
                <b>Guia del analista.</b> Una cobertura amplia y diversificada reduce el riesgo de que la herramienta
                dependa de un solo sector, pais o clase de instrumento. Antes de optimizar, valida que la muestra
                incluya suficientes activos, benchmark y diversidad economica para que las correlaciones y la frontera
                eficiente tengan una interpretacion financiera razonable.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # FILA 4: tabla con buscador.
        with st.container(border=True):
            st.markdown("**Tabla de activos disponibles**")
            search = st.text_input("Buscar activo en la tabla", "", placeholder="Ticker, RIC, grupo o tipo...")
            table = data.metadata.copy()
            if search:
                mask = table.astype(str).apply(lambda col: col.str.contains(search, case=False, na=False)).any(axis=1)
                table = table.loc[mask]
            st.dataframe(table, use_container_width=True, hide_index=True, height=430)

    elif page == "Riesgo y Rentabilidad":
        st.subheader("Modulo 1: Riesgo y Rentabilidad")
        row = selected_indicators.loc[selected_indicators["Activo"] == focus_asset]
        row = row.iloc[0] if not row.empty else pd.Series(dtype=float)
        selected_only = selected_indicators[selected_indicators["Activo"] != benchmark].copy()
        best_sharpe = selected_only.loc[selected_only["Sharpe Ratio"].idxmax()] if not selected_only["Sharpe Ratio"].dropna().empty else pd.Series(dtype=float)
        worst_dd = selected_only.loc[selected_only["Max Drawdown"].idxmin()] if not selected_only["Max Drawdown"].dropna().empty else pd.Series(dtype=float)

        # FILA 1: cinco KPI cards ejecutivas.
        m = st.columns(5)
        with m[0]:
            kpi_card("📈", "Retorno geometrico", format_pct(row.get("Retorno anualizado", np.nan)), f"{focus_asset} | anualizado realizado")
        with m[1]:
            kpi_card("➗", "Retorno medio", format_pct(row.get("Media periodo", np.nan)), f"{focus_asset} | base por periodo")
        with m[2]:
            kpi_card("⚠️", "Volatilidad", format_pct(row.get("Volatilidad anualizada", np.nan)), f"{focus_asset} | desviacion anualizada")
        with m[3]:
            kpi_card("🏆", "Mejor Sharpe", format_num(best_sharpe.get("Sharpe Ratio", np.nan)), f"{best_sharpe.get('Activo', 'Sin dato')} | retorno por unidad de riesgo")
        with m[4]:
            kpi_card("📉", "Peor Drawdown", format_pct(worst_dd.get("Max Drawdown", np.nan)), f"{worst_dd.get('Activo', 'Sin dato')} | mayor caida historica")

        price_view = prices[selected_with_benchmark].dropna(how="all")
        ret_view = returns[selected_with_benchmark].dropna(how="all")
        base100 = price_view / price_view.dropna().iloc[0] * 100
        drawdowns = price_view.apply(lambda s: s / s.cummax() - 1)

        # FILA 2: series temporales.
        col_a, col_b = st.columns(2)
        with col_a:
            with st.container(border=True):
                fig = px.line(
                    base100,
                    title=chart_title("Evolucion base 100", "Precios normalizados para comparar crecimiento acumulado."),
                    labels={"value": "Indice base 100", "Date": "Fecha", "variable": "Activo"},
                    color_discrete_sequence=[PRIMARY, SECONDARY, NAVY],
                )
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(polish(fig), use_container_width=True)
        with col_b:
            with st.container(border=True):
                fig = px.line(
                    drawdowns,
                    title=chart_title("Drawdown historico", "Caidas porcentuales desde maximos acumulados."),
                    labels={"value": "Drawdown", "Date": "Fecha", "variable": "Activo"},
                    color_discrete_sequence=[RED, SECONDARY, NAVY],
                )
                fig.update_yaxes(tickformat=".1%")
                fig.update_layout(hovermode="x unified")
                st.plotly_chart(polish(fig), use_container_width=True)

        # FILA 3: distribucion y dispersion.
        col_c, col_d = st.columns(2)
        with col_c:
            with st.container(border=True):
                hist_asset = st.selectbox("Activo para histograma de retornos", selected_with_benchmark, index=selected_with_benchmark.index(focus_asset), key="hist_asset_m1")
                hist_data = ret_view[[hist_asset]].dropna().rename(columns={hist_asset: "Retorno"})
                fig = px.histogram(
                    hist_data,
                    x="Retorno",
                    nbins=50,
                    marginal="box",
                    title=chart_title("Histograma de retornos", f"Distribucion empirica de retornos por periodo para {hist_asset}."),
                    color_discrete_sequence=[SECONDARY],
                )
                fig.update_traces(marker_color=SECONDARY)
                fig.update_xaxes(tickformat=".1%", title="Retorno por periodo")
                fig.update_yaxes(title="Frecuencia")
                st.plotly_chart(polish(rounded_bars(fig), show_legend=False), use_container_width=True)
        with col_d:
            with st.container(border=True):
                fig = px.scatter(
                    selected_indicators,
                    x="Volatilidad anualizada",
                    y="Retorno anualizado",
                    color_discrete_sequence=[PRIMARY],
                    text="Activo",
                    hover_data=["Sharpe Ratio", "Sortino Ratio", "Beta", "Max Drawdown", "Omega Ratio"],
                    title=chart_title("Riesgo vs Retorno", "X: volatilidad anualizada | Y: retorno geometrico realizado."),
                )
                fig.update_traces(
                    mode="markers+text",
                    textposition="top center",
                    textfont=dict(size=10),
                    marker=dict(size=12, color=PRIMARY, line=dict(width=1, color="white")),
                )
                fig.update_xaxes(tickformat=".1%", title="Volatilidad anualizada")
                fig.update_yaxes(tickformat=".1%", title="Retorno geometrico realizado")
                st.plotly_chart(polish(fig, show_legend=False), use_container_width=True)

        with st.container(border=True):
            st.markdown("**Tabla dinamica de indicadores**")
            st.dataframe(
                pct_cols(
                    selected_indicators,
                    ["Retorno anualizado", "Media periodo", "Desviacion periodo", "Retorno medio anualizado", "Volatilidad anualizada", "Varianza anualizada", "Max Drawdown", "VaR 95% periodo", "CVaR 95% periodo"],
                ),
                use_container_width=True,
                hide_index=True,
                height=350,
            )

    elif page == "Optimizacion":
        st.subheader("Modulo 2: Optimizacion de Portafolio")
        if result2 is None:
            st.warning("Selecciona al menos dos activos con datos suficientes para optimizar.")
        else:
            summary = result2["summary"]
            recommended = summary.loc[summary["Portafolio"] == "Recomendado por aversion"].iloc[0]
            # FILA 0: encabezado dinamico de perfil.
            with st.container(border=True):
                p1, p2 = st.columns([0.78, 0.22])
                with p1:
                    st.markdown(
                        f"""
                        <div class="profile-box">
                            <h3 style="margin:0;color:{NAVY};">Perfil Seleccionado: {profile_name}</h3>
                            <p style="margin:8px 0 0 0;color:{MUTED};">{profile["description"]}</p>
                            <p style="margin:6px 0 0 0;color:{MUTED};"><b>Fundamento:</b> {profile["basis"]}</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with p2:
                    st.markdown(
                        f"""
                        <div class="gamma-pill">
                            <div style="color:{MUTED};font-size:12px;text-transform:uppercase;font-weight:700;">Aversión usada</div>
                            <b>{risk_aversion:.1f}</b>
                            <div style="color:{MUTED};font-size:12px;">Parametro Gamma</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            # FILA 1: KPI del portafolio recomendado.
            o1, o2, o3, o4, o5 = st.columns(5)
            with o1:
                kpi_card("📈", "Retorno recomendado", format_pct(recommended["Retorno anualizado"]), "Retorno esperado anual del portafolio recomendado.")
            with o2:
                kpi_card("⚠️", "Volatilidad", format_pct(recommended["Volatilidad anualizada"]), "Riesgo anualizado del portafolio recomendado.")
            with o3:
                kpi_card("⚖️", "Sharpe", format_num(recommended["Sharpe Ratio"]), "Compensacion por unidad de volatilidad.")
            with o4:
                kpi_card("🧩", "Indice HHI", format_num(recommended["Concentracion HHI"]), "Concentracion de pesos; menor sugiere mas diversificacion.")
            with o5:
                kpi_card("🎯", "Utilidad neta", format_num(recommended["Utilidad media-varianza"]), "U = E(R) - 0.5 x Gamma x Varianza.")

            # FILA 2: frontera eficiente a ancho completo.
            with st.container(border=True):
                sim, front, cml = result2["simulated"], result2["frontier"], result2["cml"]
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=sim["Volatilidad anualizada"], y=sim["Retorno anualizado"], mode="markers", marker=dict(size=4, color=sim["Sharpe Ratio"], colorscale="Viridis", showscale=True, colorbar=dict(title="Sharpe")), name="Portafolios simulados", opacity=0.30))
                fig.add_trace(go.Scatter(x=front["Volatilidad anualizada"], y=front["Retorno anualizado"], mode="lines", name="Frontera eficiente", line=dict(color="#b42318", width=4)))
                fig.add_trace(go.Scatter(x=cml["Volatilidad anualizada"], y=cml["Retorno CML"], mode="lines", name="Capital Market Line", line=dict(color="#2563eb", dash="dash", width=3)))
                fig.add_trace(go.Scatter(x=[0], y=[result2["rf_annual"]], mode="markers+text", text=["Rf"], textposition="bottom right", marker=dict(size=11, color="#475569"), name="Tasa libre de riesgo"))
                for _, item in summary.iterrows():
                    fig.add_trace(go.Scatter(x=[item["Volatilidad anualizada"]], y=[item["Retorno anualizado"]], mode="markers+text", text=[item["Portafolio"]], textposition="top center", marker=dict(size=15, symbol="star", line=dict(width=1, color="white")), name=item["Portafolio"]))
                fig.update_layout(title=chart_title("La Frontera Eficiente", "Simulaciones, frontera optimizada, CML y portafolios clave."), xaxis_title="Volatilidad anualizada", yaxis_title="Retorno anualizado", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                st.plotly_chart(polish(fig), use_container_width=True)

            # FILA 3: desglose de asignacion.
            col_c, col_d = st.columns(2)
            with col_c:
                with st.container(border=True):
                    portfolio_options = {
                        "Portafolio recomendado por aversion": "Peso recomendado aversion",
                        "Portafolio tangente max Sharpe": "Peso tangente max Sharpe",
                        "Portafolio minima varianza": "Peso minima varianza",
                    }
                    selected_portfolio_label = st.selectbox("Portafolio para desglose", list(portfolio_options.keys()))
                    selected_weight_col = portfolio_options[selected_portfolio_label]
                    donut_data = result2["weights"][["Activo", selected_weight_col]].copy()
                    donut_data = donut_data[donut_data[selected_weight_col] > 0.0001]
                    fig = px.pie(
                        donut_data,
                        names="Activo",
                        values=selected_weight_col,
                        hole=0.58,
                        title=chart_title("Composicion porcentual", f"Pesos del {selected_portfolio_label.lower()}."),
                        color_discrete_sequence=[PRIMARY, SECONDARY, NAVY, "#38bdf8", "#14b8a6", "#64748b"],
                    )
                    fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="white", width=2)))
                    st.plotly_chart(polish(fig), use_container_width=True)
            with col_d:
                with st.container(border=True):
                    st.markdown("**Tabla completa de pesos**")
                    weights_table = result2["weights"][["Activo", selected_weight_col]].copy()
                    weights_table = weights_table.merge(data.metadata[["Ticker", "Tipo", "Grupo"]], left_on="Activo", right_on="Ticker", how="left")
                    weights_table = weights_table.rename(columns={selected_weight_col: "Peso Exacto"})
                    weights_table = weights_table[["Activo", "Tipo", "Grupo", "Peso Exacto"]].sort_values("Peso Exacto", ascending=False)
                    st.dataframe(pct_cols(weights_table, ["Peso Exacto"]), use_container_width=True, hide_index=True, height=420)

            with st.container(border=True):
                st.markdown("**Resumen tecnico de portafolios optimos**")
                st.dataframe(pct_cols(summary, ["Retorno anualizado", "Volatilidad anualizada", "Varianza anualizada"]), use_container_width=True, hide_index=True, height=190)

    elif page == "CAPM":
        st.subheader("Modulo 3: CAPM y Security Market Line")
        if result3 is None or result3["capm_table"].empty:
            st.warning("Selecciona al menos un activo distinto del benchmark para calcular CAPM.")
        else:
            capm_table = result3["capm_table"]
            capm_params = result3["params"].iloc[0]
            selected_capm_asset = st.selectbox("Activo foco para CAPM", capm_table["Activo"].tolist(), key="capm_focus")
            selected_row = capm_table.loc[capm_table["Activo"] == selected_capm_asset].iloc[0]
            selected_reg = result3["regressions"][selected_capm_asset]

            # FILA 1: KPI superiores del modelo.
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                kpi_card("🛡️", "Tasa Libre de Riesgo", format_pct(capm_params["RF anual"]), "RF anual usada para CAPM y retornos excedentes.")
            with c2:
                kpi_card("📊", "Retorno Benchmark", format_pct(capm_params["Retorno anual benchmark"]), f"Retorno medio anualizado de {benchmark}.")
            with c3:
                kpi_card("💎", "Prima de mercado", format_pct(capm_params["Prima de mercado anual"]), "Rm - Rf exigido por el mercado.")
            with c4:
                kpi_card("🎯", "Activos analizados", f"{int(capm_params['Activos analizados'])}", "Cantidad de activos bajo CAPM.")

            # FILA 2: SML + interpretacion automatica.
            row_model_left, row_model_right = st.columns([0.70, 0.30])
            valid_capm = capm_table.dropna(subset=["Beta OLS", "Retorno esperado CAPM", "Retorno historico anualizado"]).copy()
            with row_model_left:
                with st.container(border=True):
                    if not valid_capm.empty:
                        beta_min = min(0.0, float(valid_capm["Beta OLS"].min()) - 0.20)
                        beta_max = float(valid_capm["Beta OLS"].max()) + 0.20
                        beta_axis = np.linspace(beta_min, beta_max if beta_max > beta_min else beta_min + 1.0, 100)
                        rf_ann = capm_params["RF anual"]
                        premium = capm_params["Prima de mercado anual"]
                        sml_y = rf_ann + premium * beta_axis
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=beta_axis, y=sml_y, mode="lines", name="SML / CAPM", line=dict(color=PRIMARY, width=3)))
                        for position, color in [("Por encima de la SML", GREEN), ("Cerca de la SML", SECONDARY), ("Por debajo de la SML", RED), ("Sin datos suficientes", MUTED)]:
                            subset = valid_capm[valid_capm["Posicion respecto a SML"] == position]
                            if subset.empty:
                                continue
                            fig.add_trace(go.Scatter(x=subset["Beta OLS"], y=subset["Retorno historico anualizado"], mode="markers+text", text=subset["Activo"], textposition="top center", marker=dict(size=12, color=color, line=dict(width=1, color="white")), name=position))
                        fig.update_layout(title=chart_title("Security Market Line (SML)", "Recta teorica CAPM y posicion relativa de cada activo."), xaxis_title="Beta historica", yaxis_title="Retorno anual", yaxis_tickformat=".1%")
                        st.plotly_chart(polish(fig), use_container_width=True)
                    else:
                        st.info("No hay suficientes observaciones validas para graficar la SML.")
            with row_model_right:
                beta_text = interpret_beta(selected_row["Beta OLS"])
                r2_text = interpret_r2(selected_row["R2"])
                alpha_text = interpret_jensen_alpha(selected_row["Alpha Jensen anual"])
                significance = "R2 alto: la relacion con el mercado es estadisticamente mas informativa." if not pd.isna(selected_row["R2"]) and selected_row["R2"] >= 0.60 else "R2 bajo/medio: interpretar beta y alfa junto con otros indicadores de riesgo."
                st.markdown(
                    f"""
                    <div class="interpretation-box">
                        <h4 style="margin-top:0;">Interpretacion automatica</h4>
                        <p><b>Activo foco:</b> {selected_capm_asset}</p>
                        <p><b>Beta:</b> {beta_text}</p>
                        <p><b>R2:</b> {r2_text}</p>
                        <p><b>Alfa de Jensen:</b> {alpha_text}</p>
                        <p><b>Significancia practica:</b> {significance}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            # FILA 3: regresion individual y KPIs econometricos.
            with st.container(border=True):
                st.markdown("**Analisis de regresion individual**")
                r1, r2, r3, r4, r5 = st.columns(5)
                with r1:
                    kpi_card("β", "Beta OLS", format_num(selected_row["Beta OLS"]), "Sensibilidad historica frente al benchmark.")
                with r2:
                    kpi_card("R²", "Bondad de ajuste", format_num(selected_row["R2"]), "Proporcion explicada por el mercado.")
                with r3:
                    kpi_card("📌", "Retorno CAPM", format_pct(selected_row["Retorno esperado CAPM"]), "Retorno exigido segun beta.")
                with r4:
                    kpi_card("📈", "Retorno observado", format_pct(selected_row["Retorno historico anualizado"]), "Retorno historico anualizado.")
                with r5:
                    kpi_card("α", "Alfa Jensen", format_pct(selected_row["Alpha Jensen anual"]), "Exceso frente al retorno CAPM.")

                reg_data = selected_reg["regression_data"].copy()
                if len(reg_data) >= 10 and not pd.isna(selected_reg["Beta OLS"]):
                    reg_data["Exceso benchmark"] = reg_data["market"] - reg_data["rf"]
                    reg_data["Exceso activo"] = reg_data["asset"] - reg_data["rf"]
                    line_x = np.linspace(reg_data["Exceso benchmark"].min(), reg_data["Exceso benchmark"].max(), 100)
                    line_y = selected_reg["Alpha por periodo"] + selected_reg["Beta OLS"] * line_x
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=reg_data["Exceso benchmark"], y=reg_data["Exceso activo"], mode="markers", name="Observaciones", marker=dict(size=5, color=PRIMARY), opacity=0.45))
                    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines", name="Recta OLS", line=dict(color=RED, width=3)))
                    fig.update_layout(title=chart_title(f"Regresion OLS: {selected_capm_asset} vs {benchmark}", "Excesos de retorno del activo contra excesos del mercado."), xaxis_title="Rm - Rf", yaxis_title="Ri - Rf", xaxis_tickformat=".1%", yaxis_tickformat=".1%")
                    st.plotly_chart(polish(fig), use_container_width=True)
                else:
                    st.info("Este activo no tiene suficientes observaciones para mostrar la regresion OLS.")

            with st.container(border=True):
                st.markdown("**Tabla comparativa CAPM vs historico**")
                st.dataframe(
                    pct_cols(capm_table, ["Retorno esperado CAPM", "Retorno historico anualizado", "Alpha Jensen anual", "Volatilidad anualizada", "Max Drawdown", "VaR 95%", "CVaR 95%"]),
                    use_container_width=True,
                    hide_index=True,
                    height=300,
                )

    with st.sidebar:
        st.divider()
        st.download_button(
            "Exportar resultados en Excel",
            data=make_excel_download(result1, result2, result3),
            file_name="PortfolioLab_resultados.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

    st.caption("PortfolioLab listo para ejecutarse localmente o publicarse en Streamlit Cloud.")


if __name__ == "__main__":
    run_app()
