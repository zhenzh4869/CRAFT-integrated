# Select the virtual environment and use the hourlevel venv, Python 3.9.0.

from __future__ import annotations

# %% [Notebook cell 1]
import numpy as np
from pathlib import Path
import pandas as pd
from typing import Dict, Tuple, List, Any, Optional

# %% [Notebook cell 2]
"""
Data loading utilities.
"""

import pandas as pd
from typing import Dict, Tuple, List, Any, Optional


# -----------------------------
# Helpers
# -----------------------------
SHEETS_SKIP_REMARK = {
    "PRICE_E", "PRICE_Q", "CF", "DEMAND", "PROC_COEF",
    "COST_CAP", "COST_FOM", "VOM_OTHER", "INIT_COHORT",
    "WS_POT", "LOAD_MW"
}


def _read_sheet(xls: pd.ExcelFile, sheet_name: str) -> pd.DataFrame:
    """
    Read a sheet with the user's convention:
      Row 1: header
      Row 2: remark (units/notes)
      Row 3+: data
    For selected sheets, skip only the remark row (row index 1).
    """
    if sheet_name in SHEETS_SKIP_REMARK:
        return pd.read_excel(xls, sheet_name, skiprows=[1])
    return pd.read_excel(xls, sheet_name)


def _require_cols(df: pd.DataFrame, cols: List[str], sheet: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Sheet '{sheet}' missing columns: {missing}")


def _read_kv_sheet(df: pd.DataFrame, key_col="key", val_col="value") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for _, r in df.iterrows():
        if pd.isna(r[key_col]):
            continue
        k = str(r[key_col]).strip()
        out[k] = r[val_col]
    return out


def _filter_province(df: pd.DataFrame, province: Optional[str], sheet: str, required: bool) -> pd.DataFrame:
    """
    Province filter:
      - If province is None: return df
      - If province is not None:
          - If 'province' exists: filter and require non-empty
          - If 'province' does not exist:
              - if required=True: raise
              - else: keep df (global parameter)
    """
    if province is None:
        return df

    if "province" not in df.columns:
        if required:
            raise ValueError(f"Sheet '{sheet}' requires column 'province' when province='{province}' is requested.")
        return df

    out = df[df["province"] == province].copy()
    if out.empty:
        raise ValueError(f"Sheet '{sheet}': no rows found for province='{province}'.")
    return out


def _to_numeric_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Coerce given columns to numeric; non-convertible becomes NaN."""
    for c in cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def _strip_str_col(df: pd.DataFrame, col: str) -> pd.DataFrame:
    df[col] = df[col].astype(str).str.strip()
    return df


# -----------------------------
# Main loader
# -----------------------------
def load_inputs_from_excel(filepath: str, province: Optional[str] = None) -> Dict[str, Any]:
    """
    Load all parameters for ONE province (or no province filtering if province is None).
    Returns a dict suitable to feed build_chem_power_h2_model(...).

    Key conventions enforced:
      - PRICE_E/PRICE_Q/CF/DEMAND/PROC_COEF/COST_CAP/COST_FOM/VOM_OTHER/INIT_COHORT:
          row 1 header, row 2 remark, row 3+ data (skiprows=[1])
      - Avoid itertuples/getattr completely (safe for column names like 'MeOH-CH')
      - Heat price pi_Q is YEARLY only: pi_Q[y]
      - CF is hourly and read from columns: province, r, y, m, d, s, cf
      - If CF sheet only contains year 2025, it will be automatically copied to all model years in Y
    """
    xls = pd.ExcelFile(filepath)

    # ---------- SET ----------
    df_set = _read_sheet(xls, "SET")
    _require_cols(df_set, ["key", "value"], "SET")
    kv = _read_kv_sheet(df_set)

    # y0 = int(kv["y0"])
    y0 = 2030
    y_end = int(kv["y_end"])
    # y_step = int(kv.get("y_step", 5))
    y_step = 10
    Delta_h = float(kv.get("Delta_h", 4.0))

    R = [x.strip() for x in str(kv.get("R", "W,PV")).split(",") if x.strip()]
    K = [x.strip() for x in str(kv.get("K", "NH3,MeOH-CH,MeOH-CO2")).split(",") if x.strip()]

    M = list(range(1, 13))
    S = list(range(1, int(24 / Delta_h) + 1))
    Y = list(range(y0, y_end + 1, y_step))
    I = ["W", "PV", "EL", "HS", "BAT", "HCCGT"] + K

    # ---------- hourly calendar helper for CF ----------
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
    }

    def _mdh_to_t(mm: int, dd: int, hh: int) -> int:
        if mm not in days_in_month:
            raise ValueError(f"CF month m={mm} is invalid.")
        if dd < 1 or dd > days_in_month[mm]:
            raise ValueError(f"CF day d={dd} is invalid for month m={mm}.")
        if hh < 1 or hh > 24:
            raise ValueError(f"CF hour s={hh} must be in 1..24.")
        day_of_year = sum(days_in_month[m0] for m0 in range(1, mm)) + dd
        return (day_of_year - 1) * 24 + hh

    # ---------- WEIGHT ----------
    df_w = _read_sheet(xls, "WEIGHT")
    _require_cols(df_w, ["m", "w_m"], "WEIGHT")
    df_w = _to_numeric_cols(df_w, ["m", "w_m"]).dropna(subset=["m", "w_m"]).copy()
    w_m = dict(zip(df_w["m"].astype(int), df_w["w_m"].astype(float)))

    # ---------- PRICE_E (province-required) ----------
    df_pe = _read_sheet(xls, "PRICE_E")
    df_pe = _filter_province(df_pe, province, "PRICE_E", required=True)
    _require_cols(df_pe, ["y", "m", "s", "buy", "sell"], "PRICE_E")
    df_pe = _to_numeric_cols(df_pe, ["y", "m", "s", "buy", "sell"]).dropna(
        subset=["y", "m", "s", "buy", "sell"]
    ).copy()

    pi_buy = {(int(y), int(m), int(s)): float(buy)
              for y, m, s, buy in zip(df_pe["y"], df_pe["m"], df_pe["s"], df_pe["buy"])}
    pi_sell = {(int(y), int(m), int(s)): float(sell)
               for y, m, s, sell in zip(df_pe["y"], df_pe["m"], df_pe["s"], df_pe["sell"])}

    # ---------- LOAD_MW (province-required; special handling for Inner Mongolia) ----------
    df_load = _read_sheet(xls, "LOAD_MW")
    _require_cols(df_load, ["province", "y", "m", "s", "load_mw"], "LOAD_MW")

    df_load = _to_numeric_cols(df_load, ["y", "m", "s", "load_mw"]).dropna(
        subset=["province", "y", "m", "s", "load_mw"]
    ).copy()
    df_load = _strip_str_col(df_load, "province")

    if province is None:
        raise ValueError("LOAD_MW is province-specific; please call load_inputs_from_excel(..., province='<name>').")

    if province == "Inner Mongolia":
        sub_e = df_load[df_load["province"] == "East_Inner_Mongolia"].copy()
        sub_w = df_load[df_load["province"] == "West_Inner_Mongolia"].copy()

        if sub_e.empty or sub_w.empty:
            raise ValueError(
                "LOAD_MW: province='Inner Mongolia' requires BOTH 'East_Inner_Mongolia' and 'West_Inner_Mongolia' rows."
            )

        sub = pd.concat([sub_e, sub_w], ignore_index=True)
        sub = sub.groupby(["y", "m", "s"], as_index=False)["load_mw"].sum().copy()
    else:
        sub = df_load[df_load["province"] == province].copy()
        if sub.empty:
            raise ValueError(f"Sheet 'LOAD_MW': no rows found for province='{province}'.")
        sub = sub[["y", "m", "s", "load_mw"]].copy()

    LOAD_MW = {(int(y), int(m), int(s)): float(v)
               for y, m, s, v in zip(sub["y"], sub["m"], sub["s"], sub["load_mw"])}

    # ---------- PRICE_Q (YEARLY heat price; province-required) ----------
    df_pq = _read_sheet(xls, "PRICE_Q")
    df_pq = _filter_province(df_pq, province, "PRICE_Q", required=True)
    _require_cols(df_pq, ["y", "price_Q"], "PRICE_Q")

    if any(c in df_pq.columns for c in ["m", "s"]):
        raise ValueError("Sheet 'PRICE_Q' must be YEARLY only: columns (province optional), y, price_Q. Do NOT include m/s.")

    df_pq = _to_numeric_cols(df_pq, ["y", "price_Q"]).dropna(subset=["y", "price_Q"]).copy()
    pi_Q = dict(zip(df_pq["y"].astype(int), df_pq["price_Q"].astype(float)))

    # ---------- CF (province-required; hourly m,d,s; allow only 2025 then copy to all years) ----------
    df_cf = _read_sheet(xls, "CF")
    df_cf = _filter_province(df_cf, province, "CF", required=True)
    _require_cols(df_cf, ["r", "y", "m", "d", "s", "cf"], "CF")

    df_cf = _strip_str_col(df_cf, "r")
    df_cf = _to_numeric_cols(df_cf, ["y", "m", "d", "s", "cf"]).dropna(
        subset=["r", "y", "m", "d", "s", "cf"]
    ).copy()

    cf_years_in_file = sorted(pd.unique(df_cf["y"].astype(int)).tolist())

    # Case A: only 2025 is provided -> replicate to all Y
    if cf_years_in_file == [2025]:
        df_cf_base = df_cf.copy()
        replicated = []
        for yy in Y:
            tmp = df_cf_base.copy()
            tmp["y"] = int(yy)
            replicated.append(tmp)
        df_cf = pd.concat(replicated, ignore_index=True)

    # Case B: some other single year only -> also replicate from that single year
    elif len(cf_years_in_file) == 1:
        base_year = cf_years_in_file[0]
        df_cf_base = df_cf.copy()
        replicated = []
        for yy in Y:
            tmp = df_cf_base.copy()
            tmp["y"] = int(yy)
            replicated.append(tmp)
        df_cf = pd.concat(replicated, ignore_index=True)

    # Case C: multiple years provided -> keep as is, but later validate full coverage
    # Build CF[(r,y,t)]
    cf_rows = []
    for _, row in df_cf.iterrows():
        r = str(row["r"]).strip()
        y = int(row["y"])
        mm = int(row["m"])
        dd = int(row["d"])
        ss = int(row["s"])
        cf = float(row["cf"])
        # Skip February 29.
        if mm == 2 and dd == 29:
            continue
        t = _mdh_to_t(mm, dd, ss)
        cf_rows.append((r, y, t, cf))

    seen_cf_keys = set()
    for r, y, t, _ in cf_rows:
        key = (r, y, t)
        if key in seen_cf_keys:
            raise ValueError(f"CF has duplicate row for key (r,y,t)=({r},{y},{t}).")
        seen_cf_keys.add(key)

    CF = {(r, y, t): cf for r, y, t, cf in cf_rows}

    # ---------- DEMAND (province-required) ----------
    df_dem = _read_sheet(xls, "DEMAND")
    df_dem = _filter_province(df_dem, province, "DEMAND", required=True)
    _require_cols(df_dem, ["y", "NH3", "MeOH-CH", "MeOH-CO2", "REFH2"], "DEMAND")

    df_dem = _to_numeric_cols(df_dem, ["y", "NH3", "MeOH-CH", "MeOH-CO2", "REFH2"]).dropna(
        subset=["y", "NH3", "MeOH-CH", "MeOH-CO2", "REFH2"]
    ).copy()

    y_dem = df_dem["y"].astype(int)
    D_NH3 = dict(zip(y_dem, df_dem["NH3"].astype(float)))
    D_MeOH_CH = dict(zip(y_dem, df_dem["MeOH-CH"].astype(float)))
    D_MeOH_CO2 = dict(zip(y_dem, df_dem["MeOH-CO2"].astype(float)))
    D_REFH2 = dict(zip(y_dem, df_dem["REFH2"].astype(float)))

    # ---------- PROC_COEF ----------
    df_pc = _read_sheet(xls, "PROC_COEF")
    df_pc = _filter_province(df_pc, province, "PROC_COEF", required=False)
    _require_cols(df_pc, ["k", "aE", "aQ", "aH2"], "PROC_COEF")

    df_pc = _strip_str_col(df_pc, "k")
    df_pc = _to_numeric_cols(df_pc, ["aE", "aQ", "aH2"]).dropna(subset=["k", "aE", "aQ", "aH2"]).copy()
    df_pc = df_pc[df_pc["k"].isin(K)].copy()

    aE = dict(zip(df_pc["k"], df_pc["aE"].astype(float)))
    aQ = dict(zip(df_pc["k"], df_pc["aQ"].astype(float)))
    aH2 = dict(zip(df_pc["k"], df_pc["aH2"].astype(float)))

    missing_k = [kk for kk in K if kk not in aE or kk not in aQ or kk not in aH2]
    if missing_k:
        raise ValueError(f"PROC_COEF missing numeric coefficients for k(s): {missing_k}")

    # ---------- VOM_OTHER ----------
    df_vom = _read_sheet(xls, "VOM_OTHER")
    df_vom = _filter_province(df_vom, province, "VOM_OTHER", required=False)
    _require_cols(df_vom, ["k", "y", "c_vom_other"], "VOM_OTHER")

    df_vom = _strip_str_col(df_vom, "k")
    df_vom = _to_numeric_cols(df_vom, ["y", "c_vom_other"]).dropna(subset=["k", "y", "c_vom_other"]).copy()
    df_vom = df_vom[df_vom["k"].isin(K)].copy()

    c_vom_other = {(str(k), int(y)): float(v)
                   for k, y, v in zip(df_vom["k"], df_vom["y"], df_vom["c_vom_other"])}

    # ---------- TECH_PARAM ----------
    df_tp = _read_sheet(xls, "TECH_PARAM")
    _require_cols(df_tp, ["name", "key1", "value"], "TECH_PARAM")
    df_tp["key1"] = df_tp["key1"].fillna("").astype(str).str.strip()

    def _get_scalar(name: str) -> float:
        sub = df_tp[(df_tp["name"] == name) & (df_tp["key1"] == "")]
        if sub.empty:
            raise ValueError(f"Missing scalar parameter '{name}' in TECH_PARAM.")
        return float(pd.to_numeric(sub["value"].iloc[0], errors="raise"))

    eta_EL = _get_scalar("eta_EL")
    eta_ch_HS = _get_scalar("eta_ch_HS")
    eta_dis_HS = _get_scalar("eta_dis_HS")
    gamma_ch_HS = _get_scalar("gamma_ch_HS")
    gamma_dis_HS = _get_scalar("gamma_dis_HS")
    eta_c_BAT = _get_scalar("eta_c_BAT")
    eta_d_BAT = _get_scalar("eta_d_BAT")
    RU_EL = _get_scalar("RU_EL")
    RD_EL = _get_scalar("RD_EL")
    eta_HCCGT = _get_scalar("eta_HCCGT")
    RU_HCCGT = _get_scalar("RU_HCCGT")
    RD_HCCGT = _get_scalar("RD_HCCGT")

    RU_k: Dict[str, float] = {}
    RD_k: Dict[str, float] = {}
    df_ramp = df_tp[df_tp["name"].isin(["RU_k", "RD_k"])][["name", "key1", "value"]].copy()
    df_ramp["value"] = pd.to_numeric(df_ramp["value"], errors="coerce")
    df_ramp = df_ramp.dropna(subset=["value"]).copy()
    for name, key1, val in zip(df_ramp["name"], df_ramp["key1"], df_ramp["value"]):
        kname = str(key1).strip()
        if name == "RU_k":
            RU_k[kname] = float(val)
        else:
            RD_k[kname] = float(val)

    # ---------- COST_CAP ----------
    df_cc = _read_sheet(xls, "COST_CAP")
    df_cc = _filter_province(df_cc, province, "COST_CAP", required=True)
    _require_cols(df_cc, ["i", "y", "c_cap"], "COST_CAP")

    df_cc = _strip_str_col(df_cc, "i")
    df_cc = _to_numeric_cols(df_cc, ["y", "c_cap"]).dropna(subset=["i", "y", "c_cap"]).copy()
    c_cap = {(str(i), int(y)): float(v) for i, y, v in zip(df_cc["i"], df_cc["y"], df_cc["c_cap"])}

    # ---------- COST_FOM ----------
    df_cfom = _read_sheet(xls, "COST_FOM")
    df_cfom = _filter_province(df_cfom, province, "COST_FOM", required=True)
    _require_cols(df_cfom, ["i", "y", "c_fom"], "COST_FOM")

    df_cfom = _strip_str_col(df_cfom, "i")
    df_cfom = _to_numeric_cols(df_cfom, ["y", "c_fom"]).dropna(subset=["i", "y", "c_fom"]).copy()
    c_fom = {(str(i), int(y)): float(v) for i, y, v in zip(df_cfom["i"], df_cfom["y"], df_cfom["c_fom"])}

    # ---------- LIFE ----------
    df_life = _read_sheet(xls, "LIFE")
    _require_cols(df_life, ["i", "L"], "LIFE")
    df_life = _filter_province(df_life, province, "LIFE", required=False)
    df_life = _strip_str_col(df_life, "i")
    df_life["L"] = pd.to_numeric(df_life["L"], errors="coerce")
    df_life = df_life.dropna(subset=["i", "L"]).copy()
    L = {str(i): int(Li) for i, Li in zip(df_life["i"], df_life["L"])}

    # ---------- DISCOUNT ----------
    df_disc = _read_sheet(xls, "DISCOUNT")
    if set(["key", "value"]).issubset(df_disc.columns):
        kvd = _read_kv_sheet(df_disc)
        r_disc = float(kvd["r_disc"])
        delta_y = {y: (1.0 + r_disc) ** (-(y - y0)) for y in Y}
    elif set(["y", "delta"]).issubset(df_disc.columns):
        r_disc = None
        df_disc = _to_numeric_cols(df_disc, ["y", "delta"]).dropna(subset=["y", "delta"]).copy()
        delta_y = dict(zip(df_disc["y"].astype(int), df_disc["delta"].astype(float)))
    else:
        raise ValueError("Sheet 'DISCOUNT' must be key/value with r_disc OR columns (y, delta).")

    # ---------- INIT_COHORT ----------
    V0: List[int] = []
    A_init: Dict[Tuple[str, int], float] = {}
    if "INIT_COHORT" in xls.sheet_names:
        df_init = _read_sheet(xls, "INIT_COHORT")
        df_init = _filter_province(df_init, province, "INIT_COHORT", required=True)
        _require_cols(df_init, ["i", "tau", "A_init"], "INIT_COHORT")

        df_init = _strip_str_col(df_init, "i")
        df_init = _to_numeric_cols(df_init, ["tau", "A_init"]).dropna(subset=["i", "tau", "A_init"]).copy()

        V0 = sorted(pd.unique(df_init["tau"].astype(int)))
        A_init = {(str(i), int(tau)): float(v)
                  for i, tau, v in zip(df_init["i"], df_init["tau"], df_init["A_init"])}

    # -----------------------------
    # Validation
    # -----------------------------
    missing_y_pq = [y for y in Y if y not in pi_Q]
    if missing_y_pq:
        raise ValueError(f"PRICE_Q missing yearly heat price for y={missing_y_pq}")

    for y in Y:
        for mm in M:
            for s in S:
                if (y, mm, s) not in pi_buy or (y, mm, s) not in pi_sell:
                    raise ValueError(f"PRICE_E missing (y,m,s)=({y},{mm},{s})")

    # Validate full CF hourly coverage after replication
    T = list(range(1, 8760 + 1))
    for y in Y:
        for r in R:
            for t in T:
                if (str(r), y, t) not in CF:
                    raise ValueError(f"CF missing (r,y,t)=({r},{y},{t})")

    for i in I:
        if i not in L:
            raise ValueError(f"LIFE missing lifetime for i='{i}'")
        for y in Y:
            if (i, y) not in c_cap:
                raise ValueError(f"COST_CAP missing (i,y)=({i},{y})")
            if (i, y) not in c_fom:
                raise ValueError(f"COST_FOM missing (i,y)=({i},{y})")

    for y in Y:
        if y not in D_NH3 or y not in D_MeOH_CH or y not in D_MeOH_CO2 or y not in D_REFH2:
            raise ValueError(f"DEMAND missing some product demand for y={y}")

    # ---------- WS_POT + WS_POT_ratio ----------
    df_wspot = _read_sheet(xls, "WS_POT")
    df_wspot_ratio = _read_sheet(xls, "WS_POT_ratio")

    _require_cols(df_wspot, ["PROVINCE", "Solar", "Wind"], "WS_POT")
    _require_cols(df_wspot_ratio, ["province", "AMM", "METH", "H2"], "WS_POT_ratio")

    df_wspot = _strip_str_col(df_wspot, "PROVINCE")
    df_wspot_ratio = _strip_str_col(df_wspot_ratio, "province")

    if province is None:
        raise ValueError("WS_POT is province-specific; please call load_inputs_from_excel(..., province='<name>').")

    sub_pot = df_wspot[df_wspot["PROVINCE"] == province].copy()
    if sub_pot.empty:
        raise ValueError(f"Sheet 'WS_POT': no rows found for PROVINCE='{province}'.")

    sub_pot = _to_numeric_cols(sub_pot, ["Solar", "Wind"]).dropna(subset=["Solar", "Wind"]).copy()
    if sub_pot.empty:
        raise ValueError(f"Sheet 'WS_POT': Solar/Wind is NaN for PROVINCE='{province}'.")

    ws_solar_mw = float(sub_pot.iloc[0]["Solar"]) * 1000.0
    ws_wind_mw = float(sub_pot.iloc[0]["Wind"]) * 1000.0

    sub_ratio = df_wspot_ratio[df_wspot_ratio["province"] == province].copy()
    if sub_ratio.empty:
        raise ValueError(f"Sheet 'WS_POT_ratio': no rows found for province='{province}'.")

    sub_ratio = _to_numeric_cols(sub_ratio, ["AMM", "METH", "H2"]).dropna(subset=["AMM", "METH", "H2"]).copy()
    if sub_ratio.empty:
        raise ValueError(f"Sheet 'WS_POT_ratio': AMM/METH/H2 is NaN for province='{province}'.")

    r_amm = float(sub_ratio.iloc[0]["AMM"])
    r_meth = float(sub_ratio.iloc[0]["METH"])
    r_h2 = float(sub_ratio.iloc[0]["H2"])

    r_sum = r_amm + r_meth + r_h2
    if r_sum == 0:
        r_amm = 0.33333333
        r_meth = 0.33333333
        r_h2 = 0.33333333
    elif not (abs(r_sum - 1.0) <= 1e-6):
        raise ValueError(
            f"Sheet 'WS_POT_ratio': ratios for province='{province}' do not sum to 1. "
            f"Got AMM={r_amm}, METH={r_meth}, H2={r_h2}, sum={r_sum}."
        )

    WS_POT = {
        "PV": {
            "AMM": ws_solar_mw * r_amm,
            "METH": ws_solar_mw * r_meth,
            "H2": ws_solar_mw * r_h2,
        },
        "W": {
            "AMM": ws_wind_mw * r_amm,
            "METH": ws_wind_mw * r_meth,
            "H2": ws_wind_mw * r_h2,
        },
    }

    return {
        "Y": Y, "M": M, "S": S, "R": R, "K": K, "I": I,
        "Delta_h": Delta_h,
        "w_m": w_m,
        "CF": CF,  # CF[(r,y,t)] after 2025 replication if needed
        "pi_buy": pi_buy, "pi_sell": pi_sell, "pi_Q": pi_Q,
        "D_NH3": D_NH3, "D_MeOH_CH": D_MeOH_CH, "D_MeOH_CO2": D_MeOH_CO2, "D_REFH2": D_REFH2,
        "aE": aE, "aQ": aQ, "aH2": aH2,
        "c_vom_other": c_vom_other,
        "eta_EL": eta_EL, "eta_HCCGT": eta_HCCGT,
        "eta_ch_HS": eta_ch_HS, "eta_dis_HS": eta_dis_HS,
        "gamma_ch_HS": gamma_ch_HS, "gamma_dis_HS": gamma_dis_HS,
        "eta_c_BAT": eta_c_BAT, "eta_d_BAT": eta_d_BAT,
        "RU_k": RU_k, "RD_k": RD_k, "RU_EL": RU_EL, "RD_EL": RD_EL,
        "RU_HCCGT": RU_HCCGT, "RD_HCCGT": RD_HCCGT,
        "c_cap": c_cap, "c_fom": c_fom,
        "L": L,
        "r_disc": r_disc,
        "delta_y": delta_y,
        "V0": V0,
        "A_init": A_init,
        "WS_POT": WS_POT,
        "LOAD_MW": LOAD_MW,
    }

# %% [Notebook cell 3]
"""
Model-related code: v3 updated to 8,760 hours.

Key changes in this version:
1) Electricity purchase/sale variables e_buy/e_sell remain in MW; because the internal model time step is 1 h, hourly MW can be billed directly as MWh.
2) pi_buy/pi_sell are in CNY/MWh. The 150 CNY/MWh transmission and distribution fee is already included in the electricity purchase/sale prices and is not added again in the objective function.
3) Electricity sales incur an additional capacity-price cost: CAPACITY_PRICE_SELL = 200 CNY/MWh.
4) T_D_FEE_CNY_PER_MWH and CAPACITY_PRICE_SELL_CNY_PER_MWH are added to vars_out for Excel post-processing and cost decomposition.
5) Retain the previous demand-constraint changes:
   - annual_demand_only=True: annual conservation for ammonia/methanol, equal monthly total production across the three months within each season; refinery hydrogen use is exactly equal hour by hour within each season.
   - annual_demand_only=False: equal monthly total production for ammonia/methanol; refinery hydrogen use is exactly equal hour by hour throughout the year.
6) Retain the minimum reactor-load constraint: active NH3, MeOH-CH, and MeOH-CO2 pathways must operate at >= 40% of rated load.
"""

import gurobipy as gp
from gurobipy import GRB


def build_chem_power_h2_model(
    Y, M, S, R, K, I,
    Delta_h,
    w_m,
    CF, WS_POT, WS_POT_upper, LOAD_MW,
    pi_buy, pi_sell, pi_Q,
    D_NH3, D_MeOH_CH, D_MeOH_CO2, D_REFH2,
    aE, aQ, aH2,
    c_vom_other,
    eta_EL, eta_HCCGT, eta_ch_HS, eta_dis_HS, gamma_ch_HS, gamma_dis_HS,
    eta_c_BAT, eta_d_BAT,
    RU_k, RD_k, RU_EL, RD_EL, RU_HCCGT, RD_HCCGT,
    c_cap, c_fom,
    L,
    r_disc, delta_y,
    max_load_interact, buy_equal_sale, HCCGT_open,
    V0=None, A_init=None,
    model_name="chem_power_h2_8760",
    c="AMM",                  # "AMM" / "METH" / "H2"
    annual_demand_only: bool = True,
    coup: bool = True,
):
    """
    8760-hour version.

    Key assumptions:
    1) CF is hourly:            CF[(r, y, t)] for t = 1..8760
    2) Other time-series data remain monthly-typical-day:
         pi_buy[(y,m,s)], pi_sell[(y,m,s)], LOAD_MW[(y,m,s)]
       and are mapped to each hour t by:
         m = month_of_t[t], s = seg6_of_t[t]
    3) Heat price is yearly:    pi_Q[y]
       (backward-compatible fallback to pi_Q[(y,m,s)] if provided that way)
    4) Internal model step is 1 hour
    5) Continuity is broken ONLY at seasonal boundaries:
         Mar-1 00:00, Jun-1 00:00, Sep-1 00:00, Dec-1 00:00
       for battery SOC, electrolyzer load, H2 storage, HCCGT load, reactor load.
    6) Ramping parameters RU/RD are defined on the original segment length Delta_h,
       so hourly ramp coefficients are RU/Delta_h and RD/Delta_h.
    """

    # -----------------------------
    # basic setup
    # -----------------------------
    V0 = V0 or []
    A_init = A_init or {}

    c = str(c).upper().strip()
    if c not in {"AMM", "METH", "H2"}:
        raise ValueError("Argument c must be one of {'AMM','METH','H2'}.")

    # active chemical products
    if c == "AMM":
        K_active = ["NH3"]
        demand_prod = {"NH3": D_NH3}
        refinery_active = False
    elif c == "METH":
        K_active = ["MeOH-CH", "MeOH-CO2"]
        demand_prod = {"MeOH-CH": D_MeOH_CH, "MeOH-CO2": D_MeOH_CO2}
        refinery_active = False
    else:  # c == "H2"
        K_active = []
        demand_prod = {}
        refinery_active = True

    K_inactive = [k for k in K if k not in K_active]

    # Original segment duration in the input (e.g., 4 h).
    Delta_h_seg = float(Delta_h)

    # The internal time step of the 8,760-hour model is fixed at 1 h.
    Delta_h = 1.0

    # Grid fee parameters.
    # Note: the transmission and distribution fee is already included in pi_buy / pi_sell; it is used here only for post-processing cost decomposition and is not added separately to the objective function.
    T_D_FEE_CNY_PER_MWH = 150.0
    # Capacity-price cost: an additional 200 CNY is paid for each 1 MWh of electricity sold.
    CAPACITY_PRICE_SELL_CNY_PER_MWH = 200.0

    # non-leap-year calendar
    days_in_month = {
        1: 31, 2: 28, 3: 31, 4: 30, 5: 31, 6: 30,
        7: 31, 8: 31, 9: 30, 10: 31, 11: 30, 12: 31,
    }

    # 8760 hour index
    T = list(range(1, 8760 + 1))
    D = list(range(1, 365 + 1))

    # -----------------------------
    # calendar mappings
    # -----------------------------
    month_of_t = {}
    day_of_t = {}
    hour_of_day = {}
    seg6_of_t = {}
    day_of_month_of_t = {}

    hours_in_month = {m: [] for m in M}
    days_in_month_idx = {m: [] for m in M}
    hours_in_day = {d: [] for d in D}
    month_of_day = {}
    day_of_month = {}

    t_counter = 1
    d_counter = 1
    seg_len_hours = int(Delta_h_seg)

    for mm in range(1, 13):
        for dom in range(1, days_in_month[mm] + 1):
            month_of_day[d_counter] = mm
            day_of_month[d_counter] = dom
            days_in_month_idx[mm].append(d_counter)

            for hh in range(1, 25):
                month_of_t[t_counter] = mm
                day_of_t[t_counter] = d_counter
                hour_of_day[t_counter] = hh
                seg6_of_t[t_counter] = (hh - 1) // seg_len_hours + 1
                day_of_month_of_t[t_counter] = dom

                hours_in_month[mm].append(t_counter)
                hours_in_day[d_counter].append(t_counter)
                t_counter += 1

            d_counter += 1

    # -----------------------------
    # seasonal-boundary breakpoints
    # only active when annual_demand_only == True
    # -----------------------------
    if annual_demand_only:
        seasonal_break_ts = set()
        for tt in T:
            mm = month_of_t[tt]
            dom = day_of_month_of_t[tt]
            hh = hour_of_day[tt]
            if hh == 1 and (
                (mm == 3 and dom == 1) or
                (mm == 6 and dom == 1) or
                (mm == 9 and dom == 1) or
                (mm == 12 and dom == 1)
            ):
                seasonal_break_ts.add(tt)
    else:
        seasonal_break_ts = set()

    # -----------------------------
    # helpers
    # -----------------------------
    def alive_taus(i, y):
        Li = L[i]
        return [tau for tau in Y if (y - Li) < tau <= y]

    def alive_init_taus(i, y):
        Li = L[i]
        return [tau for tau in V0 if (y - Li) < tau <= y]

    def get_month_seg(tt):
        mm = month_of_t[tt]
        ss = seg6_of_t[tt]
        return mm, ss

    def get_pi_buy(y, tt):
        mm, ss = get_month_seg(tt)
        return float(pi_buy[(y, mm, ss)])

    def get_pi_sell(y, tt):
        mm, ss = get_month_seg(tt)
        return float(pi_sell[(y, mm, ss)])

    def get_load_mw(y, tt):
        mm, ss = get_month_seg(tt)
        return float(LOAD_MW[(y, mm, ss)])

    def get_pi_Q(y, tt):
        if isinstance(pi_Q, dict):
            if y in pi_Q:
                return float(pi_Q[y])
            mm, ss = get_month_seg(tt)
            if (y, mm, ss) in pi_Q:
                return float(pi_Q[(y, mm, ss)])
        mm, ss = get_month_seg(tt)
        raise KeyError(
            f"Heat price missing for y={y}; also missing fallback key (y,m,s)=({y},{mm},{ss})."
        )

    # convert segment-ramping to hourly-ramping
    RU_EL_h = RU_EL / Delta_h_seg
    RD_EL_h = RD_EL / Delta_h_seg
    RU_HCCGT_h = RU_HCCGT / Delta_h_seg
    RD_HCCGT_h = RD_HCCGT / Delta_h_seg
    RU_k_h = {k: RU_k[k] / Delta_h_seg for k in K}
    RD_k_h = {k: RD_k[k] / Delta_h_seg for k in K}

    SEASONS = {
        "WIN": [12, 1, 2],
        "SPR": [3, 4, 5],
        "SUM": [6, 7, 8],
        "AUT": [9, 10, 11],
    }

    # -----------------------------
    # model
    # -----------------------------
    m = gp.Model(model_name)

    # -----------------------------
    # decision variables
    # -----------------------------
    # capacity
    A = m.addVars(I, Y, lb=0.0, name="A")
    Kcap = m.addVars(I, Y, lb=0.0, name="K")

    # hourly operation
    g = m.addVars(R, Y, T, lb=0.0, name="g")
    e_buy = m.addVars(Y, T, lb=0.0, name="e_buy")
    e_sell = m.addVars(Y, T, lb=0.0, name="e_sell")

    # battery
    p_c = m.addVars(Y, T, lb=0.0, name="p_c")
    p_d = m.addVars(Y, T, lb=0.0, name="p_d")
    SOC = m.addVars(Y, T, lb=0.0, name="SOC")

    # electrolyzer + H2
    p_EL = m.addVars(Y, T, lb=0.0, name="p_EL")
    h_EL = m.addVars(Y, T, lb=0.0, name="h_EL")

    h_ch = m.addVars(Y, T, lb=0.0, name="h_ch")
    h_dis = m.addVars(Y, T, lb=0.0, name="h_dis")
    S_H2 = m.addVars(Y, T, lb=0.0, name="S_H2")

    h_ref = m.addVars(Y, T, lb=0.0, name="h_ref")

    # HCCGT
    p_HCCGT = m.addVars(Y, T, lb=0.0, name="p_HCCGT")
    h_HCCGT = m.addVars(Y, T, lb=0.0, name="h_HCCGT")

    # reactors
    x = m.addVars(K, Y, T, lb=0.0, name="x")

    # derived loads
    L_E = m.addVars(Y, T, lb=0.0, name="L_E")
    H = m.addVars(Y, T, lb=0.0, name="H")

    # -----------------------------
    # separate optimization enforcement
    # -----------------------------
    for k0 in K_inactive:
        for y in Y:
            if k0 in I:
                m.addConstr(A[k0, y] == 0.0, name=f"forbid_A_inactive[{k0},{y}]")
            for tt in T:
                m.addConstr(x[k0, y, tt] == 0.0, name=f"zero_x_inactive[{k0},{y},{tt}]")

    if not refinery_active:
        for y in Y:
            for tt in T:
                m.addConstr(h_ref[y, tt] == 0.0, name=f"zero_href[{y},{tt}]")

    # -----------------------------
    # capacity mapping by survival window
    # -----------------------------
    for i in I:
        for y in Y:
            expr = gp.quicksum(A[i, tau] for tau in alive_taus(i, y))
            if V0:
                expr += gp.quicksum(A_init.get((i, tau), 0.0) for tau in alive_init_taus(i, y))
            m.addConstr(Kcap[i, y] == expr, name=f"cap_alive[{i},{y}]")

    # -----------------------------
    # VRE generation
    # -----------------------------
    for r in R:
        for y in Y:
            for tt in T:
                m.addConstr(
                    g[r, y, tt] <= CF[(r, y, tt)] * Kcap[r, y],
                    name=f"gen_ub[{r},{y},{tt}]"
                )
            m.addConstr(
                Kcap[r, y] <= WS_POT[r][c] * WS_POT_upper,
                name=f"WS_POT_{r}[{y}]"
            )

    # -----------------------------
    # battery
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(p_c[y, tt] <= Kcap["BAT", y], name=f"bat_pc_ub[{y},{tt}]")
            m.addConstr(p_d[y, tt] <= Kcap["BAT", y], name=f"bat_pd_ub[{y},{tt}]")
            m.addConstr(SOC[y, tt] <= 4.0 * Kcap["BAT", y], name=f"bat_soc_ub[{y},{tt}]")

        # SOC dynamics only within season blocks
        for tt in T[1:]:
            if tt in seasonal_break_ts:
                continue
            tt_prev = tt - 1
            m.addConstr(
                SOC[y, tt] == SOC[y, tt_prev]
                + eta_c_BAT * p_c[y, tt]
                - (1.0 / eta_d_BAT) * p_d[y, tt],
                name=f"bat_soc_dyn[{y},{tt}]"
            )

    # -----------------------------
    # electrolyzer
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(p_EL[y, tt] <= Kcap["EL", y], name=f"el_ub[{y},{tt}]")
            m.addConstr(h_EL[y, tt] == eta_EL * p_EL[y, tt], name=f"h_el[{y},{tt}]")

        # ramp only within season blocks
        for tt in T[1:]:
            if tt in seasonal_break_ts:
                continue
            tt_prev = tt - 1
            m.addConstr(
                p_EL[y, tt] - p_EL[y, tt_prev] <= RU_EL_h * Kcap["EL", y],
                name=f"el_rup[{y},{tt}]"
            )
            m.addConstr(
                p_EL[y, tt_prev] - p_EL[y, tt] <= RD_EL_h * Kcap["EL", y],
                name=f"el_rdn[{y},{tt}]"
            )

    # -----------------------------
    # HCCGT
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(p_HCCGT[y, tt] <= Kcap["HCCGT", y], name=f"hccgt_ub[{y},{tt}]")
            m.addConstr(
                h_HCCGT[y, tt] == p_HCCGT[y, tt] / eta_HCCGT,
                name=f"hccgt_fuel[{y},{tt}]"
            )

        # ramp only within season blocks
        for tt in T[1:]:
            if tt in seasonal_break_ts:
                continue
            tt_prev = tt - 1
            m.addConstr(
                p_HCCGT[y, tt] - p_HCCGT[y, tt_prev] <= RU_HCCGT_h * Kcap["HCCGT", y],
                name=f"hccgt_rup[{y},{tt}]"
            )
            m.addConstr(
                p_HCCGT[y, tt_prev] - p_HCCGT[y, tt] <= RD_HCCGT_h * Kcap["HCCGT", y],
                name=f"hccgt_rdn[{y},{tt}]"
            )

    # -----------------------------
    # hydrogen storage
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(S_H2[y, tt] <= Kcap["HS", y], name=f"hs_stock_ub[{y},{tt}]")
            m.addConstr(h_ch[y, tt] <= gamma_ch_HS * Kcap["HS", y], name=f"hs_ch_ub[{y},{tt}]")
            m.addConstr(h_dis[y, tt] <= gamma_dis_HS * Kcap["HS", y], name=f"hs_dis_ub[{y},{tt}]")
            

        # stock dynamics only within season blocks
        for tt in T[1:]:
            if tt in seasonal_break_ts:
                continue
            tt_prev = tt - 1
            m.addConstr(
                S_H2[y, tt] == S_H2[y, tt_prev]
                + eta_ch_HS * h_ch[y, tt]
                - (1.0 / eta_dis_HS) * h_dis[y, tt],
                name=f"hs_dyn[{y},{tt}]"
            )
        # Closure between hour 8,760 and the first hour must be enforced.
        m.addConstr(
            S_H2[y, T[-1]] == S_H2[y, T[0]],
            name=f"hs_annual_closure[{y}]"
            )

    

    # -----------------------------
    # minimum load requirement for synthesis reactors
    # -----------------------------
    MINLOAD_REACTORS = {"NH3", "MeOH-CH", "MeOH-CO2"}
    MINLOAD_FRAC = 0.40

    # Sort model years and define previous-year mapping
    Y_sorted = sorted(list(Y))
    prev_y_map = {
        y: (Y_sorted[idx - 1] if idx > 0 else None)
        for idx, y in enumerate(Y_sorted)
    }

    def _get_annual_demand(k, y):
        """
        Get annual demand for product k in year y.
        If k is not active in the current run, return 0.
        """
        if k not in demand_prod:
            return 0.0

        Ddict = demand_prod[k]

        if hasattr(Ddict, "get"):
            return float(Ddict.get(y, 0.0))

        return float(Ddict[y])

    # Demand-adjusted minimum-load coefficient:
    # theta[k,y] = 1 if demand is increasing or unchanged;
    # theta[k,y] = current demand / previous demand if demand is decreasing;
    # theta[k,y] = 0 if both current and previous demand are zero.
    minload_scale = {}

    for k in K:
        if (k in MINLOAD_REACTORS) and (k in K_active):
            for y in Y:
                y_prev = prev_y_map[y]
                D_curr = _get_annual_demand(k, y)

                if y_prev is None:
                    # First model year: no previous year for comparison
                    theta = 1.0 if D_curr > 0.0 else 0.0
                else:
                    D_prev = _get_annual_demand(k, y_prev)

                    if D_prev <= 0.0:
                        # If previous demand is zero:
                        # - current > 0 means demand is rising from zero, use 1
                        # - current = 0 means no production, use 0
                        theta = 1.0 if D_curr > 0.0 else 0.0
                    else:
                        if D_curr >= D_prev:
                            theta = 1.0
                        else:
                            theta = D_curr / D_prev

                minload_scale[(k, y)] = float(theta)

    # -----------------------------
    # reactor capacity and ramping
    # -----------------------------
    for k in K:
        for y in Y:
            for tt in T:
                # upper bound: reactor hourly load cannot exceed installed capacity
                m.addConstr(
                    x[k, y, tt] <= Kcap[k, y],
                    name=f"rx_ub[{k},{y},{tt}]"
                )

                # lower bound:
                # active synthesis reactors must operate at
                # >= 40% * demand-adjustment coefficient * installed capacity
                #
                # If annual demand falls to zero, theta becomes 0,
                # so the lower bound becomes x >= 0 and will not cause infeasibility.
                if (k in MINLOAD_REACTORS) and (k in K_active):
                    theta = minload_scale[(k, y)]
                    m.addConstr(
                        x[k, y, tt] >= MINLOAD_FRAC * theta * Kcap[k, y],
                        name=f"rx_minload_scaled[{k},{y},{tt}]"
                    )

            # ramp only within season blocks
            for tt in T[1:]:
                if tt in seasonal_break_ts:
                    continue
                tt_prev = tt - 1

                m.addConstr(
                    x[k, y, tt] - x[k, y, tt_prev] <= RU_k_h[k] * Kcap[k, y],
                    name=f"rx_rup[{k},{y},{tt}]"
                )

                m.addConstr(
                    x[k, y, tt_prev] - x[k, y, tt] <= RD_k_h[k] * Kcap[k, y],
                    name=f"rx_rdn[{k},{y},{tt}]"
                )

    # -----------------------------
    # derived electric load and heat demand
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(
                L_E[y, tt] == gp.quicksum(aE[k] * x[k, y, tt] for k in K),
                name=f"load_e_def[{y},{tt}]"
            )
            m.addConstr(
                H[y, tt] == gp.quicksum(aQ[k] * x[k, y, tt] for k in K),
                name=f"heat_def[{y},{tt}]"
            )

    # -----------------------------
    # hydrogen balance
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(
                h_EL[y, tt] + h_dis[y, tt]
                == h_ch[y, tt]
                + gp.quicksum(aH2[k] * x[k, y, tt] for k in K)
                + h_ref[y, tt]
                + h_HCCGT[y, tt],
                name=f"h2_bal[{y},{tt}]"
            )

    # -----------------------------
    # power balance
    # -----------------------------
    for y in Y:
        for tt in T:
            m.addConstr(
                gp.quicksum(g[r, y, tt] for r in R)
                + e_buy[y, tt]
                + p_d[y, tt]
                + p_HCCGT[y, tt]
                == p_EL[y, tt]
                + L_E[y, tt]
                + p_c[y, tt]
                + e_sell[y, tt],
                name=f"pwr_bal[{y},{tt}]"
            )

    # -----------------------------
    # demand satisfaction
    # -----------------------------
    for y in Y:
        if annual_demand_only:
            # =========================================================
            # Case 1: annual_demand_only = True
            #
            # Chemical products:
            #   1) annual total demand must be satisfied
            #   2) within each season, the three months have equal MONTHLY total output
            #   3) between seasons, monthly output can differ
            #
            # Refinery H2:
            #   1) annual total refinery H2 demand must be satisfied
            #   2) within each season, every hour has the same h_ref
            #   3) between seasons, h_ref can differ
            # =========================================================

            # chemical products: annual demand + in-season equal monthly total
            for kk, Ddict in demand_prod.items():
                # annual total
                m.addConstr(
                    gp.quicksum(x[kk, y, tt] for tt in T) == float(Ddict[y]),
                    name=f"dem_{kk}[{y}]"
                )

                # monthly total production
                month_total = {}
                for mm in M:
                    month_total[mm] = gp.quicksum(
                        x[kk, y, tt] for tt in hours_in_month[mm]
                    )

                # within each season, the three months have equal monthly total output
                for seas, months in SEASONS.items():
                    m1, m2, m3 = months
                    m.addConstr(
                        month_total[m1] == month_total[m2],
                        name=f"month_eq_{kk}[{y},{seas},{m1}={m2}]"
                    )
                    m.addConstr(
                        month_total[m1] == month_total[m3],
                        name=f"month_eq_{kk}[{y},{seas},{m1}={m3}]"
                    )

            # refinery H2: annual demand + in-season fully flat hourly demand
            if refinery_active:
                # annual total
                m.addConstr(
                    gp.quicksum(h_ref[y, tt] for tt in T) == float(D_REFH2[y]),
                    name=f"dem_REFH2[{y}]"
                )

                # within each season, every hour has the same refinery H2 demand
                # between seasons, the hourly level can differ
                for seas, months in SEASONS.items():
                    season_hours = []
                    for mm in months:
                        season_hours.extend(hours_in_month[mm])

                    if len(season_hours) >= 2:
                        tt0 = season_hours[0]
                        for tt in season_hours[1:]:
                            m.addConstr(
                                h_ref[y, tt] == h_ref[y, tt0],
                                name=f"flat_REFH2_season[{y},{seas},{tt}]"
                            )

        else:
            # =========================================================
            # Case 2: annual_demand_only = False
            #
            # Chemical products:
            #   1) no seasonal variation
            #   2) every month has equal total output
            #   3) daily output does not need to be equal
            #
            # Refinery H2:
            #   1) fully rigid
            #   2) every hour in the whole year has the same h_ref
            # =========================================================

            # chemical products: equal monthly total output across all months
            for kk, Ddict in demand_prod.items():
                monthly_dem = float(Ddict[y]) / 12.0

                for mm in M:
                    m.addConstr(
                        gp.quicksum(x[kk, y, tt] for tt in hours_in_month[mm]) == monthly_dem,
                        name=f"dem_month_{kk}[{y},{mm}]"
                    )

            # refinery H2: fully flat hourly demand across the whole year
            if refinery_active:
                hourly_ref = float(D_REFH2[y]) / 8760.0

                for tt in T:
                    m.addConstr(
                        h_ref[y, tt] == hourly_ref,
                        name=f"dem_hour_REFH2[{y},{tt}]"
                    )

    # -----------------------------
    # annual net electricity purchase = sale
    # -----------------------------
    if buy_equal_sale is True:
        for y in Y:
            m.addConstr(
                gp.quicksum(e_buy[y, tt] - e_sell[y, tt] for tt in T) == 0.0,
                name=f"net_grid[{y}]"
            )

    # -----------------------------
    # annual electricity sale cap
    # -----------------------------
    to_grid_max = 0.2
    load_max = max_load_interact

    for y in Y:
        m.addConstr(
            gp.quicksum(e_sell[y, tt] for tt in T)
            <=
            to_grid_max * gp.quicksum(g[r, y, tt] for r in R for tt in T),
            name=f"sell_cap_annual[{y}]"
        )

    # -----------------------------
    # disable H2 gas turbine when annual_demand_only == False; only the seasonal case may use the gas turbine.
    # -----------------------------
    if HCCGT_open is False:
        for y in Y:
            m.addConstr(
                Kcap["HCCGT", y] == 0.0,
                name=f"disable_HCCGT[{y}]"
            )

    # -----------------------------
    # hourly electricity buy/sell cap using mapped typical-day load
    # -----------------------------
    for y in Y:
        for tt in T:
            load_cap = get_load_mw(y, tt)
            m.addConstr(
                e_sell[y, tt] <= load_max * load_cap,
                name=f"sell_cap_hour[{y},{tt}]"
            )
            m.addConstr(
                e_buy[y, tt] <= load_max * load_cap,
                name=f"buy_cap_hour[{y},{tt}]"
            )

    # -----------------------------
    # global no-coupling switch
    # -----------------------------
    if not coup:
        for y in Y:
            for tt in T:
                m.addConstr(e_sell[y, tt] == 0.0, name=f"sell_zero[{y},{tt}]")
                m.addConstr(e_buy[y, tt] == 0.0, name=f"buy_zero[{y},{tt}]")

    # -----------------------------
    # objective
    # -----------------------------
    obj = gp.LinExpr()

    for y in Y:
        disc = delta_y[y]

        fixed_cost_y = gp.quicksum(
            (c_cap[(i, y)] + c_fom[(i, y)]) * Kcap[i, y]
            for i in I
        )

        op_cost_y = gp.LinExpr()
        for tt in T:
            buy_price = get_pi_buy(y, tt)
            sell_price = get_pi_sell(y, tt) 
            heat_price = get_pi_Q(y, tt)

            op_cost_y += (
                buy_price * e_buy[y, tt]
                - sell_price * e_sell[y, tt]
                + CAPACITY_PRICE_SELL_CNY_PER_MWH * e_sell[y, tt]
                + heat_price * H[y, tt]
                + gp.quicksum(c_vom_other[(k, y)] * x[k, y, tt] for k in K)
            )

        obj += disc * (fixed_cost_y + op_cost_y)

    m.setObjective(obj, GRB.MINIMIZE)
    m.Params.OutputFlag = 1
    m.Params.Threads = 20
    m.Params.Method = 2
    m.Params.Crossover = 0
    m.Params.BarConvTol = 0.1
    m.Params.BarIterLimit = 200
    m.Params.Presolve = 2
    m.Params.NumericFocus = 2
    m.Params.BarHomogeneous = 1
    m.Params.FeasibilityTol = 0.01
    m.Params.OptimalityTol = 0.01

    vars_out = {
        # capacities
        "Kcap": Kcap,
        "A": A,

        # power system
        "g": g,
        "e_buy": e_buy,
        "e_sell": e_sell,

        # grid-related fee parameters for post-processing
        "T_D_FEE_CNY_PER_MWH": T_D_FEE_CNY_PER_MWH,
        "CAPACITY_PRICE_SELL_CNY_PER_MWH": CAPACITY_PRICE_SELL_CNY_PER_MWH,

        # battery
        "p_c": p_c,
        "p_d": p_d,
        "SOC": SOC,

        # electrolyzer & hydrogen
        "p_EL": p_EL,
        "h_EL": h_EL,

        # hydrogen storage
        "h_ch": h_ch,
        "h_dis": h_dis,
        "S_H2": S_H2,

        # refinery H2
        "h_ref": h_ref,

        # HCCGT
        "p_HCCGT": p_HCCGT,
        "h_HCCGT": h_HCCGT,

        # chemical production
        "x": x,

        # derived loads
        "L_E": L_E,
        "H": H,

        # calendar maps for post-processing
        "T": T,
        "D": D,
        "month_of_t": month_of_t,
        "day_of_t": day_of_t,
        "hour_of_day": hour_of_day,
        "seg6_of_t": seg6_of_t,
        "hours_in_month": hours_in_month,
        "hours_in_day": hours_in_day,
        "days_in_month": days_in_month,
        "seasonal_break_ts": seasonal_break_ts,
    }

    return m, vars_out

# %% [Notebook cell 4]
"""
Compute key metrics such as cost breakdown, electrolyzer full-load hours, and equivalent cycle counts for hydrogen storage and battery storage.
"""

import pandas as pd
from gurobipy import GRB


def compute_efc_flh(model, vars_out, Y, eta_ch_HS, eps=1e-7, c=None):
    """
    Post-process for 8760-hour model:
      - EFC_BAT[y] = annual charged electricity / (4*K_BAT + eps)
      - FLH_EL[y]  = annual electrolyzer electricity / (K_EL + eps)
      - EFC_HS[y]  = annual H2 charged into storage (stock-side) / (K_HS + eps)

    Notes:
      - Uses hourly variables (y,t), no typical-day weights.
      - EFC_HS uses stock-side inflow: eta_ch_HS * h_ch
    """
    if model.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        raise RuntimeError(f"Model not solved to optimal/suboptimal. Status={model.Status}")

    p_c = vars_out["p_c"]        # (y,t)
    p_EL = vars_out["p_EL"]      # (y,t)
    h_ch = vars_out["h_ch"]      # (y,t)
    Kcap = vars_out["Kcap"]      # (i,y)
    T = vars_out["T"]            # 1..8760

    EFC_BAT = {}
    FLH_EL = {}
    EFC_HS = {}

    for y in Y:
        # battery
        charged_mwh = sum(p_c[y, t].X for t in T)
        K_BAT = Kcap["BAT", y].X
        EFC_BAT[y] = charged_mwh / (4.0 * K_BAT + eps)

        # electrolyzer
        el_mwh = sum(p_EL[y, t].X for t in T)
        K_EL = Kcap["EL", y].X
        FLH_EL[y] = el_mwh / (K_EL + eps)

        # hydrogen storage
        hs_charge_stock = sum(eta_ch_HS * h_ch[y, t].X for t in T)
        K_HS = Kcap["HS", y].X
        EFC_HS[y] = hs_charge_stock / (K_HS + eps)

    meta = {"c": (str(c).upper().strip() if c is not None else None)}
    return EFC_BAT, FLH_EL, EFC_HS, meta


def compute_annual_cost_breakdown(model, vars_out, data):
    """
    Return a DataFrame with yearly cost components (NOT discounted):
      y, CAP, FOM, E_BUY, E_SELL, HEAT, VOM_OTHER, TOTAL

    8760-hour version:
      - operation variables are indexed by (y,t)
      - pi_buy / pi_sell are still stored as (y,m,s), so we map each hour t to (m,s)
      - pi_Q is yearly: pi_Q[y]
    """
    if model.SolCount == 0:
        return pd.DataFrame(columns=["y", "CAP", "FOM", "E_BUY", "E_SELL", "HEAT", "VOM_OTHER", "TOTAL"])

    Y = data["Y"]
    c_cap, c_fom = data["c_cap"], data["c_fom"]
    pi_buy, pi_sell, pi_Q = data["pi_buy"], data["pi_sell"], data["pi_Q"]
    c_vom_other = data["c_vom_other"]
    I, K = data["I"], data["K"]

    Kcap = vars_out["Kcap"]
    e_buy, e_sell = vars_out["e_buy"], vars_out["e_sell"]
    H = vars_out["H"]
    x = vars_out["x"]

    T = vars_out["T"]
    month_of_t = vars_out["month_of_t"]
    seg6_of_t = vars_out["seg6_of_t"]

    rows = []
    for y in Y:
        CAP = sum(c_cap[(i, y)] * Kcap[i, y].X for i in I)
        FOM = sum(c_fom[(i, y)] * Kcap[i, y].X for i in I)

        E_BUY = 0.0
        E_SELL = 0.0
        HEAT = 0.0
        VOM_OTHER = 0.0

        for t in T:
            mm = month_of_t[t]
            s = seg6_of_t[t]

            E_BUY += pi_buy[(y, mm, s)] * e_buy[y, t].X
            E_SELL += pi_sell[(y, mm, s)] * e_sell[y, t].X
            HEAT += pi_Q[y] * H[y, t].X
            VOM_OTHER += sum(c_vom_other[(k, y)] * x[k, y, t].X for k in K)

        TOTAL = CAP + FOM + E_BUY - E_SELL + HEAT + VOM_OTHER
        rows.append({
            "y": y,
            "CAP": CAP,
            "FOM": FOM,
            "E_BUY": E_BUY,
            "E_SELL": E_SELL,
            "HEAT": HEAT,
            "VOM_OTHER": VOM_OTHER,
            "TOTAL": TOTAL,
        })

    return pd.DataFrame(rows)


def compute_vre_curtailment(vars_out, data, tol=1e-9):
    """
    Compute implicit curtailment from:
      curt[r,y,t] = CF[r,y,t] * Kcap[r,y] - g[r,y,t]

    Returns:
      - curt_df: long DF with columns [r,y,t,m,d,h,curt_MW,avail_MW,gen_MW]
      - annual_df: annual aggregation by [r,y] in MWh/year
    """
    g = vars_out["g"]        # (r,y,t) MW
    Kcap = vars_out["Kcap"]  # (i,y) MW
    CF = data["CF"]          # (r,y,t)
    Y, R = data["Y"], data["R"]

    T = vars_out["T"]
    month_of_t = vars_out["month_of_t"]
    day_of_t = vars_out["day_of_t"]
    hour_of_day = vars_out["hour_of_day"]

    rows = []
    annual = {(r, y): {"avail_MWh": 0.0, "gen_MWh": 0.0, "curt_MWh": 0.0} for r in R for y in Y}

    for r in R:
        for y in Y:
            K = Kcap[r, y].X
            for t in T:
                avail = CF[(r, y, t)] * K
                gen = g[r, y, t].X
                curt = avail - gen
                if curt < tol:
                    curt = 0.0

                rows.append({
                    "r": r,
                    "y": y,
                    "t": t,
                    "m": month_of_t[t],
                    "d": day_of_t[t],
                    "h": hour_of_day[t],
                    "avail_MW": avail,
                    "gen_MW": gen,
                    "curt_MW": curt,
                })

                # Delta_h = 1 hour
                annual[(r, y)]["avail_MWh"] += avail
                annual[(r, y)]["gen_MWh"] += gen
                annual[(r, y)]["curt_MWh"] += curt

    curt_df = pd.DataFrame(rows)

    annual_rows = []
    for (r, y), d in annual.items():
        annual_rows.append({"r": r, "y": y, **d})
    annual_df = pd.DataFrame(annual_rows).sort_values(["r", "y"])

    return curt_df, annual_df

# %% [Notebook cell 5]
"""
Write results to Excel.
8,760-hour model version:
- Internal model variables are stored by (y,t).
- When writing to Excel, variables are converted back to a unified (y,m,d,s) structure for convenient downstream reading.

Key changes in this version:
1) Add the grid_TD_fee_monthly sheet to report the transmission and distribution fee decomposition for monthly electricity purchases and sales in each year.
   Note: the transmission and distribution fee is already included in pi_buy / pi_sell. This sheet is for reporting decomposition only and does not represent an additional objective-function cost.
2) Add the grid_capacity_cost_monthly sheet to report the monthly capacity cost associated with electricity sales in each year.
3) If annual_cost contains a TOTAL column, GRID_CAPACITY_COST is additionally included so that annual_cost is consistent with the modified objective-function accounting.
"""

from pathlib import Path
import pandas as pd
from gurobipy import GRB


def _build_readme_df(meta, sheet_desc_rows):
    rows = []
    rows += [
        ("Section", "Run information"),
        ("province", meta["province"]),
        ("chemical", meta["chem"]),
        ("solver_status", meta["status"]),
        ("status_code", meta["status_code"]),
        ("objective_value_discounted", meta["obj_val_discounted"]),
        ("note", "annual_cost sheet is NOT discounted; it is reconstructed year-by-year from the optimal solution."),
        ("note", "grid_TD_fee_monthly is a reporting decomposition only because T&D fee is already embedded in pi_buy/pi_sell."),
        ("note", "grid_capacity_cost_monthly is an additional selling-side capacity-cost item already included in the modified objective."),
        ("", ""),
        ("Section", "Solver status explanation"),
        ("OPTIMAL", "Global optimum found."),
        ("SUBOPTIMAL", "Feasible solution found but optimality gap remains."),
        ("INFEASIBLE", "No feasible solution exists."),
        ("UNBOUNDED", "Objective is unbounded (model issue)."),
        ("INF_OR_UNBD", "Infeasible or unbounded (needs diagnosis)."),
        ("TIME_LIMIT", "Stopped due to time limit; may have incumbent solution."),
        ("", ""),
        ("Section", "Sheet descriptions and units"),
    ]
    rows += sheet_desc_rows
    return pd.DataFrame(rows, columns=["Item", "Description"])


def _tupledict_to_long_df(td, value_attr="X", value_name="value"):
    rows = []
    for k, v in td.items():
        if not isinstance(k, tuple):
            k = (k,)
        val = getattr(v, value_attr) if hasattr(v, value_attr) else float(v)
        rows.append((*k, val))
    if not rows:
        return pd.DataFrame(columns=["key", value_name])
    nkey = len(rows[0]) - 1
    cols = [f"k{i+1}" for i in range(nkey)] + [value_name]
    return pd.DataFrame(rows, columns=cols)


def _tupledict_hourly_to_ymds_df(td, vars_out, value_attr="X", value_name="value"):
    """
    Convert tupledict keyed by:
      - (y, t)      -> columns [y, m, d, s, value]
      - (r, y, t)   -> columns [r, y, m, d, s, value]
      - (k, y, t)   -> columns [k, y, m, d, s, value]
    """
    month_of_t = vars_out["month_of_t"]
    day_of_t = vars_out["day_of_t"]
    hour_of_day = vars_out["hour_of_day"]

    rows = []
    for k, v in td.items():
        if not isinstance(k, tuple):
            k = (k,)

        val = getattr(v, value_attr) if hasattr(v, value_attr) else float(v)

        if len(k) == 2:
            y, t = k
            rows.append({
                "y": int(y),
                "m": int(month_of_t[int(t)]),
                "d": int(day_of_t[int(t)]),
                "s": int(hour_of_day[int(t)]),
                value_name: val,
            })

        elif len(k) == 3:
            first, y, t = k
            rows.append({
                "k1": first,
                "y": int(y),
                "m": int(month_of_t[int(t)]),
                "d": int(day_of_t[int(t)]),
                "s": int(hour_of_day[int(t)]),
                value_name: val,
            })

        else:
            rows.append({**{f"k{i+1}": kk for i, kk in enumerate(k)}, value_name: val})

    return pd.DataFrame(rows)


def compute_grid_fee_monthly(vars_out: dict, data: dict | None = None):
    """
    Compute monthly grid-related fee sheets.

    Assumptions:
    1) e_buy and e_sell are MW.
    2) Internal model step is 1 hour, so hourly MW equals hourly MWh.
    3) T&D fee is already embedded in pi_buy/pi_sell; grid_TD_fee_monthly is reporting-only.
    4) Capacity price is an additional objective cost paid when selling electricity.
    """
    e_buy = vars_out["e_buy"]
    e_sell = vars_out["e_sell"]
    hours_in_month = vars_out["hours_in_month"]

    if data is not None and "Y" in data:
        Y = data["Y"]
    else:
        Y = sorted({k[0] for k in e_buy.keys()})

    if data is not None and "M" in data:
        M = data["M"]
    else:
        M = sorted(hours_in_month.keys())

    td_fee = float(vars_out.get("T_D_FEE_CNY_PER_MWH", 150.0))
    capacity_price_sell = float(vars_out.get("CAPACITY_PRICE_SELL_CNY_PER_MWH", 200.0))

    td_rows = []
    capacity_rows = []

    for y in Y:
        for mm in M:
            buy_mwh = sum(e_buy[y, tt].X for tt in hours_in_month[mm])
            sell_mwh = sum(e_sell[y, tt].X for tt in hours_in_month[mm])

            td_buy_cost = td_fee * buy_mwh
            td_sell_cost = td_fee * sell_mwh
            td_total_cost = td_buy_cost + td_sell_cost

            td_rows.append({
                "y": y,
                "m": mm,
                "e_buy_MWh": buy_mwh,
                "e_sell_MWh": sell_mwh,
                "T_D_fee_CNY_per_MWh": td_fee,
                "T_D_buy_cost_CNY": td_buy_cost,
                "T_D_sell_cost_CNY": td_sell_cost,
                "T_D_total_cost_CNY": td_total_cost,
            })

            capacity_cost = capacity_price_sell * sell_mwh

            capacity_rows.append({
                "y": y,
                "m": mm,
                "e_sell_MWh": sell_mwh,
                "capacity_price_CNY_per_MWh": capacity_price_sell,
                "capacity_cost_CNY": capacity_cost,
            })

    td_df = pd.DataFrame(td_rows)
    capacity_df = pd.DataFrame(capacity_rows)
    return td_df, capacity_df


def _add_capacity_cost_to_annual_cost(annual_cost: pd.DataFrame, capacity_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add selling-side capacity cost to annual_cost if not already present.
    This keeps annual_cost TOTAL consistent with the modified objective.
    """
    if annual_cost is None or annual_cost.empty or capacity_df is None or capacity_df.empty:
        return annual_cost

    cap_annual = (
        capacity_df.groupby("y", as_index=False)["capacity_cost_CNY"]
        .sum()
        .rename(columns={"capacity_cost_CNY": "GRID_CAPACITY_COST"})
    )

    out = annual_cost.copy()
    if "GRID_CAPACITY_COST" not in out.columns:
        out = out.merge(cap_annual, on="y", how="left")
        out["GRID_CAPACITY_COST"] = out["GRID_CAPACITY_COST"].fillna(0.0)
        if "TOTAL" in out.columns:
            out["TOTAL"] = out["TOTAL"] + out["GRID_CAPACITY_COST"]

    return out


def save_chemical_results_one_run_full(
    out_root: str,
    province: str,
    chem: str,
    model,
    vars_out: dict,
    data: dict,
    efc_bat: dict | None = None,
    flh_el: dict | None = None,
    efc_hs: dict | None = None,
):
    out_dir = Path(out_root) / province
    out_dir.mkdir(parents=True, exist_ok=True)
    xlsx_path = out_dir / f"{chem}.xlsx"

    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.TIME_LIMIT: "TIME_LIMIT",
    }
    meta = {
        "province": province,
        "chem": chem,
        "status_code": int(model.Status),
        "status": status_map.get(model.Status, str(model.Status)),
        "obj_val_discounted": float(model.ObjVal) if model.SolCount > 0 else None,
    }

    sheets = {}
    sheet_desc = []

    # ---- 1) core outputs ----
    if "Kcap" in vars_out:
        sheets["Kcap"] = _tupledict_to_long_df(vars_out["Kcap"], value_name="Kcap")
        sheet_desc.append((
            "Kcap",
            "Installed (in-service) capacity by (i,y). "
            "Unit: MW (power tech) / t-cap for HS / t/h proxy for reactors per your formulation."
        ))

    if "g" in vars_out:
        df = _tupledict_hourly_to_ymds_df(vars_out["g"], vars_out, value_name="g")
        if "k1" in df.columns:
            df = df.rename(columns={"k1": "r"})
        sheets["gen_vre"] = df
        sheet_desc.append(("gen_vre", "Renewable generation g[r,y,m,d,s]. Unit: MW. r in {W,PV}."))

    if "e_buy" in vars_out:
        sheets["grid_buy"] = _tupledict_hourly_to_ymds_df(vars_out["e_buy"], vars_out, value_name="e_buy")
        sheet_desc.append(("grid_buy", "Grid purchase e_buy[y,m,d,s]. Unit: MW; with 1-h step it equals MWh per hour."))

    if "e_sell" in vars_out:
        sheets["grid_sell"] = _tupledict_hourly_to_ymds_df(vars_out["e_sell"], vars_out, value_name="e_sell")
        sheet_desc.append(("grid_sell", "Grid sale e_sell[y,m,d,s]. Unit: MW; with 1-h step it equals MWh per hour."))

    # battery
    if "p_c" in vars_out:
        sheets["bat_charge"] = _tupledict_hourly_to_ymds_df(vars_out["p_c"], vars_out, value_name="p_c")
        sheet_desc.append(("bat_charge", "Battery charging power p_c[y,m,d,s]. Unit: MW."))

    if "p_d" in vars_out:
        sheets["bat_discharge"] = _tupledict_hourly_to_ymds_df(vars_out["p_d"], vars_out, value_name="p_d")
        sheet_desc.append(("bat_discharge", "Battery discharging power p_d[y,m,d,s]. Unit: MW."))

    if "SOC" in vars_out:
        sheets["bat_soc"] = _tupledict_hourly_to_ymds_df(vars_out["SOC"], vars_out, value_name="SOC")
        sheet_desc.append(("bat_soc", "Battery SOC[y,m,d,s]. Unit: MWh. SOC upper bound is 4*K_BAT."))

    # electrolyzer + hydrogen
    if "p_EL" in vars_out:
        sheets["el_power"] = _tupledict_hourly_to_ymds_df(vars_out["p_EL"], vars_out, value_name="p_EL")
        sheet_desc.append(("el_power", "Electrolyzer power p_EL[y,m,d,s]. Unit: MW."))

    if "h_EL" in vars_out:
        sheets["h2_prod_el"] = _tupledict_hourly_to_ymds_df(vars_out["h_EL"], vars_out, value_name="h_EL")
        sheet_desc.append(("h2_prod_el", "Hydrogen production by electrolysis h_EL[y,m,d,s]. Unit: tH2/h."))

    if "h_ref" in vars_out:
        sheets["h2_refinery"] = _tupledict_hourly_to_ymds_df(vars_out["h_ref"], vars_out, value_name="h_ref")
        sheet_desc.append(("h2_refinery", "Refinery hydrogen flow h_ref[y,m,d,s]. Unit: tH2/h. In AMM/METH runs should be 0."))

    # hydrogen storage
    if "S_H2" in vars_out:
        sheets["h2_stock"] = _tupledict_hourly_to_ymds_df(vars_out["S_H2"], vars_out, value_name="S_H2")
        sheet_desc.append(("h2_stock", "Hydrogen inventory S_H2[y,m,d,s]. Unit: tH2."))

    if "h_ch" in vars_out:
        sheets["h2_charge"] = _tupledict_hourly_to_ymds_df(vars_out["h_ch"], vars_out, value_name="h_ch")
        sheet_desc.append(("h2_charge", "Hydrogen charging to storage h_ch[y,m,d,s]. Unit: tH2/h."))

    if "h_dis" in vars_out:
        sheets["h2_discharge"] = _tupledict_hourly_to_ymds_df(vars_out["h_dis"], vars_out, value_name="h_dis")
        sheet_desc.append(("h2_discharge", "Hydrogen discharging from storage h_dis[y,m,d,s]. Unit: tH2/h."))

    # reactor production
    if "x" in vars_out:
        df = _tupledict_hourly_to_ymds_df(vars_out["x"], vars_out, value_name="x")
        if "k1" in df.columns:
            df = df.rename(columns={"k1": "k"})
        sheets["reactor_output"] = df
        sheet_desc.append(("reactor_output", "Reactor production x[k,y,m,d,s]. Unit: t product/h. In H2 run all k should be 0."))

    # HCCGT
    if "p_HCCGT" in vars_out:
        sheets["p_HCCGT"] = _tupledict_hourly_to_ymds_df(vars_out["p_HCCGT"], vars_out, value_name="p_HCCGT")
        sheet_desc.append(("p_HCCGT", "HCCGT power p_HCCGT[y,m,d,s]. Unit: MW."))

    if "h_HCCGT" in vars_out:
        sheets["h_HCCGT"] = _tupledict_hourly_to_ymds_df(vars_out["h_HCCGT"], vars_out, value_name="h_HCCGT")
        sheet_desc.append(("h_HCCGT", "H2 consumption for HCCGT h_HCCGT[y,m,d,s]. Unit: tH2/h."))

    # derived loads
    if "L_E" in vars_out:
        sheets["load_elec"] = _tupledict_hourly_to_ymds_df(vars_out["L_E"], vars_out, value_name="L_E")
        sheet_desc.append(("load_elec", "Plant electricity load L_E[y,m,d,s]. Unit: MW."))

    if "H" in vars_out:
        sheets["load_heat"] = _tupledict_hourly_to_ymds_df(vars_out["H"], vars_out, value_name="H")
        sheet_desc.append(("load_heat", "Plant heat demand H[y,m,d,s]. Unit: MWh_th/h (equiv. MW_th over one hour). Heat price is yearly pi_Q[y]."))

    # ---- 2) yearly KPIs ----
    if efc_bat is not None:
        df = pd.DataFrame([{"y": y, "EFC_BAT": float(v)} for y, v in efc_bat.items()]).sort_values("y")
        sheets["EFC_BAT"] = df
        sheet_desc.append(("EFC_BAT", "Battery equivalent full cycles per year. Unit: cycles/year. Definition: annual charged electricity / (4*K_BAT + eps)."))

    if flh_el is not None:
        df = pd.DataFrame([{"y": y, "FLH_EL": float(v)} for y, v in flh_el.items()]).sort_values("y")
        sheets["FLH_EL"] = df
        sheet_desc.append(("FLH_EL", "Electrolyzer equivalent full-load hours per year. Unit: hours/year. Definition: annual EL electricity / (K_EL + eps)."))

    if efc_hs is not None:
        df = pd.DataFrame([{"y": y, "EFC_HS": float(v)} for y, v in efc_hs.items()]).sort_values("y")
        sheets["EFC_HS"] = df
        sheet_desc.append(("EFC_HS", "Hydrogen storage equivalent full cycles per year. Unit: cycles/year. Definition: annual stock-side H2 charged into storage / (K_HS + eps), i.e. sum(eta_ch_HS*h_ch)/(K_HS + eps)."))

    # ---- 3) annual cost breakdown and grid fee monthly outputs ----
    if model.SolCount > 0:
        grid_td_fee_monthly, grid_capacity_cost_monthly = compute_grid_fee_monthly(vars_out, data)

        sheets["grid_TD_fee_monthly"] = grid_td_fee_monthly
        sheet_desc.append((
            "grid_TD_fee_monthly",
            "Monthly transmission & distribution fee decomposition. Unit: CNY/month. "
            "Fee = 150 CNY/MWh * (e_buy_MWh + e_sell_MWh). Reporting only; already embedded in pi_buy/pi_sell."
        ))

        sheets["grid_capacity_cost_monthly"] = grid_capacity_cost_monthly
        sheet_desc.append((
            "grid_capacity_cost_monthly",
            "Monthly selling-side capacity cost. Unit: CNY/month. "
            "Cost = 200 CNY/MWh * e_sell_MWh. This cost is included in the modified objective."
        ))

        annual_cost = compute_annual_cost_breakdown(model, vars_out, data)
        annual_cost = _add_capacity_cost_to_annual_cost(annual_cost, grid_capacity_cost_monthly)
        sheets["annual_cost"] = annual_cost
        sheet_desc.append((
            "annual_cost",
            "Yearly cost breakdown (NOT discounted). Components include GRID_CAPACITY_COST if TOTAL exists. "
            "Unit: CNY/year (or your monetary unit)."
        ))

        I, Y = data["I"], data["Y"]
        Kcap = vars_out["Kcap"]
        cap_rows = []
        for y in Y:
            for i in I:
                cap_rows.append({
                    "y": y,
                    "i": i,
                    "CAP": data["c_cap"][(i, y)] * Kcap[i, y].X,
                    "FOM": data["c_fom"][(i, y)] * Kcap[i, y].X,
                })
        sheets["annual_cap_fom_by_tech"] = pd.DataFrame(cap_rows)
        sheet_desc.append(("annual_cap_fom_by_tech", "Technology-level annualized CAP and FOM costs by (i,y). Unit: CNY/year (or your monetary unit)."))

    # ---- 4) implicit curtailment (derived) ----
    if ("g" in vars_out) and ("Kcap" in vars_out) and ("CF" in data):
        curt_df, curt_annual_df = compute_vre_curtailment(vars_out, data)

        sheets["vre_curt"] = curt_df
        sheet_desc.append((
            "vre_curt",
            "Implicit curtailment by (r,y,m,d,s). "
            "curt_MW = CF[r,y,t]*Kcap[r,y] - gen_MW. "
            "Also reports avail_MW and gen_MW. Unit: MW."
        ))

        sheets["vre_curt_annual"] = curt_annual_df
        sheet_desc.append((
            "vre_curt_annual",
            "Annual VRE availability, generation and curtailment aggregated over 8760 hours. "
            "Unit: MWh/year."
        ))

    # ---- 5) readme FIRST ----
    readme_df = _build_readme_df(meta, sheet_desc_rows=sheet_desc)

    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        readme_df.to_excel(writer, sheet_name="readme", index=False)
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)

    return str(xlsx_path)

# %% [Notebook cell 7]
province_list = ['Xinjiang']
sce ='case2'
out_root = sce + "/chemical_result"
chem_list = ['AMM','METH','H2']

# %% [Notebook cell 8]
# -----------------------------
# Example calling code (loop) - 8760 version; Note: seasonal adjustment of the production plan also changes monthly demand; production can be shifted across days. Reactor loads are now also allowed to vary.
# -----------------------------
from gurobipy import GRB
if sce == 'case2':
    seasonal = True    # Allow grid interaction and seasonal production-plan adjustment, enable the hydrogen gas turbine, limit maximum grid interaction to 30% of the contemporaneous grid load, and allow annual electricity purchases and sales to be non-net-zero.
    max_load_interact=0.3
    buy_equal_sale=False
    HCCGT_open=True
    coup = True

else:
    raise ValueError(f"Unknown scenario: {sce}")

for prov in province_list:
    data = load_inputs_from_excel(sce + "/inputs_template.xlsx", province=prov)

    for chem in chem_list:
        model, vars_out = build_chem_power_h2_model(
            Y=data["Y"], M=data["M"], S=data["S"], R=data["R"], K=data["K"], I=data["I"],
            Delta_h=data["Delta_h"],                 # Reset to 1.0 inside the model (8,760-hour model).
            w_m=data["w_m"],                        # Essentially unused in the 8,760-hour main model, but retained in the interface.
            CF=data["CF"],                          # Now CF[(r,y,t)].
            WS_POT=data["WS_POT"],
            WS_POT_upper=0.3,                       # Use at most 30% of local wind and solar potential.
            LOAD_MW=data["LOAD_MW"],                # Still (y,m,s); mapped to hourly values inside the model.
            pi_buy=data["pi_buy"],
            pi_sell=data["pi_sell"],
            pi_Q=data["pi_Q"],                      
            D_NH3=data["D_NH3"],
            D_MeOH_CH=data["D_MeOH_CH"],
            D_MeOH_CO2=data["D_MeOH_CO2"],
            D_REFH2=data["D_REFH2"],
            aE=data["aE"], aQ=data["aQ"], aH2=data["aH2"],
            c_vom_other=data["c_vom_other"],
            eta_EL=data["eta_EL"],
            eta_HCCGT=data["eta_HCCGT"],
            eta_ch_HS=data["eta_ch_HS"],
            eta_dis_HS=data["eta_dis_HS"],
            gamma_ch_HS=data["gamma_ch_HS"],
            gamma_dis_HS=data["gamma_dis_HS"],
            eta_c_BAT=data["eta_c_BAT"],
            eta_d_BAT=data["eta_d_BAT"],
            RU_k=data["RU_k"],
            RD_k=data["RD_k"],
            RU_EL=data["RU_EL"],
            RD_EL=data["RD_EL"],
            RU_HCCGT=data["RU_HCCGT"],
            RD_HCCGT=data["RD_HCCGT"],
            c_cap=data["c_cap"],
            c_fom=data["c_fom"],
            L=data["L"],
            r_disc=data["r_disc"],
            delta_y=data["delta_y"],
            max_load_interact=max_load_interact,
            buy_equal_sale=buy_equal_sale,
            HCCGT_open=HCCGT_open,
            V0=data["V0"],
            A_init=data["A_init"],
            c=chem,
            model_name=f"chem_power_h2_8760_{prov}_{chem}",
            annual_demand_only=seasonal,
            coup=coup
        )

        model.optimize()

        # ---- compute efc_bat, flh_el ----
        if model.Status in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
            efc_bat, flh_el,efc_hs, meta = compute_efc_flh(
                model=model,
                vars_out=vars_out,
                Y=data["Y"],
                eta_ch_HS=data["eta_ch_HS"],
                eps=1e-7,
                c=chem
            )

            print(f"[{prov}-{chem}] meta={meta}")
            print(f"[{prov}-{chem}] EFC_BAT sample:", list(efc_bat.items())[:3])
            print(f"[{prov}-{chem}] FLH_EL sample:", list(flh_el.items())[:3])
        else:
            efc_bat, flh_el = None, None
            print(f"[{prov}-{chem}] not solved. status={model.Status}")

        # ---- save ----
        xlsx_path = save_chemical_results_one_run_full(
            out_root=out_root,
            province=prov,
            chem=chem,
            model=model,
            vars_out=vars_out,
            data=data,
            efc_bat=efc_bat,
            flh_el=flh_el,
            efc_hs=efc_hs,
        )

        print(f"Saved: {xlsx_path}")

# %% [Notebook cell 9]
