# %% [Notebook cell 1]
import math
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# %% [Notebook cell 2]
from pathlib import Path
from typing import Dict, List, Tuple
import math

import pandas as pd
import gurobipy as gp
from gurobipy import GRB


# =========================================================
# User parameters
# =========================================================
# Fixed cost: uniform constant, unit: CNY/grid
FIXED_COST_POWER = 10_000_000.0
FIXED_COST_CHEM = 10_000_000.0

cons_cost = 5500  # CNY/(MW·km)
Crf = 0.05  # 50-year lifetime + 5% interest rate
C_ENE_POWER = cons_cost * Crf / 8760  # Cost at 8,760 full-load hours
C_ENE_CHEM = cons_cost * Crf / 8760   # Cost at 8,760 full-load hours

# Added: annual utilization rate of chemical-to-power transmission lines (0-1)
CTP_UTILIZATION = 0.1
C_ENE_CTP = cons_cost * Crf / 8760    # Still defined on a full-load basis; divided by utilization in the actual model

# Added: adjustable range of chemical-sector share
CHEM_SHARE_LB_FACTOR = 0.01
CHEM_SHARE_UB_FACTOR = 10


# Gurobi parameters
MIP_GAP = 0.0003
TIME_LIMIT = 3600
THREADS = 0   # 0 means use the default

# =========================================================
# Basic utility functions
# =========================================================
def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
Calculate the great-circle distance between two points (km)
    """
    r = 6371.0
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    c = 2 * math.asin(math.sqrt(a))
    return r * c


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_and_validate_sector(x: str) -> str:
    x = str(x).strip().lower()
    if x not in {"power", "chem"}:
        raise ValueError(f"sector 必须为 'power' 或 'chem'，但收到: {x}")
    return x


def normalize_and_validate_tech(x: str) -> str:
    x = str(x).strip().lower()
    if x not in {"wind", "pv"}:
        raise ValueError(f"tech 必须为 'wind' 或 'pv'，但收到: {x}")
    return x


# =========================================================
# Data loading and validation
# =========================================================
def load_input_data(xlsx_path: str) -> Dict[str, pd.DataFrame]:
    """
Read Excel input
    """
    xlsx_path = Path(xlsx_path)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {xlsx_path}")

    required_sheets = [
        "grid_resource",
        "wind_offshore_resource",
        "cap_target",
        "load_centers",
        "load_share",
        "gen_target_GWh",
        "ctp_MWh",
    ]
    xls = pd.ExcelFile(xlsx_path)

    for s in required_sheets:
        if s not in xls.sheet_names:
            raise ValueError(f"缺少 sheet: {s}")

    df_grid = pd.read_excel(xlsx_path, sheet_name="grid_resource")
    df_off = pd.read_excel(xlsx_path, sheet_name="wind_offshore_resource")
    df_cap = pd.read_excel(xlsx_path, sheet_name="cap_target")
    df_load = pd.read_excel(xlsx_path, sheet_name="load_centers")
    df_share = pd.read_excel(xlsx_path, sheet_name="load_share")
    df_gen_target = pd.read_excel(xlsx_path, sheet_name="gen_target_GWh")
    df_ctp = pd.read_excel(xlsx_path, sheet_name="ctp_MWh")

    # -----------------------------
    # grid_resource
    # -----------------------------
    for col in ["province", "grid_id"]:
        if col not in df_grid.columns:
            raise ValueError(f"grid_resource 缺少字段: {col}")
    for col in ["lon", "lat", "pot_wind", "pot_pv", "cf_wind", "cf_pv", "lcoe_wind", "lcoe_pv"]:
        if col not in df_grid.columns:
            raise ValueError(f"grid_resource 缺少字段: {col}")

    # -----------------------------
    # wind_offshore_resource
    # -----------------------------
    for col in ["province", "grid_id"]:
        if col not in df_off.columns:
            raise ValueError(f"wind_offshore_resource 缺少字段: {col}")
    for col in ["lon", "lat", "pot_wind_offshore", "cf_wind_offshore", "lcoe_wind_offshore"]:
        if col not in df_off.columns:
            raise ValueError(f"wind_offshore_resource 缺少字段: {col}")

    # -----------------------------
    # cap_target
    # -----------------------------
    for col in ["province", "sector", "cap_wind_mw", "cap_pv_mw", "cap_wind_offshore_mw"]:
        if col not in df_cap.columns:
            raise ValueError(f"cap_target 缺少字段: {col}")

    # -----------------------------
    # load_centers
    # -----------------------------
    for col in ["province", "sector", "load_center_id", "lon", "lat"]:
        if col not in df_load.columns:
            raise ValueError(f"load_centers 缺少字段: {col}")

    # -----------------------------
    # load_share
    # -----------------------------
    for col in ["province", "sector", "tech", "load_center_id", "share"]:
        if col not in df_share.columns:
            raise ValueError(f"load_share 缺少字段: {col}")

    # -----------------------------
    # gen_target_GWh
    # -----------------------------
    for col in ["province", "sector", "gen_wind_gwh", "gen_pv_gwh", "gen_wind_offshore_gwh"]:
        if col not in df_gen_target.columns:
            raise ValueError(f"gen_target_GWh 缺少字段: {col}")

    # -----------------------------
    # ctp_MWh
    # -----------------------------
    for col in ["province", "ctp_MWh"]:
        if col not in df_ctp.columns:
            raise ValueError(f"ctp_MWh 缺少字段: {col}")

    # Normalize sector / tech
    df_cap["sector"] = df_cap["sector"].map(normalize_and_validate_sector)
    df_load["sector"] = df_load["sector"].map(normalize_and_validate_sector)
    df_share["sector"] = df_share["sector"].map(normalize_and_validate_sector)
    df_share["tech"] = df_share["tech"].map(normalize_and_validate_tech)
    df_gen_target["sector"] = df_gen_target["sector"].map(normalize_and_validate_sector)

    # Convert to string
    df_grid["province"] = df_grid["province"].astype(str)
    df_grid["grid_id"] = df_grid["grid_id"].astype(str)

    df_off["province"] = df_off["province"].astype(str)
    df_off["grid_id"] = df_off["grid_id"].astype(str)

    df_cap["province"] = df_cap["province"].astype(str)

    df_load["province"] = df_load["province"].astype(str)
    df_load["load_center_id"] = df_load["load_center_id"].astype(str)

    df_share["province"] = df_share["province"].astype(str)
    df_share["load_center_id"] = df_share["load_center_id"].astype(str)

    df_ctp["province"] = df_ctp["province"].astype(str)

    # Basic non-empty validation
    if df_grid.empty:
        raise ValueError("grid_resource 为空")
    if df_cap.empty:
        raise ValueError("cap_target 为空")
    if df_load.empty:
        raise ValueError("load_centers 为空")
    if df_share.empty:
        raise ValueError("load_share 为空")

    return {
        "grid": df_grid,
        "offshore": df_off,
        "cap": df_cap,
        "gen_target": df_gen_target,
        "load": df_load,
        "share": df_share,
        "ctp": df_ctp,
    }


def validate_shares(df_share: pd.DataFrame, tol: float = 1e-6) -> None:
    """
Validate whether share sums to 1 for each (province, sector, tech)
    tech still only includes:
      - wind: total onshore + offshore wind
      - pv
    """
    grouped = (
        df_share.groupby(["province", "sector", "tech"], as_index=False)["share"]
        .sum()
        .rename(columns={"share": "share_sum"})
    )

    bad = grouped[(grouped["share_sum"] - 1.0).abs() > tol]
    if not bad.empty:
        raise ValueError(
            "load_share 中存在 (province, sector, tech) 的 share 加总不为 1：\n"
            f"{bad.to_string(index=False)}"
        )


def validate_cap_targets(df_cap: pd.DataFrame) -> None:
    """
Validate that cap_target is unique for each (province, sector)
    """
    dup = df_cap.groupby(["province", "sector"]).size().reset_index(name="n")
    dup = dup[dup["n"] > 1]
    if not dup.empty:
        raise ValueError(
            "cap_target 中同一 (province, sector) 出现多行，请整理为唯一：\n"
            f"{dup.to_string(index=False)}"
        )


def validate_same_share_for_wind_pv(df_share: pd.DataFrame, tol: float = 1e-6) -> None:
    """
Validate that wind / pv shares are identical for each (province, sector, load_center_id)
    """
    pivot = (
        df_share.pivot_table(
            index=["province", "sector", "load_center_id"],
            columns="tech",
            values="share",
            aggfunc="sum"
        )
        .reset_index()
    )

    if "wind" not in pivot.columns or "pv" not in pivot.columns:
        raise ValueError("load_share 必须同时包含 wind 和 pv 两类技术的 share")

    bad = pivot[(pivot["wind"] - pivot["pv"]).abs() > tol]
    if not bad.empty:
        raise ValueError(
            "发现 wind 和 pv 的 share 不一致。跨部门送电要求两者一致，请检查：\n"
            f"{bad.to_string(index=False)}"
        )


# =========================================================
# Provincial data preparation
# =========================================================
def build_province_data(
    province: str,
    data: Dict[str, pd.DataFrame],
    fixed_cost_power: float,
    fixed_cost_chem: float,
    c_ene_power: float,
    c_ene_chem: float,
    c_ene_ctp: float,
    ctp_utilization: float,
    chem_share_lb_factor: float,
    chem_share_ub_factor: float,
) -> Dict:
    """
Build model input parameters for a single province
    """
    df_grid = data["grid"].copy()
    df_off = data["offshore"].copy()
    df_cap = data["cap"].copy()
    df_load = data["load"].copy()
    df_share = data["share"].copy()
    df_gen_target = data["gen_target"].copy()
    df_ctp = data["ctp"].copy()

    dg = df_grid[df_grid["province"] == province].copy()
    doff = df_off[df_off["province"] == province].copy()
    dc = df_cap[df_cap["province"] == province].copy()
    dl = df_load[df_load["province"] == province].copy()
    ds = df_share[df_share["province"] == province].copy()
    dgen = df_gen_target[df_gen_target["province"] == province].copy()
    dctp = df_ctp[df_ctp["province"] == province].copy()

    if dg.empty:
        raise ValueError(f"{province}: grid_resource 无数据")
    if dc.empty:
        raise ValueError(f"{province}: cap_target 无数据")

    all_sectors = ["power", "chem"]
    share_techs = ["wind", "pv"]

    # -----------------------------
    # Read capacity targets
    # -----------------------------
    cap_dict = {}
    for s in all_sectors:
        row = dc[dc["sector"] == s]
        if row.empty:
            cap_dict[(s, "wind_land")] = 0.0
            cap_dict[(s, "pv")] = 0.0
            cap_dict[(s, "wind_offshore")] = 0.0
        else:
            if len(row) > 1:
                raise ValueError(f"{province}: cap_target 中部门 {s} 不唯一")
            row = row.iloc[0]
            cap_dict[(s, "wind_land")] = float(row["cap_wind_mw"])
            cap_dict[(s, "pv")] = float(row["cap_pv_mw"])
            cap_dict[(s, "wind_offshore")] = float(row["cap_wind_offshore_mw"])

    # -----------------------------
    # Read generation targets
    # -----------------------------
    gen_target_dict = {}
    for s in all_sectors:
        row = dgen[dgen["sector"] == s]
        if row.empty:
            gen_target_dict[(s, "wind_land")] = 0.0
            gen_target_dict[(s, "pv")] = 0.0
            gen_target_dict[(s, "wind_offshore")] = 0.0
        else:
            row = row.iloc[0]
            gen_target_dict[(s, "wind_land")] = float(row["gen_wind_gwh"]) * 1000.0
            gen_target_dict[(s, "pv")] = float(row["gen_pv_gwh"]) * 1000.0
            gen_target_dict[(s, "wind_offshore")] = float(row["gen_wind_offshore_gwh"]) * 1000.0

    # -----------------------------
    # Read annual chemical-to-power electricity transfer
    # -----------------------------
    if dctp.empty:
        ctp_mwh = 0.0
    else:
        if len(dctp) > 1:
            raise ValueError(f"{province}: ctp_MWh 中同一省份出现多行，请整理为唯一")
        ctp_mwh = float(dctp.iloc[0]["ctp_MWh"])

    # -----------------------------
    # Load-center set for each sector
    # -----------------------------
    load_centers_all = {
        s: dl[dl["sector"] == s]["load_center_id"].astype(str).tolist()
        for s in all_sectors
    }

    # -----------------------------
    # Identify active sectors
    # -----------------------------
    active_sectors = []
    for s in all_sectors:
        has_load = len(load_centers_all[s]) > 0
        target_sum = (
            cap_dict[(s, "wind_land")]
            + cap_dict[(s, "pv")]
            + cap_dict[(s, "wind_offshore")]
        )

        if has_load:
            active_sectors.append(s)
        else:
            if abs(target_sum) <= 1e-9:
                print(f"{province}: 部门 {s} 无负荷中心，且目标装机为0，自动跳过。")
            else:
                raise ValueError(
                    f"{province}: 部门 {s} 没有负荷中心，但目标装机不为0 "
                    f"(wind_land={cap_dict[(s, 'wind_land')]}, "
                    f"pv={cap_dict[(s, 'pv')]}, "
                    f"wind_offshore={cap_dict[(s, 'wind_offshore')]})，模型不可行。"
                )

    if len(active_sectors) == 0:
        raise ValueError(f"{province}: power 和 chem 两个部门都无有效数据，无法建模。")

    if ctp_mwh > 1e-9:
        if "power" not in active_sectors or "chem" not in active_sectors:
            raise ValueError(f"{province}: ctp_MWh > 0，但 power 或 chem 负荷中心缺失，无法构建跨部门送电网络。")

    load_centers = {s: load_centers_all[s] for s in active_sectors}

    # -----------------------------
    # Onshore grid set
    # -----------------------------
    land_grids = dg["grid_id"].astype(str).tolist()

    # -----------------------------
    # Offshore grid set
    # -----------------------------
    offshore_grids = doff["grid_id"].astype(str).tolist()

    # -----------------------------
    # Onshore grid parameters
    # -----------------------------
    pot_land_wind = {}
    pot_pv = {}
    cf_land_wind = {}
    cf_pv = {}
    lcoe_land_wind = {}
    lcoe_pv = {}
    land_lonlat = {}

    for _, row in dg.iterrows():
        g = str(row["grid_id"])
        land_lonlat[g] = (float(row["lon"]), float(row["lat"]))

        pot_land_wind[g] = float(row["pot_wind"])
        pot_pv[g] = float(row["pot_pv"])

        cf_land_wind[g] = float(row["cf_wind"])
        cf_pv[g] = float(row["cf_pv"])

        lcoe_land_wind[g] = float(row["lcoe_wind"])
        lcoe_pv[g] = float(row["lcoe_pv"])

    # -----------------------------
    # Offshore grid parameters
    # -----------------------------
    pot_offshore_wind = {}
    cf_offshore_wind = {}
    lcoe_offshore_wind = {}
    offshore_lonlat = {}

    for _, row in doff.iterrows():
        h = str(row["grid_id"])
        offshore_lonlat[h] = (float(row["lon"]), float(row["lat"]))

        pot_offshore_wind[h] = float(row["pot_wind_offshore"])
        cf_offshore_wind[h] = float(row["cf_wind_offshore"])
        lcoe_offshore_wind[h] = float(row["lcoe_wind_offshore"])

    # -----------------------------
    # Load-center longitude and latitude
    # -----------------------------
    load_lonlat = {}
    for _, row in dl.iterrows():
        s = row["sector"]
        if s not in active_sectors:
            continue
        n = str(row["load_center_id"])
        load_lonlat[(s, n)] = (float(row["lon"]), float(row["lat"]))

    # -----------------------------
    # Weights: tech still only includes wind / pv
    # -----------------------------
    omega = {}
    for s in active_sectors:
        for k in share_techs:
            sub = ds[(ds["sector"] == s) & (ds["tech"] == k)].copy()

            if sub.empty:
                raise ValueError(f"{province}: load_share 缺少 sector={s}, tech={k} 的数据")

            share_centers = set(sub["load_center_id"].astype(str).tolist())
            load_center_set = set(load_centers[s])
            if share_centers != load_center_set:
                raise ValueError(
                    f"{province}: sector={s}, tech={k} 的 load_share 负荷中心集合"
                    f"与 load_centers 不一致。\n"
                    f"share中有: {sorted(share_centers)}\n"
                    f"load_centers中有: {sorted(load_center_set)}"
                )

            for _, row in sub.iterrows():
                n = str(row["load_center_id"])
                omega[(s, n, k)] = float(row["share"])

    # -----------------------------
    # Cross-sector electricity-transfer share (use the wind column directly)
    # -----------------------------
    omega_chem_export = {}
    omega_power_import = {}

    if "chem" in active_sectors:
        sub_chem = ds[(ds["sector"] == "chem") & (ds["tech"] == "wind")].copy()
        for _, row in sub_chem.iterrows():
            omega_chem_export[str(row["load_center_id"])] = float(row["share"])

    if "power" in active_sectors:
        sub_power = ds[(ds["sector"] == "power") & (ds["tech"] == "wind")].copy()
        for _, row in sub_power.iterrows():
            omega_power_import[str(row["load_center_id"])] = float(row["share"])

    chem_share_bounds = {}
    for nc, s0 in omega_chem_export.items():
        chem_share_bounds[nc] = (
            chem_share_lb_factor * s0,
            chem_share_ub_factor * s0
        )

    # -----------------------------
    # Distance: onshore grid -> load center
    # -----------------------------
    dist_land = {}
    for g in land_grids:
        glon, glat = land_lonlat[g]
        for s in active_sectors:
            for n in load_centers[s]:
                llon, llat = load_lonlat[(s, n)]
                dist_land[(g, s, n)] = haversine_km(glon, glat, llon, llat)

    # -----------------------------
    # Distance: offshore grid -> load center
    # -----------------------------
    dist_offshore = {}
    for h in offshore_grids:
        hlon, hlat = offshore_lonlat[h]
        for s in active_sectors:
            for n in load_centers[s]:
                llon, llat = load_lonlat[(s, n)]
                dist_offshore[(h, s, n)] = haversine_km(hlon, hlat, llon, llat)

    # -----------------------------
    # Distance: chemical load center -> power load center
    # -----------------------------
    dist_ctp = {}
    if "chem" in active_sectors and "power" in active_sectors:
        for nc in load_centers["chem"]:
            clon, clat = load_lonlat[("chem", nc)]
            for np_ in load_centers["power"]:
                plon, plat = load_lonlat[("power", np_)]
                dist_ctp[(nc, np_)] = haversine_km(clon, clat, plon, plat)

    # -----------------------------
    # Fixed cost and distance-based cost
    # -----------------------------
    F_land = {}
    for g in land_grids:
        if "power" in active_sectors:
            F_land[(g, "power")] = float(fixed_cost_power)
        if "chem" in active_sectors:
            F_land[(g, "chem")] = float(fixed_cost_chem)

    F_offshore = {}
    for h in offshore_grids:
        if "power" in active_sectors:
            F_offshore[(h, "power")] = float(fixed_cost_power)
        if "chem" in active_sectors:
            F_offshore[(h, "chem")] = float(fixed_cost_chem)

    c_ene = {}
    if "power" in active_sectors:
        c_ene["power"] = float(c_ene_power)
    if "chem" in active_sectors:
        c_ene["chem"] = float(c_ene_chem)

    # -----------------------------
    # Capacity-potential feasibility check
    # -----------------------------
    total_land_wind = sum(pot_land_wind[g] for g in land_grids)
    total_pv = sum(pot_pv[g] for g in land_grids)
    total_offshore = sum(pot_offshore_wind[h] for h in offshore_grids)

    for s in active_sectors:
        if cap_dict[(s, "wind_land")] > total_land_wind + 1e-9:
            raise ValueError(
                f"{province}: 部门 {s} 陆上风电目标 {cap_dict[(s, 'wind_land')]} MW "
                f"大于全省陆上风电潜力 {total_land_wind} MW，模型必不可行。"
            )
        if cap_dict[(s, "pv")] > total_pv + 1e-9:
            raise ValueError(
                f"{province}: 部门 {s} 光伏目标 {cap_dict[(s, 'pv')]} MW "
                f"大于全省光伏潜力 {total_pv} MW，模型必不可行。"
            )
        if cap_dict[(s, "wind_offshore")] > total_offshore + 1e-9:
            raise ValueError(
                f"{province}: 部门 {s} 海上风电目标 {cap_dict[(s, 'wind_offshore')]} MW "
                f"大于全省海上风电潜力 {total_offshore} MW，模型必不可行。"
            )

    # -----------------------------
    # Generation-potential feasibility check
    # -----------------------------
    total_pot_gen_land_wind = sum(pot_land_wind[g] * 8760.0 * cf_land_wind[g] for g in land_grids)
    total_pot_gen_pv = sum(pot_pv[g] * 8760.0 * cf_pv[g] for g in land_grids)
    total_pot_gen_offshore = sum(pot_offshore_wind[h] * 8760.0 * cf_offshore_wind[h] for h in offshore_grids)

    for s in active_sectors:
        if gen_target_dict[(s, "wind_land")] > total_pot_gen_land_wind + 1e-6:
            raise ValueError(f"{province}: {s} 陆风发电目标超限")
        if gen_target_dict[(s, "pv")] > total_pot_gen_pv + 1e-6:
            print(gen_target_dict[(s, "pv")], total_pot_gen_pv)
            raise ValueError(f"{province}: {s} 光伏发电目标超限")
        if gen_target_dict[(s, "wind_offshore")] > total_pot_gen_offshore + 1e-6:
            raise ValueError(f"{province}: {s} 海风发电目标超限")

    return {
        "province": province,
        "sectors": active_sectors,
        "load_centers": load_centers,

        "land_grids": land_grids,
        "offshore_grids": offshore_grids,

        "pot_land_wind": pot_land_wind,
        "pot_pv": pot_pv,
        "pot_offshore_wind": pot_offshore_wind,

        "cf_land_wind": cf_land_wind,
        "cf_pv": cf_pv,
        "cf_offshore_wind": cf_offshore_wind,

        "lcoe_land_wind": lcoe_land_wind,
        "lcoe_pv": lcoe_pv,
        "lcoe_offshore_wind": lcoe_offshore_wind,

        "omega": omega,
        "omega_chem_export": omega_chem_export,
        "omega_power_import": omega_power_import,
        "chem_share_bounds": chem_share_bounds,

        "dist_land": dist_land,
        "dist_offshore": dist_offshore,
        "dist_ctp": dist_ctp,

        "cap_target": cap_dict,
        "gen_target": gen_target_dict,
        "ctp_mwh": ctp_mwh,
        "F_land": F_land,
        "F_offshore": F_offshore,
        "c_ene": c_ene,
        "c_ene_ctp": float(c_ene_ctp),
        "ctp_utilization": float(ctp_utilization),

        "grid_info": dg.copy(),
        "offshore_info": doff.copy(),
        "load_info": dl.copy(),
    }


# =========================================================
# Provincial model construction and solution
# =========================================================
def solve_one_province_model(prov_data: Dict) -> Tuple[gp.Model, Dict]:
    """
Solve the model for a single province
    """
    province = prov_data["province"]
    sectors = prov_data["sectors"]
    load_centers = prov_data["load_centers"]

    land_grids = prov_data["land_grids"]
    offshore_grids = prov_data["offshore_grids"]

    pot_land_wind = prov_data["pot_land_wind"]
    pot_pv = prov_data["pot_pv"]
    pot_offshore_wind = prov_data["pot_offshore_wind"]

    cf_land_wind = prov_data["cf_land_wind"]
    cf_pv = prov_data["cf_pv"]
    cf_offshore_wind = prov_data["cf_offshore_wind"]

    lcoe_land_wind = prov_data["lcoe_land_wind"]
    lcoe_pv = prov_data["lcoe_pv"]
    lcoe_offshore_wind = prov_data["lcoe_offshore_wind"]

    omega = prov_data["omega"]
    omega_chem_export = prov_data["omega_chem_export"]
    omega_power_import = prov_data["omega_power_import"]
    chem_share_bounds = prov_data["chem_share_bounds"]

    dist_land = prov_data["dist_land"]
    dist_offshore = prov_data["dist_offshore"]
    dist_ctp = prov_data["dist_ctp"]

    cap_target = prov_data["cap_target"]
    gen_target = prov_data["gen_target"]
    ctp_mwh = prov_data["ctp_mwh"]

    F_land = prov_data["F_land"]
    F_offshore = prov_data["F_offshore"]
    c_ene = prov_data["c_ene"]
    c_ene_ctp = prov_data["c_ene_ctp"]
    ctp_utilization = prov_data["ctp_utilization"]

    model = gp.Model(f"joint_siting_{province}")

    if MIP_GAP is not None:
        model.setParam("MIPGap", MIP_GAP)
    if TIME_LIMIT is not None:
        model.setParam("TimeLimit", TIME_LIMIT)
    if THREADS is not None:
        model.setParam("Threads", THREADS)

    # --------------------------------------------------
    # Variables
    # Onshore
    # --------------------------------------------------
    x_land_wind = model.addVars(
        [(g, s) for g in land_grids for s in sectors],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x_land_wind"
    )

    x_pv = model.addVars(
        [(g, s) for g in land_grids for s in sectors],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x_pv"
    )

    e_land_wind = model.addVars(
        [(g, n, s) for g in land_grids for s in sectors for n in load_centers[s]],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="e_land_wind"
    )

    e_pv = model.addVars(
        [(g, n, s) for g in land_grids for s in sectors for n in load_centers[s]],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="e_pv"
    )

    y_land = model.addVars(
        [(g, s) for g in land_grids for s in sectors],
        vtype=GRB.BINARY,
        name="y_land"
    )

    # --------------------------------------------------
    # Variables
    # Offshore
    # --------------------------------------------------
    x_offshore_wind = model.addVars(
        [(h, s) for h in offshore_grids for s in sectors],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="x_offshore_wind"
    )

    e_offshore_wind = model.addVars(
        [(h, n, s) for h in offshore_grids for s in sectors for n in load_centers[s]],
        lb=0.0,
        vtype=GRB.CONTINUOUS,
        name="e_offshore_wind"
    )

    y_offshore = model.addVars(
        [(h, s) for h in offshore_grids for s in sectors],
        vtype=GRB.BINARY,
        name="y_offshore"
    )

    # --------------------------------------------------
    # Variables
    # Chemical load center -> power load center
    # --------------------------------------------------
    has_ctp_network = ("chem" in sectors) and ("power" in sectors)
    if has_ctp_network:
        z_ctp = model.addVars(
            [(nc, np_) for nc in load_centers["chem"] for np_ in load_centers["power"]],
            lb=0.0,
            vtype=GRB.CONTINUOUS,
            name="z_ctp"
        )

        alpha_ctp = model.addVars(
            load_centers["chem"],
            lb=0.0,
            ub=1.0,
            vtype=GRB.CONTINUOUS,
            name="alpha_ctp"
        )
    else:
        z_ctp = {}
        alpha_ctp = {}

    # --------------------------------------------------
    # Objective function
    # --------------------------------------------------
    gen_cost_land_wind = gp.quicksum(
        lcoe_land_wind[g] * 8760.0 * cf_land_wind[g] * x_land_wind[g, s]
        for g in land_grids for s in sectors
    )

    gen_cost_pv = gp.quicksum(
        lcoe_pv[g] * 8760.0 * cf_pv[g] * x_pv[g, s]
        for g in land_grids for s in sectors
    )

    gen_cost_offshore_wind = gp.quicksum(
        lcoe_offshore_wind[h] * 8760.0 * cf_offshore_wind[h] * x_offshore_wind[h, s]
        for h in offshore_grids for s in sectors
    )

    trans_cost_land_wind = gp.quicksum(
        c_ene[s] / (cf_land_wind[g] + 1e-12) * dist_land[(g, s, n)] * e_land_wind[g, n, s]
        for g in land_grids for s in sectors for n in load_centers[s]
    )

    trans_cost_pv = gp.quicksum(
        c_ene[s] / (cf_pv[g] + 1e-12) * dist_land[(g, s, n)] * e_pv[g, n, s]
        for g in land_grids for s in sectors for n in load_centers[s]
    )

    trans_cost_offshore_wind = gp.quicksum(
        c_ene[s] / (cf_offshore_wind[h] + 1e-12) * dist_offshore[(h, s, n)] * e_offshore_wind[h, n, s]
        for h in offshore_grids for s in sectors for n in load_centers[s]
    )

    if has_ctp_network:
        trans_cost_ctp = gp.quicksum(
            c_ene_ctp / (ctp_utilization + 1e-12) * dist_ctp[(nc, np_)] * z_ctp[nc, np_]
            for nc in load_centers["chem"]
            for np_ in load_centers["power"]
        )
    else:
        trans_cost_ctp = gp.LinExpr(0.0)

    fixed_cost_land = gp.quicksum(
        F_land[(g, s)] * y_land[g, s]
        for g in land_grids for s in sectors
    )

    fixed_cost_offshore = gp.quicksum(
        F_offshore[(h, s)] * y_offshore[h, s]
        for h in offshore_grids for s in sectors
    )

    gen_cost = gen_cost_land_wind + gen_cost_pv + gen_cost_offshore_wind
    trans_cost = trans_cost_land_wind + trans_cost_pv + trans_cost_offshore_wind + trans_cost_ctp
    fixed_cost = fixed_cost_land + fixed_cost_offshore

    model.setObjective(gen_cost + trans_cost + fixed_cost, GRB.MINIMIZE)

    # --------------------------------------------------
    # Constraint 1: unique assignment of onshore grids
    # --------------------------------------------------
    if len(sectors) >= 2:
        for g in land_grids:
            model.addConstr(
                gp.quicksum(y_land[g, s] for s in sectors) <= 1,
                name=f"one_sector_land[{g}]"
            )

    # --------------------------------------------------
    # Constraint 2: unique assignment of offshore grids
    # --------------------------------------------------
    if len(sectors) >= 2:
        for h in offshore_grids:
            model.addConstr(
                gp.quicksum(y_offshore[h, s] for s in sectors) <= 1,
                name=f"one_sector_offshore[{h}]"
            )

    # --------------------------------------------------
    # Constraint 3: linkage between onshore capacity and assignment
    # --------------------------------------------------
    for g in land_grids:
        for s in sectors:
            model.addConstr(
                x_land_wind[g, s] <= pot_land_wind[g] * y_land[g, s],
                name=f"cap_link_land_wind[{g},{s}]"
            )
            model.addConstr(
                x_pv[g, s] <= pot_pv[g] * y_land[g, s],
                name=f"cap_link_pv[{g},{s}]"
            )

    # --------------------------------------------------
    # Constraint 4: linkage between offshore capacity and assignment
    # --------------------------------------------------
    for h in offshore_grids:
        for s in sectors:
            model.addConstr(
                x_offshore_wind[h, s] <= pot_offshore_wind[h] * y_offshore[h, s],
                name=f"cap_link_offshore_wind[{h},{s}]"
            )

    # --------------------------------------------------
    # Constraint 5: provincial capacity and generation targets
    # --------------------------------------------------
    for s in sectors:
        model.addConstr(
            gp.quicksum(x_land_wind[g, s] for g in land_grids) >= cap_target[(s, "wind_land")],
            name=f"cap_target_land_wind[{s}]"
        )
        model.addConstr(
            gp.quicksum(x_pv[g, s] for g in land_grids) >=cap_target[(s, "pv")],
            name=f"cap_target_pv[{s}]"
        )
        model.addConstr(
            gp.quicksum(x_offshore_wind[h, s] for h in offshore_grids) >= cap_target[(s, "wind_offshore")],
            name=f"cap_target_offshore_wind[{s}]"
        )

    for s in sectors:
        model.addConstr(
            gp.quicksum(8760.0 * cf_land_wind[g] * x_land_wind[g, s] for g in land_grids)
            >= gen_target[(s, "wind_land")],
            name=f"gen_target_land_wind[{s}]"
        )

        model.addConstr(
            gp.quicksum(8760.0 * cf_pv[g] * x_pv[g, s] for g in land_grids)
            >= gen_target[(s, "pv")],
            name=f"gen_target_pv[{s}]"
        )

        model.addConstr(
            gp.quicksum(8760.0 * cf_offshore_wind[h] * x_offshore_wind[h, s] for h in offshore_grids)
            >= gen_target[(s, "wind_offshore")],
            name=f"gen_target_offshore_wind[{s}]"
        )

        for g in land_grids:
            if cf_pv[g] < 0.1:
                model.addConstr(
                    x_pv[g, s] == 0,
                    name=f"cf_threshold_pv[{g},{s}]"
                )

    # --------------------------------------------------
    # Constraint 6: onshore wind generation allocation
    # --------------------------------------------------
    for g in land_grids:
        for s in sectors:
            model.addConstr(
                gp.quicksum(e_land_wind[g, n, s] for n in load_centers[s]) ==
                8760.0 * cf_land_wind[g] * x_land_wind[g, s],
                name=f"gen_alloc_land_wind[{g},{s}]"
            )

    # --------------------------------------------------
    # Constraint 7: PV generation allocation
    # --------------------------------------------------
    for g in land_grids:
        for s in sectors:
            model.addConstr(
                gp.quicksum(e_pv[g, n, s] for n in load_centers[s]) ==
                8760.0 * cf_pv[g] * x_pv[g, s],
                name=f"gen_alloc_pv[{g},{s}]"
            )

    # --------------------------------------------------
    # Constraint 8: offshore wind generation allocation
    # --------------------------------------------------
    for h in offshore_grids:
        for s in sectors:
            model.addConstr(
                gp.quicksum(e_offshore_wind[h, n, s] for n in load_centers[s]) ==
                8760.0 * cf_offshore_wind[h] * x_offshore_wind[h, s],
                name=f"gen_alloc_offshore_wind[{h},{s}]"
            )
    





    # --------------------------------------------------
    # Constraint 9: proportional absorption of total wind generation (onshore + offshore)
    # power still uses fixed omega
    # chem uses endogenous share alpha_ctp
    # --------------------------------------------------
    for s in sectors:
        total_wind_gen_s = gen_target[(s, "wind_offshore")]+gen_target[(s, "wind_land")]
    
        for n in load_centers[s]:
            if s == "chem" and has_ctp_network:
                share_expr = alpha_ctp[n]
            else:
                share_expr = omega[(s, n, "wind")]

            model.addConstr(
                gp.quicksum(e_land_wind[g, n, s] for g in land_grids)
                +
                gp.quicksum(e_offshore_wind[h, n, s] for h in offshore_grids)
                >=
                share_expr * total_wind_gen_s,
                name=f"share_alloc_wind_total[{s},{n}]"
            )

            model.addConstr(
                gp.quicksum(e_land_wind[g, n, s]/8760/ (cf_land_wind[g]+1e-12)  for g in land_grids)
                >=
                share_expr * cap_target[(s, "wind_land")],
                name=f"share_alloc_wind_land_cap[{s},{n}]"
            )
            model.addConstr(
                gp.quicksum(e_offshore_wind[h, n, s]/8760/ (cf_offshore_wind[h]+1e-12) for h in offshore_grids)
                >=
                share_expr *cap_target[(s, "wind_offshore")],
                name=f"share_alloc_wind_offshore_cap[{s},{n}]"
            )

        
        



    # --------------------------------------------------
    # Constraint 10: proportional absorption of PV generation
    # power still uses fixed omega
    # chem uses endogenous share alpha_ctp
    # --------------------------------------------------
    for s in sectors:
        total_pv_gen_s = gen_target[(s, "pv")]

        for n in load_centers[s]:
            if s == "chem" and has_ctp_network:
                share_expr = alpha_ctp[n]
            else:
                share_expr = omega[(s, n, "pv")]

            model.addConstr(
                gp.quicksum(e_pv[g, n, s] for g in land_grids)
                >=
                share_expr * total_pv_gen_s,
                name=f"share_alloc_pv[{s},{n}]"
            )
            model.addConstr(
                gp.quicksum(e_pv[g, n, s]/8760/ (cf_pv[g]+1e-12)  for g in land_grids)
                >=
                share_expr *cap_target[(s, "pv")],
                name=f"share_alloc_pv_cap[{s},{n}]"
            )


    # --------------------------------------------------
    # Constraint 11: annual chemical-to-power electricity transfer
    # --------------------------------------------------
    if has_ctp_network:
        model.addConstr(
            gp.quicksum(z_ctp[nc, np_] for nc in load_centers["chem"] for np_ in load_centers["power"]) == ctp_mwh,
            name="ctp_total_transfer"
        )

        # When chemical energy use = 0, fix share at its initial value to avoid meaningless variable drift
        if (gen_target[("chem", "pv")]<=1e-2) and (gen_target[("chem", "wind_land")]<=1e-2):
            for nc in load_centers["chem"]:
                model.addConstr(
                    alpha_ctp[nc] == omega_chem_export[nc],
                    name=f"alpha_ctp_fix_zero_transfer[{nc}]"
                )

        # Constraint 12: endogenize chemical-sector share, bounded at 10%-300% of the initial share
        for nc in load_centers["chem"]:
            lb_nc, ub_nc = chem_share_bounds[nc]
            model.addConstr(alpha_ctp[nc] >= lb_nc, name=f"alpha_ctp_lb[{nc}]")
            model.addConstr(alpha_ctp[nc] <= ub_nc, name=f"alpha_ctp_ub[{nc}]")

        # Constraint 13: optimized chemical-sector shares sum to 1
        model.addConstr(
                gp.quicksum(alpha_ctp[nc] for nc in load_centers["chem"]) == 1.0,
                name="alpha_ctp_sum_to_one"
            )

        # Constraint 13.1: maximum endogenous share cannot exceed the province's maximum original share
        max_initial_share = max(omega_chem_export[nc] for nc in load_centers["chem"])

        for nc in load_centers["chem"]:
            model.addConstr(
                alpha_ctp[nc] <= max_initial_share,
                name=f"alpha_ctp_max_initial_cap[{nc}]"
            )

        # Constraint 14: chemical load centers bear export volumes according to endogenous shares
        for nc in load_centers["chem"]:
            model.addConstr(
                gp.quicksum(z_ctp[nc, np_] for np_ in load_centers["power"]) ==
                alpha_ctp[nc] * ctp_mwh,
                name=f"ctp_export_share_endogenous[{nc}]"
            )

        # Constraint 15: power load centers receive imports according to fixed shares
        for np_ in load_centers["power"]:
            model.addConstr(
                gp.quicksum(z_ctp[nc, np_] for nc in load_centers["chem"]) ==
                omega_power_import[np_] * ctp_mwh,
                name=f"ctp_import_share[{np_}]"
            )
        # # --------------------------------------------------
        # # Added constraint: maximum chemical-to-power transmission distance of 500 km
        # # Connections longer than 500 km are not allowed
        # # --------------------------------------------------
        # MAX_CTP_DISTANCE_KM = 600.0

        # for nc in load_centers["chem"]:
        #     for np_ in load_centers["power"]:
        #         if dist_ctp[(nc, np_)] > MAX_CTP_DISTANCE_KM:
        #             model.addConstr(
        #                 z_ctp[nc, np_] == 0,
        #                 name=f"ctp_max_distance[{nc},{np_}]"
        #             )

    model.optimize()

    result = {
        "province": province,
        "status": model.Status,
        "objval": model.ObjVal if model.SolCount > 0 else None,

        "x_land_wind": x_land_wind,
        "x_pv": x_pv,
        "x_offshore_wind": x_offshore_wind,

        "e_land_wind": e_land_wind,
        "e_pv": e_pv,
        "e_offshore_wind": e_offshore_wind,

        "y_land": y_land,
        "y_offshore": y_offshore,

        "z_ctp": z_ctp,
        "alpha_ctp": alpha_ctp,

        "gen_cost_expr": gen_cost,
        "trans_cost_expr": trans_cost,
        "fixed_cost_expr": fixed_cost,
        "trans_cost_ctp_expr": trans_cost_ctp,
    }
    return model, result


# =========================================================
# Result extraction
# =========================================================
def extract_one_province_results(model: gp.Model, result: Dict, prov_data: Dict) -> Dict[str, pd.DataFrame]:
    """
Extract single-province results as DataFrames
    """
    province = prov_data["province"]
    sectors = prov_data["sectors"]
    load_centers = prov_data["load_centers"]

    land_grids = prov_data["land_grids"]
    offshore_grids = prov_data["offshore_grids"]

    cf_land_wind = prov_data["cf_land_wind"]
    cf_pv = prov_data["cf_pv"]
    cf_offshore_wind = prov_data["cf_offshore_wind"]

    dist_land = prov_data["dist_land"]
    dist_offshore = prov_data["dist_offshore"]
    dist_ctp = prov_data["dist_ctp"]
    ctp_mwh = prov_data["ctp_mwh"]

    omega_chem_export = prov_data["omega_chem_export"]
    chem_share_bounds = prov_data["chem_share_bounds"]

    x_land_wind = result["x_land_wind"]
    x_pv = result["x_pv"]
    x_offshore_wind = result["x_offshore_wind"]

    e_land_wind = result["e_land_wind"]
    e_pv = result["e_pv"]
    e_offshore_wind = result["e_offshore_wind"]

    y_land = result["y_land"]
    y_offshore = result["y_offshore"]
    z_ctp = result["z_ctp"]
    alpha_ctp = result["alpha_ctp"]

    if model.SolCount == 0:
        raise RuntimeError(f"{province}: 模型没有可行解，无法提取结果")

    # --------------------------------------------------
    # 1) Capacity results (onshore + offshore)
    # --------------------------------------------------
    rows_cap = []

    for g in land_grids:
        for s in sectors:
            val = x_land_wind[g, s].X
            if abs(val) > 1e-3:
                rows_cap.append({
                    "province": province,
                    "grid_type": "land",
                    "grid_id": g,
                    "sector": s,
                    "tech": "wind_land",
                    "capacity_mw": val,
                    "cf": cf_land_wind[g],
                    "annual_generation_mwh": 8760.0 * cf_land_wind[g] * val,
                })

    for g in land_grids:
        for s in sectors:
            val = x_pv[g, s].X
            if abs(val) > 1e-3:
                rows_cap.append({
                    "province": province,
                    "grid_type": "land",
                    "grid_id": g,
                    "sector": s,
                    "tech": "pv",
                    "capacity_mw": val,
                    "cf": cf_pv[g],
                    "annual_generation_mwh": 8760.0 * cf_pv[g] * val,
                })

    for h in offshore_grids:
        for s in sectors:
            val = x_offshore_wind[h, s].X
            if abs(val) > 1e-3:
                rows_cap.append({
                    "province": province,
                    "grid_type": "offshore",
                    "grid_id": h,
                    "sector": s,
                    "tech": "wind_offshore",
                    "capacity_mw": val,
                    "cf": cf_offshore_wind[h],
                    "annual_generation_mwh": 8760.0 * cf_offshore_wind[h] * val,
                })

    df_cap = pd.DataFrame(rows_cap)

    # --------------------------------------------------
    # 2) Grid-assignment results: onshore
    # --------------------------------------------------
    rows_assign_land = []
    for g in land_grids:
        yp = y_land[g, "power"].X if "power" in sectors else 0.0
        yc = y_land[g, "chem"].X if "chem" in sectors else 0.0

        if yp > 0.5 and yc < 0.5:
            assigned_sector = "power"
        elif yc > 0.5 and yp < 0.5:
            assigned_sector = "chem"
        elif yp < 0.5 and yc < 0.5:
            assigned_sector = "none"
        else:
            assigned_sector = "both_error"

        rows_assign_land.append({
            "province": province,
            "grid_type": "land",
            "grid_id": g,
            "y_power": yp,
            "y_chem": yc,
            "assigned_sector": assigned_sector,
        })
    df_assign_land = pd.DataFrame(rows_assign_land)

    # --------------------------------------------------
    # 3) Grid-assignment results: offshore
    # --------------------------------------------------
    rows_assign_offshore = []
    for h in offshore_grids:
        yp = y_offshore[h, "power"].X if "power" in sectors else 0.0
        yc = y_offshore[h, "chem"].X if "chem" in sectors else 0.0

        if yp > 0.5 and yc < 0.5:
            assigned_sector = "power"
        elif yc > 0.5 and yp < 0.5:
            assigned_sector = "chem"
        elif yp < 0.5 and yc < 0.5:
            assigned_sector = "none"
        else:
            assigned_sector = "both_error"

        rows_assign_offshore.append({
            "province": province,
            "grid_type": "offshore",
            "grid_id": h,
            "y_power": yp,
            "y_chem": yc,
            "assigned_sector": assigned_sector,
        })
    df_assign_offshore = pd.DataFrame(rows_assign_offshore)

    # --------------------------------------------------
    # 4) Generation-allocation results
    # --------------------------------------------------
    rows_alloc = []

    for g in land_grids:
        for s in sectors:
            for n in load_centers[s]:
                val = e_land_wind[g, n, s].X
                if abs(val) > 1e-2:
                    rows_alloc.append({
                        "province": province,
                        "grid_type": "land",
                        "grid_id": g,
                        "sector": s,
                        "load_center_id": n,
                        "tech": "wind_land",
                        "generation_mwh": val,
                        "distance_km": dist_land[(g, s, n)],
                    })

    for g in land_grids:
        for s in sectors:
            for n in load_centers[s]:
                val = e_pv[g, n, s].X
                if abs(val) > 1e-2:
                    rows_alloc.append({
                        "province": province,
                        "grid_type": "land",
                        "grid_id": g,
                        "sector": s,
                        "load_center_id": n,
                        "tech": "pv",
                        "generation_mwh": val,
                        "distance_km": dist_land[(g, s, n)],
                    })

    for h in offshore_grids:
        for s in sectors:
            for n in load_centers[s]:
                val = e_offshore_wind[h, n, s].X
                if abs(val) > 1e-2:
                    rows_alloc.append({
                        "province": province,
                        "grid_type": "offshore",
                        "grid_id": h,
                        "sector": s,
                        "load_center_id": n,
                        "tech": "wind_offshore",
                        "generation_mwh": val,
                        "distance_km": dist_offshore[(h, s, n)],
                    })

    df_alloc = pd.DataFrame(rows_alloc)

    # --------------------------------------------------
    # 5) Chemical-to-power transfer matching results
    # --------------------------------------------------
    rows_ctp = []
    if "chem" in sectors and "power" in sectors:
        for nc in load_centers["chem"]:
            for np_ in load_centers["power"]:
                val = z_ctp[nc, np_].X
                if abs(val) > 1e-2:
                    rows_ctp.append({
                        "province": province,
                        "chem_load_center_id": nc,
                        "power_load_center_id": np_,
                        "transfer_mwh": val,
                        "distance_km": dist_ctp[(nc, np_)],
                    })
    df_ctp_alloc = pd.DataFrame(rows_ctp)

    # --------------------------------------------------
    # 6) Chemical-to-power share adjustment results
    # --------------------------------------------------
    rows_share = []
    if "chem" in sectors and "power" in sectors:
        for nc in load_centers["chem"]:
            s0 = omega_chem_export[nc]
            lb_nc, ub_nc = chem_share_bounds[nc]
            s_opt = alpha_ctp[nc].X
            rows_share.append({
                "province": province,
                "chem_load_center_id": nc,
                "share_initial": s0,
                "share_optimized": s_opt,
                "share_lower_bound": lb_nc,
                "share_upper_bound": ub_nc,
            })
    df_ctp_share = pd.DataFrame(rows_share)

    # --------------------------------------------------
    # 7) Provincial summary results
    # --------------------------------------------------
    gen_cost_value = result["gen_cost_expr"].getValue()
    trans_cost_value = result["trans_cost_expr"].getValue()
    fixed_cost_value = result["fixed_cost_expr"].getValue()
    trans_cost_ctp_value = result["trans_cost_ctp_expr"].getValue()
    total_cost_value = model.ObjVal

    summary_rows = []
    for s in sectors:
        wind_land_cap = sum(x_land_wind[g, s].X for g in land_grids)
        pv_cap = sum(x_pv[g, s].X for g in land_grids)
        wind_offshore_cap = sum(x_offshore_wind[h, s].X for h in offshore_grids)

        wind_land_gen = sum(8760.0 * cf_land_wind[g] * x_land_wind[g, s].X for g in land_grids)
        pv_gen = sum(8760.0 * cf_pv[g] * x_pv[g, s].X for g in land_grids)
        wind_offshore_gen = sum(8760.0 * cf_offshore_wind[h] * x_offshore_wind[h, s].X for h in offshore_grids)

        active_land_grids = sum(1 for g in land_grids if y_land[g, s].X > 0.5)
        active_offshore_grids = sum(1 for h in offshore_grids if y_offshore[h, s].X > 0.5)

        summary_rows.append({
            "province": province,
            "sector": s,
            "wind_land_capacity_mw": wind_land_cap,
            "pv_capacity_mw": pv_cap,
            "wind_offshore_capacity_mw": wind_offshore_cap,
            "wind_total_capacity_mw": wind_land_cap + wind_offshore_cap,

            "wind_land_generation_mwh": wind_land_gen,
            "pv_generation_mwh": pv_gen,
            "wind_offshore_generation_mwh": wind_offshore_gen,
            "wind_total_generation_mwh": wind_land_gen + wind_offshore_gen,

            "active_land_grid_count": active_land_grids,
            "active_offshore_grid_count": active_offshore_grids,

            "generation_cost_total": gen_cost_value,
            "transmission_cost_total": trans_cost_value,
            "fixed_cost_total": fixed_cost_value,
            "objective_total": total_cost_value,
        })

    df_summary = pd.DataFrame(summary_rows)

    # --------------------------------------------------
    # 8) Cross-sector electricity-transfer summary results
    # --------------------------------------------------
    df_cross_summary = pd.DataFrame([{
        "province": province,
        "chem_to_power_transfer_mwh": ctp_mwh,
        "chem_to_power_transmission_cost": trans_cost_ctp_value,
    }])

    return {
        "grid_capacity": df_cap,
        "grid_assignment_land": df_assign_land,
        "grid_assignment_offshore": df_assign_offshore,
        "generation_allocation": df_alloc,
        "chem_to_power_allocation": df_ctp_alloc,
        "chem_to_power_share_adjusted": df_ctp_share,
        "province_summary": df_summary,
        "cross_sector_summary": df_cross_summary,
    }


# =========================================================
# Save results
# =========================================================
def save_one_province_results(province: str, results: Dict[str, pd.DataFrame], output_dir: str) -> None:
    """
Save single-province results
    """
    out_dir = Path(output_dir) / province
    ensure_dir(out_dir)

    for name, df in results.items():
        df.to_csv(out_dir / f"{name}.csv", index=False, encoding="utf-8-sig")

    xlsx_path = out_dir / f"{province}_joint_siting_results.xlsx"
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for name, df in results.items():
            df.to_excel(writer, sheet_name=name[:31], index=False)


# =========================================================
# Main workflow: solve province by province
# =========================================================
def run_all_provinces(
    input_xlsx: str,
    output_dir: str,
    fixed_cost_power: float = FIXED_COST_POWER,
    fixed_cost_chem: float = FIXED_COST_CHEM,
    c_ene_power: float = C_ENE_POWER,
    c_ene_chem: float = C_ENE_CHEM,
    c_ene_ctp: float = C_ENE_CTP,
    ctp_utilization: float = CTP_UTILIZATION,
    chem_share_lb_factor: float = CHEM_SHARE_LB_FACTOR,
    chem_share_ub_factor: float = CHEM_SHARE_UB_FACTOR,
    province_list: List[str] = None,
) -> None:
    """
Main function for solving province by province
    """
    ensure_dir(Path(output_dir))

    data = load_input_data(input_xlsx)

    validate_shares(data["share"])
    validate_cap_targets(data["cap"])
    validate_same_share_for_wind_pv(data["share"])

    all_provinces = sorted(data["grid"]["province"].astype(str).unique().tolist())

    if province_list is None:
        province_list = all_provinces
    else:
        province_list = [str(p) for p in province_list]

    print("待求解省份：")
    for p in province_list:
        print(f" - {p}")

    for province in province_list:
        print("=" * 80)
        print(f"开始求解省份: {province}")

        prov_data = build_province_data(
            province=province,
            data=data,
            fixed_cost_power=fixed_cost_power,
            fixed_cost_chem=fixed_cost_chem,
            c_ene_power=c_ene_power,
            c_ene_chem=c_ene_chem,
            c_ene_ctp=c_ene_ctp,
            ctp_utilization=ctp_utilization,
            chem_share_lb_factor=chem_share_lb_factor,
            chem_share_ub_factor=chem_share_ub_factor,
        )

        model, result = solve_one_province_model(prov_data)

        status = model.Status
        if status == GRB.OPTIMAL:
            print(f"{province}: 求解完成，达到最优。Obj = {model.ObjVal:,.2f}")
        elif status == GRB.TIME_LIMIT and model.SolCount > 0:
            print(f"{province}: 达到时限，但有可行解。当前 Obj = {model.ObjVal:,.2f}")
        elif status == GRB.INFEASIBLE:
            print(f"{province}: 模型不可行。")
            out_dir = Path(output_dir) / province
            ensure_dir(out_dir)
            model.computeIIS()
            model.write(str(out_dir / f"{province}_model.ilp"))
            raise RuntimeError(f"{province}: 模型不可行，请检查输入数据。")
        else:
            print(f"{province}: 求解状态码 = {status}")
            if model.SolCount == 0:
                raise RuntimeError(f"{province}: 无可行解，状态码 = {status}")

        prov_results = extract_one_province_results(model, result, prov_data)
        save_one_province_results(province, prov_results, output_dir)

        print(f"{province}: 结果已保存到 {Path(output_dir) / province}")

    print("=" * 80)
    print("全部省份求解完成。")


# %% [Notebook cell 3]


# =========================================================
# User parameters
# =========================================================
case='case2'
INPUT_XLSX = r"togrid/input_joint_siting_"+f"{case}.xlsx"
OUTPUT_DIR = r"togrid/outputs_joint_siting/"+f"{case}"




# =========================================================
# Example run
# =========================================================
if __name__ == "__main__":
    # To run only selected provinces, provide province_list
    # province_list = ["Anhui", "Shandong"]
    province_list =['Xinjiang']
    run_all_provinces(
        input_xlsx=INPUT_XLSX,
        output_dir=OUTPUT_DIR,
        fixed_cost_power=FIXED_COST_POWER,
        fixed_cost_chem=FIXED_COST_CHEM,
        c_ene_power=C_ENE_POWER,
        c_ene_chem=C_ENE_CHEM,
        province_list=province_list,
    )


# %% [Notebook cell 4]
import re
from pathlib import Path

import numpy as np
import pandas as pd

# =========================================================
# User parameters
# =========================================================
for case in  ["case2"]:

    province_list = ['Xinjiang']

    all_data_csv = Path(r"togrid/all_data.csv")
    result_root = Path(r"togrid/outputs_joint_siting") / case

    # suitable encoding: 0 = suitable
    SUIT_VALUE = 0

    # 1 km -> 50 km aggregation scale
    BLOCK_SIZE = 50

    # Extend the boundary outward by 10 km (approximated by the number of 1 km pixels)
    BUFFER_CELLS = 30

    # Output file names
    ALLOC_FILE_NAME = "grid_capacity_1km_allocated.csv"
    SUMMARY_FILE_NAME = "grid_capacity_1km_allocation_summary.csv"


    # =========================================================
    # Utility functions
    # =========================================================
    pattern_1km = re.compile(r"^r(\d+)_c(\d+)$")
    pattern_50km = re.compile(r"^R(\d+)_C(\d+)$")


    def parse_id_to_row_col(x: str):
        m = pattern_1km.match(str(x))
        if m is None:
            raise ValueError(f"id 格式不符合要求: {x}，应为 r{{row}}_c{{col}}")
        return int(m.group(1)), int(m.group(2))


    def parse_grid50_id(x: str):
        m = pattern_50km.match(str(x))
        if m is None:
            raise ValueError(f"50km grid_id 格式不符合要求: {x}，应为 R{{row}}_C{{col}}")
        return int(m.group(1)), int(m.group(2))


    def row_col_to_grid50_id(row: int, col: int, block_size: int = 50) -> str:
        grid50_row = (row - 1) // block_size + 1
        grid50_col = (col - 1) // block_size + 1
        return f"R{grid50_row}_C{grid50_col}"


    def clean_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")


    def normalize_tech(tech: str):
        tech = str(tech).strip().lower()
        if tech == "wind_land":
            return "wind"
        elif tech in {"pv", "solar"}:
            return "pv"
        elif tech == "wind":
            return "wind"
        return None


    def get_side_neighbors(grid50_id: str):
        """
Return four-neighbor adjacency:
        {
            "top": gid,
            "bottom": gid,
            "left": gid,
            "right": gid
        }
        """
        r, c = parse_grid50_id(grid50_id)
        neighbors = {}
        if r - 1 >= 1:
            neighbors["top"] = f"R{r-1}_C{c}"
        neighbors["bottom"] = f"R{r+1}_C{c}"
        if c - 1 >= 1:
            neighbors["left"] = f"R{r}_C{c-1}"
        neighbors["right"] = f"R{r}_C{c+1}"
        return neighbors


    def build_core_and_extension_indices(
        source_grid50_id: str,
        occupied_grids_global: set,
        grid_to_pixels: dict,
        df_1km: pd.DataFrame,
        buffer_cells: int = 10,
    ):
        """
For a given 50 km grid, construct:
        1) Core area: all 1 km pixels within the grid
        2) Extension area: 10 km strips extending outward from the four sides
        - Only four-neighbor adjacency is considered
        - If the adjacent 50 km grid in a direction is occupied by any installed capacity, disable the entire buffer on that side
        - Corner blocks are excluded

        Returns:
        core_idx: np.array
        ext_idx: np.array
        disabled_sides: list[str]
        """
        core_idx = grid_to_pixels.get(source_grid50_id, np.array([], dtype=int))

        r, c = parse_grid50_id(source_grid50_id)
        row_min = (r - 1) * BLOCK_SIZE + 1
        row_max = r * BLOCK_SIZE
        col_min = (c - 1) * BLOCK_SIZE + 1
        col_max = c * BLOCK_SIZE

        neighbors = get_side_neighbors(source_grid50_id)

        ext_idx_list = []
        disabled_sides = []

        for side, ngid in neighbors.items():
            # Disable the buffer on this side if the adjacent 50 km grid is occupied by any installed capacity
            if ngid in occupied_grids_global:
                disabled_sides.append(side)
                continue

            ng_idx = grid_to_pixels.get(ngid, np.array([], dtype=int))
            if len(ng_idx) == 0:
                continue

            sub = df_1km.loc[ng_idx, ["row", "col"]].copy()

            if side == "top":
                mask = (
                    (sub["row"] >= row_min - buffer_cells) &
                    (sub["row"] <= row_min - 1) &
                    (sub["col"] >= col_min) &
                    (sub["col"] <= col_max)
                )
            elif side == "bottom":
                mask = (
                    (sub["row"] >= row_max + 1) &
                    (sub["row"] <= row_max + buffer_cells) &
                    (sub["col"] >= col_min) &
                    (sub["col"] <= col_max)
                )
            elif side == "left":
                mask = (
                    (sub["row"] >= row_min) &
                    (sub["row"] <= row_max) &
                    (sub["col"] >= col_min - buffer_cells) &
                    (sub["col"] <= col_min - 1)
                )
            elif side == "right":
                mask = (
                    (sub["row"] >= row_min) &
                    (sub["row"] <= row_max) &
                    (sub["col"] >= col_max + 1) &
                    (sub["col"] <= col_max + buffer_cells)
                )
            else:
                mask = np.zeros(len(sub), dtype=bool)

            selected = sub.index[mask].to_numpy()
            if len(selected) > 0:
                ext_idx_list.append(selected)

        if len(ext_idx_list) > 0:
            ext_idx = np.unique(np.concatenate(ext_idx_list))
        else:
            ext_idx = np.array([], dtype=int)

        return core_idx, ext_idx, disabled_sides


    # =========================================================
    # 1. Read 1 km base data and recover the corresponding 50 km grid
    # =========================================================
    df_1km = pd.read_csv(all_data_csv)

    required_cols_1km = [
        "id", "lon", "lat",
        "lcoe_solar", "lcoe_wind",
        "solar_suitable", "wind_suitable",
        "solar_pot", "wind_pot",
    ]
    missing_1km = [c for c in required_cols_1km if c not in df_1km.columns]
    if missing_1km:
        raise ValueError(f"all_data.csv 缺少字段: {missing_1km}")

    row_col = df_1km["id"].map(parse_id_to_row_col)
    df_1km["row"] = row_col.map(lambda x: x[0])
    df_1km["col"] = row_col.map(lambda x: x[1])
    df_1km["host_grid_id_50km"] = [
        row_col_to_grid50_id(r, c, block_size=BLOCK_SIZE)
        for r, c in zip(df_1km["row"], df_1km["col"])
    ]

    # Convert numeric columns to numeric type
    for c in ["lcoe_solar", "lcoe_wind", "solar_pot", "wind_pot", "solar_suitable", "wind_suitable"]:
        df_1km[c] = clean_numeric(df_1km[c])

    # Prebuild the 1 km pixel index for each 50 km grid to accelerate subsequent lookup
    grid_to_pixels = {
        grid_id: sub.index.to_numpy()
        for grid_id, sub in df_1km.groupby("host_grid_id_50km")
    }


    # =========================================================
    # 2. Pre-read nationwide 50 km capacity results
    #    Used for:
    #    - global occupied_grids identification
    #    - global competition filtering in extension areas
    # =========================================================
    cap_frames = []

    for province in province_list:
        province_dir = result_root / province
        grid_capacity_file = province_dir / "grid_capacity.csv"
        if not grid_capacity_file.exists():
            continue

        df_cap_tmp = pd.read_csv(grid_capacity_file)
        required_cols_cap = ["province", "grid_type", "grid_id", "sector", "tech", "capacity_mw"]
        if any(c not in df_cap_tmp.columns for c in required_cols_cap):
            continue

        df_cap_tmp = df_cap_tmp[df_cap_tmp["grid_type"] == "land"].copy()
        if df_cap_tmp.empty:
            continue

        df_cap_tmp = df_cap_tmp[["province", "grid_id", "sector", "tech", "capacity_mw"]].copy()
        df_cap_tmp["capacity_mw"] = clean_numeric(df_cap_tmp["capacity_mw"])
        df_cap_tmp = df_cap_tmp[df_cap_tmp["capacity_mw"].notna()].copy()
        df_cap_tmp = df_cap_tmp[df_cap_tmp["capacity_mw"] > 0].copy()
        df_cap_tmp["tech_norm"] = df_cap_tmp["tech"].map(normalize_tech)
        df_cap_tmp = df_cap_tmp[df_cap_tmp["tech_norm"].notna()].copy()

        if not df_cap_tmp.empty:
            cap_frames.append(df_cap_tmp)

    if len(cap_frames) == 0:
        raise ValueError("全国没有读取到任何陆上 land 的正装机结果。")

    df_cap_all = pd.concat(cap_frames, ignore_index=True)

    # Treat a grid as occupied if any sector has positive capacity for any technology
    occupied_grids_global = set(df_cap_all["grid_id"].astype(str).unique().tolist())

    # Prebuild core and extension areas for each source 50 km grid (by sector)
    grid_sector_global = df_cap_all[["province", "grid_id", "sector"]].drop_duplicates().copy()

    core_map = {}             # key=(province, grid_id, sector) -> np.array
    ext_map = {}              # key=(province, grid_id, sector) -> set(idx)
    disabled_sides_map = {}   # key=(province, grid_id, sector) -> list[str]
    ext_before_count_map = {} # key=(province, grid_id, sector) -> int

    for _, gs in grid_sector_global.iterrows():
        province = str(gs["province"])
        gid = str(gs["grid_id"])
        sector = str(gs["sector"])

        core_idx, ext_idx, disabled_sides = build_core_and_extension_indices(
            source_grid50_id=gid,
            occupied_grids_global=occupied_grids_global,
            grid_to_pixels=grid_to_pixels,
            df_1km=df_1km,
            buffer_cells=BUFFER_CELLS,
        )

        key = (province, gid, sector)
        core_map[key] = core_idx
        ext_map[key] = set(ext_idx.tolist())
        disabled_sides_map[key] = disabled_sides
        ext_before_count_map[key] = len(ext_idx)

    # =========================================================
    # 3. Competition filtering: extension areas only
    #    If power and chem both extend into the same 1 km pixel in an extension area,
    #    neither can select it
    # =========================================================
    power_union = set()
    chem_union = set()

    for (province, gid, sector), s in ext_map.items():
        if sector == "power":
            power_union |= s
        elif sector == "chem":
            chem_union |= s

    competition_pixels = power_union & chem_union

    competition_removed_count_map = {}

    if len(competition_pixels) > 0:
        for key in list(ext_map.keys()):
            before = len(ext_map[key])
            ext_map[key] = ext_map[key] - competition_pixels
            after = len(ext_map[key])
            competition_removed_count_map[key] = before - after
    else:
        for key in list(ext_map.keys()):
            competition_removed_count_map[key] = 0


    # =========================================================
    # 4. Run the original downscaling algorithm province by province (candidate set = core area + filtered extension area)
    # =========================================================
    # Global residual potential to avoid repeated over-allocation of the same technology in the same 1 km pixel
    residual_map = {}
    for _, r in df_1km[["id", "solar_pot", "wind_pot"]].iterrows():
        pid = r["id"]
        wind_pot = float(r["wind_pot"]) if pd.notna(r["wind_pot"]) else 0.0
        solar_pot = float(r["solar_pot"]) if pd.notna(r["solar_pot"]) else 0.0
        residual_map[(pid, "wind")] = max(0.0, wind_pot)
        residual_map[(pid, "pv")] = max(0.0, solar_pot)

    for province in province_list:
        province_dir = result_root / province

        df_cap = df_cap_all[df_cap_all["province"].astype(str) == province].copy()

        if df_cap.empty:
            print(f"[提示] {province}: grid_type=land 后无数据")
            pd.DataFrame(columns=[
                "id", "lon", "lat", "grid_id_50km", "host_grid_id_50km", "sector", "tech", "allocated_capacity_mw"
            ]).to_csv(province_dir / ALLOC_FILE_NAME, index=False, encoding="utf-8-sig")

            pd.DataFrame(columns=[
                "province", "grid_id_50km", "sector", "tech",
                "target_capacity_mw", "allocated_capacity_mw", "unallocated_capacity_mw",
                "core_pixel_count", "extension_pixel_count", "competition_removed_count",
                "eligible_pixel_count", "eligible_pot_sum_mw", "disabled_buffer_sides"
            ]).to_csv(province_dir / SUMMARY_FILE_NAME, index=False, encoding="utf-8-sig")
            continue

        alloc_rows = []
        summary_rows = []

        # Fixed order to ensure reproducibility
        df_cap = df_cap.sort_values(["grid_id", "sector", "tech_norm"]).reset_index(drop=True)

        for _, row in df_cap.iterrows():
            grid50_id = str(row["grid_id"])
            sector = str(row["sector"])
            tech_norm = str(row["tech_norm"])
            target_capacity = float(row["capacity_mw"])

            key = (province, grid50_id, sector)

            core_idx = core_map.get(key, np.array([], dtype=int))
            ext_idx = np.array(sorted(list(ext_map.get(key, set()))), dtype=int)

            if len(core_idx) == 0 and len(ext_idx) == 0:
                summary_rows.append({
                    "province": province,
                    "grid_id_50km": grid50_id,
                    "sector": sector,
                    "tech": tech_norm,
                    "target_capacity_mw": target_capacity,
                    "allocated_capacity_mw": 0.0,
                    "unallocated_capacity_mw": target_capacity,
                    "core_pixel_count": 0,
                    "extension_pixel_count": 0,
                    "competition_removed_count": competition_removed_count_map.get(key, 0),
                    "eligible_pixel_count": 0,
                    "eligible_pot_sum_mw": 0.0,
                    "disabled_buffer_sides": "|".join(disabled_sides_map.get(key, [])),
                })
                continue

            candidate_idx = np.unique(np.concatenate([core_idx, ext_idx])) if len(ext_idx) > 0 else core_idx
            sub = df_1km.loc[candidate_idx].copy()

            # Select suitable / lcoe / pot columns by technology
            if tech_norm == "wind":
                suit_col = "wind_suitable"
                lcoe_col = "lcoe_wind"
                pot_col = "wind_pot"
            else:
                suit_col = "solar_suitable"
                lcoe_col = "lcoe_solar"
                pot_col = "solar_pot"

            # Eligible-pixel filtering
            sub = sub[
                (sub[suit_col] == SUIT_VALUE) &
                (sub[pot_col].notna()) &
                (sub[pot_col] > 0) &
                (sub[lcoe_col].notna()) &
                np.isfinite(sub[lcoe_col])
            ].copy()

            if sub.empty:
                summary_rows.append({
                    "province": province,
                    "grid_id_50km": grid50_id,
                    "sector": sector,
                    "tech": tech_norm,
                    "target_capacity_mw": target_capacity,
                    "allocated_capacity_mw": 0.0,
                    "unallocated_capacity_mw": target_capacity,
                    "core_pixel_count": int(len(core_idx)),
                    "extension_pixel_count": int(len(ext_idx)),
                    "competition_removed_count": competition_removed_count_map.get(key, 0),
                    "eligible_pixel_count": 0,
                    "eligible_pot_sum_mw": 0.0,
                    "disabled_buffer_sides": "|".join(disabled_sides_map.get(key, [])),
                })
                continue

            # Include current residual potential to avoid repeated over-allocation
            sub["available_pot"] = [
                residual_map.get((pid, tech_norm), 0.0)
                for pid in sub["id"]
            ]
            sub = sub[sub["available_pot"] > 1e-12].copy()

            if sub.empty:
                summary_rows.append({
                    "province": province,
                    "grid_id_50km": grid50_id,
                    "sector": sector,
                    "tech": tech_norm,
                    "target_capacity_mw": target_capacity,
                    "allocated_capacity_mw": 0.0,
                    "unallocated_capacity_mw": target_capacity,
                    "core_pixel_count": int(len(core_idx)),
                    "extension_pixel_count": int(len(ext_idx)),
                    "competition_removed_count": competition_removed_count_map.get(key, 0),
                    "eligible_pixel_count": 0,
                    "eligible_pot_sum_mw": 0.0,
                    "disabled_buffer_sides": "|".join(disabled_sides_map.get(key, [])),
                })
                continue

            # Sort by LCOE in ascending order
            sub = sub.sort_values([lcoe_col, "id"], ascending=[True, True]).copy()

            remaining = target_capacity
            allocated_total = 0.0
            eligible_pot_sum = float(sub["available_pot"].sum())

            for _, px in sub.iterrows():
                if remaining <= 1e-12:
                    break

                pid = px["id"]
                avail = residual_map.get((pid, tech_norm), 0.0)
                if avail <= 1e-12:
                    continue

                alloc = min(avail, remaining)

                if alloc > 1e-12:
                    alloc_rows.append({
                        "id": pid,
                        "lon": px["lon"],
                        "lat": px["lat"],
                        "grid_id_50km": grid50_id,                    # source 50 km grid
                        "host_grid_id_50km": px["host_grid_id_50km"], # physical host 50 km grid
                        "sector": sector,
                        "tech": tech_norm,
                        "allocated_capacity_mw": alloc,
                    })
                    allocated_total += alloc
                    remaining -= alloc
                    residual_map[(pid, tech_norm)] = max(0.0, avail - alloc)

            summary_rows.append({
                "province": province,
                "grid_id_50km": grid50_id,
                "sector": sector,
                "tech": tech_norm,
                "target_capacity_mw": target_capacity,
                "allocated_capacity_mw": allocated_total,
                "unallocated_capacity_mw": max(0.0, target_capacity - allocated_total),
                "core_pixel_count": int(len(core_idx)),
                "extension_pixel_count": int(len(ext_idx)),
                "competition_removed_count": competition_removed_count_map.get(key, 0),
                "eligible_pixel_count": int(len(sub)),
                "eligible_pot_sum_mw": eligible_pot_sum,
                "disabled_buffer_sides": "|".join(disabled_sides_map.get(key, [])),
            })

        # =====================================================
        # 5. Export provincial results
        # =====================================================
        df_alloc_out = pd.DataFrame(alloc_rows)
        if not df_alloc_out.empty:
            df_alloc_out = (
                df_alloc_out.groupby(
                    ["id", "lon", "lat", "grid_id_50km", "host_grid_id_50km", "sector", "tech"],
                    as_index=False
                )["allocated_capacity_mw"]
                .sum()
                .sort_values(["grid_id_50km", "sector", "tech", "id"])
                .reset_index(drop=True)
            )
        else:
            df_alloc_out = pd.DataFrame(columns=[
                "id", "lon", "lat", "grid_id_50km", "host_grid_id_50km", "sector", "tech", "allocated_capacity_mw"
            ])

        df_summary_out = pd.DataFrame(summary_rows)
        if not df_summary_out.empty:
            df_summary_out = df_summary_out.sort_values(
                ["grid_id_50km", "sector", "tech"]
            ).reset_index(drop=True)
        else:
            df_summary_out = pd.DataFrame(columns=[
                "province", "grid_id_50km", "sector", "tech",
                "target_capacity_mw", "allocated_capacity_mw", "unallocated_capacity_mw",
                "core_pixel_count", "extension_pixel_count", "competition_removed_count",
                "eligible_pixel_count", "eligible_pot_sum_mw", "disabled_buffer_sides"
            ])

        alloc_outfile = province_dir / ALLOC_FILE_NAME
        summary_outfile = province_dir / SUMMARY_FILE_NAME

        df_alloc_out.to_csv(alloc_outfile, index=False, encoding="utf-8-sig")
        df_summary_out.to_csv(summary_outfile, index=False, encoding="utf-8-sig")

        print(f"[完成] {province}")
        print(f"  - 1km 分配结果: {alloc_outfile}")
        print(f"  - 50km 回填汇总: {summary_outfile}")

