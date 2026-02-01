import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import geopandas as gpd
import torch
import torch.nn as nn
import statsmodels.api as sm
import plotly.graph_objects as go
import pandas_datareader.data as web
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.integrate import odeint
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.api import VARMAX, VAR
from shapely.geometry import Polygon, MultiPolygon
from shapely.affinity import scale, translate
from matplotlib.patches import Polygon as MplPolygon
from shapely.ops import unary_union
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from torch.utils.data import DataLoader, TensorDataset

# --- Global Configuration ---
BASE_DIR = r"G:\jupyter\2025APMCM"
TARIFF_DIR = os.path.join(BASE_DIR, "Tariff Data")
TRADE_DIR = os.path.join(BASE_DIR, "Trade Data")
MACRO_DIR = os.path.join(BASE_DIR, "US_Macroeconomic")
OUTPUT_DIR = os.path.join(BASE_DIR, "Cleaned_Data")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.size'] = 12
sns.set_theme(style="whitegrid", font="Times New Roman")

# ==============================================================================
# FILE 01: Tariff Data Cleaning & Index Construction
# ==============================================================================

SECTORS = {
    'Soybean': ['1201', '120190'],
    'Auto': ['8703'],
    'Chips': ['8541', '8542']
}


def clean_rate_column(series):
    s = series.astype(str).str.lower().str.strip()
    s = s.replace({'free': '0', 'nan': '0', 'nm': '0'})
    s = s.str.replace('%', '', regex=False)
    vals = pd.to_numeric(s, errors='coerce').fillna(0.0)
    return vals


def get_clean_rates(year):
    folder_path = os.path.join(TARIFF_DIR, f"tariff_data_{year}")
    file_xlsx = os.path.join(folder_path, f"tariff_database_{year}.xlsx")
    file_txt = os.path.join(folder_path, f"tariff_database_{year}.txt")

    df = None
    if os.path.exists(file_xlsx):
        df = pd.read_excel(file_xlsx, dtype={'hts8': str})
    elif os.path.exists(file_txt):
        sep = ',' if year == 2025 else '|'
        df = pd.read_csv(file_txt, sep=sep, encoding='latin1', dtype={'hts8': str}, on_bad_lines='skip')

    if df is not None:
        df.columns = [c.lower().strip() for c in df.columns]
        if 'mfn_ad_val_rate' in df.columns:
            vals = clean_rate_column(df['mfn_ad_val_rate'])
            return vals[vals < 10.0]
    return pd.Series(dtype=float)


years = range(2015, 2026)
yearly_means = []
data_2025 = None

for y in years:
    rates = get_clean_rates(y)
    if not rates.empty:
        yearly_means.append({'Year': y, 'Mean_Rate': rates.mean() * 100})
        if y == 2025:
            data_2025 = rates * 100

ts_df = pd.DataFrame(yearly_means)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
sns.lineplot(data=ts_df, x='Year', y='Mean_Rate', marker='o', markersize=8, ax=axes[0], linewidth=2.5, color='#2b579a')
axes[0].set_ylim(3.0, 5.0)
if data_2025 is not None:
    sns.histplot(data_2025, bins=60, ax=axes[1], color='#c44e52', stat='probability', kde=False)
    axes[1].set_xlim(0, 15)
    axes[1].axvline(x=10, color='k', linestyle='--', linewidth=2)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_1_Tariff_Baseline.png"), dpi=300)
plt.close()


def get_sector_name(hts_code):
    hts = str(hts_code)
    for sector, prefixes in SECTORS.items():
        for p in prefixes:
            if hts.startswith(p):
                return sector
    return None


all_data = []
for year in range(2015, 2026):
    folder_path = os.path.join(TARIFF_DIR, f"tariff_data_{year}")
    file_xlsx = os.path.join(folder_path, f"tariff_database_{year}.xlsx")
    file_txt = os.path.join(folder_path, f"tariff_database_{year}.txt")

    df = None
    if os.path.exists(file_xlsx):
        df = pd.read_excel(file_xlsx, dtype={'hts8': str})
    elif os.path.exists(file_txt):
        sep = ',' if year == 2025 else '|'
        df = pd.read_csv(file_txt, sep=sep, encoding='latin1', dtype={'hts8': str}, on_bad_lines='skip')

    if df is not None:
        df.columns = [c.lower().strip() for c in df.columns]
        if 'hts8' in df.columns and 'mfn_ad_val_rate' in df.columns:
            df['hts8'] = df['hts8'].astype(str).str.replace('.', '', regex=False).str.strip()
            df['Base_Rate'] = clean_rate_column(df['mfn_ad_val_rate'])
            df['Sector'] = df['hts8'].apply(get_sector_name)
            df_target = df.dropna(subset=['Sector']).copy()

            for sector in SECTORS.keys():
                sector_slice = df_target[df_target['Sector'] == sector].copy()
                sector_slice = sector_slice[sector_slice['Base_Rate'] < 10.0]
                if not sector_slice.empty:
                    X = sector_slice['Base_Rate'].values.reshape(-1, 1)
                    if len(X) >= 10:
                        clf = IsolationForest(random_state=42, contamination='auto', n_estimators=100)
                        preds = clf.fit_predict(X)
                        valid_mask = preds == 1
                        clean_slice = sector_slice[valid_mask].copy()
                    else:
                        clean_slice = sector_slice.copy()

                    clean_slice['Year'] = year
                    all_data.append(clean_slice[['Year', 'Sector', 'hts8', 'Base_Rate']])

if all_data:
    pd.concat(all_data, ignore_index=True).to_csv(os.path.join(OUTPUT_DIR, "Sector_Base_Tariffs.csv"), index=False)


def clean_hts(val):
    return str(val).replace('.', '').strip()


def get_trade_weights():
    files = ["US_Import_Cars_Chips_2015_2024.csv", "Soybean_Competition_Exports_to_China_2015_2024.csv"]
    dfs = []
    for f in files:
        p = os.path.join(TRADE_DIR, f)
        if os.path.exists(p):
            d = pd.read_csv(p, usecols=['refYear', 'cmdCode', 'primaryValue'])
            d.columns = ['Year', 'hts', 'Value']
            d['hts'] = d['hts'].apply(clean_hts)
            d['hts_4'] = d['hts'].str.slice(0, 4)
            dfs.append(d)
    if dfs:
        return pd.concat(dfs).groupby(['Year', 'hts_4'])['Value'].sum().reset_index()
    return pd.DataFrame()


weights = get_trade_weights()
results = []

for y in range(2015, 2026):
    folder = os.path.join(TARIFF_DIR, f"tariff_data_{y}")
    f_xlsx = os.path.join(folder, f"tariff_database_{y}.xlsx")
    f_txt = os.path.join(folder, f"tariff_database_{y}.txt")

    df = None
    if os.path.exists(f_xlsx):
        df = pd.read_excel(f_xlsx, dtype={'hts8': str})
    elif os.path.exists(f_txt):
        sep = ',' if y == 2025 else '|'
        df = pd.read_csv(f_txt, sep=sep, encoding='latin1', dtype={'hts8': str}, on_bad_lines='skip')

    if df is not None:
        df.columns = [c.lower().strip() for c in df.columns]
        if 'mfn_ad_val_rate' in df.columns:
            df['hts8'] = df['hts8'].apply(clean_hts)
            df['hts_4'] = df['hts8'].str.slice(0, 4)
            df['Rate'] = clean_rate_column(df['mfn_ad_val_rate'])
            df = df[df['Rate'] < 10.0]

            res = {'Year': y, 'General_Tau': df['Rate'].mean()}
            w_year = 2024 if y == 2025 else y
            t_w = weights[weights['Year'] == w_year]
            tariff_agg = df.groupby('hts_4')['Rate'].mean().reset_index()
            merged = pd.merge(tariff_agg, t_w, on='hts_4', how='inner')

            for sec, prefixes in SECTORS.items():
                p_list = [p[:4] for p in prefixes]
                sec_data = merged[merged['hts_4'].isin(p_list)]
                if not sec_data.empty and sec_data['Value'].sum() > 0:
                    res[f'{sec}_Tau'] = (sec_data['Rate'] * sec_data['Value']).sum() / sec_data['Value'].sum()
                else:
                    raw_match = df[df['hts_4'].isin(p_list)]
                    res[f'{sec}_Tau'] = raw_match['Rate'].mean() if not raw_match.empty else 0.0
            results.append(res)

final_df = pd.DataFrame(results)
if not final_df.empty:
    final_df['Simulated_Shock'] = np.where(final_df['Year'] == 2025, np.maximum(final_df['General_Tau'], 0.10),
                                           final_df['General_Tau'])
    final_df.to_csv(os.path.join(OUTPUT_DIR, "Annual_Tariff_Index.csv"), index=False)

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.axvspan(2015, 2017.5, color='green', alpha=0.05)
    ax1.axvspan(2017.5, 2024.5, color='orange', alpha=0.05)
    ax1.axvspan(2024.5, 2026, color='red', alpha=0.1)
    sns.lineplot(data=final_df, x='Year', y='General_Tau', ax=ax1, marker='o', color='gray', linestyle='--')
    shock_data = final_df[final_df['Year'] >= 2024].copy()
    shock_data.loc[shock_data['Year'] == 2024, 'Simulated_Shock'] = shock_data.loc[
        shock_data['Year'] == 2024, 'General_Tau']
    sns.lineplot(data=shock_data, x='Year', y='Simulated_Shock', ax=ax1, marker='s', markersize=8, linewidth=3,
                 color='#d62728')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_2_Tariff_Shock_Index.png"), dpi=300)
    plt.close()

# ==============================================================================
# FILE 02: Soybean Competition Model (Lotka-Volterra)
# ==============================================================================

TRADE_FILE = os.path.join(TRADE_DIR, "Soybean_Competition_Exports_to_China_2015_2024.csv")
df_raw = pd.read_csv(TRADE_FILE)

val_col = next((c for c in ['fobvalue', 'cifvalue', 'primaryValue'] if c in df_raw.columns and df_raw[c].sum() > 1e6),
               None)
year_col = next((c for c in ['refPeriodId', 'period', 'refYear'] if
                 c in df_raw.columns and str(df_raw[c].iloc[0]).startswith('20')), 'refPeriodId')

df = df_raw.copy()
df['Year'] = df[year_col].astype(str).str.slice(0, 4).astype(int)
country_map = {'USA': 'USA', 'United States': 'USA', 'Brazil': 'BRA', 'BRA': 'BRA', 'Argentina': 'ARG', 'ARG': 'ARG'}
df['Country'] = df['reporterISO'].map(country_map)
df = df.dropna(subset=['Country'])
df['Value_Billion'] = df[val_col] / 1e9
history_data = df.groupby(['Year', 'Country'])['Value_Billion'].sum().unstack(fill_value=0).reindex(
    range(2015, 2025)).fillna(0)


def lv_model(x, t, params):
    U, B, A = x
    r_u, r_b, r_a = params[0:3]
    a_ub, a_ua = params[3:5]
    a_bu, a_ba = params[5:7]
    a_au, a_ab = params[7:9]
    K = 160.0
    U, B, A = max(U, 0), max(B, 0), max(A, 0)
    dUdt = r_u * U * (1 - (U + a_ub * B + a_ua * A) / K)
    dBdt = r_b * B * (1 - (B + a_bu * U + a_ba * A) / K)
    dAdt = r_a * A * (1 - (A + a_au * U + a_ab * B) / K)
    return [dUdt, dBdt, dAdt]


def objective(params, t, data_true):
    x0 = data_true[0]
    try:
        x_pred = odeint(lv_model, x0, t, args=(params,))
        return np.mean((x_pred - data_true) ** 2)
    except:
        return 1e6


t_train = np.arange(len(history_data))
data_train = history_data[['USA', 'BRA', 'ARG']].values
initial_guess = [0.3, 0.5, 0.1, 0.8, 0.1, 0.2, 0.1, 0.1, 0.1]
bnds = [(0.05, 1.5), (0.05, 1.5), (0.01, 0.5)] + [(0, 2.0)] * 6

result = minimize(objective, initial_guess, args=(t_train, data_train), bounds=bnds, method='L-BFGS-B',
                  options={'maxiter': 1000})
lv_params = result.x


def calculate_shock_params(base_params, tariff_rate=0.20, substitution_elasticity=2.5):
    r_u, r_b, r_a, a_ub, a_ua, a_bu, a_ba, a_au, a_ab = base_params
    profit_shock = 1.0 / (1.0 + tariff_rate)
    r_u_new = r_u * profit_shock * 0.5
    shock_factor = np.log(1 + tariff_rate * substitution_elasticity)
    a_ub_new = a_ub + shock_factor * 0.5
    r_b_new = r_b * (1 + tariff_rate * 0.5)
    return [r_u_new, r_b_new, r_a, a_ub_new, a_ua, a_bu, a_ba, a_au, a_ab]


shock_params = calculate_shock_params(lv_params, tariff_rate=0.20)
future_years = np.arange(2024, 2031)
t_future = np.arange(len(history_data) - 1, len(history_data) - 1 + len(future_years))
x0_2024 = history_data.iloc[-1][['USA', 'BRA', 'ARG']].values

usa_trajectories = []
for _ in range(500):
    noise = np.random.normal(1, 0.2, len(shock_params))
    current_params = np.array(shock_params) * noise
    current_params[3] = max(current_params[3], 0.05)
    pred = odeint(lv_model, x0_2024, t_future, args=(tuple(current_params),))
    usa_trajectories.append(pred[:, 0])

usa_sims = np.array(usa_trajectories)
usa_mean = np.mean(usa_sims, axis=0)
usa_lower = np.percentile(usa_sims, 5, axis=0)
usa_upper = np.percentile(usa_sims, 95, axis=0)

plt.figure(figsize=(10, 6))
plt.plot(np.arange(2015, 2025), history_data['USA'], 'o-', color='black')
plt.plot(future_years, usa_mean, '--', color='#d62728')
plt.fill_between(future_years, usa_lower, usa_upper, color='#d62728', alpha=0.2)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4S_Sensitivity.png"), dpi=300)
plt.close()

pred_shock = odeint(lv_model, x0_2024, t_future, args=(tuple(shock_params),))
df_shock = pd.DataFrame(pred_shock, index=future_years, columns=['USA', 'BRA', 'ARG'])

fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')
ax.view_init(elev=25, azim=135)
ax.plot(history_data['USA'], history_data['BRA'], history_data['ARG'], color='gray', linestyle='--', marker='o')
ax.plot(df_shock['USA'], df_shock['BRA'], df_shock['ARG'], color='#d62728', linewidth=3.5, marker='^')
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_4_Phase_Portrait.png"), dpi=300)
plt.close()

try:
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world.to_crs(epsg=4326)
    target_countries = ['United States of America', 'Brazil', 'Argentina', 'China']
    filtered_world = world[world['name'].isin(target_countries)].copy()

    name_to_code = {'United States of America': 'USA', 'Brazil': 'BRA', 'Argentina': 'ARG', 'China': 'CHN'}
    country_geometries_transformed = {}
    country_centroids_display = {}
    layout_positions = {'USA': (0.25, 0.75), 'BRA': (0.25, 0.45), 'ARG': (0.25, 0.15), 'CHN': (0.75, 0.45)}

    for idx, row in filtered_world.iterrows():
        code = name_to_code.get(row['name'])
        if not code: continue
        geom = row.geometry
        if code == 'USA' and isinstance(geom, MultiPolygon):
            geom = max(geom.geoms, key=lambda p: p.area)

        minx, miny, maxx, maxy = geom.bounds
        w, h = maxx - minx, maxy - miny
        scale_factor = 0.45 / max(w, h) if max(w, h) > 0 else 1
        scaled_geom = scale(geom, xfact=scale_factor, yfact=scale_factor, origin='centroid')
        target_x, target_y = layout_positions[code]
        curr_x, curr_y = scaled_geom.centroid.x, scaled_geom.centroid.y
        display_geom = translate(scaled_geom, xoff=target_x - curr_x, yoff=target_y - curr_y)
        country_geometries_transformed[code] = display_geom
        country_centroids_display[code] = (target_x, target_y)

    fig, axes = plt.subplots(1, 3, figsize=(22, 10), constrained_layout=True)
    china_centroid = country_centroids_display.get('CHN', (0.75, 0.45))
    colors = {'USA': '#1f77b4', 'BRA': '#ff7f0e', 'ARG': '#2ca02c'}

    years_to_plot = [2024, 2025, 2030]
    for i, year in enumerate(years_to_plot):
        ax = axes[i]
        if year in df_shock.index:
            current_data = df_shock.loc[year]
        else:
            current_data = history_data.loc[year]

        ax.set_xlim(0, 1);
        ax.set_ylim(0, 1);
        ax.axis('off')
        for code, geom in country_geometries_transformed.items():
            if isinstance(geom, Polygon):
                polys = [geom]
            else:
                polys = geom.geoms
            for p in polys:
                ax.add_patch(MplPolygon(list(p.exterior.coords), closed=True, facecolor='#f0f0f0', edgecolor='#555555',
                                        linewidth=0.8))

        ax.scatter(china_centroid[0], china_centroid[1], marker='*', s=800, color='gold', edgecolor='black')
        for code in ['USA', 'BRA', 'ARG']:
            if code in country_centroids_display:
                val = current_data[code] if code in current_data else 0
                center = country_centroids_display[code]
                area = (val / 100) * 15000
                ax.scatter(center[0], center[1], s=max(area, 100), color=colors[code], alpha=0.8, edgecolor='black')
                ax.annotate('', xy=china_centroid, xytext=center,
                            arrowprops=dict(arrowstyle='->', color=colors[code], lw=1 + (val / 100) * 8,
                                            shrinkA=np.sqrt(area) / 2.5, shrinkB=20))

    plt.savefig(os.path.join(OUTPUT_DIR, "Soybean_Trade_LargeFont_AdjustedPad.png"), dpi=300)
    plt.close()
except Exception:
    pass

# ==============================================================================
# FILE 03: Supply Chain Transmission & VARX
# ==============================================================================

files_map = {
    'US_CPI_Total': 'US_CPI_2015_2025.csv',
    'Chip_PPI': 'US_PCU33443344_2015_2025.csv',
    'Auto_PPI': 'US_PCU336336_2015_2025.csv',
    'Auto_CPI': 'US_CUSR0000SETA01_2015_2025.csv',
    'JPY_Rate': 'US_DEXJPUS_2015_2025.csv'
}
dfs = []
for label, fname in files_map.items():
    fpath = os.path.join(MACRO_DIR, fname)
    if os.path.exists(fpath):
        df = pd.read_csv(fpath)
        date_col = [c for c in df.columns if 'date' in c.lower()][0]
        val_col = [c for c in df.columns if c != date_col][0]
        df = df[[date_col, val_col]].copy()
        df.columns = ['Date', label]
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        df[label] = pd.to_numeric(df[label], errors='coerce')
        df = df.resample('ME').mean().interpolate(method='linear')
        dfs.append(df)

if len(dfs) >= 4:
    df_final = pd.concat(dfs, axis=1).dropna()
    df_norm = df_final / df_final.loc['2016-01-31'] * 100
    df_norm = df_norm['2016-01-01':'2024-12-31']
    df_norm.to_csv(os.path.join(OUTPUT_DIR, "Supply_Chain_Dataset.csv"))

    plt.figure(figsize=(14, 7))
    plt.plot(df_norm.index, df_norm['Chip_PPI'], label='Upstream: Chip PPI')
    plt.plot(df_norm.index, df_norm['Auto_PPI'], label='Midstream: Auto Manuf. PPI')
    plt.plot(df_norm.index, df_norm['Auto_CPI'], label='Downstream: New Vehicle CPI')
    plt.plot(df_norm.index, df_norm['US_CPI_Total'], label='Macro: General CPI')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_5_Supply_Chain_Indices.png"), dpi=300)
    plt.close()

    df_stationary = pd.DataFrame()
    for col in df_norm.columns:
        series = df_norm[col].dropna()
        if adfuller(series.diff().dropna())[1] <= 0.05:
            df_stationary[col] = series.diff().dropna()
        else:
            df_stationary[col] = series.diff().diff().dropna()

    endog = df_stationary[['Auto_PPI', 'Auto_CPI']].dropna()
    exog = df_stationary[['Chip_PPI']].loc[endog.index]

    model = VARMAX(endog, order=(3, 0), exog=exog)
    results = model.fit(disp=False)
    exog_idx = results.model.exog_names.index('Chip_PPI')
    irf = results.impulse_responses(steps=24, impulse=exog_idx)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    irf['Auto_PPI'].plot(ax=axes[0]);
    axes[0].set_title('Response of Auto Manufacturing PPI')
    irf['Auto_CPI'].plot(ax=axes[1]);
    axes[1].set_title('Response of New Vehicle CPI')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_7_VARX_Impulse_Response.png"), dpi=300)
    plt.close()

    exog_hedge = df_stationary[['Chip_PPI', 'JPY_Rate']].loc[endog.index]
    model_hedge = VARMAX(endog, order=(3, 0), exog=exog_hedge)
    results_hedge = model_hedge.fit(disp=False)

    irf_chip = results_hedge.impulse_responses(steps=12, impulse=results_hedge.model.exog_names.index('Chip_PPI'))
    irf_jpy = results_hedge.impulse_responses(steps=12, impulse=results_hedge.model.exog_names.index('JPY_Rate'))
    hedge_ratio = -irf_chip['Auto_PPI'].max() / irf_jpy['Auto_PPI'].min()

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    irf_chip['Auto_PPI'].plot(ax=axes[0], color='crimson')
    irf_jpy['Auto_PPI'].plot(ax=axes[1], color='mediumseagreen')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_8_Hedging_Effect.png"), dpi=300)
    plt.close()

    avg_car_price_usd = 30000
    costs = {
        "Export": 30000 + (30000 * 0.05 * 0.25) + (30000 * 0.10),
        "FDI USA": 30000 * 1.15,
        "FDI Mexico": 30000 * 1.05 + 30000 * 0.03
    }
    best_strategy = min(costs, key=costs.get)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.bar(costs.keys(), costs.values(), color=['steelblue', 'lightgray', 'lightgray'])
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_9_Scenario_Comparison.png"), dpi=300)
    plt.close()

# ==============================================================================
# FILE 04: Laffer Curve & Hybrid Forecasting
# ==============================================================================

SOYBEAN_FILE = os.path.join(TRADE_DIR, "Soybean_Competition_Exports_to_China_2015_2024.csv")
AUTOS_CHIPS_FILE = os.path.join(TRADE_DIR, "US_Import_Cars_Chips_2015_2024.csv")
TARIFF_FILE = os.path.join(OUTPUT_DIR, "Annual_Tariff_Index.csv")
GDP_FILE = os.path.join(MACRO_DIR, "US_GDP_2015_2025.csv")

if os.path.exists(SOYBEAN_FILE) and os.path.exists(AUTOS_CHIPS_FILE):
    df_soy = pd.read_csv(SOYBEAN_FILE)
    us_exports = df_soy[df_soy['reporterISO'] == 'USA'].copy()
    us_exports['Year'] = pd.to_numeric(us_exports['refPeriodId'].astype(str).str[:4])
    soy_vol = us_exports.groupby('Year')['fobvalue'].sum().reset_index().rename(columns={'fobvalue': 'Soybean_Value'})

    df_ac = pd.read_csv(AUTOS_CHIPS_FILE)
    df_ac['Year'] = pd.to_numeric(df_ac['refPeriodId'].astype(str).str[:4])
    ac_vol = df_ac.groupby('Year')['primaryValue'].sum().reset_index().rename(
        columns={'primaryValue': 'AutoChip_Value'})

    df_vol = pd.merge(soy_vol, ac_vol, on='Year', how='outer').fillna(0)
    df_vol['Trade_Volume_Proxy'] = (df_vol['Soybean_Value'] + df_vol['AutoChip_Value']) / 1e9
    df_vol = df_vol[df_vol['Year'].between(2015, 2024)]

    df_tariffs = pd.read_csv(TARIFF_FILE, usecols=['Year', 'General_Tau'])
    df_gdp = pd.read_csv(GDP_FILE)
    date_col = [col for col in df_gdp.columns if 'date' in col.lower()][0]
    val_col = [col for col in df_gdp.columns if col != date_col][0]
    df_gdp['Year'] = pd.to_datetime(df_gdp[date_col]).dt.year
    annual_gdp = df_gdp.groupby('Year')[val_col].mean().reset_index().rename(columns={val_col: 'US_GDP'})

    model_data = pd.merge(df_vol, df_tariffs, on='Year')
    model_data = pd.merge(model_data, annual_gdp, on='Year').set_index('Year').sort_index()

    sarimax_model = sm.tsa.SARIMAX(endog=model_data['Trade_Volume_Proxy'], exog=model_data[['US_GDP']],
                                   order=(1, 0, 0)).fit(disp=False)

    future_years = range(2025, 2029)
    future_gdp = [model_data['US_GDP'].iloc[-1] * (1.02) ** i for i in range(1, len(future_years) + 1)]
    forecast_obj = sarimax_model.get_forecast(steps=len(future_years),
                                              exog=pd.DataFrame({'US_GDP': future_gdp}, index=future_years))
    forecast_df = pd.DataFrame({'Forecast': forecast_obj.predicted_mean.values,
                                'Lower_CI': forecast_obj.conf_int().iloc[:, 0].values,
                                'Upper_CI': forecast_obj.conf_int().iloc[:, 1].values}, index=future_years)

    plt.figure(figsize=(14, 8))
    plt.plot(model_data.index, model_data['Trade_Volume_Proxy'], marker='o', label='Historical')
    plt.plot(forecast_df.index, forecast_df['Forecast'], marker='s', linestyle='--', label='Forecast')
    plt.fill_between(forecast_df.index, forecast_df['Lower_CI'], forecast_df['Upper_CI'], alpha=0.3)
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_10_Baseline_Trade_Forecast_Professional.png"), dpi=300)
    plt.close()


    def predict_revenue_linear(base_volume, base_tariff, tariff_range, elasticity):
        results = []
        for new_tariff in tariff_range:
            price_change_pct = (new_tariff - base_tariff) / (1 + base_tariff)
            volume_change_pct = elasticity * price_change_pct
            new_volume = max(0, base_volume * (1 + volume_change_pct))
            results.append({'Tariff_Rate': new_tariff, 'Predicted_Revenue': new_volume * new_tariff})
        return pd.DataFrame(results)


    base_tariff_2024 = model_data['General_Tau'].iloc[-1] / 100.0
    base_volume_2025 = forecast_df.loc[2025, 'Forecast']
    tariff_schedule = np.linspace(0, 0.60, 200)

    plt.figure(figsize=(12, 8))
    for eps, col in zip([-0.5, -1.5, -2.5], ['green', 'navy', 'crimson']):
        sim_df = predict_revenue_linear(base_volume_2025, base_tariff_2024, tariff_schedule, eps)
        plt.plot(sim_df['Tariff_Rate'] * 100, sim_df['Predicted_Revenue'], color=col, label=f'Elasticity {eps}')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_11S_Laffer_Sensitivity.png"), dpi=300)
    plt.close()

    # LSTM / Hybrid Logic
    try:
        monthly_data = web.DataReader(['IMPGS', 'INDPRO'], 'fred', '2000-01-01', '2024-12-31').ffill()
        monthly_data.columns = ['Trade_Volume', 'GDP_Proxy']
    except:
        date_rng = pd.date_range('2000-01-01', '2024-12-31', freq='MS')
        monthly_data = pd.DataFrame(
            {'Trade_Volume': np.random.randn(len(date_rng)) + 3000, 'GDP_Proxy': np.random.randn(len(date_rng)) + 100},
            index=date_rng)


    class LSTMModel(nn.Module):
        def __init__(self, input_size=1, hidden_layer_size=100, output_size=1, dropout_rate=0.2):
            super().__init__()
            self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True, dropout=dropout_rate)
            self.linear = nn.Linear(hidden_layer_size, output_size)

        def forward(self, input_seq):
            lstm_out, _ = self.lstm(input_seq)
            return self.linear(lstm_out[:, -1, :])


    def create_sequences(input_data, tw):
        inout_seq = []
        for i in range(len(input_data) - tw):
            inout_seq.append((input_data[i:i + tw], input_data[i + tw:i + tw + 1]))
        return inout_seq


    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_diff = monthly_data[['Trade_Volume']].diff().dropna()
    scaled_data_diff = scaler.fit_transform(data_diff)
    sequences = create_sequences(scaled_data_diff, 12)

    X_train = torch.FloatTensor([seq[0] for seq in sequences])
    y_train = torch.FloatTensor([seq[1] for seq in sequences])
    model = LSTMModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()

    for epoch in range(50):
        model.train()
        y_pred = model(X_train)
        loss = loss_fn(y_pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Hybrid Model
    sarimax_monthly = sm.tsa.SARIMAX(monthly_data['Trade_Volume'], exog=monthly_data[['GDP_Proxy']],
                                     order=(1, 1, 1)).fit(disp=False)
    residuals = sarimax_monthly.resid.dropna()
    scaled_resid = scaler.fit_transform(residuals.values.reshape(-1, 1))
    resid_seq = create_sequences(scaled_resid, 12)
    X_resid = torch.FloatTensor([s[0] for s in resid_seq])
    y_resid = torch.FloatTensor([s[1] for s in resid_seq])

    model_resid = LSTMModel(hidden_layer_size=50)
    optimizer_resid = torch.optim.Adam(model_resid.parameters(), lr=0.001)
    for epoch in range(50):
        model_resid.train()
        y_p = model_resid(X_resid)
        l = loss_fn(y_p, y_resid)
        optimizer_resid.zero_grad()
        l.backward()
        optimizer_resid.step()


# ==============================================================================
# FILE 05: Game Theory & Reshoring Index
# ==============================================================================

def calculate_utility(player, strategy_us, strategy_cn):
    base_loss = -0.16 * 100 - 0.18 * 100
    if player == 'US':
        if strategy_us == 'Impose Tariff':
            val = 15 + base_loss
            return val - 15 if strategy_cn == 'Retaliate' else val
        return 0
    elif player == 'China':
        if strategy_us == 'Impose Tariff':
            val = -20
            return val + 5 if strategy_cn == 'Retaliate' else val - 5
        return 0


payoff_t_r = (
calculate_utility('US', 'Impose Tariff', 'Retaliate'), calculate_utility('China', 'Impose Tariff', 'Retaliate'))
payoff_t_nr = (
calculate_utility('US', 'Impose Tariff', 'No Retaliate'), calculate_utility('China', 'Impose Tariff', 'No Retaliate'))
payoff_nt_nr = (0, 0)
payoff_nt_r = (0, -5)

val_matrix = np.array([[sum(payoff_t_r), sum(payoff_t_nr)], [sum(payoff_nt_r), sum(payoff_nt_nr)]])
text_matrix = np.array([[f"{payoff_t_r}", f"{payoff_t_nr}"], [f"{payoff_nt_r}", f"{payoff_nt_nr}"]])

plt.figure(figsize=(10, 8))
sns.heatmap(val_matrix, annot=text_matrix, fmt='', cmap='coolwarm', vmin=-60, vmax=5)
plt.savefig(os.path.join(OUTPUT_DIR, "Figure_12_Payoff_Matrix_Final.png"), dpi=300)
plt.close()

try:
    df_fred = web.DataReader(['INDPRO', 'PAYEMS', 'IMPGS', 'BOPGSTB'], 'fred', '2015-01-01', '2024-12-31').ffill()
    scaler_idx = MinMaxScaler(feature_range=(0.01, 1))
    df_norm = pd.DataFrame(scaler_idx.fit_transform(df_fred), index=df_fred.index, columns=df_fred.columns)
    df_processed = df_norm.copy()
    df_processed['IMPGS'] = 1 - df_norm['IMPGS']


    def get_entropy_weights(data):
        P = data / data.sum(axis=0)
        E = - (1 / np.log(len(data))) * (P * np.log(P)).sum(axis=0)
        D = 1 - E
        return D / D.sum()


    w = get_entropy_weights(df_processed)
    df_fred['Reshoring_Index'] = (df_processed * w).sum(axis=1) * 100

    plt.figure(figsize=(16, 8))
    plt.plot(df_fred.index, df_fred['Reshoring_Index'], label='Reshoring Index')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_13_Reshoring_Index_Fixed_Final.png"), dpi=300)
    plt.close()

    df_contrib = pd.DataFrame(index=df_processed.index)
    for col in df_processed.columns:
        df_contrib[col] = df_processed[col] * w[col]

    plt.figure(figsize=(18, 10))
    plt.stackplot(df_contrib.index, df_contrib.T, labels=df_contrib.columns)
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_14_Index_Decomposition_Final.png"), dpi=300)
    plt.close()
except:
    pass

try:
    df_high = pd.read_csv(os.path.join(MACRO_DIR, "Chip_High_IC.csv"), index_col=0, parse_dates=True)
    df_low = pd.read_csv(os.path.join(MACRO_DIR, "Chip_Low_Discrete.csv"), index_col=0, parse_dates=True)
    df_prod = pd.read_csv(os.path.join(MACRO_DIR, "Chip_Production.csv"), index_col=0, parse_dates=True)
    df_chips = pd.concat([df_high, df_low, df_prod], axis=1).dropna()
    df_index = df_chips / df_chips.iloc[0] * 100

    fig, ax1 = plt.subplots(figsize=(12, 7))
    ax1.plot(df_index.index, df_index.iloc[:, 0], color='red', label='High End')
    ax1.plot(df_index.index, df_index.iloc[:, 1], color='blue', linestyle='--', label='Low End')
    ax2 = ax1.twinx()
    ax2.plot(df_index.index, df_index.iloc[:, 2], color='black', alpha=0.4, label='Capacity')
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_16_Real_Chip_Analysis.png"), dpi=300)
    plt.close()
except:
    pass

# ==============================================================================
# FILE 06: Final Validation & Visualization (Regime Shift & Sankey & VAR)
# ==============================================================================

if 'history_data' in locals():
    pre_war = history_data.loc[2015:2017]
    trade_war = history_data.loc[2018:2024]

    res_pre = minimize(objective, initial_guess, args=(np.arange(len(pre_war)), pre_war.values), bounds=bnds,
                       method='L-BFGS-B')
    res_war = minimize(objective, initial_guess, args=(np.arange(len(trade_war)), trade_war.values), bounds=bnds,
                       method='L-BFGS-B')

    labels_2017 = ["USA", "Brazil", "Argentina", "China"]
    labels_2018 = labels_2017
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    fig_sankey = make_subplots(rows=1, cols=2, specs=[[{'type': 'domain'}, {'type': 'domain'}]])
    if 2017 in history_data.index and 2018 in history_data.index:
        d17 = history_data.loc[2017]
        d18 = history_data.loc[2018]
        fig_sankey.add_trace(go.Sankey(node=dict(label=labels_2017, color=colors),
                                       link=dict(source=[0, 1, 2], target=[3, 3, 3], value=d17.values)), 1, 1)
        fig_sankey.add_trace(go.Sankey(node=dict(label=labels_2018, color=colors),
                                       link=dict(source=[0, 1, 2], target=[3, 3, 3], value=d18.values)), 1, 2)
        fig_sankey.write_html(os.path.join(OUTPUT_DIR, "Figure_17_Sankey_Diagram.html"))

if 'df_gdp' in locals() and 'df_cpi' in locals() and 'df_tariffs' in locals():
    df_macro = pd.merge(df_tariffs[['Year', 'General_Tau']], annual_gdp, on='Year')
    df_macro = pd.merge(df_macro, pd.read_csv(os.path.join(MACRO_DIR, "US_CPI_2015_2025.csv")).assign(
        Year=lambda x: pd.to_datetime(x.iloc[:, 0]).dt.year).groupby('Year').mean().reset_index(), on='Year').set_index(
        'Year')

    df_pct = df_macro.pct_change().dropna()
    model_var = VAR(df_pct)
    res_var = model_var.fit(1)
    irf = res_var.irf(5)

    fig, axes = plt.subplots(2, 1, figsize=(10, 12), sharex=True)
    irf.plot(orth=False, impulse='General_Tau', response='US_GDP', ax=axes[0])
    irf.plot(orth=False, impulse='General_Tau', response=df_pct.columns[-1], ax=axes[1])  # CPI
    plt.savefig(os.path.join(OUTPUT_DIR, "Figure_18_Macro_IRF.png"), dpi=300)
    plt.close()