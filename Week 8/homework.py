# Question 2: Two advantages of using an API over web scraping are that it delivers structured, easily parsable data and operates within officially supported terms of service, reducing legal and ethical risks. However, researchers face limitations such as strict rate limits that restrict data collection volume, and the risk of changing endpoints or API version deprecation, which can break collection pipelines over time. To document API data provenance for replication, researchers should record the exact API version, the specific endpoints and query parameters used, the exact date and time of data extraction, and provide the scripts used to pull the data.

import os
from dotenv import load_dotenv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
from fredapi import Fred
import pyreadr
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error

vote_data = pd.read_csv("Week 8/votes/1976-2020-president.csv")
vote_data = vote_data[vote_data["party_detailed"].isin(["DEMOCRAT", "REPUBLICAN"])].copy()
vote_data = (vote_data.groupby(["year", "candidate", "party_detailed"], as_index=False).agg(candidatevotes=("candidatevotes", "sum"),totalvotes=("totalvotes", "sum")))
vote_data = vote_data[(~vote_data["candidate"].isin(["OTHER", ""])) &(vote_data["candidate"].notna())].copy()
vote_data["vote_pct"] = vote_data["candidatevotes"] / vote_data["totalvotes"]
election_years = np.sort(vote_data["year"].unique())

load_dotenv("Week 8/.env")
fred_api_key = os.getenv("API_KEY")
fred = Fred(api_key=fred_api_key)

obs_start = f"{int(election_years.min())}-01-01"
obs_end   = f"{int(election_years.max())}-06-30"

unrate = fred.get_series("UNRATE", observation_start=obs_start, observation_end=obs_end)
unrate = unrate.to_frame(name="unemployment_rate")
unrate.index = pd.to_datetime(unrate.index)
unrate = unrate.resample("Q").mean().reset_index().rename(columns={"index": "date"})
unrate["year"] = unrate["date"].dt.year
unrate["quarter"] = unrate["date"].dt.quarter
unemployment_data = unrate[(unrate["year"].isin(election_years)) & (unrate["quarter"] <= 2)][["year", "quarter", "unemployment_rate"]].copy()

gdp = fred.get_series("GDP", observation_start=obs_start, observation_end=obs_end)
gdp = gdp.to_frame(name="gdp")
gdp.index = pd.to_datetime(gdp.index)
gdp = gdp.resample("Q").mean().reset_index().rename(columns={"index": "date"})
gdp["year"] = gdp["date"].dt.year
gdp["quarter"] = gdp["date"].dt.quarter
gdp_data = gdp[(gdp["year"].isin(election_years)) & (gdp["quarter"] <= 2)][["year", "quarter", "gdp"]].copy()

cpi = fred.get_series("CPIAUCSL", observation_start=obs_start, observation_end=obs_end)
cpi = cpi.to_frame(name="cpi")
cpi.index = pd.to_datetime(cpi.index)
cpi = cpi.resample("Q").mean().reset_index().rename(columns={"index": "date"})
cpi["year"] = cpi["date"].dt.year
cpi["quarter"] = cpi["date"].dt.quarter
cpi_data = cpi[(cpi["year"].isin(election_years)) &(cpi["quarter"] <= 2)][["year", "quarter", "cpi"]].copy()

inflation_data = cpi_data.sort_values(["year", "quarter"]).copy()
inflation_data["inflation_rate"] = ((inflation_data["cpi"] / inflation_data["cpi"].shift(2) - 1) * 100)

combined_long = (unemployment_data.merge(gdp_data, on=["year", "quarter"], how="outer").merge(inflation_data[["year", "quarter", "cpi"]], on=["year", "quarter"], how="outer").sort_values(["year", "quarter"]))

combined_wide = combined_long.pivot_table(index="year", columns="quarter",values=["unemployment_rate", "gdp", "cpi"], aggfunc="first")
combined_wide.columns = [f"{var}_Q{q}" for var, q in combined_wide.columns]
combined_wide = combined_wide.reset_index()
forecast_data = vote_data.merge(combined_wide, on="year", how="left").copy()

forecast_data["incumbent"] = 0
forecast_data.loc[(forecast_data["candidate"] == "FORD, GERALD") & (forecast_data["year"] == 1976), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CARTER, JIMMY") & (forecast_data["year"] == 1980), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "REAGAN, RONALD") & (forecast_data["year"] == 1984), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE H.W.") & (forecast_data["year"] == 1992), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "CLINTON, BILL") & (forecast_data["year"] == 1996), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "BUSH, GEORGE W.") & (forecast_data["year"] == 2004), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "OBAMA, BARACK H.") & (forecast_data["year"] == 2012), "incumbent"] = 1
forecast_data.loc[(forecast_data["candidate"] == "TRUMP, DONALD J.") & (forecast_data["year"] == 2020), "incumbent"] = 1
forecast_data["gdp_change"] = forecast_data["gdp_Q2"] - forecast_data["gdp_Q1"]
forecast_data["cpi_change"] = forecast_data["cpi_Q2"] - forecast_data["cpi_Q1"]
forecast_data["unemploy_change"] = forecast_data["unemployment_rate_Q2"] - forecast_data["unemployment_rate_Q1"]
poll_census_obj = pyreadr.read_r("Week 8/poll_census_data.rds")
poll_census_data = list(poll_census_obj.values())[0]
forecast_econ = forecast_data[["year", "gdp_change", "cpi_change", "unemploy_change"]].drop_duplicates()
state_data = poll_census_data.merge(forecast_econ, on="year", how="left")

# Q6 and Q7

dspic = fred.get_series("DSPIC96", observation_start=obs_start, observation_end=obs_end)
dspic = dspic.to_frame(name="dspic")
dspic.index = pd.to_datetime(dspic.index)
dspic = dspic.resample("Q").mean().reset_index().rename(columns={"index": "date"})
dspic["year"] = dspic["date"].dt.year
dspic["quarter"] = dspic["date"].dt.quarter

dspic_data = dspic[(dspic["year"].isin(election_years)) & (dspic["quarter"] <= 2)][["year", "quarter", "dspic"]].copy()
dspic_wide = dspic_data.pivot_table(index="year", columns="quarter", values="dspic").reset_index()
dspic_wide.columns = ["year", "dspic_Q1", "dspic_Q2"]
dspic_wide["dspic_change"] = dspic_wide["dspic_Q2"] - dspic_wide["dspic_Q1"]
state_data_improved = state_data.merge(dspic_wide[["year", "dspic_change"]], on="year", how="left")
model_features = ["poll_avg", "year", "party_simplified", "white", "black", "asian", "hispanic", "gdp_change", "cpi_change", "unemploy_change", "dspic_change"]
state_data_clean = state_data_improved.dropna(subset=model_features + ["vote_pct"]).copy()
state_data_encoded = pd.get_dummies(state_data_clean, columns=["party_simplified"], drop_first=True)
features_encoded = [col for col in state_data_encoded.columns if col in model_features or col.startswith("party_simplified_")]
train_df = state_data_encoded[state_data_encoded["year"] < 2020]
test_df = state_data_encoded[state_data_encoded["year"] == 2020]

X_train = train_df[features_encoded]
y_train = train_df["vote_pct"]
X_test = test_df[features_encoded]
y_test = test_df["vote_pct"]

rf_model = RandomForestRegressor(n_estimators=100, random_state=19880106)
rf_model.fit(X_train, y_train)

test_df_results = test_df.copy()
test_df_results["rf_pred"] = rf_model.predict(X_test)

baseline_ols = smf.ols("vote_pct ~ poll_avg + year + C(party_simplified) + white + black + asian + hispanic",data=state_data_clean[state_data_clean["year"] < 2020]).fit()
test_df_results["baseline_pred"] = baseline_ols.predict(state_data_clean[state_data_clean["year"] == 2020])

metrics = {"Model": ["Baseline OLS", "Improved Random Forest"],
    "MAE": [mean_absolute_error(test_df_results["vote_pct"], test_df_results["baseline_pred"]),
        mean_absolute_error(test_df_results["vote_pct"], test_df_results["rf_pred"])],
    "RMSE": [np.sqrt(mean_squared_error(test_df_results["vote_pct"], test_df_results["baseline_pred"])),
        np.sqrt(mean_squared_error(test_df_results["vote_pct"], test_df_results["rf_pred"]))]}

metrics_df = pd.DataFrame(metrics)
print(metrics_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.scatter(test_df_results["vote_pct"], test_df_results["baseline_pred"], alpha=0.6, label="Baseline OLS", color="blue", marker="o")
plt.scatter(test_df_results["vote_pct"], test_df_results["rf_pred"],alpha=0.6, label="Random Forest", color="orange", marker="s")
min_val = min(test_df_results["vote_pct"].min(), test_df_results["baseline_pred"].min())
max_val = max(test_df_results["vote_pct"].max(), test_df_results["baseline_pred"].max())
plt.plot([min_val, max_val], [min_val, max_val], 'k--', label="Perfect Prediction")
plt.title("2020 Out-of-Sample Forecast: Predicted vs. Actual Vote Share")
plt.xlabel("Actual Vote Share")
plt.ylabel("Predicted Vote Share")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("Week 8/Figure_1.png", dpi=300)
plt.show()


#To improve the out-of-sample forecaster, I utilized a hold out evaluation design by training the model on pre-2020 elections and testing it on the 2020 election year. I implemented two specific model improvements: incorporating Real Disposable Personal Income via the FRED API to capture pocketbook voting dynamics, and switching the model family from Ordinary Least Squares to a Random Forest Regressor to account for non-linear interactions among state level demographics and economic indicators. The comparative performance is documented in the accompanying metrics table and predicted vs actual scatter plot. Contrary to my expectations, the baseline OLS model outperformed the Random Forest out-of-sample, achieving both a lower Mean Absolute Error, 0.0176 vs 0.0179, and Root Mean Squared Error, 0.0225 vs. 0.0258. As i illustrated in the scatter plot, this performance gap is primarily driven by the tree-based model's inability to estimate beyond its training data; the Random Forest under-predicted the most extreme actual vote shares, those above 0.9, whereas the linear OLS model successfully captured these outliers.

