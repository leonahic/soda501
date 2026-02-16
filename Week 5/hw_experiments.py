#The ITT estimand measures the average difference in outcomes between all users assigned to the treatment group and those assigned to the control group, regardless of whether the treatment was successfully delivered or if the users engaged with it. ITT is the default because it captures net impact of shipping a feature, accounting for operational friction like delivery failures and low adoption rates, while strictly preserving the randomization to prevent selection bias. However, a research audience might prefer the Local Average Treatment Effect if the goal is to understand the theoretical efficacy of the feature itself rather than the effectiveness of the deployment; estimating this requires instrumental variable that rely on the exclusion restriction assumption and monotonicity.

import numpy as np
import pandas as pd
import scipy.special as sc
import statsmodels.formula.api as smf

np.random.seed(123)

n_users = 100000 
n_days  = 14

user_id = np.arange(1, n_users + 1)

platform = np.random.choice(
    ["ios", "android", "web"],
    size=n_users,
    replace=True,
    p=[0.35, 0.35, 0.30]
)

cluster_id = np.random.randint(1, 501, size=n_users)

baseline_activity = np.random.gamma(shape=2.0, scale=2.0, size=n_users)

signup_cohort = np.random.choice(
    ["cohort_A", "cohort_B", "cohort_C"],
    size=n_users,
    replace=True,
    p=[0.40, 0.35, 0.25]
)

users = pd.DataFrame({
    "user_id": user_id,
    "platform": platform,
    "cluster_id": cluster_id,
    "baseline_activity": baseline_activity,
    "signup_cohort": signup_cohort
})

users["pre_metric"] = users["baseline_activity"] + np.random.normal(0, 0.5, size=n_users)
users.to_csv("Penn State/Spring 2026/SoDA 501/Week 5/data/raw/users.csv", index=False)
users["block"] = pd.qcut(users["baseline_activity"], 10, labels=False) + 1  # 1..10
users["treat"] = (
    users.groupby("block")["user_id"]
    .transform(lambda s: (np.random.rand(len(s)) < 0.5).astype(int))
)

assignment = users[[
    "user_id", "treat", "block", "platform", "cluster_id",
    "signup_cohort", "baseline_activity", "pre_metric"
]].copy()

assignment["assignment_date"] = np.datetime64("2026-04-16")
assignment.to_csv("Penn State/Spring 2026/SoDA 501/Week 5/data/raw/assignment_table.csv", index=False)

dt_assign = assignment.copy()
dt_assign["dummy"] = 1

dt_days = pd.DataFrame({"day": np.arange(1, n_days + 1)})
dt_days["dummy"] = 1

logs = dt_assign.merge(dt_days, on="dummy", how="outer").drop(columns=["dummy"])

logs["date"] = logs["assignment_date"] + pd.to_timedelta(logs["day"] - 1, unit="D")
logs["dow"] = logs["date"].dt.dayofweek + 1
logs["logged_ok"] = (np.random.rand(len(logs)) < 0.98).astype(int)
logs["base_rate"] = np.exp(
    -1.2
    + 0.15 * np.log1p(logs["baseline_activity"])
    + 0.05 * (logs["platform"] == "ios").astype(float)
    + 0.03 * (logs["platform"] == "android").astype(float)
    + 0.02 * (logs["dow"].isin([6, 7])).astype(float)
    + 0.01 * logs["day"]
)
logs["click_rate"] = logs["base_rate"] * np.exp(0.05 * logs["treat"])
logs["clicks"] = np.random.poisson(lam=logs["click_rate"].to_numpy())

lin = (
    -5.0
    + 0.08 * logs["clicks"]
    + 0.10 * np.log1p(logs["baseline_activity"])
    + 0.15 * logs["treat"]
    + 0.02 * (logs["dow"].isin([6, 7])).astype(float)
)
logs["purchase_prob"] = sc.expit(lin.to_numpy())

logs["purchase"] = (np.random.rand(len(logs)) < logs["purchase_prob"].to_numpy()).astype(int)
logs["active"] = ((logs["clicks"] > 0) | (logs["purchase"] > 0)).astype(int)
logs["clicks"] = logs["clicks"].where(logs["logged_ok"] == 1, np.nan)
logs["purchase"] = logs["purchase"].where(logs["logged_ok"] == 1, np.nan)
logs["active"] = logs["active"].where(logs["logged_ok"] == 1, np.nan)
logs.to_csv("Penn State/Spring 2026/SoDA 501/Week 5/data/raw/event_logs.csv", index=False)


# Question 4

user = (
    logs.groupby([
        "user_id", "treat", "block", "platform", "cluster_id",
        "signup_cohort", "baseline_activity", "pre_metric"
    ], as_index=False)
    .agg(
        post_clicks=("clicks", "sum"),
        post_purchases=("purchase", "sum"),
        days_observed=("active", lambda x: x.notna().sum()),
        missing_share=("active", lambda x: x.isna().mean()),
        days_active=("active", "sum")
    )
)

user["retained_any"] = (user["days_active"] >= 1).astype(int)

user.to_csv("Penn State/Spring 2026/SoDA 501/Week 5/data/processed/analysis_dataset.csv", index=False)


retention_outcomes = ["days_active", "retained_any"]
results_list = []

for outcome in retention_outcomes:
    mean_treat = user.loc[user["treat"] == 1, outcome].mean()
    mean_ctrl  = user.loc[user["treat"] == 0, outcome].mean()
    diff_means = mean_treat - mean_ctrl
    model = smf.ols(
        f"{outcome} ~ treat + baseline_activity + pre_metric + C(block)", 
        data=user
    ).fit(cov_type="cluster", cov_kwds={"groups": user["cluster_id"]})
    reg_ate = model.params["treat"]
    reg_se  = model.bse["treat"]
    reg_p   = model.pvalues["treat"]
    
    results_list.append({
        "outcome": outcome,
        "ate_diff_means": diff_means,
        "ate_regression": reg_ate,
        "reg_se": reg_se,
        "reg_p_value": reg_p
    })
ate_retention = pd.DataFrame(results_list)
output_path = "Penn State/Spring 2026/SoDA 501/Week 5/outputs/tables/ate_retention.csv"
ate_retention.to_csv(output_path, index=False)

print(ate_retention)

# Question 5

p_compliance = 0.6 
df_iv = assignment.copy()

n_treated = df_iv[df_iv["treat"] == 1].shape[0]
compliance_draws = (np.random.rand(n_treated) < p_compliance).astype(int)

df_iv["received"] = 0
df_iv.loc[df_iv["treat"] == 1, "received"] = compliance_draws

dt_assign_iv = df_iv.copy()
dt_assign_iv["dummy"] = 1
dt_days_iv = pd.DataFrame({"day": np.arange(1, n_days + 1)})
dt_days_iv["dummy"] = 1

logs_iv = dt_assign_iv.merge(dt_days_iv, on="dummy", how="outer").drop(columns=["dummy"])

logs_iv["date"] = logs_iv["assignment_date"] + pd.to_timedelta(logs_iv["day"] - 1, unit="D")
logs_iv["dow"] = logs_iv["date"].dt.dayofweek + 1
logs_iv["logged_ok"] = (np.random.rand(len(logs_iv)) < 0.98).astype(int)

logs_iv["base_rate"] = np.exp(
    -1.2
    + 0.15 * np.log1p(logs_iv["baseline_activity"])
    + 0.05 * (logs_iv["platform"] == "ios").astype(float)
    + 0.03 * (logs_iv["platform"] == "android").astype(float)
    + 0.02 * (logs_iv["dow"].isin([6, 7])).astype(float)
    + 0.01 * logs_iv["day"]
)

logs_iv["click_rate"] = logs_iv["base_rate"] * np.exp(0.05 * logs_iv["received"])

logs_iv["clicks"] = np.random.poisson(lam=logs_iv["click_rate"].to_numpy())

lin_iv = (
    -5.0
    + 0.08 * logs_iv["clicks"]
    + 0.10 * np.log1p(logs_iv["baseline_activity"])
    + 0.15 * logs_iv["received"]
    + 0.02 * (logs_iv["dow"].isin([6, 7])).astype(float)
)
logs_iv["purchase_prob"] = sc.expit(lin_iv.to_numpy())
logs_iv["purchase"] = (np.random.rand(len(logs_iv)) < logs_iv["purchase_prob"].to_numpy()).astype(int)

user_iv = (
    logs_iv.groupby(["user_id", "treat", "received"], as_index=False)
    .agg(
        post_purchases=("purchase", "sum"),
        post_clicks=("clicks", "sum")
    )
)
user_iv["converted"] = (user_iv["post_purchases"] > 0).astype(int)

itt_model = smf.ols("converted ~ treat", data=user_iv).fit()
itt_est = itt_model.params["treat"]

first_stage_model = smf.ols("received ~ treat", data=user_iv).fit()
first_stage_est = first_stage_model.params["treat"]

tot_est = itt_est / first_stage_est

results_table = pd.DataFrame({
    "Estimator": ["ITT (Intent-to-Treat)", "TOT (IV / LATE)"],
    "Estimate": [itt_est, tot_est],
    "Calculation": ["Slope of Y ~ Treat", "ITT / (Slope of Received ~ Treat)"]
})

output_path = "Penn State/Spring 2026/SoDA 501/Week 5/outputs/tables/itt_vs_tot.csv"
results_table.to_csv(output_path, index=False)

print(results_table)

#The TOT estimate is larger in magnitude than the ITT estimate because the ITT calculates the average effect across everyone assigned to the treatment group, including non-compliers who did not actually receive the intervention and thus experienced zero effect. This inclusion of non-compliers dilutes the ITT estimate toward zero. The TOT estimator corrects for this dilution by dividing the ITT by the compliance rate (approximately 60% in this simulation), thereby scaling the estimate up to reflect the full impact of the treatment on only those who actually received it.

