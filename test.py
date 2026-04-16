import numpy as np
import pandas as pd

# ── STEP 1: Define water usage efficiency per datacenter location ──
# WUE = Water Usage Effectiveness (liters of water per kWh of energy)
# Source: Li et al. 2023 "Making AI Less Thirsty" arxiv.org/abs/2304.03271
# Hot climates need more evaporative cooling → more water per kWh
WUE = {
    "us-east-iowa": 1.8,
    "us-west-oregon": 0.5,
    "us-south-arizona": 3.0
}

# ── STEP 2: Calculate energy cost per token ──
# Paper says one average request uses 0.004 kWh across ~800 tokens
# Dividing gives us the energy cost of a single token
energy_per_token = 0.004 / 800  # kWh per token

# ── STEP 3: Define model energy multipliers ──
# GPT-4o is a larger model and uses more compute than smaller models
# These multipliers scale water usage based on model size
model_multiplier = {
    "gpt-4o": 1.0,
    "gpt-4o-mini": 0.3,
    "gpt-3.5-turbo": 0.2
}

# ── STEP 4: Generate 200 synthetic API calls ──
# We don't have real data yet, so we simulate realistic API calls
# seed(42) ensures we get the same random numbers every run → reproducibility
np.random.seed(42)
n = 200

regions = list(WUE.keys())
models = list(model_multiplier.keys())

data = []

for _ in range(n):
    tokens = np.random.randint(50, 2000)
    region = np.random.choice(regions)
    model = np.random.choice(models)
    hour = np.random.randint(0, 24)

    # Core formula from Li et al. 2023:
    # water = tokens × energy_per_token × WUE[region] × model_multiplier
    water = (tokens * energy_per_token * WUE[region] * model_multiplier[model])

    data.append([tokens, region, model, hour, water])

df = pd.DataFrame(data, columns=["tokens", "region", "model", "hour", "water_litres"])

print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"\nWater range: {df.water_litres.min():.6f} to {df.water_litres.max():.6f} litres")

# ── STEP 5: Ordinal encoding (baseline — has a known flaw) ──
# ML models need numbers not strings
# Problem: model assumes Arizona(2) is "twice" Iowa(1) — not true
df["region_code"] = df["region"].map({
    "us-west-oregon": 0,
    "us-east-iowa": 1,
    "us-south-arizona": 2
})

df["model_code"] = df["model"].map({
    "gpt-3.5-turbo": 0,
    "gpt-4o-mini": 1,
    "gpt-4o": 2
})

# ── STEP 6: Define features (X) and label (y) ──
# Features = inputs the model uses to make predictions
# Label = what we are trying to predict (water usage)
X = df[["tokens", "region_code", "model_code"]].values
y = df["water_litres"].values

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst row of X:", X[0])
print("First row of y:", y[0])

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# ── STEP 7: Train/test split ──
# Hold back 20% of data so we can test on data the model has never seen
# Without this, we'd be testing on exam questions the model memorized
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── STEP 8: Train baseline linear regression (ordinal encoding) ──
model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"MAE: {mae:.6f} litres")
print(f"\nModel coefficients: {model.coef_}")
print(f"\nModel intercept: {model.intercept_:.6f}")

# ── STEP 9: One-hot encoding (fixes the ordinal encoding problem) ──
# Creates separate yes/no columns for each category
# Now Iowa is [0,1,0] not 1 — no fake math between locations
df_encoded = pd.get_dummies(df, columns=["region", "model"])
print(df_encoded.columns.tolist())
print(df_encoded.head(3))

# ── STEP 10: Retrain with one-hot encoded features ──
# Exclude the label and old ordinal columns from features
feature_cols = [c for c in df_encoded.columns
                if c not in ["water_litres", "region_code", "model_code"]]

X2 = df_encoded[feature_cols].values
y2 = df_encoded["water_litres"].values

X2_train, X2_test, y2_train, y2_test = train_test_split(
    X2, y2, test_size=0.2, random_state=42
)

model12 = LinearRegression()
model12.fit(X2_train, y2_train)

y2_pred = model12.predict(X2_test)
mae2 = mean_absolute_error(y2_test, y2_pred)

print(f"Old MAE: 0.002531 liters (36% error)")
print(f"New MAE: {mae2:.6f} litres")
print(f"Improvement: {((0.002531 - mae2) / 0.002531 * 100):.1f}%")

# ── STEP 11: Decision tree (best model) ──
# Decision trees learn IF/THEN rules from data
# Better than linear regression here because water usage multiplies
# across features — not a straight line relationship
from sklearn.tree import DecisionTreeRegressor

model3 = DecisionTreeRegressor(max_depth=5, random_state=42)
model3.fit(X2_train, y2_train)

y3_pred = model3.predict(X2_test)
mae3 = mean_absolute_error(y2_test, y3_pred)

print(f"\nModel Comparison:")
print(f"Linear Regression (ordinal):  MAE = 0.002531")
print(f"Linear Regression (one-hot):  MAE = {mae2:.6f}")
print(f"Decision Tree (one-hot):      MAE = {mae3:.6f}")
print(f"\nBest model: {'Decision Tree' if mae3 < mae2 else 'Linear Regression'}")