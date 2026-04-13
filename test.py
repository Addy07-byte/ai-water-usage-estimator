import numpy as np
import pandas as pd


#WUE values from Li et al. 2023 (L/Kwh)
WUE = {
    "us-east-iowa": 1.8,
    "us-west-oregon":0.5,
    "us-south-arizona": 3.0

}

#Energy per token (derived from paper: 0.004 kwh per request, ~800 tokens average)

energy_per_token = 0.004/800 #kwh per token

#model energy multipliers (Gpt-4o uses more compute than mini)
model_multiplier = { "gpt-4o": 1.0,
                    "gpt-4o-mini": 0.3,
                    "gpt-3.5-turbo": 0.2
                    }

#Generate 200 synthetic API calls
np.random.seed(42) #makes results reproducible

n = 200

regions = list(WUE.keys())
models = list(model_multiplier.keys())

data = []

for _ in range(n):
    tokens = np.random.randint(50, 2000)
    region = np.random.choice(regions)
    model = np.random.choice(models)
    hour = np.random.randint(0,24)

    water = (tokens * energy_per_token * WUE[region] * model_multiplier[model])

    data.append([tokens, region, model, hour, water])

df = pd.DataFrame(data, columns= ["tokens", "region", "model", "hour", "water_litres"])

print(df.head(10))
print(f"\nDataset shape: {df.shape}")
print(f"\n Water range : {df.water_litres.min():.6f} to {df.water_litres.max():.6f} litres")


#convert categorial features to numbers
#ML models need numbers, not strings 
df["region_code"] = df["region"].map({""
"us-west-oregon": 0, "us-east-iowa":1, "us-south-arizona":2})

df["model_code"] = df["model"].map({
    "gpt-3.5-turbo" : 0,
    "gpt-4o-mini": 1,
    "gpt-4o":2
})

# Features (X) and label (y)
X = df[["tokens", "region_code", "model_code"]].values
y = df["water_litres"].values

print("X shape:", X.shape)
print("y shape:", y.shape)
print("\nFirst row of X: ", X[0])
print("First row of y:", y[0])


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#Split data - 80% train, 20% test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


#Train linear regrssion
model = LinearRegression()

model.fit(X_train, y_train)

#Evaluate
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)

print(f"Training samples: {X_train.shape[0]}")
print(f"Test samples: {X_test.shape[0]}")
print(f"MAE: {mae:.6f} litres")
print(f"\nmodel coefficients: {model.coef_}")
print(f"\n model intercept: {model.intercept_:.6f}")

# one-hot encoding
df_encoded = pd.get_dummies(df, columns= ["region", "model"])
print(df_encoded.columns.tolist())
print(df_encoded.head(3))


# New feature set with one hot encoding

feature_cols = [c for c in df_encoded.columns
                if c not in ["water_litres", "regiomn_code", "model_code"]]

X2 = df_encoded[feature_cols].values
Y2 = df_encoded["water_litres"].values

X2_train, X2_test, Y2_train, Y


