import numpy as np
import matplotlib.pyplot as plt
import math

#============================
#REal training data
#x = total tokens from real API calls
# y = water usage litres
#water per token = 0.000519L (research estimate)
#=============================

x_train = np.array([17.0, 92.0, 533.0])
y_train = np.array([17 * 0.000519, 92 * 0.000519, 533 * 0.000519])

print(f"x_train (tokens): {x_train}")
print(f"y_train (litres): {y_train}")

#==========================
# Cost fucntion - measures how wrong w and b are
#==========================

def compute_cost(x,y,w,b):
    m = x.shape[0]
    cost = 0
    for i in range(m):
        f_wb = w * x[i] + b
        cost = cost + (f_wb - y[i]) ** 2
    total_cost = 1/(2 * m) * cost
    return total_cost


# ======================
# Compute gradient - which direction to move
#=======================

def compute_gradient(x,y,w,b):
    m = x.shape[0]
    dj_dw = 0
    dj_db = 0
    for i in range(m):
        f_wb = w * x[i] + b
        dj_dw += (f_wb - y[i]) * x[i]
        dj_db += (f_wb -y[i])
    dj_dw = dj_dw/m
    dj_db = dj_db/m
    return dj_dw, dj_db


#==================================
# Gradient descent - finds best w and b automatically
#==================================

def gradient_descent(x,y,w_in, b_in, alpha, num_iters):
    J_history = []
    w = w_in
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient(x, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost(x,y,w,b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: cost {J_history[-1]:0.2e} w:{w:0.6f} b:{b:0.6f}")
    return w, b, J_history



# ============================================
# TRAIN THE MODEL — run gradient descent
# ============================================
w_init = 0
b_init = 0
iterations = 1000
alpha = 1.0e-7  # smaller alpha because our x values are larger

w_final, b_final, J_hist = gradient_descent(
    x_train, y_train, w_init, b_init, alpha, iterations
)

print(f"\nTraining complete!")
print(f"w = {w_final:.6f}")
print(f"b = {b_final:.6f}")

# Test predictions
print(f"\nPredictions:")
print(f"17  tokens → {w_final*17  + b_final:.6f} liters")
print(f"92  tokens → {w_final*92  + b_final:.6f} liters")
print(f"533 tokens → {w_final*533 + b_final:.6f} liters")

# ============================================
# SAVE w AND b FOR USE IN THE APP
# ============================================
print(f"\nCopy these into your Streamlit app:")
print(f"W_TRAINED = {w_final}")
print(f"B_TRAINED = {b_final}")

# Plot cost curve
plt.plot(J_hist)
plt.title('Cost vs Iterations')
plt.xlabel('Iteration')
plt.ylabel('Cost J(w,b)')
plt.savefig('cost_curve.png')
plt.show()


# ============================================
# Z-SCORE NORMALIZATION
# Scales all features to similar range (-2 to +2)
# Essential for Sprint 2 (multiple features)
# Formula: x_norm = (x - mean) / std_deviation
# ============================================
def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)      # mean of each feature
    sigma = np.std(X, axis=0)    # std deviation of each feature
    X_norm = (X - mu) / sigma    # normalize
    return X_norm, mu, sigma

# ============================================
# TEST NORMALIZATION — Sprint 2 prep
# Two features: tokens + model_type
# ============================================
X_multi = np.array([
    [17,  0],   # "Hi" on gpt-3.5-turbo
    [92,  1],   # "What is ML?" on gpt-4o
    [533, 1],   # "Explain..." on gpt-4o
])

X_norm, X_mu, X_sigma = zscore_normalize_features(X_multi)
print(f"\nSprint 2 Feature Scaling Test:")
print(f"Original range:   {np.ptp(X_multi, axis=0)}")
print(f"Normalized range: {np.ptp(X_norm, axis=0)}")
print(f"mu:    {X_mu}")
print(f"sigma: {X_sigma}")


# ============================================
# SPRINT 2 — MULTI-FEATURE LINEAR REGRESSION
# Feature 1: total_tokens
# Feature 2: model_type (0=gpt-3.5, 1=gpt-4o)
# Using vectorized gradient descent + zscore norm
# ============================================

# Training data with 2 features
X_train_multi = np.array([
    [17,  0],   # "Hi" on gpt-3.5-turbo
    [92,  1],   # "What is ML?" on gpt-4o
    [533, 1],   # "Explain..." on gpt-4o
])

# TO (model-specific rates — meaningful)
# Water per token by model (research-based estimates)
# gpt-3.5-turbo is ~2.5x lighter than gpt-4o
WATER_PER_TOKEN = {
    "gpt-3.5": 0.000284,   # lighter model, less compute
    "gpt-4o":  0.000519,   # heavier model, more compute
}

y_train_multi = np.array([
    17  * WATER_PER_TOKEN["gpt-3.5"],  # "Hi" on gpt-3.5
    92  * WATER_PER_TOKEN["gpt-4o"],   # "What is ML?" on gpt-4o
    533 * WATER_PER_TOKEN["gpt-4o"],   # "Explain..." on gpt-4o
])

print(f"\nModel-specific y_train:")
print(f"Hi gpt-3.5:   {y_train_multi[0]:.6f}L")
print(f"What is ML:   {y_train_multi[1]:.6f}L")
print(f"Explain gpt4: {y_train_multi[2]:.6f}L")

# Normalize features
X_norm_multi, X_mu_multi, X_sigma_multi = zscore_normalize_features(X_train_multi)
print(f"\nSprint 2 — Normalized training data:")
print(f"X_norm:\n{X_norm_multi}")

# Multi-feature cost function
def compute_cost_multi(X, y, w, b):
    m = X.shape[0]
    cost = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        cost += (f_wb - y[i]) ** 2
    return (1 / (2 * m)) * cost

# Multi-feature gradient
def compute_gradient_multi(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros(n)
    dj_db = 0
    for i in range(m):
        f_wb = np.dot(X[i], w) + b
        err = f_wb - y[i]
        for j in range(n):
            dj_dw[j] += err * X[i, j]
        dj_db += err
    return dj_dw / m, dj_db / m

# Multi-feature gradient descent
def gradient_descent_multi(X, y, w_in, b_in, alpha, num_iters):
    J_history = []
    w = w_in.copy()
    b = b_in
    for i in range(num_iters):
        dj_dw, dj_db = compute_gradient_multi(X, y, w, b)
        w = w - alpha * dj_dw
        b = b - alpha * dj_db
        if i < 100000:
            J_history.append(compute_cost_multi(X, y, w, b))
        if i % math.ceil(num_iters / 10) == 0:
            print(f"Iteration {i:4}: cost {J_history[-1]:0.2e}  w:{w}  b:{b:.6f}")
    return w, b, J_history

# Train on normalized data
w_init_multi = np.zeros(2)
b_init_multi = 0
alpha_multi = 0.1    # can use larger alpha after normalization
iterations_multi = 1000

w_multi, b_multi, J_hist_multi = gradient_descent_multi(
    X_norm_multi, y_train_multi,
    w_init_multi, b_init_multi,
    alpha_multi, iterations_multi
)

print(f"\nSprint 2 Training Complete!")
print(f"w = {w_multi}  (w[0]=tokens weight, w[1]=model_type weight)")
print(f"b = {b_multi:.6f}")

# Test predictions
print(f"\nSprint 2 Predictions:")
for tokens, model, label in [(17, 0, "Hi gpt-3.5"), (92, 1, "What is ML gpt-4o"), (533, 1, "Explain gpt-4o")]:
    x_new = np.array([tokens, model])
    x_new_norm = (x_new - X_mu_multi) / X_sigma_multi
    pred = np.dot(x_new_norm, w_multi) + b_multi
    print(f"{label}: {pred:.6f} liters")

# Plot Sprint 2 cost curve
plt.figure()
plt.plot(J_hist_multi)
plt.title('Sprint 2: Cost vs Iterations (Multi-feature)')
plt.xlabel('Iteration')
plt.ylabel('Cost J(w,b)')
plt.savefig('cost_curve_sprint2.png')
plt.show()

print(f"\nSprint 2 model parameters for Streamlit app:")
print(f"W_MULTI = {w_multi.tolist()}")
print(f"B_MULTI = {b_multi}")
print(f"X_MU    = {X_mu_multi.tolist()}")
print(f"X_SIGMA = {X_sigma_multi.tolist()}")


# ============================================
# SIGMOID FUNCTION — Sprint 3 classification prep
# Converts any number to probability between 0 and 1
# g(z) = 1 / (1 + e^(-z))
# Used in logistic regression for classification
# ============================================
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Test sigmoid behavior
print("\nSigmoid function test:")
print(f"sigmoid(-10) = {sigmoid(-10):.6f}  (close to 0 = NO)")
print(f"sigmoid(0)   = {sigmoid(0):.6f}  (exactly 0.5 = uncertain)")
print(f"sigmoid(10)  = {sigmoid(10):.6f}  (close to 1 = YES)")

# Visualize sigmoid curve
z_values = np.linspace(-10, 10, 100)
plt.figure()
plt.plot(z_values, sigmoid(z_values), c='b', label='sigmoid(z)')
plt.axhline(y=0.5, color='r', linestyle='--', label='threshold = 0.5')
plt.axvline(x=0, color='g', linestyle='--', label='z = 0')
plt.title('Sigmoid Function — Sprint 3 Classification Prep')
plt.xlabel('z (wx + b)')
plt.ylabel('Probability')
plt.legend()
plt.savefig('sigmoid_curve.png')
plt.show()
print("Sigmoid curve saved to sigmoid_curve.png")