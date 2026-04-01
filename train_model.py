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