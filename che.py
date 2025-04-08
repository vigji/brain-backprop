# %%

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
#  Generate XOR Dataset
# -----------------------------
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

# XOR targets
Y = np.array([[0], [1], [1], [0]], dtype=float)

# -----------------------------
#  Activation Functions
# -----------------------------
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_deriv(sig_x):
    # derivative wrt the *output* of the sigmoid
    return sig_x * (1 - sig_x)

def forward_pass(X, W_in, W_out):
    """
    Feedforward pass: X -> hidden -> output.
    Returns (hidden_activations, output_activations).
    """
    h = sigmoid(X.dot(W_in))
    y = sigmoid(h.dot(W_out))
    return h, y

# -----------------------------
#  CHL Update (Single-Step Clamped Hidden)
# -----------------------------
def chl_update(X, Y, W_in, W_out, lr=0.1, alpha=1.0):
    """
    Perform one update of Contrastive Hebbian Learning in a feed-forward style,
    but with a single-step hidden-unit 'adjustment' in the clamped phase.
    
    :param X: input data (batch_size x input_dim)
    :param Y: targets (batch_size x output_dim)
    :param W_in: weights input->hidden
    :param W_out: weights hidden->output
    :param lr: learning rate
    :param alpha: scaling for hidden 'error feedback' in the clamped phase
    """
    # ---- Phase 1: "Free phase" ----
    h_free, y_free = forward_pass(X, W_in, W_out)

    # ---- Phase 2: "Clamped phase" ----
    # We want the output to be Y, so let's do one step of gradient-like feedback
    # to adjust the hidden state from h_free towards what would produce Y.

    # 1. Compute output error & derivative
    err_out   = (Y - y_free)                       # shape: (batch, output_dim)
    delta_out = err_out * sigmoid_deriv(y_free)    # shape: (batch, output_dim)

    # 2. Backpropagate to hidden (just the error signal, not changing W yet)
    err_hidden   = delta_out.dot(W_out.T)          # shape: (batch, hidden_dim)
    delta_hidden = err_hidden * sigmoid_deriv(h_free)

    # 3. Define "clamped" hidden state as old hidden + alpha * delta_hidden
    h_clamped = h_free + alpha * delta_hidden
    # Could optionally clip: h_clamped = np.clip(h_clamped, 0.0, 1.0)
    
    # 4. Output is forced to Y (the perfect target):
    y_clamped = Y

    # ---- Weight updates from correlation differences ----
    # correlation in free phase:
    corr_free_in  = np.einsum('bi,bj->ij', X,       h_free)    / X.shape[0]  # input->hidden
    corr_free_out = np.einsum('bi,bj->ij', h_free,  y_free)    / X.shape[0]  # hidden->output

    # correlation in clamped phase:
    corr_clamped_in  = np.einsum('bi,bj->ij', X,       h_clamped) / X.shape[0]
    corr_clamped_out = np.einsum('bi,bj->ij', h_clamped, y_clamped) / X.shape[0]

    # delta weights
    dW_in  = (corr_clamped_in  - corr_free_in)
    dW_out = (corr_clamped_out - corr_free_out)

    # update
    W_in  += lr * dW_in
    W_out += lr * dW_out

    return W_in, W_out

# -----------------------------
#  (Optional) Standard Backprop
# -----------------------------
def backprop_update(X, Y, W_in, W_out, lr=0.1):
    """
    Standard feedforward backprop for reference/comparison.
    """
    # forward
    h, y_pred = forward_pass(X, W_in, W_out)

    # output error
    error_out = (Y - y_pred)
    delta_out = error_out * sigmoid_deriv(y_pred)

    # hidden error
    error_h = delta_out.dot(W_out.T)
    delta_h = error_h * sigmoid_deriv(h)

    # weight gradients
    dW_out = h.T.dot(delta_out) / X.shape[0]
    dW_in = X.T.dot(delta_h) / X.shape[0]

    W_in += lr * dW_in
    W_out += lr * dW_out
    return W_in, W_out

# -----------------------------
#  Experiment Setup
# -----------------------------
#np.random.seed(42)

input_dim = 2
hidden_dim = 2
output_dim = 1

# -- Initialize weights for CHL
W_in_chl = np.random.randn(input_dim, hidden_dim) * 0.5
W_out_chl = np.random.randn(hidden_dim, output_dim) * 0.5

# -- Initialize weights for BP (comparison)
W_in_bp = W_in_chl.copy()
W_out_bp = W_out_chl.copy()

epochs = 20000
lr = 1
alpha = 1.0  # how strongly we adjust hidden in the "clamped" phase

loss_chl = []
loss_bp  = []

# -----------------------------
#  Training
# -----------------------------
for epoch in range(epochs):
    # ---- CHL forward + update
    h_free, y_free = forward_pass(X, W_in_chl, W_out_chl)
    chl_err = np.mean(0.5 * (Y - y_free)**2)
    loss_chl.append(chl_err)
    W_in_chl, W_out_chl = chl_update(X, Y, W_in_chl, W_out_chl, lr=lr, alpha=alpha)

    # ---- BP forward + update
    h_bp, y_bp = forward_pass(X, W_in_bp, W_out_bp)
    bp_err = np.mean(0.5 * (Y - y_bp)**2)
    loss_bp.append(bp_err)
    W_in_bp, W_out_bp = backprop_update(X, Y, W_in_bp, W_out_bp, lr=lr)

# -----------------------------
#  Final results
# -----------------------------
print("Final MSE with CHL:", loss_chl[-1])
print("Final MSE with BP: ", loss_bp[-1])

# Predictions
_, y_chl_pred = forward_pass(X, W_in_chl, W_out_chl)
_, y_bp_pred  = forward_pass(X, W_in_bp,  W_out_bp)

print("\nCHL Predictions:")
print(y_chl_pred.round(3))
print("BP Predictions:")
print(y_bp_pred.round(3))

# Quick plot of losses
plt.plot(loss_chl, label='CHL MSE')
plt.plot(loss_bp,  label='BP MSE')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title("Contrastive Hebbian vs. Backprop on XOR")
plt.show()

# %%
