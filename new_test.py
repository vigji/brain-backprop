# %%
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

# ----- Create a toy binary classification dataset -----
# We'll use a "circle" decision boundary: label=1 if x^2 + y^2 < radius^2, else 0
N = 400  # number of samples
X = np.random.uniform(-1.5, 1.5, size=(N,2))
radius = 1.0
y = (X[:,0]**2 + X[:,1]**2 < radius**2).astype(np.float32)  # 0 or 1

# ----- Network Hyperparams -----
H1 = 5   # hidden1 size
H2 = 3   # hidden2 size
lr_backprop = 0.05
lr_heuristic = 0.001
epochs = 100

# Utility: Sigmoid + derivative
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# We will track how the 'middle neuron' in H2 (index=1) changes over epochs.
# By "middle neuron," let's pick h2_1 specifically.

# -------------------------
# 1) Define a "backprop" model and training
# -------------------------

class MLP_Backprop:
    def __init__(self, input_dim=2, h1=5, h2=3):
        # Initialize all weights/biases
        self.W1 = 0.1 * np.random.randn(input_dim, h1)   # [2 x 5]
        self.b1 = np.zeros((1, h1))
        self.W2 = 0.1 * np.random.randn(h1, h2)          # [5 x 3]
        self.b2 = np.zeros((1, h2))
        self.W3 = 0.1 * np.random.randn(h2, 1)           # [3 x 1]
        self.b3 = np.zeros((1, 1))

        # We'll store the "middle neuron" (h2_1) incoming weights each epoch
        self.h2_1_traces_in = []  # track W2[:,1] over epochs
        self.h2_1_traces_out = [] # track W3[1, :] over epochs

    def forward(self, X):
        # Layer 1
        z1 = X.dot(self.W1) + self.b1      # [N x 5]
        a1 = np.maximum(z1, 0)            # ReLU

        # Layer 2
        z2 = a1.dot(self.W2) + self.b2     # [N x 3]
        a2 = np.maximum(z2, 0)            # ReLU

        # Output
        z3 = a2.dot(self.W3) + self.b3     # [N x 1]
        a3 = sigmoid(z3)                  # Sigmoid
        return a1, a2, a3

    def backward_and_update(self, X, a1, a2, a3, y_true, lr):
        # We use binary cross-entropy:
        # L = - sum( y*log(a3) + (1-y)*log(1-a3) ), average over N
        # For gradient: dL/dz3 = (a3 - y)
        N = X.shape[0]
        y_true = y_true.reshape(-1, 1)  # shape [N,1] for consistency

        # Backprop into output
        dL_dz3 = (a3 - y_true) / N    # [N x 1]

        # Grad w.r.t W3, b3
        dW3 = a2.T.dot(dL_dz3)        # [3 x 1]
        db3 = np.sum(dL_dz3, axis=0, keepdims=True)  # [1 x 1]

        # Backprop into hidden2
        dz2 = dL_dz3.dot(self.W3.T)  # [N x 3], via chain rule
        dz2[a2 <= 0] = 0             # ReLU grad
        dW2 = a1.T.dot(dz2)          # [5 x 3]
        db2 = np.sum(dz2, axis=0, keepdims=True)   # [1 x 3]

        # Backprop into hidden1
        dz1 = dz2.dot(self.W2.T)     # [N x 5]
        dz1[a1 <= 0] = 0             # ReLU grad
        dW1 = X.T.dot(dz1)           # [2 x 5]
        db1 = np.sum(dz1, axis=0, keepdims=True)   # [1 x 5]

        # Update
        self.W3 -= lr * dW3
        self.b3 -= lr * db3
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

    def train_one_epoch(self, X, y, lr):
        a1, a2, a3 = self.forward(X)
        self.backward_and_update(X, a1, a2, a3, y, lr)

        # For tracking the "middle neuron" in hidden2
        # i.e. index=1
        self.h2_1_traces_in.append(self.W2[:, 1].copy())  # incoming from layer1
        self.h2_1_traces_out.append(self.W3[1, :].copy()) # outgoing to layer3

    def predict(self, X):
        # Return binary predictions
        _, _, a3 = self.forward(X)
        return (a3 > 0.5).astype(int)


# -------------------------
# 2) Define a "heuristic" model and training
# -------------------------

class MLP_Heuristic:
    """
    We'll use the same architecture, but the weight update rule is:

    - Forward pass to get final output: y_pred = (a3 > 0.5)
    - Check if correct or not.
    - For each sample, for each neuron:
      If final output is correct AND that neuron's ReLU activation > 0, then increase its incoming/outgoing weights by a small amount.
      If final output is incorrect AND that neuron's ReLU activation > 0, then decrease its incoming/outgoing weights.

    This is obviously *not* the true gradient, just a simple 'reinforce if good' approach.
    """
    def __init__(self, input_dim=2, h1=5, h2=3):
        self.W1 = 0.1 * np.random.randn(input_dim, h1)
        self.b1 = np.zeros((1, h1))
        self.W2 = 0.1 * np.random.randn(h1, h2)
        self.b2 = np.zeros((1, h2))
        self.W3 = 0.1 * np.random.randn(h2, 1)
        self.b3 = np.zeros((1, 1))

        self.h2_1_traces_in = []
        self.h2_1_traces_out = []

    def forward(self, X):
        z1 = X.dot(self.W1) + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1.dot(self.W2) + self.b2
        a2 = np.maximum(z2, 0)
        z3 = a2.dot(self.W3) + self.b3
        a3 = sigmoid(z3)
        return a1, a2, a3

    def train_one_epoch(self, X, y, lr):
        """
        We'll do a forward pass on *each sample*, determine correctness,
        then do a simple 'reinforce/punish' for each active neuron in Hidden1 & Hidden2.
        """
        N = X.shape[0]
        for i in range(N):
            # Single-sample forward pass
            x_i = X[i].reshape(1,-1)           # shape [1,2]
            y_i = y[i]                         # 0 or 1
            a1, a2, a3 = self.forward(x_i)     # each shaped [1, #neurons]
            pred_i = 1 if a3[0,0] > 0.5 else 0
            correct = (pred_i == y_i)

            # We'll do 'reinforce' if correct, 'punish' if incorrect.
            sign = +1 if correct else -1

            # Let's say a2[0,j] is the ReLU activation of the j-th neuron in hidden2.
            # If that > 0, we update the weights leading into j and from j -> output by sign*lr*(some factor).
            # We'll do a simple approach: W2[:,j] += sign * lr * a1 (if a2[0,j] > 0).
            # Similarly, W3[j,:] += sign * lr * a2[0,j].
            # We'll do something similar for hidden1 as well, if we want. (But let's focus on hidden2 for the demonstration.)

            # Hidden2 updates:
            for j in range(a2.shape[1]):
                if a2[0,j] > 0:  # active
                    # incoming (from a1 to h2_j)
                    # We'll increment W2[:,j] in proportion to a1's values
                    self.W2[:, j] += sign * lr * a1[0,:]
                    self.b2[0, j] += sign * lr  # small shift in bias

                    # outgoing (from h2_j to output)
                    self.W3[j, 0] += sign * lr * a2[0,j]
                    self.b3[0, 0] += sign * lr

            # Also do hidden1 in a similar way, if we want the entire net to adapt
            for k in range(a1.shape[1]):
                if a1[0,k] > 0:
                    self.W1[:, k] += sign * lr * x_i[0,:]
                    self.b1[0, k] += sign * lr

        # After processing all samples in this epoch, record the middle neuron's W2/W3 for analysis
        self.h2_1_traces_in.append(self.W2[:, 1].copy())   # incoming from hidden1 to h2_1
        self.h2_1_traces_out.append(self.W3[1, :].copy())  # outgoing from h2_1 to output

    def predict(self, X):
        _, _, a3 = self.forward(X)
        return (a3 > 0.5).astype(int)

# -------------------------
# 3) Run both training methods + track results
# -------------------------

# Shuffle dataset for convenience
idx = np.arange(N)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# Split into train/test
split = int(0.8 * N)
X_train, y_train = X[:split], y[:split]
X_test,  y_test  = X[split:], y[split:]

model_bp = MLP_Backprop()
model_hx = MLP_Heuristic()

acc_bp_list = []
acc_hx_list = []

for ep in range(epochs):
    # Backprop epoch
    model_bp.train_one_epoch(X_train, y_train, lr_backprop)
    # Heuristic epoch
    model_hx.train_one_epoch(X_train, y_train, lr_heuristic)

    # Evaluate on test set
    pred_bp = model_bp.predict(X_test)
    pred_hx = model_hx.predict(X_test)
    acc_bp = (pred_bp.flatten() == y_test).mean()
    acc_hx = (pred_hx.flatten() == y_test).mean()

    acc_bp_list.append(acc_bp)
    acc_hx_list.append(acc_hx)

# -------------------------
# 4) Plot results
# -------------------------
plt.figure()
plt.plot(acc_bp_list, label='Backprop Accuracy')
plt.plot(acc_hx_list, label='Heuristic Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (Test)')
plt.title('Backprop vs. Heuristic Accuracy')
plt.legend()
plt.show()

# Track how the middle neuron (h2_1) changes its incoming/outgoing weights
bp_in = np.array(model_bp.h2_1_traces_in)   # shape [epochs, 5]
bp_out = np.array(model_bp.h2_1_traces_out) # shape [epochs, 1]

hx_in = np.array(model_hx.h2_1_traces_in)   # shape [epochs, 5]
hx_out = np.array(model_hx.h2_1_traces_out) # shape [epochs, 1]

# For illustration, let's just plot the L2 norm of those weights over time
plt.figure()
plt.plot(np.linalg.norm(bp_in, axis=1), label='Backprop (W2[:,1] norm)')
plt.plot(np.linalg.norm(hx_in, axis=1), label='Heuristic (W2[:,1] norm)')
plt.xlabel('Epoch')
plt.ylabel('L2 Norm of Middle Neuron Incoming Weights')
plt.title('Evolution of h2_1 Incoming Weights')
plt.legend()
plt.show()

# %%
