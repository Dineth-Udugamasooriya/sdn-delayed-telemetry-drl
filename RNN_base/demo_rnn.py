import torch
import torch.nn as nn
import torch.optim as optim

# Use a fixed seed so the generated dataset is identical across runs.
# This makes experiments easier to reproduce and compare.
torch.manual_seed(42)

# Create synthetic sequences that follow simple increasing trends. The RNN
# will later use the first five values to predict the sixth value.
num_samples = 1000
seq_len = 5

# Give every sequence a different initial value between 0.20 and 0.40.
starts = torch.rand(num_samples, 1) * 0.2 + 0.2

# Give every sequence a different positive slope between 0.03 and 0.07,
# so each sequence increases at a slightly different rate.
slopes = torch.rand(num_samples, 1) * 0.04 + 0.03

# Create the time indices 0 through 5. The extra time step is needed because
# the final value will be held out as the prediction target.
time_steps = torch.arange(seq_len + 1).float().unsqueeze(0)

# Generate each sequence using: value = start + slope * time.
# Small random noise prevents the prediction task from being perfectly
# deterministic while keeping the underlying trend easy to learn.
values = starts + slopes * time_steps
values += torch.randn_like(values) * 0.005

# Keep all values within the range [0, 1], matching the intended input scale.
values = values.clamp(0.0, 1.0)

# Use the first five time steps as the RNN input. unsqueeze(-1) adds the
# feature dimension, producing the shape (samples, sequence length, features).
# The sixth value is the expected output for each sequence.
X = values[:, :seq_len].unsqueeze(-1)
y = values[:, seq_len]

# Train on the first 800 sequences and reserve the remaining 200 for testing
# how well the model generalizes to sequences it has not seen before.
X_train, X_test = X[:800], X[800:]
y_train, y_test = y[:800], y[800:]

class SequencePredictor(nn.Module):
    def __init__(self, hidden_size=16):
        super().__init__()

        # The model receives one numeric feature at each time step and keeps
        # a hidden representation with `hidden_size` values. batch_first=True
        # means input tensors use the shape (batch, sequence, features).
        self.rnn = nn.RNN(
            input_size=1,
            hidden_size=hidden_size,
            batch_first=True,
        )

        # Convert the RNN's final hidden representation into one predicted
        # value: the next value in the sequence.
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # `outputs` contains a hidden state for every time step, while `h_n`
        # contains the final hidden state for the sequence.
        outputs, h_n = self.rnn(x)

        # There is one RNN layer, so select its final hidden state. The result
        # has shape (batch, hidden_size) and summarizes the whole input.
        final_hidden = h_n[-1]

        # Apply the prediction layer and remove the final size-one dimension,
        # returning one scalar prediction per sequence in the batch.
        return self.fc(final_hidden).squeeze(-1)


model = SequencePredictor()
optimizer = optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

for epoch in range(100):
    # Enable training behavior before the forward pass. This is important
    # for models that contain layers such as dropout or batch normalization.
    model.train()

    # Run the complete training set through the RNN in one batch. The model
    # returns one prediction for the next value in each input sequence.
    predictions = model(X_train)
    loss = loss_fn(predictions, y_train)

    # Remove gradients left over from the previous epoch before calculating
    # the gradients for the current loss.
    optimizer.zero_grad()
    loss.backward()  # Backpropagate through time (BPTT)

    # Protect against excessively large gradients, which can make recurrent
    # networks unstable during backpropagation through long sequences.
    nn.utils.clip_grad_norm_(
        model.parameters(), max_norm=1.0
    )

    # Update the RNN and output-layer weights using the clipped gradients.
    optimizer.step()

    # Print periodically instead of every epoch so the training output stays
    # readable while still showing whether the loss is decreasing.
    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1}, "
            f"loss = {loss.item():.6f}"
        )

# Switch to evaluation behavior before measuring generalization on data that
# was not used to update the model's parameters.
model.eval()

# Disable autograd during testing because no parameter updates are needed.
# This reduces memory use and makes evaluation faster.
with torch.no_grad():
    predictions = model(X_test)
    test_loss = loss_fn(predictions, y_test)

    # Show a few individual predictions so the numerical test loss is easier
    # to interpret.
    for i in range(5):
        print(
            f"Actual: {y_test[i].item():.3f}, "
            f"Predicted: {predictions[i].item():.3f}"
        )

    print("Test MSE:", test_loss.item())