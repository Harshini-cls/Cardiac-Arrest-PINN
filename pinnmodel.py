import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

X_train = pd.read_csv("X_train.csv").values
y_train = pd.read_csv("y_train.csv").values.reshape(-1, 1)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)

class PINNModel(nn.Module):
    def __init__(self, input_dim):
        super(PINNModel, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.25),

            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

model = PINNModel(X_train.shape[1])

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

epochs = 300

for epoch in range(epochs):
    model.train()

    y_pred = model(X_train)

    data_loss = criterion(y_pred, y_train)

    physics_loss = torch.mean((y_pred - torch.clamp(y_pred, 0, 1)) ** 2)

    loss = data_loss + 0.1 * physics_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 25 == 0:
        print(f"Epoch {epoch+1}/{epochs} | Loss: {loss.item():.4f}")

torch.save(model.state_dict(), "cardiac_pinn_model_weights.pth")
torch.save(model, "cardiac_pinn_full_model.pth")

print("Training completed")
print("Saved: cardiac_pinn_model_weights.pth")
print("Saved: cardiac_pinn_full_model.pth")