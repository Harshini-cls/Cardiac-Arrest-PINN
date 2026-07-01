import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay

X_test = pd.read_csv("X_test.csv").values
y_test = pd.read_csv("y_test.csv").values.reshape(-1, 1)

X_test_tensor = torch.tensor(X_test, dtype=torch.float32)

class PINNModel(nn.Module):
    def __init__(self, input_dim):
        super(PINNModel, self).__init__()
        self.network = nn.Sequential(
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
        return self.network(x)

input_dim = X_test.shape[1]
model = PINNModel(input_dim)
model.load_state_dict(torch.load("cardiac_pinn_model_weights.pth"))
model.eval()

with torch.no_grad():
    y_prob = model(X_test_tensor).numpy()

y_pred = (y_prob >= 0.5).astype(int)

fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - PINN Cardiac Arrest Prediction")
plt.legend()
plt.savefig("roc_curve.png")
plt.show()

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
plt.show()

print("ROC curve saved as roc_curve.png")
print("Confusion matrix saved as confusion_matrix.png")