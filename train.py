import torch
import torch.nn as nn
import torch.optim as optim
import json
from utils import get_dataloaders
from model import get_model

train_loader, test_loader, classes = get_dataloaders("data/training", "data/testing", batch_size=32)

model = get_model(num_classes=len(classes))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)

num_epochs = 5
history = {"loss": [], "accuracy": []}

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = running_loss / len(train_loader)
    train_accuracy = 100 * correct / total

    history["loss"].append(avg_loss)
    history["accuracy"].append(train_accuracy)

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}, Accuracy: {train_accuracy:.2f}%")


model.eval()
test_correct = 0
test_total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        test_total += labels.size(0)
        test_correct += (predicted == labels).sum().item()

test_accuracy = 100 * test_correct / test_total
print(f"Test Accuracy: {test_accuracy:.2f}%")


torch.save({
    'model_state_dict': model.state_dict(),
    'test_accuracy': test_accuracy,
    'class_names': classes
}, "brain_tumor_model.pth")

with open("train_history.json", "w") as f:
    json.dump(history, f)

print("Model ve eğitim geçmişi kaydedildi.")
