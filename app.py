import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
from model import get_model
import matplotlib.pyplot as plt
import json


checkpoint = torch.load("brain_tumor_model.pth", map_location='cpu')
num_classes = len(checkpoint['class_names'])

model = get_model(num_classes=num_classes, pretrained=False)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

test_accuracy = checkpoint.get('test_accuracy', None)
class_names = checkpoint.get('class_names', ['glioma', 'meningioma', 'notumor', 'pituitary'])


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])


st.title("🧠 Brain Tumor MRI Sınıflandırma")
st.write("MRI görüntüsünü yükleyin ve tümör tipini tahmin edin.")

if test_accuracy is not None:
    st.metric(label="Model Test Doğruluğu", value=f"{test_accuracy:.2f}%")


st.subheader("📊 Eğitim Süreci (Loss & Accuracy Grafiği)")

try:
    with open("train_history.json", "r") as f:
        history = json.load(f)
        epochs = list(range(1, len(history["loss"]) + 1))

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].plot(epochs, history["loss"], marker='o')
        ax[0].set_title("Kayıp (Loss)")
        ax[0].set_xlabel("Epoch")
        ax[0].set_ylabel("Loss")

        ax[1].plot(epochs, history["accuracy"], color='green', marker='o')
        ax[1].set_title("Doğruluk (Accuracy)")
        ax[1].set_xlabel("Epoch")
        ax[1].set_ylabel("Accuracy (%)")

        st.pyplot(fig)
except FileNotFoundError:
    st.warning("Eğitim geçmişi (train_history.json) bulunamadı. Lütfen modeli yeniden eğitip tekrar deneyin.")


st.subheader("🔍 Görüntü Tahmini")

uploaded_file = st.file_uploader("Görüntü seçin", type=["jpg","png","jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Yüklenen Görüntü', use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    
    st.write(f"Tahmin: **{class_names[pred.item()]}**")
