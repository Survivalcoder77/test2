"""
Handwritten Digit Recognizer
----------------------------
Trains a CNN on MNIST, then serves a web app where you can
draw a digit on an HTML5 canvas and get an instant prediction.

Usage:
    python3 digit_recognizer.py
    Then open http://localhost:5000 in your browser.
"""

import os
import base64
import io

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from flask import Flask, request, jsonify, render_template_string

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_PATH = "digit_cnn.pth"
EPOCHS = 5
BATCH_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model ──────────────────────────────────────────────────────────────────────
class DigitCNN(nn.Module):
    """Simple two-conv-block CNN for 28×28 grayscale digit images."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 256), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Training ───────────────────────────────────────────────────────────────────
def train_model():
    """Download MNIST and train the CNN; save weights to MODEL_PATH."""
    print(f"Training on {DEVICE} ...")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])

    train_set = datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_set  = datasets.MNIST("./data", train=False, download=True, transform=transform)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = DigitCNN().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_loss = running_loss / len(train_loader)

        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                preds = model(images).argmax(dim=1)
                correct += (preds == labels).sum().item()
        acc = correct / len(test_set) * 100
        print(f"  Epoch {epoch}/{EPOCHS}  loss={avg_loss:.4f}  test_acc={acc:.2f}%")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved → {MODEL_PATH}\n")
    return model


def load_model():
    model = DigitCNN().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    return model


# ── Image preprocessing ────────────────────────────────────────────────────────
def preprocess_image(pil_image: Image.Image) -> torch.Tensor:
    """Convert a PIL image from the canvas to a normalised 1×28×28 tensor."""
    gray = pil_image.convert("L").resize((28, 28), Image.LANCZOS)
    arr = np.array(gray, dtype=np.float32)
    arr = (arr / 255.0 - 0.1307) / 0.3081
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0).to(DEVICE)


# ── HTML page ──────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Handwritten Digit Recognizer</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background: #1e1e2e;
    color: #cdd6f4;
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
    padding: 40px 20px;
    min-height: 100vh;
  }
  h1 { font-size: 1.8rem; margin-bottom: 6px; color: #89b4fa; }
  p.sub { font-size: 0.95rem; color: #6c7086; margin-bottom: 24px; }

  #canvas {
    background: #000;
    border: 2px solid #313244;
    border-radius: 12px;
    cursor: crosshair;
    touch-action: none;
  }

  .buttons {
    display: flex;
    gap: 12px;
    margin-top: 16px;
  }
  button {
    padding: 10px 28px;
    border: none;
    border-radius: 8px;
    font-size: 1rem;
    cursor: pointer;
    transition: opacity .15s;
  }
  button:hover { opacity: .85; }
  #btn-clear   { background: #45475a; color: #cdd6f4; }
  #btn-predict { background: #89b4fa; color: #1e1e2e; font-weight: 700; }

  .result-box {
    margin-top: 28px;
    background: #181825;
    border: 1px solid #313244;
    border-radius: 14px;
    padding: 24px 36px;
    text-align: center;
    width: 320px;
  }
  .digit-display {
    font-size: 5rem;
    font-weight: 900;
    color: #a6e3a1;
    line-height: 1;
  }
  .conf-label {
    font-size: 0.9rem;
    color: #6c7086;
    margin: 6px 0 8px;
  }
  .conf-bar-bg {
    background: #313244;
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
  }
  .conf-bar-fill {
    height: 100%;
    background: #89b4fa;
    border-radius: 999px;
    transition: width .3s ease;
    width: 0%;
  }
  .top3 {
    margin-top: 16px;
    text-align: left;
    font-size: 0.88rem;
    color: #a6adc8;
  }
  .top3-row {
    display: flex;
    justify-content: space-between;
    padding: 3px 0;
  }
  .top3-row span.bar {
    display: inline-block;
    height: 8px;
    background: #585b70;
    border-radius: 4px;
    margin-top: 2px;
    transition: width .3s ease;
  }
  .hint { margin-top: 10px; font-size: 0.82rem; color: #6c7086; }
</style>
</head>
<body>
<h1>Handwritten Digit Recognizer</h1>
<p class="sub">Draw a digit (0–9) on the canvas below</p>

<canvas id="canvas" width="320" height="320"></canvas>

<div class="buttons">
  <button id="btn-clear">Clear</button>
  <button id="btn-predict">Predict</button>
</div>

<div class="result-box">
  <div class="digit-display" id="pred-digit">–</div>
  <div class="conf-label" id="conf-text">Confidence: –</div>
  <div class="conf-bar-bg"><div class="conf-bar-fill" id="conf-bar"></div></div>
  <div class="top3" id="top3"></div>
  <div class="hint">Prediction updates automatically after each stroke</div>
</div>

<script>
const canvas = document.getElementById('canvas');
const ctx    = canvas.getContext('2d');
let drawing  = false;

ctx.fillStyle = '#000';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = '#fff';
ctx.lineWidth   = 22;
ctx.lineCap     = 'round';
ctx.lineJoin    = 'round';

function getPos(e) {
  const r = canvas.getBoundingClientRect();
  const src = e.touches ? e.touches[0] : e;
  return { x: src.clientX - r.left, y: src.clientY - r.top };
}

canvas.addEventListener('mousedown',  e => { drawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('mousemove',  e => { if (!drawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('mouseup',    () => { drawing = false; predict(); });
canvas.addEventListener('mouseleave', () => { if (drawing) { drawing = false; predict(); } });

canvas.addEventListener('touchstart', e => { e.preventDefault(); drawing = true; ctx.beginPath(); const p = getPos(e); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('touchmove',  e => { e.preventDefault(); if (!drawing) return; const p = getPos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('touchend',   () => { drawing = false; predict(); });

document.getElementById('btn-clear').onclick = () => {
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('pred-digit').textContent = '–';
  document.getElementById('conf-text').textContent  = 'Confidence: –';
  document.getElementById('conf-bar').style.width   = '0%';
  document.getElementById('top3').innerHTML = '';
};

document.getElementById('btn-predict').onclick = predict;

async function predict() {
  const dataURL = canvas.toDataURL('image/png');
  const res = await fetch('/predict', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ image: dataURL }),
  });
  const data = await res.json();

  document.getElementById('pred-digit').textContent = data.digit;
  document.getElementById('conf-text').textContent  = `Confidence: ${data.confidence.toFixed(1)}%`;
  document.getElementById('conf-bar').style.width   = `${data.confidence}%`;

  const top3El = document.getElementById('top3');
  top3El.innerHTML = '<strong>Top 3</strong>';
  data.top3.forEach(([d, p]) => {
    const row = document.createElement('div');
    row.className = 'top3-row';
    row.innerHTML = `<span>${d}</span><span>${p.toFixed(1)}%</span>`;
    top3El.appendChild(row);
    const barWrap = document.createElement('div');
    barWrap.style.cssText = 'background:#313244;border-radius:4px;height:6px;margin-bottom:4px;overflow:hidden;';
    const fill = document.createElement('span');
    fill.className = 'bar';
    fill.style.width = `${p}%`;
    fill.style.display = 'block';
    barWrap.appendChild(fill);
    top3El.appendChild(barWrap);
  });
}
</script>
</body>
</html>
"""

# ── Flask app ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
model = None


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    data_url = request.json["image"]
    # Decode base64 PNG from the canvas
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    pil_img = Image.open(io.BytesIO(img_bytes))

    tensor = preprocess_image(pil_img)
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top_probs, top_classes = probs.topk(3)
    return jsonify({
        "digit":      top_classes[0].item(),
        "confidence": float(top_probs[0]) * 100,
        "top3":       [[top_classes[i].item(), float(top_probs[i]) * 100] for i in range(3)],
    })


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import webbrowser, threading

    if os.path.exists(MODEL_PATH):
        print(f"Loading saved model from '{MODEL_PATH}' ...")
        model = load_model()
    else:
        print("No saved model found. Starting training ...")
        model = train_model()

    print("\nStarting web server at http://localhost:5001")
    print("Press Ctrl+C to quit.\n")

    threading.Timer(1.0, lambda: webbrowser.open("http://localhost:5001")).start()
    app.run(host="localhost", port=5001, debug=False)
