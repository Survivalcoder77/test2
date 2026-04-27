# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the app

```bash
python3 digit_recognizer.py
```

Opens a Flask web server at `http://localhost:5001` and auto-launches the browser. On the first run (no `digit_cnn.pth`), it trains the CNN on MNIST (~5 epochs, CPU, ~5–10 min); subsequent runs load the saved weights instantly.

To launch from Finder: double-click `숫자인식_실행.command`.

## Architecture

Everything lives in a single file, `digit_recognizer.py`, with four logical sections:

| Section | What it does |
|---|---|
| `DigitCNN` | Two conv blocks (32→64→128 channels) + two max-pools → 7×7 feature map → 256-unit FC head → 10 logits |
| `train_model` / `load_model` | Downloads MNIST via torchvision, trains with Adam/CrossEntropyLoss, saves/loads `digit_cnn.pth` |
| `preprocess_image` | Converts a PIL image from the canvas (any size, RGBA) to a normalised 1×1×28×28 tensor using MNIST stats (mean=0.1307, std=0.3081) |
| Flask routes + `HTML` | `GET /` serves an inline HTML5 canvas page; `POST /predict` receives a base64 PNG, runs inference, returns `{digit, confidence, top3}` |

The model is held in a module-level `model` variable that is populated before `app.run()` and read inside the `/predict` route.

## Key constants

```python
MODEL_PATH = "digit_cnn.pth"   # relative to CWD — run from project root
EPOCHS     = 5
BATCH_SIZE = 64
PORT       = 5001               # 5000 is occupied by macOS AirPlay Receiver
```

## Dependencies

No `requirements.txt`; install manually if needed:

```bash
pip3 install torch torchvision flask pillow numpy
```

Tested with: `torch 2.8.0`, `torchvision 0.23.0`, Python 3 system install (`/usr/bin/python3`) on macOS Darwin 25.3.0.

> **Note:** tkinter is not usable on this machine (macOS 26 Tahoe beta incompatibility with the bundled Tcl/Tk). All UI must stay web-based (Flask + HTML5 Canvas).
