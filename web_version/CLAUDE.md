# CLAUDE.md — web_version

<!-- Created: 2026-04-27 15:00 -->

This file provides guidance for the **web version** of the handwritten digit recognizer.

## Overview

Flask + HTML5 Canvas 기반 웹 애플리케이션. 브라우저에서 손글씨 숫자를 그리면 CNN 모델이 실시간으로 인식한다.

## Running the app

```bash
cd web_version
python3 app.py
```

브라우저가 자동으로 열리며 `http://localhost:5001` 에 접속된다.

## Architecture

| File | Role |
|---|---|
| `app.py` | Flask 서버 + `/predict` API 엔드포인트 |
| `model.py` | `DigitCNN` 정의, 학습(`train_model`) 및 로드(`load_model`) |
| `preprocess.py` | PIL 이미지 → 1×1×28×28 정규화 텐서 변환 |
| `templates/index.html` | HTML5 Canvas UI |
| `digit_cnn.pth` | 학습된 모델 가중치 (첫 실행 시 자동 생성) |

## Key constants

```python
MODEL_PATH = "digit_cnn.pth"
EPOCHS     = 5
BATCH_SIZE = 64
PORT       = 5001   # 5000은 macOS AirPlay Receiver가 점유
```

## Dependencies

```bash
pip3 install torch torchvision flask pillow numpy
```

## Constraints

- **tkinter 사용 불가** — macOS 26 Tahoe 베타에서 번들 Tcl/Tk 호환성 문제. UI는 반드시 웹 기반으로 유지할 것.
- MNIST 정규화 파라미터: mean=0.1307, std=0.3081
