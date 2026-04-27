# CLAUDE.md — desktop_version

<!-- Created: 2026-04-27 15:00 -->

This file provides guidance for the **desktop version** of the handwritten digit recognizer.

## Overview

PyQt6 (또는 PySide6) 기반 네이티브 데스크톱 애플리케이션. 별도 브라우저 없이 독립 실행되며, 캔버스에 손글씨 숫자를 그리면 CNN 모델이 실시간으로 인식한다.

## Running the app

```bash
cd desktop_version
python3 app.py
```

## Architecture

| File | Role |
|---|---|
| `app.py` | 애플리케이션 진입점, `QApplication` 초기화 |
| `model.py` | `DigitCNN` 정의, 학습(`train_model`) 및 로드(`load_model`) |
| `preprocess.py` | QImage → 1×1×28×28 정규화 텐서 변환 |
| `ui/main_window.py` | 메인 윈도우 (캔버스 + 결과 표시 패널) |
| `ui/canvas_widget.py` | 마우스/터치 드로잉 위젯 |
| `digit_cnn.pth` | 학습된 모델 가중치 (첫 실행 시 자동 생성) |

## Key constants

```python
MODEL_PATH = "digit_cnn.pth"
EPOCHS     = 5
BATCH_SIZE = 64
WINDOW_SIZE = (600, 400)
```

## Dependencies

```bash
pip3 install torch torchvision pyqt6 pillow numpy
```

## Constraints

- **tkinter 사용 불가** — macOS 26 Tahoe 베타에서 번들 Tcl/Tk 호환성 문제. GUI 프레임워크는 PyQt6 또는 PySide6을 사용할 것.
- MNIST 정규화 파라미터: mean=0.1307, std=0.3081
- 모델 가중치(`digit_cnn.pth`)는 web_version과 공유 가능 (동일 아키텍처 사용 시).
