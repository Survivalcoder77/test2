#!/bin/zsh
# Double-click this file in Finder to launch the Digit Recognizer app.

cd "$(dirname "$0")"

echo "================================================"
echo "  Handwritten Digit Recognizer"
echo "================================================"
echo ""

/usr/bin/python3 digit_recognizer.py

echo ""
echo "Server stopped. You can close this window."
