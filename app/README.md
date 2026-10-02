---
title: Help-desk Intent Bot
emoji: 💬
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.29.0
app_file: app.py
pinned: false
---

# Help-desk Intent Bot (CLINC150)

MLP intent classifier (150 intents) with out-of-scope detection by softmax-confidence threshold.

* `POST /predict` with `{"text": "..."}` returns intent, reply, OOS flag, top-3 intents
* `GET /health` model info, `GET /logs` download the conversation log
* `/` Gradio chat UI with a debug panel (top-3 intents and confidences)
