"""
Help-desk chatbot: FastAPI inference API + Gradio chat UI in one process.
cd 
  POST /predict   {"text": "..."}  -> intent, reply, is_oos, top-3, threshold
  GET  /health                      -> model info
  GET  /logs                        -> download the conversation log (CSV)
  /                                 -> Gradio chat UI (calls /predict over HTTP)

Run locally:  python app.py   then open http://localhost:7860
"""
import csv, datetime, json, os, re, time, uuid
from pathlib import Path

import gradio as gr
import joblib, requests, torch, torch.nn as nn, uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

HERE = Path(__file__).parent
ART = HERE / "artifacts"
PORT = int(os.environ.get("PORT", 7860))
API_URL = os.environ.get("API_URL", f"http://127.0.0.1:{PORT}/predict")
LOG_PATH = Path(os.environ.get("LOG_PATH", HERE / "chat_log.csv"))

# ------------------------------------------------------------------ model
cfg = json.load(open(ART / "config.json"))
INTENTS, THRESHOLD = cfg["intents"], cfg["threshold"]
TEMPLATES = json.load(open(ART / "templates.json"))
vectorizer = joblib.load(ART / "tfidf.joblib")
OOS_REPLY = ("Sorry, I don't think I can help with that. I handle things like banking, travel, "
             "work, cooking, your car and small talk. Could you rephrase your question?")


class MLP(nn.Module):  # identical to the training notebook
    def __init__(self, d_in, hidden, n_classes, dropout=0.3, batchnorm=True):
        super().__init__()
        layers, prev = [], d_in
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers += [nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


model = MLP(cfg["d_in"], cfg["hidden"], len(INTENTS), cfg["dropout"], cfg.get("batchnorm", True))
model.load_state_dict(torch.load(ART / "mlp.pt", map_location="cpu"))
model.eval()


def clean(t: str) -> str:  # identical to the training notebook
    t = re.sub(r"[^a-z0-9' ]", " ", t.lower())
    return re.sub(r"\s+", " ", t).strip()


def classify(text: str):
    x = torch.from_numpy(vectorizer.transform([clean(text)]).toarray()).float()
    with torch.no_grad():
        p = torch.softmax(model(x), dim=1)[0]
    conf, idx = torch.topk(p, 3)
    return [{"intent": INTENTS[i], "confidence": round(float(c), 4)} for c, i in zip(conf, idx)]


# ------------------------------------------------------------------ logging
LOG_FIELDS = ["timestamp", "session_id", "user_text", "predicted_intent", "confidence",
              "is_oos", "threshold", "top3", "reply", "latency_ms"]


def log_turn(row: dict):
    new = not LOG_PATH.exists()
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


# ------------------------------------------------------------------ API
app = FastAPI(title="Help-desk intent API", version="1.0")


class Query(BaseModel):
    text: str
    session_id: str = "api"


@app.post("/predict")
def predict(q: Query):
    t0 = time.perf_counter()
    text = q.text.strip()
    if not text:
        return JSONResponse({"error": "empty text"}, status_code=400)
    top3 = classify(text)
    is_oos = top3[0]["confidence"] < THRESHOLD
    intent = "oos" if is_oos else top3[0]["intent"]
    reply = OOS_REPLY if is_oos else TEMPLATES.get(intent, f"I can help with {intent.replace('_', ' ')}.")
    latency = round((time.perf_counter() - t0) * 1000, 1)
    log_turn({"timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
              "session_id": q.session_id, "user_text": text, "predicted_intent": intent,
              "confidence": top3[0]["confidence"], "is_oos": is_oos, "threshold": THRESHOLD,
              "top3": json.dumps(top3), "reply": reply, "latency_ms": latency})
    return {"intent": intent, "reply": reply, "is_oos": is_oos, "top3": top3,
            "threshold": THRESHOLD, "latency_ms": latency}


@app.get("/health")
def health():
    return {"status": "ok", "n_intents": len(INTENTS), "threshold": THRESHOLD,
            "hidden": cfg["hidden"], "test_in_scope_acc": cfg.get("test_in_scope_acc"),
            "test_oos_recall_at_T": cfg.get("test_oos_recall_at_T")}


@app.get("/logs")
def logs():
    if not LOG_PATH.exists():
        return JSONResponse({"detail": "no conversations logged yet"}, status_code=404)
    return FileResponse(LOG_PATH, media_type="text/csv", filename="chat_log.csv")


# ------------------------------------------------------------------ UI
def chat(message, history, session_id):
    if not message or not message.strip():
        return "", history, None, "", session_id
    r = requests.post(API_URL, json={"text": message, "session_id": session_id}, timeout=10).json()
    history = (history or []) + [{"role": "user", "content": message},
                                 {"role": "assistant", "content": r["reply"]}]
    label = {d["intent"]: d["confidence"] for d in r["top3"]}
    decision = (f"**Decision:** {'OUT-OF-SCOPE (asked to rephrase)' if r['is_oos'] else 'in-scope: ' + r['intent']}  \n"
                f"**Top-1 confidence:** {r['top3'][0]['confidence']:.3f} vs threshold T = {r['threshold']:.3f}  \n"
                f"**Latency:** {r['latency_ms']} ms")
    return "", history, label, decision, session_id


EXAMPLES = ["what's my checking account balance", "book me a flight to new york on friday",
            "add milk to my shopping list", "set a timer for 10 minutes",
            "i lost my credit card", "who won the football game last night",
            "how do i grow tomatoes on my balcony", "tell me a joke"]

with gr.Blocks(title="Help-desk Intent Bot") as demo:
    gr.Markdown("## Help-desk Intent Bot\nMLP intent classifier over 150 CLINC150 intents, "
                "with out-of-scope detection by softmax-confidence threshold.")
    sid = gr.State(lambda: uuid.uuid4().hex[:8])
    with gr.Row():
        with gr.Column(scale=3):
            bot = gr.Chatbot(height=460, label="Chat")
            box = gr.Textbox(placeholder="Type a message and press Enter", show_label=False)
            with gr.Row():
                send = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear chat")
            gr.Examples(EXAMPLES, inputs=box)
        with gr.Column(scale=2):
            gr.Markdown("### Debug panel")
            dbg = gr.Label(num_top_classes=3, label="Top-3 intents (softmax confidence, before threshold)")
            decision = gr.Markdown()
            gr.Markdown(f"Model: MLP {cfg['hidden']} on TF-IDF, T = {THRESHOLD:.3f}. "
                        "Every turn is logged; download at `/logs`.")
    outs = [box, bot, dbg, decision, sid]
    box.submit(chat, [box, bot, sid], outs)
    send.click(chat, [box, bot, sid], outs)
    clear.click(lambda: ([], None, ""), None, [bot, dbg, decision])

app = gr.mount_gradio_app(app, demo, path="/")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
