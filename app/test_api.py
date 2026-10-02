"""Smoke test for the API (no server needed): python app/test_api.py"""
import os, sys, tempfile
os.environ["LOG_PATH"] = os.path.join(tempfile.mkdtemp(), "log.csv")
sys.path.insert(0, os.path.dirname(__file__))
from fastapi.testclient import TestClient
import app as appmod

c = TestClient(appmod.app)
assert c.get("/health").json()["n_intents"] == 150
for q in ["what is my account balance", "set an alarm for 7am", "who won the cricket match yesterday"]:
    r = c.post("/predict", json={"text": q}).json()
    assert len(r["top3"]) == 3 and "reply" in r
    print(f"{q!r:45s} -> {r['intent']:20s} conf={r['top3'][0]['confidence']:.3f}")
assert c.post("/predict", json={"text": "  "}).status_code == 400
assert c.get("/logs").status_code == 200
print("all API tests passed; log rows:", sum(1 for _ in open(os.environ["LOG_PATH"])) - 1)
