# locustfile.py
from locust import HttpUser, task, between
import random

class BotUser(HttpUser):
    wait_time = between(0.2, 0.8)  # user think-time

    def on_start(self):
        r = self.client.get("/tickers")
        self.ticker = (r.json().get("tickers") or ["APPLE"])[0]
        s = self.client.post("/start_session", json={"ticker": self.ticker})
        self.session_id = s.json()["session_id"]

    @task(3)
    def state_and_hint(self):
        self.client.get("/state_and_hint", params={"session_id": self.session_id})

    @task(2)
    def take_action(self):
        action = random.choice([0,1,2])
        self.client.post("/take_action", json={"session_id": self.session_id, "action": action})
