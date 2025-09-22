// test/trade_record.test.js
const request = require("supertest");
const assert = require("assert");

// IMPORTANT: server.js should export { app, ready }
const { app, ready } = require("../server");

function randUser() {
  return "user" + Math.random().toString(36).slice(2, 8);
}

describe("UT-25: Trade Data Format", function () {
  this.timeout(10000);

  let agent;
  before(async () => {
    // wait for DB tables to exist
    if (ready && typeof ready.then === "function") {
      await ready;
    }
    agent = request.agent(app);

    // 1) register
    const username = randUser();
    const password = "pass123!";
    await agent.post("/register").send({ username, password });

    // 2) login (session cookie maintained by agent)
    await agent.post("/login").send({ username, password }).expect(302);
  });

  it("accepts a valid payload", async () => {
    const payload = {
      session_uuid: "abc-123",
      ticker: "AAPL",
      action: 1,      // 0=Hold,1=Buy,2=Sell
      correct: 1,     // 0/1 only
      reward: 12.34,  // number
    };

    const res = await agent.post("/record_trade").send(payload);
    assert.strictEqual(res.status, 200);
    assert.strictEqual(res.body.success, true);
  });

  it("rejects invalid action (must be 0/1/2)", async () => {
    const res = await agent.post("/record_trade").send({
      session_uuid: "abc-123",
      ticker: "AAPL",
      action: 3,      // invalid
      correct: 1,
      reward: 0,
    });
    assert.strictEqual(res.status, 400);
    assert.ok(res.body.error);
  });

  it("rejects invalid correct (must be 0/1)", async () => {
    const res = await agent.post("/record_trade").send({
      session_uuid: "abc-123",
      ticker: "AAPL",
      action: 0,
      correct: 2,     // invalid
      reward: 0,
    });
    assert.strictEqual(res.status, 400);
    assert.ok(res.body.error);
  });

  it("rejects non-numeric reward", async () => {
    const res = await agent.post("/record_trade").send({
      session_uuid: "abc-123",
      ticker: "AAPL",
      action: 2,
      correct: 0,
      reward: "not-a-number", // invalid
    });
    assert.strictEqual(res.status, 400);
    assert.ok(res.body.error);
  });

  it("rejects missing fields", async () => {
    const res = await agent.post("/record_trade").send({
      // session_uuid missing
      ticker: "AAPL",
      action: 1,
      correct: 1,
      reward: 1,
    });
    assert.strictEqual(res.status, 400);
    assert.ok(res.body.error);
  });
});

