// frontend/test/test_sendAction.test.js
// test/test_sendAction.test.js
const request = require("supertest");
const assert = require("assert");
const app = require("../server");  

describe("UT-30: Frontend JS – sendAction()", () => {
  const creds = { username: "testuser_jest", password: "password" };

  it("returns correct shape in API call", async () => {
    const agent = request.agent(app);

    // ok if already exists; login will still succeed
    await agent.post("/register").send(creds);
    await agent.post("/login").send(creds).expect(302);

    const payload = {
      session_uuid: "test-session-123",
      ticker: "AAPL",
      action: 1,
      correct: 1,
      reward: 5.0,
    };

    const res = await agent.post("/record_trade").send(payload);

    assert.strictEqual(res.status, 200);
    assert.strictEqual(typeof res.body, "object");
    assert.strictEqual(res.body.success, true);
    assert.strictEqual(typeof res.body.success, "boolean");
  });
});



//ut-30