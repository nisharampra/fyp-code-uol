// UT-24: Unique Users — Duplicate usernames throw error
const assert = require('assert');
const path = require('path');
const request = require('supertest');
const sqlite3 = require('sqlite3').verbose();

const app = require(path.join('..', 'server.js'));

describe('UT-24: Unique Users', function () {
  this.timeout(10000);

  before((done) => {
    const db = new sqlite3.Database('./auth.db');
    db.serialize(() => {
      db.run('DELETE FROM users', [], (err) => {
        db.close();
        done(err || undefined);
      });
    });
  });

  it('should not allow registering the same username twice', async () => {
    let res1 = await request(app)
      .post('/register')
      .type('form')
      .send({ username: 'alice', password: 'pw123' });
    assert.strictEqual(res1.status, 302);

    let res2 = await request(app)
      .post('/register')
      .type('form')
      .send({ username: 'alice', password: 'pw123' });
    assert.strictEqual(res2.status, 200);
    assert.ok(/Username already taken/i.test(res2.text));
  });
});
