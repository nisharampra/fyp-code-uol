//security test
const request = require("supertest");
const assert = require("assert");
const express = require("express");
const app = require("../server"); 
const sqlite3 = require("sqlite3").verbose();

const db = new sqlite3.Database("./auth.db");

describe("Security Tests", () => {
  describe("UT-22: SQL Injection on Login", () => {
    it("should reject login attempt using SQL injection payload", (done) => {
      request(app)
        .post("/login")
        .send({
          username: "' OR 1=1 --",
          password: "anything",
        })
        .expect(200)
        .end((err, res) => {
          if (err) return done(err);
          assert(res.text.includes("Invalid username or password"));
          done();
        });
    });

    it("should not insert injection payload into DB", (done) => {
      db.get(
        "SELECT * FROM users WHERE username = ?",
        ["' OR 1=1 --"],
        (err, row) => {
          if (err) return done(err);
          assert.strictEqual(row, undefined);
          done();
        }
      );
    });
  });

  describe("UT-23: Session Required Routes", () => {
    const protectedRoutes = ["/profile", "/practice?session_id=abc&ticker=FAKE"];

    protectedRoutes.forEach((route) => {
      it(`should redirect unauthenticated access to ${route}`, (done) => {
        request(app)
          .get(route)
          .expect(302)
          .expect("Location", /\/login/, done);
      });
    });
  });
});
