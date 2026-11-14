import assert from "node:assert/strict";
import { test } from "node:test";
import { percentile, pyFixed, pyG, pyRound, pyRoundInt } from "../src/pymath.js";

test("pyFixed rounds exact ties to even, like Python", () => {
  assert.equal(pyFixed(0.125, 2), "0.12"); // exact tie in binary: Python gives 0.12, toFixed 0.13
  assert.equal(pyFixed(0.375, 2), "0.38");
  assert.equal(pyFixed(2.5, 0), "2");
  assert.equal(pyFixed(3.5, 0), "4");
  assert.equal(pyFixed(-0.125, 2), "-0.12");
  assert.equal(pyFixed(1.005, 2), "1.00"); // 1.005 is below the tie in binary
  assert.equal(pyFixed(7.0349, 2), "7.03");
  assert.equal(pyFixed(9.95, 1), "9.9"); // 9.95 is just under in binary
  assert.equal(pyFixed(0.995, 2), "0.99");
});

test("pyRound and pyRoundInt match Python's round", () => {
  assert.equal(pyRound(0.125, 2), 0.12);
  assert.equal(pyRound(20.049999, 2), 20.05);
  assert.equal(pyRoundInt(2.5), 2);
  assert.equal(pyRoundInt(3.5), 4);
  assert.equal(pyRoundInt(-2.5), -2);
  assert.equal(pyRoundInt(59.99999999), 60);
});

test("pyG matches Python's :g", () => {
  assert.equal(pyG(6), "6");
  assert.equal(pyG(0.5), "0.5");
  assert.equal(pyG(6.5), "6.5");
  assert.equal(pyG(-44), "-44");
  assert.equal(pyG(0.9), "0.9");
  assert.equal(pyG(1234567), "1.23457e+06");
  assert.equal(pyG(0.00001), "1e-05");
  assert.equal(pyG(0.0001), "0.0001");
  assert.equal(pyG(6.1234567), "6.12346");
});

test("percentile is numpy's linear method", () => {
  assert.equal(percentile([1, 2, 3, 4], 90), 3.7);
  assert.equal(percentile([5], 90), 5);
  assert.equal(percentile([-20, -10], 50), -15);
});
