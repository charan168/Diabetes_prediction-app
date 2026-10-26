/**
 * Integration tests: real Express app + real Python model worker + real MongoDB.
 * Uses TEST_MONGODB_URI (default: local mongod, database `disease_prediction_test`).
 * Run:  npm test
 */
const test = require('node:test');
const assert = require('node:assert/strict');
const mongoose = require('mongoose');
const { loadConfig } = require('../src/config');
const { MlService } = require('../src/services/mlService');
const { createApp } = require('../src/app');
const Prediction = require('../src/models/Prediction');

const AUTH_USER = 'tester';
const AUTH_PASS = 's3cret:with:colons';
const basic = (u, p) => 'Basic ' + Buffer.from(`${u}:${p}`).toString('base64');
const GOOD_AUTH = { Authorization: basic(AUTH_USER, AUTH_PASS) };

const valid = {
  Pregnancies: 2,
  Glucose: 120,
  BloodPressure: 70,
  SkinThickness: 25,
  Insulin: 80,
  BMI: 28.5,
  DiabetesPedigreeFunction: 0.45,
  Age: 35,
};

let server;
let base;
let ml;

test.before(async () => {
  const config = loadConfig({
    ...process.env,
    MONGODB_URI: process.env.TEST_MONGODB_URI || 'mongodb://127.0.0.1:27017/disease_prediction_test',
    DASHBOARD_USER: AUTH_USER,
    DASHBOARD_PASSWORD: AUTH_PASS,
  });
  ml = new MlService({
    pythonBin: config.pythonBin,
    script: config.predictScript,
    modelPath: config.modelPath,
  });
  await ml.start();
  await mongoose.connect(config.mongoUri, { serverSelectionTimeoutMS: 5000 });
  await Prediction.deleteMany({});
  const app = createApp({ ml, config });
  await new Promise((resolve) => {
    server = app.listen(0, resolve);
  });
  base = `http://127.0.0.1:${server.address().port}/api`;
});

test.after(async () => {
  await Prediction.deleteMany({});
  server.close();
  ml.stop();
  await mongoose.disconnect();
});

const post = (body, headers = {}) =>
  fetch(`${base}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...headers },
    body: typeof body === 'string' ? body : JSON.stringify(body),
  });

test('health reports database and model ready', async () => {
  const res = await fetch(`${base}/health`);
  assert.equal(res.status, 200);
  const json = await res.json();
  assert.deepEqual(json.data, { status: 'ok', database: true, model: true });
});

test('POST /predict returns prediction + probability, with disclaimer, and saves to MongoDB', async () => {
  const res = await post(valid);
  assert.equal(res.status, 201);
  const { success, data } = await res.json();
  assert.equal(success, true);
  assert.ok([0, 1].includes(data.prediction));
  assert.equal(data.label, data.prediction === 1 ? 'Positive' : 'Negative');
  assert.ok(data.probability >= 0 && data.probability <= 1);
  assert.match(data.disclaimer, /not medical advice/i);

  const saved = await Prediction.findById(data.id).lean();
  assert.ok(saved, 'record persisted');
  assert.equal(saved.input.Glucose, 120);
  assert.equal(saved.prediction, data.prediction);
  assert.equal(saved.probability, data.probability);
  assert.ok(saved.createdAt instanceof Date);
});

test('optional fields (SkinThickness, Insulin) may be null or omitted', async () => {
  const { SkinThickness, Insulin, ...rest } = valid;
  const res = await post({ ...rest, Insulin: null });
  assert.equal(res.status, 201);
  const { data } = await res.json();
  assert.equal(data.input.SkinThickness, null);
  assert.equal(data.input.Insulin, null);
});

test('same input gives a deterministic prediction', async () => {
  const a = (await (await post(valid)).json()).data;
  const b = (await (await post(valid)).json()).data;
  assert.equal(a.probability, b.probability);
});

test('validation: missing required field -> 400 with field detail, nothing saved', async () => {
  const before = await Prediction.countDocuments();
  const { Glucose, ...rest } = valid;
  const res = await post(rest);
  assert.equal(res.status, 400);
  const json = await res.json();
  assert.equal(json.success, false);
  assert.equal(json.error.code, 'VALIDATION_ERROR');
  assert.ok(json.error.details.some((d) => d.field === 'Glucose'));
  assert.equal(await Prediction.countDocuments(), before);
});

test('validation: wrong types, ranges, non-integers, NaN-like values', async () => {
  const cases = [
    [{ ...valid, Age: '35' }, 'Age'],
    [{ ...valid, Age: 0 }, 'Age'],
    [{ ...valid, Age: 150 }, 'Age'],
    [{ ...valid, Age: 35.5 }, 'Age'],
    [{ ...valid, Glucose: -5 }, 'Glucose'],
    [{ ...valid, BMI: 500 }, 'BMI'],
    [{ ...valid, Pregnancies: 99 }, 'Pregnancies'],
    [{ ...valid, Insulin: -1 }, 'Insulin'],
    [{ ...valid, DiabetesPedigreeFunction: true }, 'DiabetesPedigreeFunction'],
    [{ ...valid, BloodPressure: [70] }, 'BloodPressure'],
  ];
  for (const [body, field] of cases) {
    const res = await post(body);
    assert.equal(res.status, 400, `expected 400 for ${field}=${JSON.stringify(body[field])}`);
    const json = await res.json();
    assert.ok(json.error.details.some((d) => d.field === field), `detail for ${field}`);
  }
});

test('validation: body that is not an object, empty body, malformed JSON', async () => {
  assert.equal((await post('[]')).status, 400);
  assert.equal((await post('null')).status, 400);
  const bad = await post('{"Age": ');
  assert.equal(bad.status, 400);
  assert.equal((await bad.json()).error.code, 'INVALID_JSON');
  const empty = await fetch(`${base}/predict`, { method: 'POST' });
  assert.equal(empty.status, 400);
});

test('unknown fields are ignored and not stored', async () => {
  const res = await post({ ...valid, isAdmin: true, $where: '1' });
  assert.equal(res.status, 201);
  const { data } = await res.json();
  const saved = await Prediction.findById(data.id).lean();
  assert.equal(saved.input.isAdmin, undefined);
});

test('oversized body is rejected with 413', async () => {
  const res = await post({ ...valid, junk: 'x'.repeat(20000) });
  assert.equal(res.status, 413);
});

test('dashboard endpoints require valid basic auth', async () => {
  for (const path of ['/predictions', '/metrics', '/model-evaluation']) {
    const none = await fetch(base + path);
    assert.equal(none.status, 401, `${path} without auth`);
    assert.match(none.headers.get('www-authenticate') || '', /Basic/);
    const wrongPass = await fetch(base + path, { headers: { Authorization: basic(AUTH_USER, 'nope') } });
    assert.equal(wrongPass.status, 401, `${path} wrong password`);
    const wrongUser = await fetch(base + path, { headers: { Authorization: basic('x', AUTH_PASS) } });
    assert.equal(wrongUser.status, 401, `${path} wrong user`);
    const garbage = await fetch(base + path, { headers: { Authorization: 'Basic !!!' } });
    assert.equal(garbage.status, 401, `${path} garbage header`);
    const ok = await fetch(base + path, { headers: GOOD_AUTH });
    assert.equal(ok.status, 200, `${path} with auth`);
  }
});

test('GET /predictions returns the latest 10, newest first', async () => {
  await Prediction.deleteMany({});
  for (let i = 0; i < 13; i++) {
    const res = await post({ ...valid, Age: 20 + i });
    assert.equal(res.status, 201);
    await new Promise((r) => setTimeout(r, 5)); // distinct timestamps
  }
  const res = await fetch(`${base}/predictions`, { headers: GOOD_AUTH });
  const { data } = await res.json();
  assert.equal(data.length, 10);
  assert.equal(data[0].input.Age, 32, 'newest first');
  assert.equal(data[9].input.Age, 23);
  const times = data.map((d) => new Date(d.createdAt).getTime());
  assert.deepEqual(times, [...times].sort((a, b) => b - a));
});

test('GET /metrics matches the database and percentages add up', async () => {
  const res = await fetch(`${base}/metrics`, { headers: GOOD_AUTH });
  const { data } = await res.json();
  const total = await Prediction.countDocuments();
  const pos = await Prediction.countDocuments({ prediction: 1 });
  assert.equal(data.total, total);
  assert.equal(data.positive.count, pos);
  assert.equal(data.negative.count, total - pos);
  assert.ok(Math.abs(data.positive.percentage + data.negative.percentage - 100) < 0.2);
});

test('GET /metrics with no data returns zeros (no divide-by-zero)', async () => {
  await Prediction.deleteMany({});
  const res = await fetch(`${base}/metrics`, { headers: GOOD_AUTH });
  const { data } = await res.json();
  assert.deepEqual(data, {
    total: 0,
    positive: { count: 0, percentage: 0 },
    negative: { count: 0, percentage: 0 },
  });
});

test('GET /model-evaluation returns held-out metrics from training', async () => {
  const res = await fetch(`${base}/model-evaluation`, { headers: GOOD_AUTH });
  const { data } = await res.json();
  assert.ok(data.test_metrics_held_out.accuracy > 0);
  assert.ok(data.n_test > 0);
  assert.ok(data.test_metrics_held_out.confusion_matrix);
});

test('unknown routes return a JSON 404; wrong method too', async () => {
  const res = await fetch(`${base}/nope`);
  assert.equal(res.status, 404);
  assert.equal((await res.json()).error.code, 'NOT_FOUND');
  assert.equal((await fetch(`${base}/predict`)).status, 404);
});

test('concurrent predictions are matched to the right responses', async () => {
  const ages = [22, 33, 44, 55, 66, 77];
  const results = await Promise.all(
    ages.map(async (age) => (await (await post({ ...valid, Age: age })).json()).data)
  );
  assert.deepEqual(results.map((r) => r.input.Age), ages);
});
