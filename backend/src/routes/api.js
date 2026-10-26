const express = require('express');
const fs = require('fs/promises');
const mongoose = require('mongoose');
const Prediction = require('../models/Prediction');
const { basicAuth } = require('../middleware/basicAuth');
const { validatePrediction } = require('../utils/validatePrediction');
const { ApiError } = require('../utils/errors');

const DISCLAIMER =
  'For demonstration purposes only. This is not medical advice and must not be used for diagnosis or treatment.';

const labelOf = (p) => (p === 1 ? 'Positive' : 'Negative');
const pct = (count, total) => (total === 0 ? 0 : Math.round((count / total) * 1000) / 10);

const serialize = (doc) => ({
  id: String(doc._id),
  input: doc.input,
  prediction: doc.prediction,
  label: labelOf(doc.prediction),
  probability: doc.probability,
  createdAt: doc.createdAt,
});

const wrap = (fn) => (req, res, next) => Promise.resolve(fn(req, res, next)).catch(next);

function createApiRouter({ ml, config }) {
  const router = express.Router();
  const auth = basicAuth({ user: config.dashboardUser, password: config.dashboardPassword });

  // Public: liveness + dependency status.
  router.get('/health', (_req, res) => {
    const dbReady = mongoose.connection.readyState === 1;
    res.status(dbReady && ml.ready ? 200 : 503).json({
      success: true,
      data: { status: dbReady && ml.ready ? 'ok' : 'degraded', database: dbReady, model: ml.ready },
    });
  });

  // Public: submit patient attributes, get a prediction, log it.
  router.post(
    '/predict',
    wrap(async (req, res) => {
      const { value, errors } = validatePrediction(req.body);
      if (errors.length) {
        throw new ApiError(400, 'VALIDATION_ERROR', 'Invalid patient data', errors);
      }
      const { prediction, probability } = await ml.predict(value);
      const doc = await Prediction.create({ input: value, prediction, probability });
      res.status(201).json({
        success: true,
        data: { ...serialize(doc), disclaimer: DISCLAIMER },
      });
    })
  );

  // Protected dashboard endpoints.
  router.get(
    '/predictions',
    auth,
    wrap(async (_req, res) => {
      const docs = await Prediction.find().sort({ createdAt: -1 }).limit(10).lean();
      res.json({ success: true, data: docs.map(serialize) });
    })
  );

  router.get(
    '/metrics',
    auth,
    wrap(async (_req, res) => {
      // Group by predicted class (0 / 1); plain $group/$sum works on any MongoDB-compatible server.
      const groups = await Prediction.aggregate([
        { $group: { _id: '$prediction', count: { $sum: 1 } } },
      ]);
      const countOf = (cls) => (groups.find((g) => g._id === cls) || { count: 0 }).count;
      const positive = countOf(1);
      const negative = countOf(0);
      const total = positive + negative;
      res.json({
        success: true,
        data: {
          total,
          positive: { count: positive, percentage: pct(positive, total) },
          negative: { count: negative, percentage: pct(negative, total) },
        },
      });
    })
  );

  // Protected: model quality measured on the held-out test split at training time.
  router.get(
    '/model-evaluation',
    auth,
    wrap(async (_req, res) => {
      let raw;
      try {
        raw = await fs.readFile(config.metricsFile, 'utf8');
      } catch {
        throw new ApiError(404, 'METRICS_NOT_FOUND', 'Run ml/train_model.py to generate model metrics');
      }
      res.json({ success: true, data: JSON.parse(raw) });
    })
  );

  return router;
}

module.exports = { createApiRouter, DISCLAIMER };
