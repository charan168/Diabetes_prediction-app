const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const { createApiRouter } = require('./routes/api');
const { notFound, errorHandler } = require('./middleware/errorHandler');

function createApp({ ml, config }) {
  const app = express();
  app.disable('x-powered-by');
  app.use(helmet());
  app.use(
    cors({
      origin: config.corsOrigin,
      allowedHeaders: ['Content-Type', 'Authorization'],
      methods: ['GET', 'POST', 'OPTIONS'],
    })
  );
  app.use(express.json({ limit: '10kb' }));
  app.use('/api', createApiRouter({ ml, config }));
  app.use(notFound);
  app.use(errorHandler);
  return app;
}

module.exports = { createApp };
