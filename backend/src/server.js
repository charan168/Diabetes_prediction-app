const mongoose = require('mongoose');
const { loadConfig } = require('./config');
const { MlService } = require('./services/mlService');
const { createApp } = require('./app');

async function main() {
  const config = loadConfig();

  // 1. Load the model ONCE (Python worker stays alive for the life of the server).
  const ml = new MlService({
    pythonBin: config.pythonBin,
    script: config.predictScript,
    modelPath: config.modelPath,
  });
  await ml.start();
  console.log('[ml] model loaded');

  // 2. Connect to MongoDB (fail fast if it's unreachable).
  await mongoose.connect(config.mongoUri, { serverSelectionTimeoutMS: 5000 });
  console.log('[db] connected to MongoDB');

  // 3. Serve.
  const app = createApp({ ml, config });
  const server = app.listen(config.port, () => {
    console.log(`[api] listening on http://localhost:${config.port}`);
  });

  const shutdown = async () => {
    console.log('\nShutting down...');
    server.close();
    ml.stop();
    await mongoose.disconnect();
    process.exit(0);
  };
  process.on('SIGINT', shutdown);
  process.on('SIGTERM', shutdown);
}

main().catch((err) => {
  console.error('Startup failed:', err.message);
  process.exit(1);
});
