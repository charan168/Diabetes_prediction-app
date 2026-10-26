const path = require('path');
require('dotenv').config({ path: path.join(__dirname, '..', '.env') });

const required = ['MONGODB_URI', 'DASHBOARD_USER', 'DASHBOARD_PASSWORD'];

function loadConfig(env = process.env) {
  const missing = required.filter((key) => !env[key]);
  if (missing.length) {
    throw new Error(
      `Missing required environment variables: ${missing.join(', ')}. Copy .env.example to .env.`
    );
  }
  const mlDir = path.resolve(__dirname, '..', '..', 'ml');
  return {
    port: Number(env.PORT) || 5000,
    nodeEnv: env.NODE_ENV || 'development',
    corsOrigin: (env.CORS_ORIGIN || 'http://localhost:5173').split(',').map((s) => s.trim()),
    mongoUri: env.MONGODB_URI,
    dashboardUser: env.DASHBOARD_USER,
    dashboardPassword: env.DASHBOARD_PASSWORD,
    pythonBin: env.PYTHON_BIN || 'python3',
    predictScript: path.join(mlDir, 'predict_service.py'),
    metricsFile: path.join(mlDir, 'artifacts', 'metrics.json'),
    modelPath: env.MODEL_PATH || path.join(mlDir, 'artifacts', 'diabetes_rf.joblib'),
  };
}

module.exports = { loadConfig };
