const { spawn } = require('child_process');
const readline = require('readline');
const { ApiError } = require('../utils/errors');

const REQUEST_TIMEOUT_MS = 10000;
const STARTUP_TIMEOUT_MS = 60000;

/**
 * Wraps one long-lived Python worker so the model is loaded exactly once.
 * Talks JSON-lines over stdin/stdout (see ml/predict_service.py).
 */
class MlService {
  constructor({ pythonBin, script, modelPath }) {
    this.pythonBin = pythonBin;
    this.script = script;
    this.modelPath = modelPath;
    this.proc = null;
    this.ready = false;
    this.nextId = 1;
    this.pending = new Map();
    this.stopping = false;
  }

  start() {
    return new Promise((resolve, reject) => {
      const env = { ...process.env, MODEL_PATH: this.modelPath, PYTHONUNBUFFERED: '1' };
      const proc = spawn(this.pythonBin, ['-I', this.script], { env, stdio: ['pipe', 'pipe', 'pipe'] });
      this.proc = proc;
      this.ready = false;

      const timer = setTimeout(() => {
        reject(new Error('Python prediction worker did not become ready in time'));
        proc.kill();
      }, STARTUP_TIMEOUT_MS);

      proc.stderr.on('data', (d) => process.stderr.write(`[python] ${d}`));

      proc.once('error', (err) => {
        clearTimeout(timer);
        reject(new Error(`Could not start Python (${this.pythonBin}): ${err.message}`));
      });

      proc.once('exit', (code, signal) => {
        clearTimeout(timer);
        const wasReady = this.ready;
        this.ready = false;
        this._failAll(new ApiError(503, 'MODEL_UNAVAILABLE', 'Prediction model is unavailable'));
        if (this.stopping) return;
        console.error(`[ml] worker exited (code=${code}, signal=${signal})`);
        if (!wasReady) {
          // Failed during startup (e.g. model file missing): surface to start() caller.
          reject(new Error(`Python worker exited early (code=${code}). Is the model trained?`));
        } else {
          // Crashed while serving: try to bring it back.
          setTimeout(() => {
            this.start().then(
              () => console.log('[ml] worker restarted'),
              (err) => console.error('[ml] restart failed:', err.message)
            );
          }, 1000);
        }
      });

      const rl = readline.createInterface({ input: proc.stdout });
      rl.on('line', (line) => {
        let msg;
        try {
          msg = JSON.parse(line);
        } catch {
          return; // ignore stray non-protocol output
        }
        if (msg.ready) {
          this.ready = true;
          clearTimeout(timer);
          resolve();
          return;
        }
        const entry = this.pending.get(msg.id);
        if (!entry) return;
        this.pending.delete(msg.id);
        clearTimeout(entry.timer);
        if (msg.ok) entry.resolve({ prediction: msg.prediction, probability: msg.probability });
        else entry.reject(new ApiError(500, 'PREDICTION_FAILED', msg.error || 'Prediction failed'));
      });
    });
  }

  predict(features) {
    if (!this.ready || !this.proc || this.proc.killed) {
      return Promise.reject(new ApiError(503, 'MODEL_UNAVAILABLE', 'Prediction model is unavailable'));
    }
    return new Promise((resolve, reject) => {
      const id = this.nextId++;
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new ApiError(504, 'PREDICTION_TIMEOUT', 'Prediction timed out'));
      }, REQUEST_TIMEOUT_MS);
      this.pending.set(id, { resolve, reject, timer });
      this.proc.stdin.write(JSON.stringify({ id, features }) + '\n');
    });
  }

  _failAll(err) {
    for (const [id, entry] of this.pending) {
      clearTimeout(entry.timer);
      entry.reject(err);
      this.pending.delete(id);
    }
  }

  stop() {
    this.stopping = true;
    if (this.proc) this.proc.kill();
  }
}

module.exports = { MlService };
