const crypto = require('crypto');
const { ApiError } = require('../utils/errors');

// Compare via SHA-256 digests so lengths always match and timing doesn't leak the secret.
const digest = (s) => crypto.createHash('sha256').update(String(s)).digest();
const safeEqual = (a, b) => crypto.timingSafeEqual(digest(a), digest(b));

function basicAuth({ user, password }) {
  return (req, _res, next) => {
    const header = req.headers.authorization || '';
    const [scheme, encoded] = header.split(' ');
    if (scheme !== 'Basic' || !encoded) {
      return next(new ApiError(401, 'UNAUTHORIZED', 'Authentication required'));
    }
    const decoded = Buffer.from(encoded, 'base64').toString('utf8');
    const idx = decoded.indexOf(':');
    const u = idx === -1 ? decoded : decoded.slice(0, idx);
    const p = idx === -1 ? '' : decoded.slice(idx + 1);
    // Evaluate both comparisons (no short-circuit).
    const okUser = safeEqual(u, user);
    const okPass = safeEqual(p, password);
    if (okUser && okPass) return next();
    return next(new ApiError(401, 'UNAUTHORIZED', 'Invalid credentials'));
  };
}

module.exports = { basicAuth };
