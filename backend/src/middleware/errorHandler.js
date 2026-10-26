const { ApiError } = require('../utils/errors');

function notFound(_req, _res, next) {
  next(new ApiError(404, 'NOT_FOUND', 'Route not found'));
}

// eslint-disable-next-line no-unused-vars
function errorHandler(err, _req, res, _next) {
  let status = err.status || 500;
  let code = err.code;
  let message = err.message;

  if (err.type === 'entity.parse.failed') {
    status = 400;
    code = 'INVALID_JSON';
    message = 'Request body is not valid JSON';
  } else if (err.type === 'entity.too.large') {
    status = 413;
    code = 'PAYLOAD_TOO_LARGE';
    message = 'Request body too large';
  } else if (!(err instanceof ApiError)) {
    // Unexpected error: log details server-side, return a generic message.
    console.error('[error]', err);
    const dbDown = ['MongooseError', 'MongoNetworkError', 'MongoServerSelectionError'].includes(err.name);
    status = dbDown ? 503 : 500;
    code = dbDown ? 'DB_UNAVAILABLE' : 'INTERNAL_ERROR';
    message = dbDown ? 'Database is unavailable' : 'Internal server error';
  }

  if (status === 401) res.set('WWW-Authenticate', 'Basic realm="dashboard", charset="UTF-8"');
  res.status(status).json({
    success: false,
    error: { code, message, ...(err.details ? { details: err.details } : {}) },
  });
}

module.exports = { notFound, errorHandler };
