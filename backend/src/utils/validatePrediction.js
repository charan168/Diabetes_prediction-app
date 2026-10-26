/**
 * Field rules for the patient input. `optional` fields may be null/omitted
 * (treated as "not measured" and imputed by the model pipeline).
 */
const FIELDS = {
  Pregnancies: { min: 0, max: 20, integer: true, label: 'Pregnancies' },
  Glucose: { min: 1, max: 500, label: 'Glucose (mg/dL)' },
  BloodPressure: { min: 1, max: 250, label: 'Diastolic blood pressure (mm Hg)' },
  SkinThickness: { min: 0, max: 100, optional: true, label: 'Skin thickness (mm)' },
  Insulin: { min: 0, max: 900, optional: true, label: 'Insulin (mu U/ml)' },
  BMI: { min: 10, max: 80, label: 'BMI' },
  DiabetesPedigreeFunction: { min: 0.01, max: 3, label: 'Diabetes pedigree function' },
  Age: { min: 1, max: 120, integer: true, label: 'Age' },
};

/** Returns { value, errors }. `value` only contains known fields. */
function validatePrediction(body) {
  const errors = [];
  const value = {};

  if (body === null || typeof body !== 'object' || Array.isArray(body)) {
    return { value, errors: [{ field: 'body', message: 'Request body must be a JSON object' }] };
  }

  for (const [name, rule] of Object.entries(FIELDS)) {
    const raw = body[name];

    if (raw === undefined || raw === null) {
      if (rule.optional) {
        value[name] = null;
      } else {
        errors.push({ field: name, message: `${rule.label} is required` });
      }
      continue;
    }
    if (typeof raw !== 'number' || !Number.isFinite(raw)) {
      errors.push({ field: name, message: `${rule.label} must be a number` });
      continue;
    }
    if (rule.integer && !Number.isInteger(raw)) {
      errors.push({ field: name, message: `${rule.label} must be a whole number` });
      continue;
    }
    if (raw < rule.min || raw > rule.max) {
      errors.push({
        field: name,
        message: `${rule.label} must be between ${rule.min} and ${rule.max}`,
      });
      continue;
    }
    value[name] = raw;
  }

  return { value, errors };
}

module.exports = { FIELDS, validatePrediction };
