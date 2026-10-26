const mongoose = require('mongoose');

const inputSchema = new mongoose.Schema(
  {
    Pregnancies: { type: Number, required: true },
    Glucose: { type: Number, required: true },
    BloodPressure: { type: Number, required: true },
    SkinThickness: { type: Number, default: null },
    Insulin: { type: Number, default: null },
    BMI: { type: Number, required: true },
    DiabetesPedigreeFunction: { type: Number, required: true },
    Age: { type: Number, required: true },
  },
  { _id: false }
);

const predictionSchema = new mongoose.Schema(
  {
    input: { type: inputSchema, required: true },
    prediction: { type: Number, enum: [0, 1], required: true },
    probability: { type: Number, min: 0, max: 1, required: true },
    createdAt: { type: Date, default: Date.now, index: -1 },
  },
  { versionKey: false, bufferCommands: false }
);

module.exports = mongoose.model('Prediction', predictionSchema);
