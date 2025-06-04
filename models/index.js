// model/index.js
// Utility for referencing trained Keras model files

const path = require("path");

const MODEL_PATHS = {
  patternCnn: path.join(__dirname, "pattern-cnn.h5"),
  trendLstm: path.join(__dirname, "trend-lstm.h5"),
  sentimentBert: path.join(__dirname, "sentiment-bert.h5"),
};

function getModelPath(modelName) {
  return MODEL_PATHS[modelName] || null;
}

module.exports = { getModelPath, MODEL_PATHS };