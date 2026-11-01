#include "classifier.h"
#include <math.h>

Classifier::Classifier() {
  motionVarianceThreshold = 8.0f; // Safe initial default before calibration
  ampDeviationThreshold = 5.0f;   // Safe initial default before calibration
  baselineMeanAmp = 0.0f;
  baselineVariance = 0.0f;
  baselineCalibrated = false;

  occupiedConsecutive = 0;
  emptyConsecutive = 0;
  currentlyOccupied = false;
  lastOccupiedTime = 0;
}

void Classifier::calibrateBaseline(float baselineAmp, float baselineVar,
                                   float baselineStd) {
  baselineMeanAmp = baselineAmp;
  baselineVariance = baselineVar;
  baselineCalibrated = true;
  currentlyOccupied = false;
  occupiedConsecutive = 0;
  emptyConsecutive = 0;

  // Dynamically calculate thresholds from the room's empty noise floor:
  // Motion threshold: Noise floor variance + 3 * standard deviation
  // (minimum 7.0)
  motionVarianceThreshold = baselineVar + 3.0f * baselineStd;
  if (motionVarianceThreshold < 7.0f) {
    motionVarianceThreshold = 7.0f;
  }

  // Amplitude deviation threshold for static obstruction: 3.5 * std dev
  // (minimum 4.5)
  ampDeviationThreshold = baselineStd * 3.5f;
  if (ampDeviationThreshold < 4.5f) {
    ampDeviationThreshold = 4.5f;
  }
}

ClassificationResult Classifier::classify(CSIFeatures features) {
  ClassificationResult result;
  result.features = features;
  result.timestamp = millis();
  result.varThreshold = motionVarianceThreshold;
  result.ampThreshold = ampDeviationThreshold;

  float diff = 0.0f;
  if (baselineCalibrated && baselineMeanAmp > 0.0f) {
    diff = fabsf(features.ampMean - baselineMeanAmp);
  }
  result.diffAmp = diff;

  // 1. Motion condition: elevated variance in sliding CSI window
  bool motionDetected = (features.ampVariance >= motionVarianceThreshold);

  // 2. Obstruction condition: sustained amplitude shift from baseline
  bool obstructionDetected =
      (baselineCalibrated && (diff >= ampDeviationThreshold));

  bool rawOccupied = (motionDetected || obstructionDetected);

  // Debounce & Hysteresis filter:
  if (rawOccupied) {
    occupiedConsecutive++;
    emptyConsecutive = 0;
    // Require 2 consecutive positive detections (500 ms) to confirm Occupied
    if (occupiedConsecutive >= 2) {
      currentlyOccupied = true;
      lastOccupiedTime = result.timestamp;
    }
  } else {
    emptyConsecutive++;
    occupiedConsecutive = 0;
    // Require at least 1500 ms elapsed and 2 consecutive empty readings to
    // clear
    if (emptyConsecutive >= 2 && (result.timestamp - lastOccupiedTime > 1500)) {
      currentlyOccupied = false;
    }
  }

  // 3. Adaptive Baseline:
  // When the room is empty and stable (variance well below motion threshold),
  // slowly adapt baseline amplitude to compensate for ESP32 thermal drift.
  if (baselineCalibrated && !currentlyOccupied &&
      (features.ampVariance < (motionVarianceThreshold * 0.7f))) {
    baselineMeanAmp = (0.95f * baselineMeanAmp) + (0.05f * features.ampMean);
  }

  result.isOccupied = currentlyOccupied;
  if (currentlyOccupied) {
    result.state = motionDetected ? STATE_HIGH_ACTIVITY : STATE_LOW_ACTIVITY;
  } else {
    result.state = STATE_EMPTY;
  }

  return result;
}

const char *Classifier::getStateName(OccupancyState state) {
  switch (state) {
  case STATE_EMPTY:
    return "EMPTY";
  case STATE_LOW_ACTIVITY:
    return "LOW";
  case STATE_HIGH_ACTIVITY:
    return "HIGH";
  default:
    return "UNKNOWN";
  }
}

void Classifier::setThresholds(float varMotion, float ampDeviation) {
  motionVarianceThreshold = varMotion;
  ampDeviationThreshold = ampDeviation;
}
