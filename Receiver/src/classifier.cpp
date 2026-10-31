#include "classifier.h"

Classifier::Classifier() {
    varianceThresholdLow  = VARIANCE_THRESHOLD_LOW;
    varianceThresholdHigh = VARIANCE_THRESHOLD_HIGH;
}

ClassificationResult Classifier::classify(CSIFeatures features) {
    ClassificationResult result;
    result.features  = features;
    result.timestamp = millis();

    if (features.ampVariance < varianceThresholdLow) {
        result.state = STATE_EMPTY;
    } else if (features.ampVariance > varianceThresholdHigh) {
        result.state = STATE_HIGH_ACTIVITY;
    } else {
        result.state = STATE_LOW_ACTIVITY;
    }

    return result;
}

const char* Classifier::getStateName(OccupancyState state) {
    switch (state) {
        case STATE_EMPTY:         return "EMPTY";
        case STATE_LOW_ACTIVITY:  return "LOW";
        case STATE_HIGH_ACTIVITY: return "HIGH";
        default:                  return "UNKNOWN";
    }
}

void Classifier::setThresholds(float varLow, float varHigh) {
    varianceThresholdLow  = varLow;
    varianceThresholdHigh = varHigh;
}
