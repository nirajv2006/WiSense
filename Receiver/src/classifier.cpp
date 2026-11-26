#include "classifier.h"

Classifier::Classifier() {
    threshEmpty  = VARIANCE_EMPTY_THRESH;
    threshMedium = VARIANCE_MEDIUM_THRESH;
    threshHigh   = VARIANCE_HIGH_THRESH;
    threshRoc    = HUMAN_ROC_THRESH;
}

ClassificationResult Classifier::classify(CSIFeatures f) {
    ClassificationResult r;
    r.features  = f;
    r.timestamp = millis();

    float var = f.ampVariance;
    float roc = f.rateOfChange;

    if (var < threshEmpty) {
        // No significant signal disturbance — room is empty
        r.state = STATE_EMPTY;
        r.level = LEVEL_NONE;
    } else if (roc < threshRoc) {
        // Variance present but signal changes slowly → mechanical source (fan, AC)
        r.state = STATE_OBJECT;
        r.level = (var < threshMedium) ? LEVEL_LOW : LEVEL_MEDIUM;
    } else if (var >= threshHigh) {
        // High variance + human-like rate of change → person moving
        r.state = STATE_HUMAN_ACTIVE;
        r.level = LEVEL_HIGH;
    } else {
        // Moderate variance + human-like rate of change → person sitting / breathing
        r.state = STATE_HUMAN_STILL;
        r.level = LEVEL_MEDIUM;
    }

    return r;
}

const char* Classifier::getStateName(OccupancyState state) {
    switch (state) {
        case STATE_EMPTY:        return "Empty";
        case STATE_OBJECT:       return "Object";
        case STATE_HUMAN_STILL:  return "Human Still";
        case STATE_HUMAN_ACTIVE: return "Human Active";
        default:                 return "Unknown";
    }
}

const char* Classifier::getLevelName(OccupancyLevel level) {
    switch (level) {
        case LEVEL_NONE:   return "None";
        case LEVEL_LOW:    return "Low";
        case LEVEL_MEDIUM: return "Medium";
        case LEVEL_HIGH:   return "High";
        default:           return "None";
    }
}

void Classifier::setThresholds(float varEmpty, float varMedium, float varHigh, float rocHuman) {
    threshEmpty  = varEmpty;
    threshMedium = varMedium;
    threshHigh   = varHigh;
    threshRoc    = rocHuman;
}
