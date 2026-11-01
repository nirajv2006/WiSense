#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "csi_processor.h"

typedef enum {
    STATE_EMPTY         = 0,
    STATE_LOW_ACTIVITY  = 1,
    STATE_HIGH_ACTIVITY = 2
} OccupancyState;

typedef struct {
    OccupancyState state;
    CSIFeatures    features;
    unsigned long  timestamp;
    bool           isOccupied;
    float          diffAmp;
    float          varThreshold;
    float          ampThreshold;
} ClassificationResult;

class Classifier {
public:
    Classifier();

    void calibrateBaseline(float baselineAmp, float baselineVar = 1.0f, float baselineStd = 1.0f);
    bool hasBaseline() const { return baselineCalibrated; }
    float getBaseline() const { return baselineMeanAmp; }
    float getBaselineVar() const { return baselineVariance; }
    float getMotionThreshold() const { return motionVarianceThreshold; }
    float getAmpThreshold() const { return ampDeviationThreshold; }

    ClassificationResult classify(CSIFeatures features);
    const char* getStateName(OccupancyState state);
    void setThresholds(float varMotion, float ampDeviation);

private:
    float motionVarianceThreshold;
    float ampDeviationThreshold;
    float baselineMeanAmp;
    float baselineVariance;
    bool  baselineCalibrated;

    // Hysteresis & debouncing
    int   occupiedConsecutive;
    int   emptyConsecutive;
    bool  currentlyOccupied;
    unsigned long lastOccupiedTime;
};

#endif
