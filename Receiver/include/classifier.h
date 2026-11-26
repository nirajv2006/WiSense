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
} ClassificationResult;

// Variance of mean CSI amplitude over the sliding window.
// Tune these for your environment using the serial CSV data.
#define VARIANCE_THRESHOLD_LOW  150.0f   // below → Empty (Not Occupied)
#define VARIANCE_THRESHOLD_HIGH 600.0f   // above → High Activity (Occupied)

class Classifier {
public:
    Classifier();

    ClassificationResult classify(CSIFeatures features);
    const char* getStateName(OccupancyState state);
    void setThresholds(float varLow, float varHigh);

private:
    float varianceThresholdLow;
    float varianceThresholdHigh;
};

#endif
