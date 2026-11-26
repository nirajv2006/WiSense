#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include "csi_processor.h"

typedef enum {
    STATE_EMPTY         = 0,   // no presence
    STATE_OBJECT        = 1,   // non-human source (fan, AC, vibration)
    STATE_HUMAN_STILL   = 2,   // human breathing / sitting still
    STATE_HUMAN_ACTIVE  = 3,   // human moving / walking
} OccupancyState;

typedef enum {
    LEVEL_NONE   = 0,
    LEVEL_LOW    = 1,
    LEVEL_MEDIUM = 2,
    LEVEL_HIGH   = 3,
} OccupancyLevel;

// ─── Tunable thresholds ───────────────────────────────────────────────────────
// Adjust based on your environment using the serial CSV data.
#define VARIANCE_EMPTY_THRESH   150.0f   // below → empty
#define VARIANCE_MEDIUM_THRESH  400.0f   // object low→medium boundary
#define VARIANCE_HIGH_THRESH    600.0f   // above + high RoC → human active
#define HUMAN_ROC_THRESH          2.5f   // mean |Δamp| below = object, above = human

typedef struct {
    OccupancyState state;
    OccupancyLevel level;
    CSIFeatures    features;
    unsigned long  timestamp;
} ClassificationResult;

class Classifier {
public:
    Classifier();

    ClassificationResult classify(CSIFeatures features);
    const char* getStateName(OccupancyState state);
    const char* getLevelName(OccupancyLevel level);
    void setThresholds(float varEmpty, float varMedium, float varHigh, float rocHuman);

private:
    float threshEmpty;
    float threshMedium;
    float threshHigh;
    float threshRoc;
};

#endif
