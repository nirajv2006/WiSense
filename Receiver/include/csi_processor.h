#ifndef CSI_PROCESSOR_H
#define CSI_PROCESSOR_H

#include <Arduino.h>

#define CSI_WINDOW_SIZE      100   // temporal sliding window (samples)
#define EXPECTED_INTERVAL_MS  20   // expected TX packet interval (ms)
#define JITTER_WINDOW_SIZE    50   // window for inter-arrival jitter

// Features computed from the CSI amplitude window
typedef struct {
    float    ampMean;          // mean subcarrier amplitude over window
    float    ampVariance;      // variance of amplitude — primary occupancy feature
    float    ampStdDev;        // standard deviation
    float    packetJitter;     // mean abs deviation of inter-arrival time from 20 ms
    float    signalVariation;  // max − min amplitude in window
    float    currentAmp;       // most recent sample
    uint32_t packetCount;      // samples currently in window
} CSIFeatures;

class CSIProcessor {
public:
    CSIProcessor();

    void addSample(float amplitude, unsigned long timestamp);
    CSIFeatures getFeatures();
    void reset();
    uint32_t getSampleCount();

private:
    float         ampBuffer[CSI_WINDOW_SIZE];
    unsigned long timestampBuffer[CSI_WINDOW_SIZE];
    unsigned long interArrivalBuffer[JITTER_WINDOW_SIZE];

    uint32_t ampIndex, timestampIndex, jitterIndex;
    uint32_t ampCount, jitterCount;
    unsigned long lastTimestamp;

    float computeMean();
    float computeVariance();
    float computeStdDev();
    float computeJitter();
    float computeSignalVariation();
};

#endif
