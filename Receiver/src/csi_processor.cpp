#include "csi_processor.h"
#include <math.h>

CSIProcessor::CSIProcessor() { reset(); }

void CSIProcessor::reset() {
    ampIndex = timestampIndex = jitterIndex = 0;
    ampCount = jitterCount = 0;
    lastTimestamp = 0;
    memset(ampBuffer,          0, sizeof(ampBuffer));
    memset(timestampBuffer,    0, sizeof(timestampBuffer));
    memset(interArrivalBuffer, 0, sizeof(interArrivalBuffer));
}

void CSIProcessor::addSample(float amplitude, unsigned long timestamp) {
    if (lastTimestamp > 0) {
        unsigned long ia = timestamp - lastTimestamp;
        interArrivalBuffer[jitterIndex] = ia;
        jitterIndex = (jitterIndex + 1) % JITTER_WINDOW_SIZE;
        if (jitterCount < JITTER_WINDOW_SIZE) jitterCount++;
    }

    ampBuffer[ampIndex]           = amplitude;
    timestampBuffer[timestampIndex] = timestamp;
    ampIndex       = (ampIndex       + 1) % CSI_WINDOW_SIZE;
    timestampIndex = (timestampIndex + 1) % CSI_WINDOW_SIZE;
    if (ampCount < CSI_WINDOW_SIZE) ampCount++;

    lastTimestamp = timestamp;
}

float CSIProcessor::computeMean() {
    if (ampCount == 0) return 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0; i < ampCount; i++) sum += ampBuffer[i];
    return sum / (float)ampCount;
}

float CSIProcessor::computeVariance() {
    if (ampCount < 2) return 0.0f;
    float mean = computeMean();
    float ss = 0.0f;
    for (uint32_t i = 0; i < ampCount; i++) {
        float d = ampBuffer[i] - mean;
        ss += d * d;
    }
    return ss / (float)ampCount;
}

float CSIProcessor::computeStdDev()  { return sqrtf(computeVariance()); }

float CSIProcessor::computeJitter() {
    if (jitterCount < 2) return 0.0f;
    float sum = 0.0f;
    for (uint32_t i = 0; i < jitterCount; i++)
        sum += fabsf((float)interArrivalBuffer[i] - EXPECTED_INTERVAL_MS);
    return sum / (float)jitterCount;
}

float CSIProcessor::computeSignalVariation() {
    if (ampCount < 2) return 0.0f;
    float mn = ampBuffer[0], mx = ampBuffer[0];
    for (uint32_t i = 1; i < ampCount; i++) {
        if (ampBuffer[i] < mn) mn = ampBuffer[i];
        if (ampBuffer[i] > mx) mx = ampBuffer[i];
    }
    return mx - mn;
}

CSIFeatures CSIProcessor::getFeatures() {
    CSIFeatures f;
    f.ampMean        = computeMean();
    f.ampVariance    = computeVariance();
    f.ampStdDev      = computeStdDev();
    f.packetJitter   = computeJitter();
    f.signalVariation = computeSignalVariation();
    f.currentAmp     = (ampCount > 0) ? ampBuffer[(ampIndex - 1 + CSI_WINDOW_SIZE) % CSI_WINDOW_SIZE] : 0.0f;
    f.packetCount    = ampCount;
    return f;
}

uint32_t CSIProcessor::getSampleCount() { return ampCount; }
