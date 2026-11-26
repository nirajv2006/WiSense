# WiSense: technical guide

How the system is built, why each design choice was made, how to collect data
and evaluate it properly, and what you can and cannot claim.

---

## 1. System architecture

```
 ┌──────────────┐  UDP, 50 pkt/s   ┌────────────────────────────┐   USB serial 921600
 │ Transmitter  │ ───────────────▶ │ Receiver (soft-AP)         │ ─────────────────────┐
 │ ESP32        │   Wi-Fi ch. of   │ esp_wifi_set_csi_rx_cb()   │                      │
 └──────────────┘   the AP         │ - keeps TX frames only     │ ◀── "S,label,level" ─┤
                                   │ - CSI,ts,rssi,n,I/Q...     │     (shown on OLED)  │
                                   └────────────────────────────┘                      │
                                                                                       ▼
 ┌──────────────────────────────────────── PC (Python) ────────────────────────────────────────┐
 │ A. Acquisition   collect.py  -> data/raw/<session>/*.csv + .json (label, people, conditions) │
 │ B. Processing    amplitude -> resample 50 Hz -> Hampel -> Butterworth 10 Hz -> AGC normalise │
 │ C. Features      21 features per 4 s window (level / variation / spectral / structure)       │
 │ D. ML            build_dataset.py -> train.py  (6 models, 4 evaluation schemes, reports)     │
 │ E. Decision      PowerAdvisor: confidence gate + asymmetric hysteresis + explanation         │
 │ F. Live          ingest.py -> SQLite (wisense.db) + live_window.npz -> dashboard.py          │
 └─────────────────────────────────────────────────────────────────────────────────────────────┘
```

**Why ingest and dashboard are separate processes.** Streamlit re-runs its
script on every interaction, so it cannot own a serial port reliably. One
process produces data (ingest) and one only reads it (dashboard), with SQLite
in between. The dashboard can be restarted or opened on several devices
without losing a single packet. This is a standard producer/consumer design
and worth a slide.

### About "ESP-IDF and ESP-CSI"

The firmware in this repo uses **PlatformIO with the Arduino-ESP32 framework**.
Arduino-ESP32 is built on top of ESP-IDF, and the CSI calls
(`esp_wifi_set_csi_config`, `esp_wifi_set_csi_rx_cb`) are the ESP-IDF Wi-Fi API.
Describe it accurately in the report: *"CSI is captured with the ESP-IDF Wi-Fi
CSI API, using the Arduino-ESP32 framework in PlatformIO."*

If you would rather switch to Espressif's **esp-csi** examples
(`csi_send` / `csi_recv`), the Python side already understands their
`CSI_DATA,...,"[...]"` output. You can record with them and use the same
pipeline unchanged. Switching is optional, so don't spend your limited time on it.

---

## 2. The CSI data format

Each received packet gives one CSI measurement. With the default configuration
(`lltf_en = true`, 20 MHz) the buffer has **128 signed bytes = 64 subcarriers ×
(imaginary, real)**.

| Buffer index | Subcarrier | Used? |
|---|---|---|
| 0 | 0 (DC) | no, carries no data |
| 1-26 | +1 … +26 | yes |
| 27-37 | +27…+31, −32…−27 | no, guard band (zeros) |
| 38-63 | −26 … −1 | yes |

That leaves **52 data subcarriers** (`config.VALID_SUBCARRIERS`).

- **Amplitude** `|H| = sqrt(I² + Q²)` is what WiSense uses.
- **Phase** `atan2(I, Q)` is available in `preprocessing.iq_to_phase`, but raw ESP32
  phase includes random offsets from carrier-frequency and sampling-time
  differences between the two boards. It needs sanitisation (e.g. linear
  detrending across subcarriers) before it is useful. Good *future work*.
- **AGC**: the receiver's automatic gain control rescales whole packets. That's
  why each packet is normalised by its mean amplitude before computing
  variation features (the raw level is kept as a separate feature).

Serial line from the receiver:
```
CSI,183520,-46,128,3 -12 4 -11 5 -10 ...
     │      │   │   └ imag0 real0 imag1 real1 ...
     │      │   └ number of values
     │      └ RSSI (dBm)
     └ receiver millis()
```

---

## 3. Data collection protocol

The dataset decides the grade more than the model does. Follow a fixed protocol and write it in the report.

**Classes** (edit in `wisense/config.py` if your team defines them differently):

| Class | Definition | Typical recording |
|---|---|---|
| low | 0-1 people | empty room, one person sitting, one person walking |
| medium | 2-3 people | mix of sitting / standing / moving |
| high | 4+ people | as many as you can gather, behaving naturally |

**Sessions.** A session is one sitting: same day, same board placement. The
model is evaluated on sessions it has never seen, so you need several.

- Minimum: **4 sessions**. Better: 6+ over different days.
- Per session: each class **2-3 recordings of 60-120 s** (`--seconds 90`).
- Vary something between sessions and record it in `--condition`
  (door open/closed, chairs moved, time of day, different people).
- Within a recording, people behave **naturally**. Do not stand perfectly still, and do
  not dance. Record activity with `--activity` (sitting / walking / mixed).
- Type the real head count with `--people`. The script warns if it doesn't fit the label.
- Keep the boards fixed (tape them down), same height (~1 m), 3-5 m apart.
- If possible, add one session in a **different room**. Even if accuracy drops,
  reporting it honestly is a strong result.

```bash
python collect.py --port /dev/cu.usbserial-0001 --session s01 --label low    --people 0 --activity empty   --condition door_closed --seconds 90
python collect.py --port /dev/cu.usbserial-0001 --session s01 --label low    --people 1 --activity walking --condition door_closed --seconds 90
python collect.py --port /dev/cu.usbserial-0001 --session s01 --label medium --people 3 --activity mixed   --condition door_closed --seconds 90
python collect.py --port /dev/cu.usbserial-0001 --session s01 --label high   --people 5 --activity mixed   --condition door_closed --seconds 90
```

**Design question for your team:** "low" currently merges *empty* and *one person*.
For energy, empty is the most valuable state to detect (everything can go to
minimum). It is also the easiest to detect, since nothing moves. If your guide
allows four bands, consider `vacant / low / medium / high`. Otherwise, at least
record empty and one-person recordings separately (`--activity empty`) so you can
analyse them separately.

---

## 4. Signal processing (why each step)

| Step | What | Why |
|---|---|---|
| Subcarrier selection | drop DC + guard | they carry no channel information |
| Resampling | interpolate to a uniform 50 Hz grid | packets arrive with jitter and losses; FFT features need uniform sampling |
| Hampel filter | replace points > 3 MAD from local median | corrupted packets produce single huge spikes |
| Butterworth low-pass | 4th order, 10 Hz, zero-phase | human motion is below ~10 Hz; higher frequencies are noise |
| Packet normalisation | divide by mean amplitude | removes AGC gain jumps |
| Windowing | 4 s windows, 1 s step | long enough to see breathing/slow movement, short enough to react |

The **same function** (`preprocess_window`) runs during training and live
inference, which prevents train/serve skew.

---

## 5. Features: which ones matter?

Grouped in `wisense/features.py`:

- **Variation** (core signal): temporal std/variance per subcarrier, IQR,
  range, coefficient of variation, mean absolute first difference, variance of
  the mean amplitude (what the current firmware uses), RSSI std.
- **Spectral**: the signal is projected onto its first principal component (the
  dominant "motion" direction across subcarriers). Then energy in the breathing
  (0.1-0.6 Hz), motion (0.6-3 Hz) and fast (3-10 Hz) bands, plus spectral centroid.
- **Structure**: how concentrated the change is (PC1 variance ratio), dispersion
  across subcarriers, adjacent-subcarrier correlation, skewness, kurtosis.
- **Level**: mean amplitude and mean RSSI. These depend on the room, distance and
  antenna angle more than on people, so expect them to overfit across sessions.

**Don't decide from intuition.** `train.py` gives you two pieces of evidence:
1. *Permutation importance* on unseen-session folds: how much macro-F1 drops when a feature is shuffled.
2. *Group ablation*: all features vs. without level vs. variation only, etc.

If "without level" scores as well as "all features", drop the level features.
The model is then more likely to transfer to new rooms. That finding is
a good report section on its own.

---

## 6. Is Random Forest the right model?

Yes, it's a sensible main model for this problem:

- small, tabular dataset (hundreds to a few thousand windows) → tree ensembles work well;
- non-linear interactions between features, no need for feature scaling;
- gives class probabilities (needed for the confidence gate) and feature importance;
- `min_samples_leaf=5`, `max_features="sqrt"` keep it from memorising.

But it has to **earn its place**. `train.py` compares it with:

| Baseline | Why it is there |
|---|---|
| Majority class | the floor: what you get by always guessing the most common class |
| Variance threshold | the current firmware method, with thresholds learned fairly on training folds |
| Logistic regression | simplest learned model; if it matches RF, prefer it (simpler, more interpretable) |
| k-NN | sensitive to scaling/room differences; shows how "local" the data is |
| Decision tree | a single interpretable tree; shows what the ensemble adds |

Your classes are **ordinal** (low < medium < high). Mistaking low for high is
worse than low for medium. Look at the confusion matrix for that, and mention
ordinal classification as possible future work.

---

## 7. Avoiding overfitting and evaluating honestly

The most common mistake in CSI projects: overlapping windows from the same
recording end up in both train and test, so the model only has to recognise
the recording, not the occupancy. `train.py` reports four schemes so you can
**show** this effect:

| Scheme | Train / test split | Interpretation |
|---|---|---|
| random | shuffled windows | optimistic, includes leakage |
| recording | whole recordings held out | same day, unseen recording |
| **session** | **one whole session held out** (headline) | unseen day / setup |
| condition | one condition held out (if it varies within sessions) | e.g. door open never seen in training |

Further rules:
- Report **macro-F1** (treats classes equally) plus accuracy, per-class precision/recall and the confusion matrix.
- Report the **spread across held-out sessions** (per-fold F1 in the report), not just the mean.
- Keep **one session untouched** until the very end. Tune on the others, then evaluate once on it.
- Do not change features or hyper-parameters after looking at the final test session.
- More sessions help more than more minutes per session.

### Suggested experiment matrix for the report

| ID | Question | How |
|---|---|---|
| E1 | Does it work at all? | `recording` scheme |
| E2 | Does it work on a new day? | `session` scheme (headline) |
| E3 | Robust to changed conditions? | `condition` scheme, or a dedicated session with door open / furniture moved |
| E4 | Which features matter? | permutation importance + ablation |
| E5 | Is RF better than simple methods? | model comparison table |
| E6 | Is the decision layer stable and safe? | `evaluate_advisor.py` |
| E7 (optional) | New room | train on room A sessions, test on room B |

---

## 8. Handling environmental noise

- **Fixed packet rate.** The TX sends every 20 ms, and Wi-Fi power-save is
  disabled on both boards so packets don't arrive in bursts.
- **Choose a quiet Wi-Fi channel.** Check nearby networks on a phone app and put the soft-AP on the least
  busy channel (`WiFi.softAP(ssid, pass, channel)`).
- **Filter by transmitter MAC.** Phones and other APs on the same channel also
  produce CSI. The receiver keeps only frames from the connected transmitter.
- **Hampel + low-pass + AGC normalisation** (section 4).
- **Temporal smoothing.** Probabilities are averaged over the last 3 windows.
- **Quality checks.** Packet rate is stored with every window and shown on the
  dashboard. Low-rate windows are skipped.
- **Known confounders**, to state in the report: ceiling fans, swinging doors, curtains
  and pets also cause CSI fluctuations. A perfectly still person looks like an
  empty room in amplitude variation. Moving the boards changes everything (re-collect).

---

## 9. Making power recommendations meaningful

`wisense/power.py` maps occupancy to a recommended level:

| Occupancy | Level | Nominal share | Example actions (recommendation only) |
|---|---|---|---|
| low | ECO | 30 % | minimum lighting, eco AC setpoint, minimum ventilation |
| medium | STANDARD | 60 % | normal lighting and setpoint |
| high | FULL | 100 % | full lighting, comfort setpoint, boosted ventilation |

What makes it more than a lookup table:
1. **Confidence gate**: if the model is below 55 % confident, the current level is kept.
2. **Asymmetric hysteresis**: step up after 5 s (comfort first), step down only after
   60 s (avoid switching off when someone sits still for a moment).
3. **Explanation**: every recommendation carries a reason shown on the dashboard.

**Evaluate it** with `evaluate_advisor.py`. It replays unseen sessions through
the advisor and reports:
- *matched*: % time the recommendation equals the level the true occupancy needs,
- *under-provided*: % time below it (comfort risk, keep this low),
- *over-provided*: % time above it (unused capacity),
- *changes per hour*: stability.

It also sweeps the step-down delay. Plot that trade-off in the report.
For a realistic test, record a few **long continuous sessions** (10-20 min) in
which people enter and leave, and label the changes with timestamps.

**Do not claim kWh or money savings.** Nothing is metered. Safe wording:
*"the system recommended a reduced level X % of the observed time; actual savings
depend on the controlled loads and would require metering."*

---

## 10. What you may and may not claim

✅ "We implemented and evaluated a CSI-based occupancy estimation pipeline on low-cost ESP32 hardware."
✅ "On sessions not used for training, the Random Forest reached macro-F1 = … (range … across sessions)."
✅ "Wi-Fi sensing is camera-free and does not capture images or identities."
❌ "A new invention." (Wi-Fi CSI sensing is a well-studied research area; cite prior work.)
❌ Accuracy numbers from the `random` scheme as the headline.
❌ Any number produced on synthetic data (`simulate.py`) as a result.
❌ Energy or cost savings in kWh / ₹.
❌ "Counts people." (It estimates occupancy bands.)

"Privacy-conscious" is fair, but say precisely what it means: no images, no
identities, no personal devices needed. People's presence and movement *are* still
sensed.

---

## 11. Suggested report / presentation outline

1. Problem & motivation: occupancy-aware energy use; why not cameras/PIR
2. Background: Wi-Fi OFDM, CSI, why bodies change it (cite prior work)
3. System architecture (section 1 diagram)
4. Data acquisition & dataset (protocol, table of sessions/classes/minutes)
5. Signal processing & features (with a CSI heatmap: empty vs. high)
6. Models & evaluation methodology (schemes, why leakage matters)
7. Results: model comparison, confusion matrix, importance/ablation, advisor trade-off
8. Dashboard demo (live, or `ingest.py --replay` of a real recording)
9. Limitations & future work (phase features, ordinal models, multi-link, more rooms)
10. Conclusion

**Demo tip:** record a good real session beforehand and run
`python ingest.py --replay "data/raw/s05/*.csv" --loop`, so the demo works even
if the room is noisy on presentation day. Use the live boards as the "wow" moment.
