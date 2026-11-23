# Learning notes

Short explanations of the key ideas in this project. These are drafts: rewrite
each one in your own words until you could say it out loud in an interview.

## Data

**Sampling rate (300 Hz).** The AliveCor device measures voltage 300 times per
second, so one sample is 3.3 ms apart. The fastest useful ECG content (QRS
edges) is well under 150 Hz, which is the most a 300 Hz signal can represent
(the Nyquist limit, half the sampling rate).

**Signal units (mV).** The .mat files store raw integers from the
analog-to-digital converter. The .hea header says the gain is 1000 units per mV,
and `wfdb` divides by it, so the signals we use are in millivolts. A normal QRS is
roughly 0.5-2 mV.

**Class imbalance.** About 59% of recordings are Normal, 9% AF, 28% Other and 3%
Noisy. A model that always says "Normal" gets 59% accuracy and is useless, which
is why accuracy is the wrong metric here.

## Signal processing

**Bandpass filter (0.5-40 Hz).** Removes baseline wander below 0.5 Hz (breathing,
movement) and muscle noise or mains hum above 40 Hz, keeping the QRS, P and T
waves. We use a Butterworth filter run forward and backward (`filtfilt`) so the
output has no time delay and R-peaks stay where they really are.

**R-peak detection.** Finding the tallest spike of each heartbeat. Our simple
detector follows Pan-Tompkins: take the slope (QRS is the steepest part),
square it, smooth over about one QRS width, threshold it, then snap to the true
maximum. neurokit2's detector is used as the reference.

**Polarity correction.** Handheld recordings are often upside down. If the
biggest deflections point down we flip the signal, otherwise the detector would
lock onto S-waves.

**Detector agreement as a quality signal.** On clean ECG, two different
detectors find the same beats. On noise they disagree, so their agreement score
works as a cheap signal-quality index.

## Features (physiology to numbers)

**RR interval.** Time between consecutive R-peaks (one cardiac cycle). In AF the
AV node gets chaotic atrial input and conducts at random, so RR intervals become
"irregularly irregular".

**RMSSD.** Root mean square of differences between neighbouring RR intervals.
It measures beat-to-beat change, which is small in sinus rhythm and large in
AF.

**Coefficient of variation (CV).** Standard deviation of RR divided by mean RR.
Dividing by the mean makes it comparable across heart rates.

**pNN50.** Fraction of neighbouring RR intervals that differ by more than 50 ms.
It is a simple, robust count of big beat-to-beat jumps.

**Poincare SD1/SD2.** Plot each RR interval against the next one. Sinus rhythm
makes a thin cigar along the diagonal (SD1/SD2 small). AF makes a round cloud
(SD1/SD2 near 1).

**Sample entropy.** If a short run of beats resembles another run, how likely
is it that the next beat matches too? Predictable rhythms score low and random
rhythms score high. It is undefined for very short recordings, so the model gets
an "is missing" flag instead.

**Why no P-wave feature?** Absent P-waves are the other AF hallmark, but P-waves
are small and hard to measure reliably on a noisy single lead. This is a known
limitation and a good next step.

## Machine learning

**Train / validation / test split.** Train fits the model. Validation chooses
between models and settings. Test is used once at the very end to estimate
real-world performance. Looking at test results while still choosing models
inflates the reported score.

**Stratified split.** Each split keeps the same class proportions as the full
dataset, so the 3% Noisy class doesn't end up almost absent from test by chance.

**Data leakage.** Any information from validation/test that sneaks into
training. Examples are fitting the scaler or imputer on all data, or choosing
features by looking at test performance. We put the imputer and scaler inside
the sklearn Pipeline so they are fit on training data only.

**Class weights.** We tell the model that mistakes on rare classes cost more,
in proportion to how rare they are. Without this, the model mostly predicts
Normal.

**F1 score.** The harmonic mean of precision (when I say AF, how often am I
right?) and recall (of real AF, how much do I catch?). Per-class F1 shows where
the model struggles.

**Macro F1.** The plain average of per-class F1, so the tiny Noisy class counts
as much as Normal. The 2017 challenge averaged over Normal, AF and Other only;
we report both.

**Confusion matrix.** A table of true class versus predicted class. The
off-diagonal cells show exactly which mistakes happen (e.g. Other predicted as
Normal).

**Permutation importance.** Shuffle one feature's values on validation data and
measure how much the score drops. A big drop means the model relies on it. This
is more honest than the random forest's built-in importance, which favours
features with many unique values.

**Random forest vs logistic regression.** Logistic regression draws straight
boundaries in feature space and is easy to interpret. A random forest combines
many decision trees and can capture thresholds and interactions (e.g. "irregular
RR AND good signal quality").
