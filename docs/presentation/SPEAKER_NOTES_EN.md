# Seven-minute presentation notes

## 1. Online EEG + NIRS hand imagery

We classify rest, imagined left-hand movement, and imagined right-hand
movement. The design follows the competition's streaming interface. This
presentation reports public-data results only; the hidden test is not
available to us.

## 2. Task and online contract

The frozen runner sends only samples available at the current time. It
scores one-second windows at 250 ms intervals. It does not score block 1.
The labels for a block become available for training only after that block
finishes. The official scorer emphasizes the weakest class F1 and the
average recall across classes, not accuracy alone.

## 3. Signal processing and model

We apply a stateful causal 1-40 Hz Butterworth filter to EEG. Each window
supplies log variance, four band-power values per channel, and C3/C4
power differences. Available HbO and HbR samples give per-channel mean,
variation, and short-window change. Three Random Forests handle EEG,
NIRS, and both together. A previous-session feature cache augments the
released labels from the current session at each block update.

## 4. Validation split

The public source lists 56 sessions from 14 people. We used N15 and N16
for development and kept N17 and N18 for an independent final check. The
N17/N18 training cache has 46 compatible sessions from other people.
Its manifest can be inspected in the package. Two public recordings lack
compatible NIRS and do not enter this joint cache. Eight test sessions
mean two independent people, so uncertainty remains substantial.

## 5. Independent N17/N18 results

After selecting probability fusion on N15/N16, we scored the N17/N18
sessions in chronological order. Mean accuracy rose from 63.67% to 65.18%,
mean worst-class F1 from 0.4050 to 0.4152, and computed points from
50.2353 to 51.2448 out of 130. The separate NIRS output was unchanged.
Neither version incurred a timing penalty on this machine. Jury design
points are not included in the computed score.
These numbers belong to the earlier tested probability-fusion source. The
optimized source was later run by the user on a Mac: 65.12% and 51.4810/130
on the same eight named sessions. The small score difference was measured
across different computers, and only terminal output was supplied.

## 6. Where the gain comes from

The earlier model replaced the hybrid decision whenever the EEG and NIRS
labels agreed. The selected model averages probabilities instead: 50%
hybrid, 35% EEG, and 15% NIRS. Six of eight sessions improved their score;
two got worse. A paired bootstrap interval for the average score gain
includes zero, so we should not promise a benefit on every new person.

## 7. Submission and limits

For the hidden test, the bundled final cache was rebuilt from all 54
compatible public recordings. The classifier uses the required Python
interface and the supplied runner and scorer were preserved. The package
contains a Windows smoke test, a subject holdout replay, per-session
outputs, and a notebook. Fourteen separate subject-excluding caches are
included, but the full 56-session public aggregate is still pending. The
Extra Trees candidate has only a one-session smoke result. Hidden-test quality and target-platform
speed can only be confirmed by the jury.
