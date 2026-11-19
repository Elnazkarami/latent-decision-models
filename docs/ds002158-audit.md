# ds002158 audit: confidence after committed vs observed decisions

**Ticket:** LDM-001 · **Audit date:** 2026-11-18 · **Snapshot:** OpenNeuro ds002158 **v1.0.2** (created 2020-04-25, DOI 10.18112/openneuro.ds002158.v1.0.2) · **Machine-readable:** [`reports/ds002158-audit.json`](../reports/ds002158-audit.json)

## Verdict

| Use | Decision | Condition |
| --- | --- | --- |
| Behavioral confidence modeling | **GO** | Trial identity, confidence scale, condition and exclusions verified across all 118 runs |
| Trial-level EEG | **GO (verified on 2 subjects)** | Use raw BrainVision data and **marker-based** alignment. Exclude sub-20 until its label conflict is resolved. Never pair runs by filename (sub-12 is mislabeled). Gradient and BCG artifacts must be removed first: see [ds002158-eeg.md](ds002158-eeg.md). |
| Trial-level fMRI | **GO for whole-trial estimates** | Onsets verified against the scanner clock. The derivative is SPM DARTEL/MNI, 1.5 mm, 5 mm FWHM (from its headers). Pre-decision and response-locked BOLD cannot be separated. See [`ds002158-fmri-audit.md`](ds002158-fmri-audit.md). |
| Published `eegprep` derivative | **NOT YET** | One concatenated file per subject; the epoch-to-trial mapping is unknown. |

## How this audit was done (no signal data downloaded)

| Source | What | Size |
| --- | --- | --- |
| GitHub mirror tree at tag `1.0.2` | full file list; git blob SHA for every small file | API |
| OpenNeuro GraphQL `snapshot(ds002158, 1.0.2)` | license, per-file sizes, versioned S3 URLs | API |
| All 118 BIDS-valid `*_task-main_*_events.tsv` + sidecars + `scans.tsv` | `ldm ds002158 fetch-metadata` (146 files) | 278 KB |
| Derivative events copies (118) | byte-compared to raw: **identical** | ~250 KB |
| BrainVision `.vhdr` / `.vmrk` text, 25 runs (1 per subject + anomalies) | marker codes, alignment | 4.1 MB |
| NIfTI headers (first 352 B of each `preproc.nii`; first 8 KB of each `bold.nii.gz`) | volume counts, TR, dims | HTTP range requests |
| bioRxiv v1 preprint Methods (PNAS returned HTTP 403) | design, exclusion rules | — |

Fetching was done through `latent_decision_models.io.openneuro`. In code, it refuses any non-text file and caps each file at 2 MB.

## Verified facts

Labels: **[D]** checked exhaustively over all events files · **[S]** sidecar · **[M]** checked on EEG markers (sampled runs) · **[H]** NIfTI headers · **[P]** preprint text · **[I]** inferred.

### Participants and runs
- 20 subjects with non-contiguous IDs: `sub-02, 04, 07–22, 24, 26` [D]. The paper reports 25 recruited and 5 excluded [P].
- One session (`ses-001`) and tasks `main` and `rest` [D]. Runs per subject: 6 for most; sub-07 has 4, sub-14 has 5 (run-005 is in `rejected/`), and sub-20 has 7 [D].
- 118 BIDS-valid runs and 5,654 trial rows. Every run has 48 trials, except **sub-19 run-001 (38)** [D].
- Each run has 4 blocks of 12 trials ordered active / observed / active / observed [D, P]. The only deviation is in the truncated sub-19 run-001.

### Events columns (identical header in all 118 files) [D]
| Column | Meaning | Codes / range | Source |
| --- | --- | --- | --- |
| `onset` | stimulus onset, **seconds from the first fMRI volume trigger** (`R128`) | 3.5–469.6 | [D][M] |
| `duration` | always 0 | — | [D] |
| `trial_type` | condition + author exclusion | `act`, `act-bad`, `obs`, `obs-bad` | [D] |
| `response_time` | active: button-press latency. Observed: **replayed virtual-hand latency** (not a participant RT) | 20 ms grid. `-0.02`/`0.0` = no response. Missing from `events.json` | [D][P] |
| `dots_difference` | evidence strength (staircase) | 0–32 | [S][D] |
| `stim_side` | side with more dots | `left`, `right` | [S] |
| `choice_accuracy` | first-order accuracy | `cor`, `err` | [S] |
| `confidence` | visual-analogue rating, 0 = sure error, 1 = sure correct | 0–1 (2 dp). **`-1` = missing** (45 rows, all `-bad`) | [S][D][P] |
| `response_time2` | time to confirm the rating (the cursor moves automatically) | `-0.02` = missing. Sidecar: "little utility" | [S] |

The sidecar's confidence unit string ("%, 0 is sure error, 1 is sure correct") is internally inconsistent. The data are proportions on [0, 1].

### Own vs observed decisions
- In observation blocks, the participant rates confidence in a decision shown by a virtual hand [P]. That decision is a **shuffled replay of the participant's own preceding active block** [P]. **209 of 235** observed blocks are an exact multiset match to the preceding active block on (RT, dots, accuracy, side) [D].
- Consequences for modeling:
  - The observed condition has no participant RT. The adapter sets `response_time_seconds = NaN` for observed trials and keeps the replayed latency as `response_latency_seconds`.
  - Observed trials are not independent of active trials: they share stimuli and outcomes.
  - In the observed condition, `choice_side` is the hand's choice, not the participant's.

### Author exclusions (`-bad`)
- Preprint rules [P]: no first-order response; no confidence; RT < 200 ms or > 500 ms; a button press during observation.
- These rules reproduce **770 of 794** flagged trials [D]. **No unflagged trial violates them** [D].
- **24 flagged trials have no reason recoverable from events.tsv**: sub-20 (21) and sub-12 (3). They may be observation-period button presses, which are not recorded in events.tsv. The adapter labels them `author_flag_reason_not_in_events`.
- Valid trials per subject: active 74–138 (mean 121.8), observed 71–139 (mean 121.2) [D]. The preprint reports 119 ± 5 [P]. The difference is unexplained.

### Timing and alignment
- **Behavior ↔ fMRI:** `onset` is exactly `t(stimulus marker) − t(first R128)` (0.0 ms residual in every matched trial) [M]. In every checked run, the `R128` count equals the raw BOLD volume count (373, or 375 for sub-02) [M][H]. TR = 1.28 s [H]. Raw and derivative volume counts are equal in 117/117 runs [H].
- **Behavior ↔ EEG (sampled runs):** 5000 Hz, 64 channels [M]. Verified marker codes:
  - `S11`/`S12`: stimulus onset (more dots left/right).
  - `S7`/`S8`: active press (chose left/right).
  - `S17`/`S18`: virtual-hand onset (left/right).
  - `R128`: volume trigger.

  The EEG response latency agrees with `response_time` to within about 25 ms. Use the marker, not `onset + response_time`, for response-locked epochs.
- Alignment summary over 25 sampled runs (`reports/ds002158-audit.json → EEG_files.alignment_sample`):
  - **verified (timing + labels):** 22 of 25 runs, including two sub-12 runs that are only correct after relabeling.
  - **sub-12 filename shift:** EEG `run-00` = events run-005, and EEG `run-005` = events run-006. Pairing by filename would silently mismatch two runs.
  - **sub-19 run-001 (`timing_only`):** 38/38 trials timing-matched, with 1 stimulus-side label disagreement. The EEG holds 285 volume triggers against 373 BOLD volumes, so the EEG recording is shorter than the scan.
  - **sub-20 runs 002 and 006:** timing matches 48/48, but the stimulus side disagrees in 20/48 and 24/48 trials, and the response side in 22/48. **Unresolved:** I could not determine whether events.tsv or the markers are wrong. Neither source fits a simple staircase rule (the rule itself doesn't reproduce in clean subjects either).
- Unverified marker codes: `S128` (high-rate train; meaning unknown), `S27`, `S38`, `S40`, `S41`, `S43`, `S44`, `S171`. They are not needed for stimulus/response alignment.

### Derivatives
- `derivatives/fmriprep` uses non-standard names (`*_preproc.nii`, `*_headmove.txt`) and is **not** fMRIPrep-style output. Files are uncompressed, 121×145×121 voxels, about 3.2 GB per run, 374 GB in total.
  - The space is suspected to be template space but is not documented, and smoothing is not stated (the preprint describes SPM12, DARTEL and 5 mm smoothing).
  - sub-19 run-001 has no derivative. sub-19 run-006 has a derivative but **no raw BOLD**.
- `derivatives/eegprep`: one EEGLAB `.set`/`.fdt` per subject, about 180 MB each (3.2 GB total). The README describes 1–10 Hz filtering, epochs −0.2 to 0.6 s around the response, CAR and ICA. Run and trial bookkeeping inside the file was not inspected.

### Storage (from OpenNeuro per-file sizes)
| Component | Size |
| --- | --- |
| Whole snapshot incl. derivatives | **460 GB** (OpenNeuro's "82 GB" summary excludes derivatives; NEMAR's ~429 figure is consistent with 460 GB ≈ 429 GiB) |
| Raw EEG / raw BOLD / eegprep / fMRI derivatives | 38.7 / 43.1 / 3.2 / 373.8 GB |
| One subject (sub-02): raw EEG 6 runs / eegprep / raw BOLD 6 runs / derivative BOLD / T1w | 2.08 / 0.18 / 2.02 / 19.1 / 0.014 GB |

**Starter subset for LDM-003:** raw EEG (`.eeg/.vhdr/.vmrk`) for sub-02 and sub-04, 4.13 GB. **Downloaded with approval on 2026-11-19;** results in [ds002158-eeg.md](ds002158-eeg.md). All 12 runs pair by timing and verify. The raw headers name the original lab files, and the raw signal still contains MR gradient and BCG artifacts.

## Hypotheses and unknowns (not verified)
1. The 24 unexplained exclusions are observation-period button presses.
2. ~~The `fmriprep` derivative is SPM-normalised and smoothed in MNI/DARTEL space.~~ **Verified (2026-11-19):** every derivative header reads "Smoothed (5x5x5) Dartel normed", MNI-152, 1.5 mm ([fMRI audit](ds002158-fmri-audit.md)).
3. Meaning of the `S128` marker train and of codes `S27/S38/S40/S41/S43/S44/S171`. A guess: S171/S43 are post-response mask or scale events at about +1.47 s, by condition.
4. Which source is correct for sub-20 labels.
5. The exact staircase implementation. For example, whether the staircase ran per side, per block, or only on non-bad trials.
6. Why 26 of 235 observed blocks are not exact replays (possibly mis-recorded or excluded trials).

## Decisions that materially affect scientific validity (please review)
1. **Confidence target is continuous on [0, 1].** This is justified by the analogue scale. Ratings exactly 0 or 1 (60 valid trials) are kept.
2. **Validity follows the authors' `-bad` flags**, so the 24 unexplained exclusions are kept excluded.
3. **sub-20 is kept in the primary behavioral analysis**, matching the published data. A sensitivity analysis excludes it (`configs/ds002158_confidence_baseline_excl_sub-20.yaml`), and both runs are reported. sub-20 must be excluded from EEG work until resolved.
4. **Observed-trial latency is used as a predictor.** It is the latency the participant *saw*, not a participant RT. Interactions with condition are included.
5. Objective accuracy is a predictor in the "behavior" model. That makes it a **post-decision, descriptive** model of confidence, not a prospective prediction.

## Reproduce
```bash
ldm ds002158 fetch-metadata --dest data/ds002158          # 146 text files, ~0.28 MB
ldm ds002158 trials --bids-root data/ds002158 --out data/ds002158_trials.csv --issues data/ds002158_issues.csv
# EEG marker check for one run (requires that run's .vhdr/.vmrk, ~0.18 MB):
ldm ds002158 align-eeg --events <events.tsv> --vmrk <eeg.vmrk>
```
