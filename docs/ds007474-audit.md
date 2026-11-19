# ds007474 audit: two-step task fMRI (model-based vs model-free learning)

- **Audit date:** 2026-11-19
- **Snapshot:** OpenNeuro ds007474 **v1.0.5** (2026-07-31, CC0, DOI 10.18112/openneuro.ds007474.v1.0.5)
- **Behavior and timing:** OSF project [`ctfzd`](https://osf.io/ctfzd), reached via the view-only link in the dataset README
- **Machine-readable summary:** [`reports/ds007474-audit.json`](../reports/ds007474-audit.json)
- **Paper:** Ding, Cockburn et al. (2026) *Cell Reports* 45(6), [doi:10.1016/j.celrep.2026.117454](https://doi.org/10.1016/j.celrep.2026.117454). Task variant: Cockburn et al., [doi:10.31234/osf.io/egq3x](https://doi.org/10.31234/osf.io/egq3x).

## Verdict

| Use | Decision | Condition |
| --- | --- | --- |
| Behavioral RL modeling | **GO** | 189 subjects, each with 259 trials (105 structural-session + 154 in the scanner). Codes verified against the task constants and checked for internal consistency. |
| Trial-level fMRI timing | **CONDITIONAL GO** | The BIDS `events.tsv` files are **empty**. Use the OSF timing via `ldm ds007474 trials --timing`. Onsets are relative to the first scanner pulse of each run (assumed to be the first saved volume; see the unknowns). sub-MBMF025002 run-1 is truncated. |
| Published fMRI derivatives | **NOT USABLE for trial-level work** | Only 66 group-level (second-level) T maps; no preprocessed or subject-level data. Any fMRI analysis needs preprocessing of the raw BOLD (259 GB). |
| Licensing | **⚠ needs a decision** | OpenNeuro data are CC0, but the **OSF project and the authors' code repository declare no license**. Analysis for research is reasonable; redistributing the OSF files or reusing the code is not. Nothing from either is committed here. |

## How this audit was done

| Source | What | Size |
| --- | --- | --- |
| OpenNeuro GraphQL `snapshot(ds007474, 1.0.5)` | file list, sizes, license, description | API |
| BIDS text metadata: README, CHANGES, 378 `events.tsv`, 378 `bold.json` | `latent_decision_models.io.openneuro` (refuses signal files) | 1.3 MB |
| NIfTI headers of all 378 `bold.nii.gz` | 8 KB HTTP range request each, no image data | — |
| OSF listing and files: 189 behavior `.csv` + 189 timing `.mat` | `ldm ds007474 fetch-behavior`. Each file is checked against OSF's SHA-256 (378/378 match). | 13.4 MB |
| Authors' GitHub code (`Gentu-Ding/2-step-task-MBMF-fMRI-code`) | read to establish codes and timing; not copied | — |

Labels:
- **[D]** checked on all files.
- **[C]** from the authors' task or analysis code.
- **[R]** dataset README.
- **[I]** inferred, consistent with the data but not stated.

## Verified facts

### Participants and files
- **189 subjects** labelled `sub-MBMFxxxx0k` [D]. The suffix `k` is the task order within the scan session, because the two-step task was paired with a habit task [R]. No `participants.tsv`.
- Each subject has 2 functional runs (`run-1`, `run-2`), T1w, T2w, single-band references and fieldmaps [D].
  - sub-MBMF025002 has no run-1 fieldmaps [R][D].
- **BOLD:** TR = 1.12 s, multiband factor 4, 96×96×72 voxels, **650 volumes per run** in 377/378 runs [D].
  - sub-MBMF025002 run-1 has **381** volumes (427 s) [D].

### The BIDS events files are empty
- All 378 `*_events.tsv` contain only the header `onset duration trial_type response_time`, with **no rows** [D]. The behavior and event timing exist only on OSF [R].

### OSF ↔ BIDS subject mapping
- Timing files `subID_xxxx_sesID_k_MBMF_fMRI.mat` map one-to-one onto BIDS `MBMFxxxx0k` for **189/189** subjects [D].
- Behavior files are named `subxxxx{s2f3|s3f2}.csv`. Their suffix does **not** encode `k`; it encodes **session order** [D]:
  - `s2f3` (95 subjects): the structural-scanner session came before the functional session.
  - `s3f2` (94 subjects): the reverse.
  - The 4-digit IDs are unique, so mapping by ID alone is unambiguous [D].
- OSF functional run k = BIDS `run-k` **[I]**, supported by one strong check. In sub-MBMF025002, all 30 no-response trials are in OSF functional run 1, which is the run whose BIDS `run-1` BOLD stops after 427 s. The task itself continued for 715 s.

### Behavior CSV (34 columns, identical header in all 189 files)
- **Trial counts:** 259 trials per subject [D]:
  - 105 with `sess_type = structural`: done in the structural scanner, **no BOLD**;
  - 154 `functional`: two runs × 77, matching the timing file's runs exactly.

| Column | Meaning | Codes | Source |
| --- | --- | --- | --- |
| `Resp1` | first-stage choice | 1 = left, 2 = right; NaN = no response | [C] (`isLeft = Resp1 == 1`) |
| `RT1` | first-stage RT (s) | median 0.42; 2 s timeout | [D][C] (`MAX_RT = 2`) |
| `doRareTrans` | rare transition on this trial | 0/1; rare rate 0.28 (task `pTransRare` = 0.3) | [D][C] |
| `outcome1` | planet (state pair) reached | 1/2 = ceil(`outcome2` / 2). **Placeholder value 1 on all 897 no-response trials** | [D] |
| `outcome2` | second-stage state | 1–4 | [D] |
| `outcomeBin`, `outcomeMag` | win shown and points shown | 0/1; `round(win × rewMagnitude × 100)` | [C][D] |
| `condContingent` | reward contingency | 1 = stimulus, 2 = state | [C] (`CONT_STIM = 1`, `CONT_STATE = 2`) |
| `condState` | P(common landing pad \| planet), 26-trial blocks | 1 ("high" uncertainty) = 0.5, 2 = 0.9 | [C][D] (empirical 0.62 vs 0.89) |
| `condReward` | magnitude scale, 26-trial blocks | 1 = U(0.30, 1.00), 2 = U(0.10, 0.19) | [C][D] |
| `isPost_ContState`, `isPost_StateLow`, `isPost_RewHigh` | the **previous** trial's condition (state contingency, `condState` = 2, `condReward` = 1) | 0/1; NaN on a run's first trial | [D] (one-trial lag on 100%) |
| `state2_1`, `state2_2` | pad (1–4) each ship would reach on this trial | `outcome2 = state2_{Resp1}` on 100% | [D] |
| `rewBinary_k`, `rewPWin_k`, `rewMagnitude` | per-pad win draw and probability (shared within a planet); magnitude scale | — | [C][D] |
| `isStay`, `isPrev*`, `PrevRT1`, `PrevRewMag` | precomputed history regressors | `isStay` reproduced exactly from `Resp1` | [D] |

- **No second-stage choice** exists in this variant: the timing file's `resp2` is empty on every trial [D].
- **Special subjects:** the five subjects the README lists (sub-MBMF0215–0219) have **only state-contingent** trials (`condContingent = 2` on all 259) [D][R]. Everyone else experienced both contingencies [D].

### Event timing (`.mat`)
- **Format:** stored as MATLAB `table` objects, which need the optional `mat-io` reader (BSD-3; `pip install -e '.[ds007474]'`) [D]. The top-level `trialEvents` table is an **empty placeholder**; the filled-in table is `task_MBMF_fMRI.trials` [D].
- **Reference time:** `tRunStart` is the time of the first scanner-pulse keypress (`waitForScannerPulse`). Functional runs have no self-paced breaks (`breakITI = Inf`) [C].
- **Events recorded:** stimulus, response, second-state, terminal-state and feedback onsets [D].
  - `tResp1 − tStim1On = RT1` exactly on all responded trials [D].
  - The first stimulus comes 0.8–1.9 s after the run's first pulse, and trials are spaced a median of 9.3 s apart [D].
- **Agreement with the CSV:** the timing file matches the CSV on condition, transition, choice, RT and reward magnitude for all 154 functional trials of all 189 subjects [D].
  - On no-response trials the timing file stores the 2 s timeout as `RT1`; the adapter keeps RT missing.
- **Scan coverage:** every run except sub-MBMF025002 run-1 has its last event 5–22 s before the end of the scan (median 16 s) [D].

### Behavioral sanity check (descriptive, not a finding)
Group mean P(stay with the same first-stage choice), within run, 189 subjects:

| Previous trial | Common transition | Rare transition |
| --- | --- | --- |
| Rewarded | 0.82 | 0.62 |
| Unrewarded | 0.39 | 0.64 |

- This is the classic reward × transition interaction (mean per-subject interaction 0.23; reward main effect 0.20). It confirms that the transition and reward codes are read correctly.
- It reproduces a well-known qualitative pattern; it is **not** a new result.

### Data quality
- **No-response trials:** 2.0% of functional trials (594/29,106); median 1 per subject. Another 303 are in the structural session.
  - 8 subjects miss more than 15 of 154: MBMF005701 (39), MBMF008801 (32), MBMF025002 (30, the truncated run), MBMF007201 (22), MBMF008901 (20), MBMF011802 (19), MBMF024402 (17), MBMF027001 (17).
- **Very fast responses:** 658 functional trials have RT < 150 ms (min 1 ms), probably anticipatory. The authors' RT exclusion rule is not documented in the data.

## Storage (OpenNeuro per-file sizes)
| Component | Size |
| --- | --- |
| Whole snapshot | 265 GB |
| Raw BOLD (378 runs) | 258.7 GB, **0.69 GB per run** |
| Anatomy / fieldmaps | 5.1 / 0.85 GB |
| Derivatives (66 group-level T maps) | 0.29 GB |
| OSF behavior + timing | 13.4 MB |

## Hypotheses and unknowns
1. **Dummy scans.** It is not verified whether the first scanner pulse corresponds to the first *saved* volume. This determines whether onsets need a constant offset; check the sequence protocol (`code/Protocol_Translator.json`) or ask the authors.
2. **OSF run k = BIDS run-k** is inferred; only one subject provides a decisive test.
3. ~~Meaning of `condState`, `condReward` and `isPost_*`~~ **Resolved (2026-11-19)** from the task code and checked on all files; see the column table and [`ds007474-two-step-model.md`](ds007474-two-step-model.md). The shown outcome is reproduced exactly from the stored schedule under the contingency rule (stimulus: pad 1 / pad 3 draw by ship; state: the reached pad's draw).
4. **The authors' trial exclusions** (fast RTs, missed trials) and subject exclusions for the paper.
5. **The structural-session trials** (105 per subject) can enlarge behavioral fits, but they were collected in a different session and have no event timing.

## Decisions that materially affect the research (please review)
1. **Licensing.** I recommend asking the authors to add a license to the OSF project and the code repository before any public derived release. Until then, this repo stores no OSF data, fixtures or code, and its tests are synthetic.
2. **fMRI scope.** There is no subject-level derivative, so any ds007474 fMRI work needs our own preprocessing (for example fMRIPrep) of 259 GB of raw BOLD. The project plan lists raw-MRI reprocessing as out of initial scope, so **the RL benchmark (LDM-007) should start behavior-only**.
3. **Trial validity** in the adapter is "a first-stage response was made". No RT-based exclusion is applied yet; that is a modeling decision for LDM-007.
4. **Structural-session trials are kept and labelled** (`run = "structural"`), not merged into the functional runs.

## Reproduce
```bash
ldm ds007474 fetch-behavior --dest data/ds007474_osf     # 378 files, 13.4 MB, SHA-256-checked
ldm ds007474 trials --osf-root data/ds007474_osf --out data/ds007474_trials.csv              # behavior only
pip install -e '.[ds007474]'                             # optional reader for MATLAB tables
ldm ds007474 trials --osf-root data/ds007474_osf --out data/ds007474_trials.csv --timing     # + event timing
```
