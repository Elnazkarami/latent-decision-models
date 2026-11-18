# Fixture provenance

Source: OpenNeuro ds002158 snapshot 1.0.2 (License: CC0, per `dataset_description.json`).
Pereira et al. (2020) PNAS, https://doi.org/10.1073/pnas.1918335117 — "This data was obtained from the OpenNeuro database."

| Fixture file | Source path | Modification |
| --- | --- | --- |
| `task-main_events.json` | `task-main_events.json` | none |
| `sub-02/ses-001/func/sub-02_ses-001_task-main_run-001_events.tsv` | same path | none (48 rows, behavioral only) |
| `sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vhdr` | same path | none (header text; no signal) |
| `sub-02/ses-001/eeg/sub-02_ses-001_task-main_run-001_eeg.vmrk` | same path | removed `Stimulus,S128` and `SyncStatus` lines to reduce size; all other markers unchanged |

No EEG/MRI signal data are included. Used only by `tests/test_ds002158.py`.
