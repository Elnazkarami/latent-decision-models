#!/bin/zsh
# Clean-room reproduction of the committed reports (docs/VALIDATION.md, part 2).
#
# Fresh clone at a pinned commit, fresh environment from requirements-lock.txt, fresh
# metadata from OpenNeuro and OSF. Large signal files already downloaded with
# `ldm ds002158 fetch-eeg` / `fetch-fmri` are reused only after their size and SHA-256
# match the fetch manifests. Every report is regenerated into the clone and compared
# with the committed copy (and, for outputs that are not committed, with REF_LOCAL).
#
# Usage: STORE=/path/to/downloads WORK=/path/to/empty/dir COMMIT=<sha> \
#        REF_LOCAL=/path/to/reference/reports zsh reproduce/cleanroom.sh
#   STORE      holds ds002158/ with eeg_fetch_manifest.json and fmri_fetch_manifest.json
#   REF_LOCAL  reference copies of reports that are not committed (ds007474: the OSF
#              source has no license; the behavioral baseline)
#   OSF_LOCAL  optional: an earlier `ldm ds007474 fetch-behavior` folder, used instead of a
#              fresh OSF download (e.g. when the OSF API is failing). Every file must match
#              that folder's fetch_manifest.json (size and SHA-256, checked against OSF's
#              published hashes when it was fetched).
set -euo pipefail
: ${STORE:?} ${WORK:?} ${COMMIT:?} ${REF_LOCAL:?}
PY=${PY:-python3.12}
REPO=https://github.com/Elnazkarami/latent-decision-models.git
SUBJECTS=(sub-02 sub-04 sub-07 sub-08 sub-09 sub-10 sub-11 sub-12 sub-13 sub-14 sub-15
          sub-16 sub-17 sub-18 sub-19 sub-21 sub-22 sub-24 sub-26)
step() { echo "== $(date +%T) $*"; }

step clone and environment
mkdir -p $WORK/data $WORK/expected
git clone -q $REPO $WORK/repo && cd $WORK/repo && git checkout -q $COMMIT
$PY -m venv .venv
.venv/bin/pip install -q -r requirements-lock.txt
.venv/bin/pip install -q --no-deps -e .
ln -s $WORK/data data
cp reports/*.json $WORK/expected/
cp $REF_LOCAL/ds007474_two_step_model.json $REF_LOCAL/ds002158_confidence_baseline.json \
   $WORK/expected/
LDM=.venv/bin/ldm

step verify stored signal files against their manifests
$LDM verify-files --manifest $STORE/ds002158/eeg_fetch_manifest.json --root $STORE/ds002158
$LDM verify-files --manifest $STORE/ds002158/fmri_fetch_manifest.json --root $STORE/ds002158

step fresh metadata and trial table
$LDM ds002158 fetch-metadata --dest data/ds002158 > /dev/null
$LDM ds002158 trials --bids-root data/ds002158 --out data/ds002158_trials.csv \
    --issues data/ds002158_issues.csv > /dev/null
.venv/bin/python - $STORE/ds002158 data/ds002158 <<'PY'
import json, os, sys
from pathlib import Path
store, dest = Path(sys.argv[1]), Path(sys.argv[2])
for man in ("eeg_fetch_manifest.json", "fmri_fetch_manifest.json"):
    for m in json.loads((store / man).read_text())["files"]:
        target = dest / m["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            os.symlink(store / m["path"], target)
PY

step behavior
$LDM confidence-baseline --config configs/ds002158_confidence_baseline.yaml > /dev/null
$LDM ds002158 compare-paper --trials data/ds002158_trials.csv \
    --out reports/ds002158-paper-comparison.json > /dev/null

step EEG preprocessing, 19 subjects
for s in $SUBJECTS; do
  $LDM ds002158 eeg-preprocess --bids-root data/ds002158 --trials data/ds002158_trials.csv \
      --subjects $s --bcg-method aas --out-dir data/derivatives/hb-aas \
      --report data/eeg-preprocess-$s.json 2>&1 | grep -iv warn | tail -1
done

step EEG analyses
E=data/derivatives/hb-aas
$LDM ds002158 eeg-decode --config configs/ds002158_eeg_decoding_confirmatory.yaml \
    --epochs-dir $E --out reports/ds002158-eeg-decoding-confirmatory.json > /dev/null
$LDM ds002158 eeg-cardiac --config configs/ds002158_eeg_cardiac_control.yaml \
    --epochs-dir $E --out reports/ds002158-eeg-cardiac-behavior.json > /dev/null
$LDM ds002158 eeg-decode --config configs/ds002158_eeg_cardiac_control.yaml \
    --epochs-dir $E --out reports/ds002158-eeg-decoding-cardiac-control-aas.json > /dev/null
$LDM ds002158 fusion --config configs/ds002158_fusion.yaml --epochs-dir $E \
    --out reports/ds002158-fusion.json > /dev/null

step fMRI pilot from the derivative images
$LDM ds002158 fmri-lss --config configs/ds002158_fmri_pilot.yaml --bids-root data/ds002158 \
    --out-dir data/derivatives/ldm-fmri --report reports/ds002158-fmri-lss-qc.json > /dev/null
$LDM ds002158 fmri-pilot --config configs/ds002158_fmri_pilot.yaml \
    --betas-dir data/derivatives/ldm-fmri --out reports/ds002158-fmri-pilot.json > /dev/null
$LDM ds002158 fmri-timing --config configs/ds002158_fmri_pilot.yaml \
    --betas-dir data/derivatives/ldm-fmri --bids-root data/ds002158 \
    --out reports/ds002158-fmri-timing.json > /dev/null
$LDM ds002158 fmri-lss --config configs/ds002158_fmri_pilot_derivative.yaml \
    --bids-root data/ds002158 --roi-ts data/derivatives/ldm-fmri/roi_ts \
    --out-dir data/derivatives/ldm-fmri-derivative \
    --report reports/ds002158-fmri-derivative-lss-qc.json > /dev/null
$LDM ds002158 fmri-pilot --config configs/ds002158_fmri_pilot_derivative.yaml \
    --betas-dir data/derivatives/ldm-fmri-derivative \
    --out reports/ds002158-fmri-derivative-pilot.json > /dev/null

step ds007474 behavior and the primary RL comparison
if [[ -n ${OSF_LOCAL:-} ]]; then
  mkdir -p data/ds007474_osf && cp -R $OSF_LOCAL/. data/ds007474_osf/
  find data/ds007474_osf -name '._*' -delete   # macOS metadata files on exFAT/FAT drives
  .venv/bin/python -I - data/ds007474_osf <<'PY'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
files = json.loads((root / "fetch_manifest.json").read_text())["files"]
bad = []
for f in files:
    p = root / f["path"].lstrip("/")
    if not (p.is_file() and len(b := p.read_bytes()) == f["bytes"] and f["sha256_ok"]
            and hashlib.sha256(b).hexdigest() == f["sha256"]):
        bad.append(f["path"])
print(f"OSF_LOCAL: {len(files)} files checked, {len(bad)} failing", bad[:5])
sys.exit(1 if bad else 0)
PY
else
  $LDM ds007474 fetch-behavior --dest data/ds007474_osf > /dev/null
fi
$LDM ds007474 trials --osf-root data/ds007474_osf --out data/ds007474_trials.csv \
    --timing > /dev/null
$LDM two-step-model compare --config configs/ds007474_two_step_model.yaml > /dev/null

step compare regenerated reports with the references
$LDM compare-reports --expected $WORK/expected --actual reports --out $WORK/comparison.json \
    --allow-new-keys --files \
    ds002158_confidence_baseline.json ds002158-paper-comparison.json \
    ds002158-eeg-decoding-confirmatory.json ds002158-eeg-cardiac-behavior.json \
    ds002158-eeg-decoding-cardiac-control-aas.json ds002158-fusion.json \
    ds002158-fmri-lss-qc.json ds002158-fmri-pilot.json ds002158-fmri-timing.json \
    ds002158-fmri-derivative-lss-qc.json ds002158-fmri-derivative-pilot.json \
    ds007474_two_step_model.json
step done
