# Copy/paste into your coding agent

You are working inside the attached `latent-decision-models` repository.

First read `AGENTS.md`, `tasks/next-task.md`, `tasks/backlog.json`, and the three files under `docs/`. Inspect the actual source and run the smoke tests before changing anything.

Execute ticket **LDM-001** only: audit OpenNeuro ds002158 BIDS events and write a verified, source-linked dataset audit. If you can access a small metadata/events sample, implement a strictly tested task adapter only after validating the actual column names and coding. If metadata are unavailable, write a fetch manifest and stop rather than guessing. Do not download EEG/fMRI scans or train deep networks.

Preserve the naming everywhere: GitHub repository/distribution `latent-decision-models`, Python import `latent_decision_models`, CLI `ldm`. There is no branded project name.

At the end, report the files changed, exact commands run, tests actually passing/failing, scientific unknowns, GO/NO-GO for using ds002158, and the next unblocked ticket.
