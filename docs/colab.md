# Stills and keyframes on Google Colab (pay-as-you-go)

Qwen-Image needs a GPU with bf16 (Ampere or newer); Kaggle's T4s give blank images. Colab rents
A100s and L4s by **compute units**: 100 units cost $9.99 and last 90 days. An A100 uses about
5.4 units an hour, an L4 about 1.7. The pipeline drives Colab from GitHub Actions with Google's
Colab CLI; you never open a notebook.

How it runs: when stills or keyframes are due, the producer packs a batch, puts it in the repo's
`batches` release and starts the **colab-run** workflow. That workflow checks your real unit
balance, opens a Colab VM (A100 first, then L4), runs the same worker as on Kaggle, downloads the
results, **stops the VM**, and starts the producer again to collect them. The review and the gates
carry on as before. Video and the 4K upscale stay on Kaggle.

## Money guards

- `colab.total_cap_units` (100) in `config.json`: the producer sizes every batch to what is left,
  counting each batch's time at the A100 rate. When it is used up, Colab batches stop and the
  issue says so. Raising it is your decision.
- `colab.min_balance_units` (3): no VM is started below this real balance.
- The VM is stopped at the end of every run, any mochizuki VM left behind is stopped at the start
  of the next run, and the job on the VM gives its VM back by itself at its deadline.

## One-time setup

1. **Buy units.** colab.research.google.com → sign in → *Pay As You Go* → 100 compute units.
2. **Sign-in for GitHub.** On your Mac, in Terminal:

   ```bash
   brew install --cask google-cloud-sdk
   gcloud auth application-default login \
     --scopes=openid,https://www.googleapis.com/auth/cloud-platform,https://www.googleapis.com/auth/userinfo.email,https://www.googleapis.com/auth/colaboratory
   ```

   A browser opens; sign in with the Google account that has the units, and allow access.
   Optional check: `pip install google-colab-cli && colab --auth adc usage` should print your balance.
3. **Give it to GitHub.** Copy the file it wrote:

   ```bash
   pbcopy < ~/.config/gcloud/application_default_credentials.json
   ```

   GitHub → the repo → Settings → Secrets and variables → Actions → New repository secret →
   name `COLAB_ADC_JSON`, paste, save. Do not paste it anywhere else.
4. **Test with one still.** GitHub → Actions → colab-run → Run workflow → mode `test`. It makes one
   style still on a Colab GPU (well under one unit), prints the units used, and attaches the still
   to the run as `colab-test`. Production starts on its own after that.

## About that secret

It holds a long-lived Google sign-in token (Google Cloud and Colab access for your account).
It stays in GitHub's encrypted secrets and is deleted from the runner after each run. To revoke it
at any time: `gcloud auth application-default revoke` on the Mac, or Google Account → Security →
Third-party connections → "Google Auth Library" → remove access. Then make a new one as above.

## Settings (`config.json` → `colab`)

| Setting | Meaning |
|---|---|
| `enabled` | the producer sends stills and keyframes to Colab |
| `stages` | which stages go to Colab |
| `gpus` | GPUs to ask for, in order (`A100`, `L4`; never T4) |
| `units_per_hour` | Colab's rate per GPU, for the cap |
| `total_cap_units` | the budget you agreed; the producer stops at it |
| `min_balance_units` | no VM below this balance |
| `batch_hours` | the most one batch may run (GitHub stops a job at 6 h) |
