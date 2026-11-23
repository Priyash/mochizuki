# State branch

Written by the producer workflow and the Claude reviewer. Do not edit by hand.

- `ledger.sqlite` — every job, verdict, approval and Kaggle batch
- `chapter.txt` — the chapter the producer works on (change it with `/chapter N` on the issue)
- `review/chN/` — images and `queue.json` for the reviewer, and `board.jpg` for you
- `reviews/` — the reviewer's verdict files, applied and removed by the next producer run
- `status.json` — what the last producer run did
