# Training MOCID on the lab server from your own laptop

How to get this repository onto the lab GPU server, use the DAUB dataset that is already
there, and train the three models of the paper's ablation (Table 2):

| Model | File | Paper row (Table 2) | Params |
|---|---|---|---|
| `MOCIDBase` | [base.py](base.py) | Base | 8.94 M |
| `MOCIDBaseFISTA` | [base_fista.py](base_fista.py) | +FISTA | 9.49 M |
| `MOCID` | [model.py](model.py) | +FISTA+DAM (MOCID) | 12.53 M |

All three are trained with the repo's own pipeline (`main.py` → `train.py`), using the team's
standard settings, the **R0 profile** (§7). How the models differ from each other and from the
paper is in [FINDINGS.md](FINDINGS.md).

---

## 1. How the pieces fit together

```
your laptop                                   lab server (pl-lawr7615)
-----------                                   ------------------------
GitHub access           --- rsync / ssh -->   NVIDIA L40S GPU (46 GB), shared
git clone / git push    <-- rsync --------    DAUB, Python environment, helper scripts
                                              NO access to github.com
```

The server cannot reach GitHub. Every copy of the code goes laptop → server with `rsync`,
and every result you want on GitHub goes server → laptop → `git push`.

## 2. Access

You need:

- **The university VPN**, connected, whenever you talk to the server.
- **An account on `pl-lawr7615.services.adelaide.edu.au`**, with your university ID as the
  username.
- **Membership of the `proj-mamba` group**, which gives read access to the data, the
  environment and the helper scripts under `/srv/proj-mamba/`. Check with `groups` once
  logged in.

Add a short name for the server to `~/.ssh/config` **on your laptop**, so the commands below
work as written:

```
Host mamba
    HostName pl-lawr7615.services.adelaide.edu.au
    User a1234567          # your university ID
```

Then `ssh mamba` logs you in.

**Optional: VS Code.** With the *Remote - SSH* extension, connect to `mamba` and open your
repo folder. Files you edit are on the server, and the integrated terminal is a server
shell. Anything that needs GitHub still has to happen in a terminal on your laptop.

## 3. Getting the repository onto the server

### First time

On your laptop:

```bash
git clone https://github.com/mvkulatunga/IRSTD-Methods-master.git
rsync -avP --exclude '__pycache__' IRSTD-Methods-master mamba:~/
```

The repo lands at `~/IRSTD-Methods-master` on the server. Home directories on the server are
private, so nobody else can see it, and nothing in it can be shared by path.

### Updating it later

Option A, **rsync again**. Simple, but it copies your laptop's working tree over the
server's, so commit or copy anything you changed on the server first:

```bash
# laptop
cd IRSTD-Methods-master && git pull
rsync -avP --exclude '__pycache__' --exclude 'MOCID/runs' ./ mamba:~/IRSTD-Methods-master/
```

Never add `--delete`: it would remove files that only exist on the server, such as training
outputs.

Option B, **a git bundle**. This carries history, so the server copy stays a real git repo
you can compare and merge:

```bash
# laptop
cd IRSTD-Methods-master && git pull
git bundle create /tmp/irstd.bundle --all
rsync -avP /tmp/irstd.bundle mamba:~/

# server
cd ~/IRSTD-Methods-master
git fetch ~/irstd.bundle 'refs/heads/*:refs/remotes/origin/*'
git log --oneline main..origin/main     # what's new
git merge --ff-only origin/main
```

### One or two files

`raw.githubusercontent.com` is reachable from the server, so a single file can be fetched
directly:

```bash
curl -O https://raw.githubusercontent.com/mvkulatunga/IRSTD-Methods-master/main/MOCID/model.py
```

There's no way to list or download the whole repo like this: the GitHub API is blocked.

## 4. DAUB on the server

DAUB is already on the server. Don't download or copy it.

| What | Where |
|---|---|
| Images | `/srv/proj-mamba/original_data/DAUB/dataN/<frame>.bmp` |
| Train split | `/srv/proj-mamba/mocid-baseline/splits/daub_train_server.txt` |
| Validation split | `/srv/proj-mamba/mocid-baseline/splits/daub_val_server.txt` |

- **17 infrared videos**, one folder each (`data5` … `data22`), frames numbered from `0.bmp`.
  The frames are 256×256 and each contains one small target, typically about 8×8 pixels.
- **The split is the paper's**, from SSTNet's released files with the paths rewritten:
  - train: `data5, 8, 9, 10, 13, 14, 16, 17, 19, 22`, 8,982 frames;
  - validation: `data6, 11, 12, 15, 18, 20, 21`, 4,795 frames.
  No video appears in both, and there is no separate test set.
- **Annotation format**: one line per frame, `<image path> x1,y1,x2,y2,class`, with class `0`:

  ```
  /srv/proj-mamba/original_data/DAUB/data6/0.bmp 96,77,104,85,0
  ```

- **Clips** (the repo's loader, `utils/data.py`): for each annotated frame, a 5-frame clip from
  the same video ending at that frame, so the target frame is last. Frames are resized to
  512×512 (bilinear) and scaled to 0–1. A clip needs 4 real earlier frames, so the first 4
  frames of every video are skipped: **8,942 training clips and 4,767 validation frames** are
  used, not 8,982 and 4,795. `MOCIDBase` uses only the target frame of each clip.

## 5. One-time setup on the server

**There is nothing to install or configure.** On this server the repo finds what it needs by
itself (`config.py`, "lab server defaults"):

- VMamba, `fvcore`, and a selective-scan CUDA kernel patched to build under CUDA 13, all in
  `/srv/proj-mamba/mocid-baseline/deps`. MOCID's DAM needs them; the stock kernel does not
  compile here;
- the train and validation split files from §4;
- `/srv/proj-mamba/runs/<your ID>` for run outputs (below).

The one thing to get right is the Python: always use the project's environment,
**`/srv/proj-mamba/venv/bin/python`** (PyTorch 2.13, CUDA 13). A plain `python` may be a
different one, such as conda's. Adding an alias to `~/.bashrc` on the server saves typing:

```bash
alias mpy=/srv/proj-mamba/venv/bin/python
export SCREENDIR=$HOME/.screen      # screen -ls / -r fail without it on this server
```

**Check the whole setup** by scoring R0's stage-1 checkpoint. It must print **88.65**; any
other number means something is set up differently from R0:

```bash
cd ~/IRSTD-Methods-master/MOCID
/srv/proj-mamba/venv/bin/python main.py --profile r0 eval \
    --ckpt /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth --no-dam
```

It takes about 2 minutes. If it fails with `ModuleNotFoundError: base` or similar, your copy
of the repo is out of date; update it (§3).

**Where outputs go.** Training writes to `runs/<tag>/` inside the `MOCID` folder. Checkpoints
are 80–140 MB each and a run keeps several, so they must not go in your home directory:
`/home` is a single 50 GB volume shared by every user, and when it fills up, runs crash while
saving and other people's sessions break too. So the first time you train, `train.py` creates
`/srv/proj-mamba/runs/<your ID>` (on the 500 GB project disk) and links `MOCID/runs` to it.
Keep it that way.

**Optional helper scripts.** `/srv/proj-mamba/mocid-baseline/scripts` has two conveniences:
`train.sh` (starts a run in a `screen` session, §7) and `progress.sh` (a run's results, §8).
Copy them if you want them; the shared copy is read-only:

```bash
cp -r /srv/proj-mamba/mocid-baseline/scripts ~/mocid-scripts
```

They expect your repo at `~/IRSTD-Methods-master`; if it's elsewhere, add
`export MOCID_DIR=/path/to/IRSTD-Methods-master/MOCID` to `~/.bashrc`.

## 6. Before you train: check the GPU

There is one GPU, shared with other groups:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader
```

If another job is running near 100% utilisation, wait. Our runs are not memory-bound (about
10 GB each), but they are badly slowed by compute contention. In one case a run that
normally takes 0.2 s per step took 12 s, which would have turned 4 hours into a week.
`train.sh` prints the GPU state before it starts, so you can back out.

## 7. Training

Training is `main.py` in the repo's `MOCID` folder. `--profile` and `--model` go **before**
`train`; the run's options go after it:

```bash
cd ~/IRSTD-Methods-master/MOCID
PY=/srv/proj-mamba/venv/bin/python

# Base
$PY main.py --profile r0 --model MOCIDBase train --tag base-jane-1

# Base + FISTA
$PY main.py --profile r0 --model MOCIDBaseFISTA train --tag bf-jane-1

# Full MOCID: stage 1 (backbone with FISTA, DAM off) then stage 2 (DAM), in one run
$PY main.py --profile r0 train --tag mocid-jane-1
```

Every run needs a **tag**, the name of its output folder `runs/<tag>/`. Put your name in it
(for example `base-jane-1`) so runs don't collide. `MOCIDBase` and `MOCIDBaseFISTA` train in
one stage of 100 epochs; the full MOCID runs 100 epochs of stage 1, then 100 of stage 2.

**Stage 2 only.** To work on the DAM without retraining stage 1, start stage 2 from an
existing stage-1 checkpoint, for example R0's:

```bash
$PY main.py --profile r0 train --tag dam-jane-1 \
    --stage1-from /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth
```

**Keep runs alive when you disconnect.** A run started as above stops if your SSH connection
drops. For anything longer than a few minutes, start it inside `screen` (`screen -S <tag>`,
then the command, then Ctrl-A D to detach), or with `nohup`:

```bash
nohup $PY -u main.py --profile r0 --model MOCIDBase train --tag base-jane-1 > runs/base-jane-1.log 2>&1 &
```

Or use the helper, which does the `screen` part, prints the GPU state first, always uses
`--profile r0`, and refuses to start from a copy of the repo that is too old to recompute
BatchNorm statistics before evaluation (§12):

```bash
bash ~/mocid-scripts/train.sh base-jane-1 --model MOCIDBase
bash ~/mocid-scripts/train.sh dam-jane-1 --stage1-from /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth
```

On a GPU you have to yourself, an epoch takes about 4–5 minutes for the full MOCID (either
stage) and less for the one-frame `MOCIDBase`, so 7–8 hours per 100 epochs, and about twice
that for the full MOCID's two stages.

Things to know before reading the numbers:

- **The DAM handover was fixed on 30 Sep 2026.** Before that, switching the DAM on at the
  start of stage 2 cost up to 64 AP50 until the FPN and head retrained around it. Stage-2
  results from before the fix aren't comparable. See [CODE-REVIEW.md](CODE-REVIEW.md) finding 1.
- **The repo's MOCID has a different front end from `MOCIDBaseFISTA`**, so it isn't a clean
  "+DAM" step on top of it. See [FINDINGS.md](FINDINGS.md) §4.
- **Under these settings the plain Base does badly** (75.93 AP50 in our run), because weight
  decay on BatchNorm and the unnormalised input hurt it far more than the full MOCID. So the
  Base → +FISTA gain comes out larger than the paper's. See `results/BASELINE-FINDINGS.md`.

### Checking a DAM change before training

After any change to the DAM (`components/dam.py`), run the check script on a stage-1
checkpoint before committing hours to a stage-2 run:

```bash
cd ~/IRSTD-Methods-master/MOCID
/srv/proj-mamba/venv/bin/python tools/check_dam.py --profile r0 \
    --ckpt /srv/proj-mamba/mocid-baseline/checkpoints/r0_fista_best.pth
```

It checks that switching the DAM on at its initial weights leaves the model's output
unchanged, that a training step gives finite gradients, and the parameter counts against
the paper. Add `--ap` to also compare AP50 on the full validation set (about 4 more minutes).

### The training settings (the R0 profile)

`train.sh` always uses `--profile r0`: the settings of the team's first full two-stage run,
R0 (`results/R0`), which the team uses as the reference for all new work. They live in
`config.py` (`PROFILES["r0"]` on top of the `Config` defaults) and `train.py`.

| Setting | R0 profile |
|---|---|
| Optimizer | SGD, momentum 0.937 (Nesterov), weight decay 5e-4 on **every** parameter |
| Learning rate | 0.01 → 1e-4: 6-epoch linear warmup, then cosine |
| Stage 2 learning rate | 1e-3 → 1e-4 |
| Epochs | 100 (MOCID stage 2: another 100), each over the whole training set (2,235 steps) |
| Batch size, input | 4; 512×512, 5-frame clips |
| Augmentation | random horizontal flip of whole clips |
| Normalisation | ÷255 only |
| Precision, clipping | mixed precision (AMP); gradients clipped at 10 |
| EMA | 0.9999 (stage 2: 0.999); evaluation and checkpoints use the EMA weights |
| What trains in stage 2 | the DAM, FPN and head; the backbone is frozen |
| Evaluation | every 2 epochs, on the 4,767 validation frames |
| BatchNorm statistics | recomputed from 300 training batches before every evaluation, for the model being evaluated (§12) |
| Metric | VOC all-points AP50 at IoU 0.5, and F1 at the best confidence threshold |
| Best checkpoint | by AP50; from epoch 40 in stage 1 and in the one-stage models, from the start in stage 2 |

These are not command-line options. To try a different setting, add a new profile to
`PROFILES` in `config.py`, give it a name, and pass it to `main.py` with `--profile`.
**Evaluate with the same profile you trained with**: the normalisation differs between
profiles, and R0's checkpoint scores 49.42 instead of 88.65 without `--profile r0`.

## 8. Watching, stopping and resuming

```bash
bash ~/mocid-scripts/progress.sh <tag>      # the latest evaluations, plus the best AP50 per stage
tail -f ~/IRSTD-Methods-master/MOCID/runs/<tag>.log   # console output, if you used nohup or train.sh
screen -ls                                  # your sessions
screen -r <tag>                             # attach; Ctrl-A then D to detach again
screen -S <tag> -X quit                     # stop a run
```

`progress.sh` shows nothing until the first evaluation, after epoch 2 (about 10 minutes). If
it is still empty after 20 minutes, check the GPU (§6).

To **resume** a run that stopped, run the same `train.sh` command with the same tag: it picks
up from the latest checkpoint in `runs/<tag>/`. For the same reason, **reusing a tag resumes
the old run instead of starting a new one.**

## 9. What a run produces

Everything is in `MOCID/runs/<tag>/`, which is `/srv/proj-mamba/runs/<your ID>/<tag>/`:

| File | Contents |
|---|---|
| `eval_log.csv` | one row per evaluation: time, tag, epoch, AP50, F1, average training loss |
| `best.pth`, `final.pth`, `last.pth` | `MOCIDBase` / `MOCIDBaseFISTA`: best-AP50 weights, final weights, full state for resuming |
| `fista_best.pth`, `fista.pth`, `fista_last.pth` | MOCID stage 1: the same three |
| `dam_best.pth`, `dam.pth`, `dam_last.pth` | MOCID stage 2: the same three |

In `eval_log.csv` the `tag` column says which stage a row belongs to: `<tag>-base`,
`<tag>-basefista`, `<tag>-fista` (MOCID stage 1) or `<tag>-dam` (MOCID stage 2). AP50 and F1
are percentages.

**Per-video results** for any checkpoint (pass the same `--model` it was trained with):

```bash
cd ~/IRSTD-Methods-master/MOCID
/srv/proj-mamba/venv/bin/python main.py --profile r0 --model MOCIDBaseFISTA eval \
    --ckpt runs/<tag>/best.pth --perseq
```

**Checkpoints saved before 1 Oct 2026** (R0's, for example) have stale BatchNorm statistics.
Add `--recal-bn 300` to `eval` to score them the way training now does; R0's stage-1
checkpoint scores 88.65 as saved and 89.55 recomputed.

When you report a run, give both the best and the final epoch. The best epoch is chosen on
the validation set, so on its own it is optimistic.

## 10. Getting results back and sharing them

To your laptop:

```bash
# laptop
rsync -avP mamba:/srv/proj-mamba/runs/<your ID>/<tag>/eval_log.csv ./
```

Checkpoints are 80–140 MB each; copy them only if you need them, and never commit them.

To teammates on the server, copy the small files to the group area:

```bash
mkdir -p /srv/proj-mamba/results/mocid-daub/<tag>
cp /srv/proj-mamba/runs/<your ID>/<tag>/eval_log.csv /srv/proj-mamba/results/mocid-daub/<tag>/
```

To GitHub, add a folder under `MOCID/results/<run-name>/` with those files and a short
`README.md` (settings, best and final numbers, anything that differs from the R0 profile), then
push from your laptop. If you committed on the server, carry the commit over with a bundle:

```bash
# server
cd ~/IRSTD-Methods-master
git bundle create ~/out.bundle origin/main..main

# laptop
rsync -avP mamba:~/out.bundle /tmp/
git fetch /tmp/out.bundle main:from-server
git cherry-pick <commit>          # each commit to bring over
git push origin main
```

## 11. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `ssh: Could not resolve hostname` or it hangs | VPN not connected |
| `Permission denied` under `/srv/proj-mamba` | not in the `proj-mamba` group yet |
| `git clone` / `git pull` hangs on the server | GitHub is blocked there; see §3 |
| `ModuleNotFoundError: base`, `base_fista` or `components.yolox` | your repo copy predates them; update it (§3) |
| `ModuleNotFoundError: classification` or `fvcore` | the repo couldn't find the shared dependencies: check you can read `/srv/proj-mamba/mocid-baseline/deps` (the `proj-mamba` group) and that your repo is up to date. Off this server, set `VMAMBA_PATH` and `PYTHONPATH` yourself |
| `unrecognized arguments: --model` | `--model` and `--profile` go **before** `train` / `eval` (`train.sh` handles this) |
| The setup check (§5) prints a number other than 88.65 | the repo or environment differs from R0's; check you used `--profile r0` and an up-to-date repo |
| `Directory '/run/screen' must have mode 777` | `SCREENDIR` isn't set; add `export SCREENDIR=$HOME/.screen` to `~/.bashrc` (§5) |
| `progress.sh` empty after 20+ minutes | another job is using the GPU (§6) |
| A new run starts at epoch 20, not 1 | the tag was used before, so it resumed; pick a new tag |
| `CUDA out of memory` | someone else is using most of the GPU memory; check `nvidia-smi` |
| `train.sh` says your copy of the repo is out of date | it predates the BatchNorm fix (§12); update it (§3) |
| A run's first log line doesn't say `BatchNorm recomputed before eval: 300 batches` | your repo copy is out of date, or `BN_RECAL_BATCHES` was changed in `config.py`; every team run should have it |
| `RuntimeError: ... iostream error` or `unexpected pos` when saving, or `No space left on device` | a full disk, usually `/home` (`df -h /home`). Make sure `MOCID/runs` points to `/srv/proj-mamba/runs/<your ID>` (§5) and clear out old files in your home directory |

## 12. Notes on the pipeline

**One pipeline for every model.** Until 1 Oct 2026, `MOCIDBase` and `MOCIDBaseFISTA` were trained
with a separate script (`ablation_train_sstnet.py`) that followed SSTNet's released training
code rather than R0's settings. It has been retired: `main.py --model` now trains all three
models with the same loop, settings and metric. Results from the retired script used a
different recipe and all 4,795 validation frames, so they aren't directly comparable with
results from `main.py`.

**`torch.compile` is off.** `train.py` only compiles the model when `MOCID_COMPILE=1` is set.
It can't work here as the server stands: PyTorch's compiler generates GPU code with Triton,
which needs Python's C header files (the `python3.12-devel` system package), and they aren't
installed. With the headers supplied by hand it does work for `MOCIDBase`, but was no faster
(0.111 against 0.114 s per step), and it fails on the full MOCID with a compiler error
(`InductorError: ValueRangeError: Invalid ranges [0:-1]`). Results don't depend on it: R0
ran without it.

**BatchNorm statistics are recomputed before each evaluation.** Evaluation uses the EMA copy
of the model, which averages its weights and its BatchNorm statistics separately. With R0's
settings the weights move fast (LR 0.01, and weight decay shrinking the BatchNorm scales), so
the averaged statistics stop matching the averaged weights, and evaluation can collapse while
training is fine. That is what happened to R0 around epochs 10–25 (AP50 fell to 0.7 while the
training loss kept falling). On a collapsed checkpoint, recomputing the statistics took AP50 from
28.87 to 75.18. Since 1 Oct 2026, `train.py` recomputes them from 300 training batches before
every evaluation (`BN_RECAL_BATCHES` in `config.py`; 0 turns it off), and saves the recomputed
model as `best` / `final`. Training itself is unchanged. In stage 2 only the layers after the
frozen backbone are recomputed.

**How the numbers relate to the paper's.** Both use VOC AP50 at IoU 0.5, but the repo's loader
evaluates 4,767 of the paper's 4,795 validation frames (§4), and several R0 settings are ones
the paper doesn't specify: the batch size, the scaling of the learning rate to it, AMP,
clipping and the normalisation. Treat comparisons with the paper's tables as approximate.
