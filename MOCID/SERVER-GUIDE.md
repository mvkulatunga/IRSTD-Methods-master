# Training MOCID on the lab server from your own laptop

How to get this repository onto the lab GPU server, use the DAUB dataset that is already
there, and train the three models we compare against the paper:

| Model | File | Paper row (Table 2) | Params |
|---|---|---|---|
| `MOCIDBase` | [base.py](base.py) | Base | 8.94 M |
| `MOCIDBaseFISTA` | [base_fista.py](base_fista.py) | +FISTA | 9.49 M |
| `MOCID` | [model.py](model.py) | +FISTA+DAM (MOCID) | 12.53 M |

How these differ from each other and from the paper is in [FINDINGS.md](FINDINGS.md).

---

## 1. How the pieces fit together

```
your laptop                                   lab server (pl-lawr7615)
-----------                                   ------------------------
GitHub access           --- rsync / ssh -->   NVIDIA L40S GPU (46 GB), shared
git clone / git push    <-- rsync --------    DAUB, Python environment, training scripts
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
  environment and the scripts under `/srv/proj-mamba/`. Check with `groups` once logged in.

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

- **Clips**: for each annotated frame, the loader builds a 5-frame clip from the same video,
  ending at that frame (the target frame is last). At the start of a video the earlier
  frames are clamped to frame 0. Each frame is letterboxed (bicubic) to 512×512, then scaled
  to 0–1 and normalised with ImageNet mean/std. `MOCIDBase` uses only the target frame.

## 5. One-time setup on the server

Copy the training scripts into your home directory, since the shared copy is read-only:

```bash
cp -r /srv/proj-mamba/mocid-baseline/scripts ~/mocid-scripts
cd ~/mocid-scripts
```

The environment is set by `env.sh`, which `train.sh` loads automatically. Add two lines to
`~/.bashrc` on the server so every new shell also has what it needs:

```bash
export SCREENDIR=$HOME/.screen                          # screen -ls / -r fail without it
export MOCID_DIR=$HOME/IRSTD-Methods-master/MOCID       # where your copy of the repo is
```

Change the second line if your repo is somewhere else.

What `env.sh` sets up, so you don't have to:

- the shared Python environment, `/srv/proj-mamba/venv` (PyTorch 2.13, CUDA 13);
- VMamba, `fvcore`, and a selective-scan CUDA kernel patched to build under CUDA 13, all
  from `/srv/proj-mamba/mocid-baseline/deps`. MOCID's DAM needs them; the stock kernel does
  not compile here;
- `TORCH_COMPILE_DISABLE=1`, because the repo's own `train.py` calls `torch.compile`, which
  crashes on this setup;
- `SCREENDIR=~/.screen`, because `screen` fails without it on this server.

Check it works:

```bash
source env.sh && $PY -c "import torch; print(torch.cuda.get_device_name(0))"   # NVIDIA L40S
```

`env.sh` also warns if your copy of the repo doesn't have `base_fista.py`. If you see that
warning, update the repo (§3).

**Where outputs go.** Checkpoints are 50–140 MB each and a run keeps three, so they don't go
in your home directory: `/home` is a single 50 GB volume shared by every user, and when it
fills up, runs crash while saving and other people's sessions break too. The first time you run
`train.sh`, it creates `/srv/proj-mamba/runs/<your ID>` (on the 500 GB project disk) and links
`~/mocid-scripts/runs` to it. Keep it that way.

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

Every run goes through `train.sh`, which starts it in a detached `screen` session named
after the run's **tag**, so it survives closing your laptop:

```bash
bash train.sh <tag> <script> [arguments]
```

Put your name in the tag (for example `base-jane-1`) so runs don't collide.

### First, a smoke test

Two epochs of 25 steps, validated on 160 clips. It takes about a minute and checks that the
data, environment and model all load:

```bash
bash train.sh smoke-jane ablation_train_sstnet.py --model MOCIDBaseFISTA --smoke
bash progress.sh smoke-jane        # two rows, AP50 0.00 is expected this early
screen -S smoke-jane -X quit       # close the finished session
rm -r runs/smoke-jane runs/smoke-jane.log
```

### The three models

```bash
# Base
bash train.sh base-jane-1 ablation_train_sstnet.py --model MOCIDBase

# Base + FISTA
bash train.sh bf-jane-1 ablation_train_sstnet.py --model MOCIDBaseFISTA

# Full MOCID: stage 1 (backbone with FISTA, DAM frozen), then stage 2 (DAM, backbone frozen)
bash train.sh mocid-s1-jane-1 fista_train_sstnet.py
bash train.sh mocid-s2-jane-1 fista_train_sstnet.py --stage 2 --init runs/mocid-s1-jane-1/best_ap50.pth
```

Start stage 2 only after stage 1 has finished.

These scripts are separate from the repo's own `main.py` / `train.py`. Why is in §12.

On a GPU you have to yourself, an epoch over the whole training set takes about 4–6 minutes
(measured: 4.2 minutes for `MOCIDBaseFISTA`), so a 100-epoch run takes 7–10 hours, and the
full MOCID's two stages about twice that.

Two things to know about the full MOCID before reading its numbers:

- **Stage 2 starts from a damaged model.** With the DAM at its initial weights, switching it
  on loses 22 AP50 before any stage-2 training. See [CODE-REVIEW.md](CODE-REVIEW.md) finding 1.
- **The repo's MOCID has a different front end from `MOCIDBaseFISTA`**, so it isn't a clean
  "+DAM" step on top of it. See [FINDINGS.md](FINDINGS.md) §4.

### The training recipe

These are the defaults: the recipe the team adopted, which takes every setting the paper
states and fills the gaps with explicit choices of ours.

| Setting | Default | Source |
|---|---|---|
| Optimizer | SGD, momentum 0.937, weight decay 5e-4 | paper |
| Learning rate | 0.01, scaled to the batch size: 6.25e-4 at batch 4 (0.01 × 4/64), down to 6.25e-6 | paper gives 0.01; the scaling is ours (SSTNet/YOLOX convention) |
| Schedule | 3-epoch warmup, then cosine, last 5 epochs flat | ours (the paper's "reduction coefficient 0.1" is not specific) |
| Stage 2 learning rate | 1e-3 → 1e-5 | ours (the repo's setting) |
| Epochs | 100 (stage 2: another 100), each over the **whole training set** (2,245 steps at batch 4) | paper |
| Batch size | 4 | ours (the paper doesn't say) |
| Input, clip length | 512×512, T = 5 | paper |
| Augmentation | **random horizontal flip of whole clips** | paper |
| Weight decay scope | conv/linear weights only, not BatchNorm or biases | ours (decaying BatchNorm collapsed training) |
| Normalisation | ÷255 then ImageNet mean/std | ours |
| Precision, clipping | fp32, no gradient clipping | ours |
| EMA | 0.9999 (stage 2: 0.999); evaluation and checkpoints use the EMA weights | ours |
| Evaluation | every epoch, all 4,795 validation frames | paper's frame count |
| Headline metric | **VOC all-points AP50** at IoU 0.5; it also picks `best_ap50.pth` | paper (confirmed by the supervisor) |

Common options, added after the script name:

| Option | Effect |
|---|---|
| `--seed N` | different initialisation and batch order (default 0) |
| `--epochs N` | shorter or longer runs |
| `--no-flip` | turn the flip augmentation off |
| `--epoch-div 5` | epochs of a random 1/5 of the training set, as in SSTNet's code (5× faster, 5× fewer steps) |
| `--lr X` | peak learning rate; the minimum becomes `X/100` |
| `--amp` | mixed precision |

Change one thing at a time, and put it in the tag.

## 8. Watching, stopping and resuming

```bash
bash progress.sh <tag>          # the last 12 epochs as a table, plus the best VOC AP50 so far
tail -f runs/<tag>.log          # live console output (Ctrl-C stops watching, not the run)
screen -ls                      # your sessions
screen -r <tag>                 # attach; Ctrl-A then D to detach again
screen -S <tag> -X quit         # stop a run
```

`progress.sh` shows nothing until the first epoch finishes, about 4–6 minutes in. If it is
still empty after 15 minutes, check the GPU (§6).

To **resume** a run that stopped, run the same `train.sh` command with the same tag: it picks
up from `runs/<tag>/last.pth`. For the same reason, **reusing a tag resumes the old run
instead of starting a new one.**

## 9. What a run produces

Everything is in `~/mocid-scripts/runs/<tag>/`, which is `/srv/proj-mamba/runs/<your ID>/<tag>/`:

| File | Contents |
|---|---|
| `log.csv` | one row per epoch |
| `per_video.csv` | VOC AP50, COCO-style AP50, recall and frames with no detection, per validation video, per epoch (the Base and Base+FISTA trainer) |
| `best_ap50.pth` | EMA weights at the best-VOC-AP50 epoch |
| `best_valloss.pth` | EMA weights at the lowest validation loss |
| `last.pth` | the full training state, for resuming |

The columns of `log.csv`:

| Column | Meaning |
|---|---|
| `voc_ap50` | **the headline number**: VOC all-points AP at IoU 0.5, the paper's AP50 |
| `coco_ap50` | COCO-style 101-point AP at IoU 0.5; usually 0.1–0.5 below `voc_ap50` |
| `pr`, `re`, `f1` | precision, recall and F1, using SSTNet's convention (recall at confidence 0.001; precision averaged up to that recall) |
| `voc_best_f1` | F1 at the best confidence threshold (the repo's own convention) |
| `train_loss`, `val_loss` | losses. Validation loss usually bottoms out long before AP50 does, which is why checkpoints are chosen by AP50 |
| `skipped` | training steps dropped for a non-finite loss; should be 0 |
| `minutes` | time per epoch |

When you report a run, give both the best epoch and the final epoch. The best epoch is
chosen on the validation set, so on its own it is optimistic.

## 10. Getting results back and sharing them

To your laptop:

```bash
# laptop
rsync -avP mamba:/srv/proj-mamba/runs/<your ID>/<tag>/{log,per_video}.csv ./
```

Checkpoints are 50–140 MB each; copy them only if you need them, and never commit them.

To teammates on the server, copy the small files to the group area:

```bash
mkdir -p /srv/proj-mamba/results/mocid-daub/<tag>
cp runs/<tag>/log.csv runs/<tag>/per_video.csv runs/<tag>.log /srv/proj-mamba/results/mocid-daub/<tag>/
```

To GitHub, add a folder under `MOCID/results/<run-name>/` with those files and a short
`README.md` (settings, best and final numbers, what differs from the defaults), then push from
your laptop. If you committed on the server, carry the commit over with a bundle:

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
| `ModuleNotFoundError: base`, `base_fista` or `components.yolox` | your repo copy predates them; update it (§3), or set `MOCID_DIR` correctly |
| `ModuleNotFoundError: classification` or `fvcore` | `env.sh` wasn't loaded; use `train.sh`, or `source env.sh` first |
| `Directory '/run/screen' must have mode 777` | `SCREENDIR` isn't set; `source env.sh` |
| `progress.sh` empty after 10+ minutes | another job is using the GPU (§6) |
| A new run starts at epoch 20, not 1 | the tag was used before, so it resumed; pick a new tag |
| `AssertionError: non-finite in F_f` during a MOCID `--smoke` run | seen once, during validation of the compressed 2-epoch smoke schedule, and not on a rerun; full runs warm up over 3 epochs and have not hit it. Rerun the smoke test |
| `CUDA out of memory` | someone else is using most of the GPU memory; check `nvidia-smi` |
| `RuntimeError: ... iostream error` or `unexpected pos` when saving, or `No space left on device` | a full disk, usually `/home` (`df -h /home`). Make sure `runs` points to `/srv/proj-mamba/runs/<your ID>` (§5) and clear out old files in your home directory |
| `val_loss nan` on the second epoch of a `--smoke` run | the smoke schedule jumps to the full learning rate after 25 steps; harmless there. In a full run, `val_loss` should stay finite and `skipped` should stay 0 |

## 12. Why these scripts, and not the repo's own `train.py`

The repo's pipeline (`python main.py train`, which runs `train.py`) does work on this server
with `TORCH_COMPILE_DISABLE=1` set, and our first full MOCID run (R0) was trained with it.
We use separate scripts for four reasons.

**1. It only trains the full MOCID.** `train.py` builds `MOCID(...)` directly and runs both
stages back to back, using `set_stage`, which expects MOCID's `backbone`, `pool`, `fpn`, `head`
and `disp`. It cannot train `MOCIDBase` or `MOCIDBaseFISTA` without changes.

**2. It evaluates fewer frames than the paper.** Both pipelines now use the paper's metric,
VOC all-points AP50, but on different frames:

| | Repo `train.py` | Our scripts | Paper |
|---|---|---|---|
| Validation frames | **4,767** | 4,795 | 4,795 |
| Training clips | 8,942 | 8,982 | 8,983 |

`utils/data.py` drops every clip without 4 real earlier frames, so the first 4 frames of each
video are never used (28 validation frames and 40 training frames). SSTNet, whose split the
paper uses, repeats frame 0 instead, as our loader does. The repo's numbers are fine for
comparing its own runs, but not directly with the paper's tables.

**3. Some of its settings differ, and none can be changed from the command line.** On the
settings the paper states (whole training set, random flip, VOC AP50) the two pipelines now
agree. They differ only where the paper says nothing:

| | Repo `train.py` | Our scripts |
|---|---|---|
| Warmup | 6 epochs, linear | 3 epochs, quadratic |
| Optimizer | SGD with Nesterov momentum | SGD |
| Precision, gradient clipping | always AMP, always clipped at 10 | fp32, no clipping (both switchable) |
| Resize | stretch, bilinear | letterbox, bicubic |
| Evaluation | every 2 epochs | every epoch |

These live in `config.py` and the training loop. Tracking down why the original Base collapsed
needed one-setting-at-a-time ablations, which is why our scripts have a switch for each.

**4. Practicalities.**

- `torch.compile` is called unconditionally and crashes here, so the environment variable is
  mandatory.
- The epoch counts are only in `config.py`.
- One command runs both stages, 200 epochs back to back.

What is *not* a reason any more: the two bugs that caused the original Base collapse, weight
decay on BatchNorm and missing input normalisation, and the unscaled learning rate were fixed
upstream in commits `6c609d6` and `afb81a6`.

**Folding everything into `train.py`** is possible and would leave the team with one pipeline.
It would need: choosing the model by name (so it can train the ablation models, in a single
stage), clips that repeat frame 0 so all 4,795 frames count, command-line switches for the
settings above, and a guard around `torch.compile`. Until then, use the scripts
in this guide for anything you want to compare with the paper.
