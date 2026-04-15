# driving-sandbox

Small **bird’s-eye (BEV) road** simulator: build a top-down map, render **perspective crops** along the lane, supervise a tiny **CNN / Transformer** on curvature-based steering (plus optional **off-ramp** intent), then **roll out** open-loop with the trained weights.

---

## Setup

From the project root (with Python 3.10+ recommended):

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux / macOS
pip install -r requirements.txt
```

Training benefits from a CUDA GPU; CPU works but is slower.

---

## Typical workflow

1. **Tune the world and experiment knobs** in `config.py` (road spline, camera, dataset size, off-ramps, simulation, etc.).
2. **Generate data** (images + `data/labels.csv`):

   ```bash
   python generate_dataset.py
   ```

3. **Train** (writes a timestamped run under `runs/<date>_<time>/` and may copy weights to `data/driving_net.pt`):

   ```bash
   python train.py
   ```

4. **Evaluate** on the test split (writes annotated previews under `data/test_pred/`):

   ```bash
   python evaluate_test.py
   ```

5. **Simulate** open-loop on the BEV map (loads latest checkpoint; writes artifacts under **`runs/<timestamp>_sim/`**, including **`driving_net.pt`** and **`config.py`** copies plus PNG/CSV/MP4s):

   ```bash
   python simulate.py
   ```

Most scripts import **`config.py`** as `cfg`; change hyperparameters there rather than hard-coding in each file.

---

## Directories

| Path | Role |
|------|------|
| **`data/`** | Generated JPGs (`train/`, `test/`, optional `train/offramp_*.jpg`, `test/offramp_*.jpg`), `labels.csv`, optional `test_labeled/`, `test_pred/`, `perturb_compare/`. |
| **`runs/`** | Training runs: `driving_net.pt`, logs, ONNX export, architecture diagram (see `train.py`). |
| **`dummy_data/`** | Created only if you use `generate_dummy_data.py` (smoke-test pipeline without the full renderer). |

---

## File reference

### Core configuration

| File | Purpose |
|------|---------|
| **`config.py`** | Single source of truth: world size, spline geometry, lane width, roadkill / off-ramp flags, dataset counts and geography, camera and normalization, model architecture switches, training and simulation hyperparameters. Read the module docstring at the top for a structured overview. |

### World and geometry

| File | Purpose |
|------|---------|
| **`generate_world.py`** | Builds **`DrivingWorld`**: BEV RGB map (grass, two-lane gray road, dashed centerline), cubic-spline **main centerline** `x(y)`, optional **roadkill** obstacles, optional **off-ramps** (quadratic Béziers with arc-length helpers). Used by dataset generation and simulation. |
| **`dataset_split.py`** | Shared **train/test row sampling** on the main road (bottom vs top BEV half), **arc-length** grids along the spline, **off-ramp** train spacing tied to main-road mean spacing, and helpers for label scaling / ramp caps. |

### Rendering and dataset

| File | Purpose |
|------|---------|
| **`perspective_camera.py`** | **BEV → forward perspective** warp (homography, adaptive far plane) producing square crops matching `CAMERA_IMAGE_SIZE`. |
| **`generate_dataset.py`** | Renders **train** and **test** crops along the road (and optional off-ramp crops), applies lateral/yaw perturbations when enabled, computes **steering** labels from scaled curvature, writes **`data/labels.csv`**. Run as main: `python generate_dataset.py`. |
| **`data_loader.py`** | **`DrivingDataset`** (PyTorch), path to active CSV (`labels.csv` vs newer `labels_new.csv`), helpers to **count** train/test rows and **perturbation** stats for tooling. |
| **`reproducibility.py`** | **`set_global_seed`**: Python / NumPy / PyTorch + deterministic cuDNN flags for repeatable runs. |

### Model and training

| File | Purpose |
|------|---------|
| **`driving_model.py`** | **`DrivingNet`**: conv backbone, optional **Transformer** head vs flatten+linear; predicts steering (**channel 0**) with scalar **`take_offramp`** concatenated to features. |
| **`train.py`** | Full training loop: DataLoaders, loss on channel 0, checkpointing under **`runs/<stamp>/`**, exports ONNX for Netron, calls **`run_architecture`** for diagrams. Run: `python train.py`. |
| **`run_architecture.py`** | Draws an **architecture schematic** and parameter breakdown into the current run folder; includes an ONNX-friendly variant for visualization tools. |
| **`evaluate_test.py`** | Loads best/latest weights, runs **`test/`** rows, writes **`data/test_pred/`** images with target vs prediction overlay. Run: `python evaluate_test.py`. |

### Simulation and analysis

| File | Purpose |
|------|---------|
| **`simulate.py`** | Loads **latest checkpoint**, steps the policy on the map (main spline by arc length; **off-ramp Bézier** after a branch merge when intent allows); optional **real-time BEV/driver** windows; each run saves **first-person + BEV** MP4s, trajectory CSV, and path overlay under **`runs/<timestamp>_sim/`**. Run: `python simulate.py`. |

### Utilities and scratch

| File | Purpose |
|------|---------|
| **`generate_dummy_data.py`** | Creates random RGB tiles and a tiny CSV under **`dummy_data/`** to sanity-check the training stack without OpenCV world rendering. |
| **`show_perturb_pairs.py`** | Builds **`data/perturb_compare/`**: side-by-side clean vs aligned-perturbed train frames and diff panels (sorted by change magnitude). Run: `python show_perturb_pairs.py`. |
| **`notes.txt`** | Loose project notes (not required for execution). |
| **`requirements.txt`** | Python dependencies (NumPy, OpenCV, SciPy, PyTorch, etc.). |

---

## Geography and labels (short)

- **Main road:** Training rows use the **bottom** BEV half (`y` toward the bottom of the image); test rows use the **top** half. `DATASET_MIX_TRAIN_TEST_GEOGRAPHY` only **shuffles order within** each half.
- **Steering:** Derived from **signed curvature** (main spline or off-ramp Bézier), **globally scaled** by max |κ| in the CSV so values stay in a bounded range; see `config.py` for clipping and perturb recentering.
- **Off-ramps:** Optional extra map arms and `train/offramp_*.jpg` / `test/offramp_*.jpg` with `take_offramp=1` in `labels.csv`. Train ramp crops are spaced along each ramp at the **same mean arc spacing** as main-road train frames (within the train half), subject to off-ramp caps in config.

---

## Simulation intent and off-ramps

Open-loop roll-out passes a scalar **`take_offramp`** tensor into `DrivingNet` each step (same role as the CSV column at training time). **Reference kinematics** use arc length on the main centerline when `SIM_PROJECT_REF_ONTO_MAIN_ROAD` is true; with `OFFRAMP_ENABLE`, the sim can **merge** onto a Bézier off-ramp when the ego crosses a branch row (`DrivingWorld.offramp_branch_y_pxs()`), but only if **intent** says to take an exit on that step:

- **`SIM_TAKE_OFFRAMP_UPPER_HALF_NAV` (default on):** `take_offramp` is **1** while ego image-`y` is in the **upper** half of the BEV (`y ≤ WORLD_IMAGE_SIZE/2`, origin top-left), and **0** in the lower half. The realtime BEV can show a **nav box** with `SIM_NAV_EXIT_INSTRUCTION_TEXT` whenever that geographic intent is active. This pattern lets a **lower-half** exit (e.g. primary branch) stay on the main road while an **upper-half** exit (e.g. a second branch / “exit 117” style row in `OFFRAMP_EXTRA_BRANCH_Y_PX`) still merges when intent is on.
- **`SIM_TAKE_OFFRAMP_UPPER_HALF_NAV` off:** `take_offramp` and merge-at-branch behavior follow the fixed flag **`SIM_TAKE_OFFRAMP`** for the whole rollout (merge at every branch when true).

Paused BEV **drag-to-relocate** can snap to main or ramp when off-ramps exist (`ramp_kinematics` in code), independent of the merge flags above.

---

## Checkpoints

- **`simulate.py`** / **`evaluate_test.py`** prefer the **newest** `runs/*/driving_net.pt` unless `data/driving_net.pt` is newer by a small margin (see `CHECKPOINT_PREFER_DATA_IF_NEWER_BY_SEC` in `config.py`).
- Each **`simulate.py`** invocation creates a fresh folder **`runs/<YYYY-mm-dd_HH-MM-SS>_sim/`** with **`sim_path.png`**, **`ego_path.csv`**, **`sim_first_person.mp4`**, **`sim_bev.mp4`**, a copy of the **`driving_net.pt`** used for the roll-out, and a copy of **`config.py`** (for reproduction; separate from training run directories).
- After training, your latest train run is summarized in **`runs/LATEST_RUN.txt`** (path + status).

---

## Headless simulation

If OpenCV GUI calls fail (e.g. over SSH), set in `config.py` before running:

- `SIM_REALTIME_BEV = False`
- `SIM_REALTIME_DRIVER_VIEW = False`

MP4s are written every run regardless of windows; use **`MAX_SIM_SPEED = True`** or set the realtime flags above for a headless-style roll-out. You can also patch flags in a one-liner before importing `simulate`.
