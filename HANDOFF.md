# Handoff: DeepReach + MPC for the 2D drone interception game

This file is for picking up the work in a new session. It covers the goal, how the game and training work, the code, what has been tried, what was learned, and what is open.

Repository: `kosmakosmakosma/SmartDrones`. Work branch: `claude/upbeat-edison-mkfeib` (last commit at time of writing: `28c932a`).

---

## 1. Working with this user

- **Language:** explain plainly. Avoid invented terms and jargon, and define symbols when you use them.
- **Questions:** when the user asks questions, answer them first. Don't implement until asked.
- **Actions:** don't take actions the user didn't ask for (no unrequested simulations or background jobs). They want to save tokens.
- **Disagreement:** scrutinize plans, ask clarifying questions when a decision is genuinely theirs, and say so when something won't work.
- **Their machine:** Windows, using PowerShell or cmd, run from the repo folder. The Python environment is `.\env\Scripts\python.exe`, or `env\Scripts\activate.bat` then `python`. Python lines must go into a `python` prompt or a `.py` file, not straight into cmd.
- **Training** runs on their machine and logs to W&B (project `SmartDrones`, entity `kosmakosmakosma-tu-delft`, group `2d_drones_mpc_2s`). They send W&B screenshots for analysis.
- **Git:** commit on the work branch and push with `git push -u origin claude/upbeat-edison-mkfeib`. No pull requests unless asked. Commit messages end with the attribution lines the session asks for.
- **This container's CPU is slow for MPC:** 8 games at 32 samples × 3 iterations take about 16 s, and cost grows with samples² × iterations. Keep any local experiments small and run them in the background.

---

## 2. The game

`CrazyflieInterception` in `dynamics/dynamics.py`: a 2D double integrator, attacker vs defender.

**State:** `[px_a, vx_a, py_a, vy_a, px_d, vx_d, py_d, vy_d]`, in metres and m/s. The target sits at the origin.

**Controls:** acceleration, limited per axis: attacker ±`accel_max_a`, defender ±`accel_max_d`.

**Speed limits:** `vel_max_a` and `vel_max_d`. After each step, `limit_state` scales a speed that is over the limit back along its direction. The Hamiltonian handles the limit: at the speed limit, outward acceleration along v is removed, and the optimum is taken over 6 candidates (`_best_acceleration`).

**Current parameters:**

| Parameter | Value |
|---|---|
| target_R | 0.2 |
| capture_R | 0.2 |
| defender_exclusion_R | 0.2 |
| accel_max_a / accel_max_d | 5 / 5 |
| vel_max_a / vel_max_d | 3.6 / 3.0 |
| box_loses | True |
| position box | ±2 m |
| tMax | 2 s |

**Rules** (`reach_fn` and `avoid_fn`):

The **attacker wins** if any of these happens:
- it reaches the target circle;
- the defender enters its keep-out zone (`defender_exclusion_R`);
- with `box_loses`, the defender leaves the box.

The **defender wins** if:
- it captures the attacker (distance ≤ capture_R);
- with `box_loses`, the attacker leaves the box, which counts like a capture.

**The value**, which is what the network learns:
- V(t,x) = min over τ ≤ t of max( reach(x_τ), max over s ≤ τ of fail(x_s) ), where fail = −avoid;
- the attacker minimises it and the defender maximises it;
- **V ≤ 0 means an attacker win**, V > 0 a defender win, and the size is the margin.

Properties to know:
- V ≥ −0.2 always (the attacker's depth into the target is at most target_R).
- The "caught" term is at most capture_R = 0.2.
- V can only go down as t grows, since more time can only help the attacker: dV/dt ≤ 0.
- **The game keeps going after a capture** in this definition. The outcome is locked (the caught term persists), but the attacker can still fly to the target, which lowers the margin. This keeps V continuous across the capture surface. The size of the value is a margin, not a physical quantity.
- **Exact ceiling:** an attacker can always fly straight at the target and ignore the defender. So V(t,x) ≤ U(t,x) = max(the closest the straight-flying attacker gets to the target edge within t, 0.2), with an extra term if that flight would leave the box. `value_upper_bound(times, states)` computes U. Once the attacker can reach the target in time (about 0.75 s from anywhere in the validation slice), V ≤ 0.2.

---

## 3. Training pipeline (DeepReach, `experiments/experiments.py`)

**Network:** sine MLP, 3 × 512, `exact` model: V = boundary_fn(x) + t · 50 · network_output. V(0,x) equals the boundary function exactly, so points at t = 0 teach nothing.

**PDE loss:** `brat_hjivi` with `--minWith target`. The residual is min( max(dV/dt − H, V − reach), V + avoid ), and the loss is its absolute value averaged over the batch. The `V + avoid` term is the floor "already caught means the defender wins", valid at every t.

**Curriculum:** 15k pretraining epochs at t = 0 (fit the boundary). Then the maximum time t_max grows linearly to tMax over `counter_end` = 100k epochs, then stays at full horizon. One epoch is one gradient step on a fresh batch.

**Batch** (`utils/dataio.py`, `ReachabilityDataset.__getitem__`), `--numpoints` points (the user now uses 45000):

| Share | Points |
|---|---|
| `--capture_fraction` 1/6 | defender placed at capture_R ± 3 cm from the attacker, random direction, random t |
| `--mpc_fraction` 0.33 | MPC labels from the replay buffer with time ≤ current t_max, drawn without repeats; if fewer are available, uniform points fill the gap |
| rest | uniform random states (velocities inside the speed disks), t uniform on [0, t_max] |
| pretraining | `--pretrain_geometric_fraction` 0.25: geometric points near the target and keep-out circles |

The geometric and learned boundary fractions are 0 after pretraining; they must be passed explicitly, because the defaults are non-zero.

**Loss terms:**
- **PDE loss** on all points.
- **MPC label loss** on the MPC points: `--mpc_loss_type l1`, `--mpc_loss_weight 1`. It is averaged over the MPC points only, so per point it pulls about 3× harder than terms averaged over the whole batch.
- **`--value_ceiling_weight`:** mean over all points of max(0, V − U). Default 0; the user uses 1.
- **`--monotonic_weight`:** mean of max(0, dV/dt). Default 0; the user uses 1.
- Both penalties are skipped during pretraining.

**Learning rate:** `--lr`, then exponential decay to `--lr_final`, from `--lr_decay_start_epoch` (default: end of the curriculum) to the last epoch. `--resume_lr` overrides the rate saved in the optimizer on resume.

**Seeds:**
- `--seed` (default 0) sets the network's starting weights and the main random stream (`run_experiment.py:245`, before the model is built).
- `--mpc_seed` seeds the MPC generator once per run start; it then advances, so every refresh is different.
- The held-out set uses a fixed seed, 12345.

### MPC label generation

Every `--mpc_refresh_epochs` (1000) from `--mpc_start_epoch` (15000), `_refresh_mpc_dataset` plays `--mpc_num_initial_states` games.

**Starting states** (`sample_mpc_initial_states`, `interception`): the defender near the target, the attacker coming in from outside with its velocity pointing inward. The user wants this distribution. Don't switch to uniform.

**Start time:** `--mpc_time_distribution`:
- `uniform` follows the curriculum;
- `current_max` starts at the current t_max;
- `tmax` (current choice) always starts at tMax. Labels with more time than the current t_max wait in the buffer until the curriculum reaches them.

**How the MPC plays** (`controllers/mpc.py`):
- **Search:** sampling-based. Candidate plans are random Gaussian noise held for blocks of `--mpc_control_hold_steps` (10), plus fixed test plans along the axes; candidate 0 is the current plan.
- **Scoring:** the reach-avoid suffix recursion J_k = max(fail_k, min(reach_k, J_{k+1})).
- **Solver:** `--mpc_game_solver maxmin` evaluates the full N × N table of attacker vs defender plans. `alternating` and `mixed` (regret matching) also exist.
- **Replanning:** closed loop (`--mpc_rollout closed_loop`): replan every `--mpc_replan_every` steps (5), dt 0.02, horizon 100 steps.
- **Training uses no network** (`--mpc_use_network false`). The game is played without the box walls (a copy of the dynamics without the box), and labels outside the domain are dropped (`--mpc_crop_to_domain`).
- **Defender keep-out:** `--mpc_defender_keep_out` rejects defender plans that enter its keep-out zone.

**End-on-event semantics: important, recently changed.**
- With `--mpc_end_on_event true` (the old setting), a game stops at the first capture, target hit or breach. Each label is then stored at its time-to-event (`mpc_label_times`).
- That turned out to bias labels upward. The MPC attacker plays the full game: it dodges, gets caught anyway, and its label is stored at the shorter time-to-event, where a straight dash would have scored at most U.
- Measured on the user's buffer: **7.5% of labels were above the exact ceiling, by 0.169 on average.** The network followed them (7% of MPC batch points above the ceiling by about 0.16), and that was most of `mpc_data_loss`.
- Fix 1: **`--mpc_end_on_event false`**. Full-length games: the MPC also plays after a capture, and labels keep their real time-to-go, which matches the network's "keeps going" definition. It also gives labels over 0–2 s evenly, instead of about 90% under 1 s. Costs about 2–2.5× more MPC time. About half the labels are states after the event; they're valid but less informative.
- Fix 2: **`--mpc_cap_labels true`**. Labels above U are lowered to U when stored, for the buffer, the new-game scores and the held-out set. The share of capped labels is logged.
- Ending at events was originally asked for to make the viewer's scenarios look realistic. It was carried into training labels without noticing the mismatch.

**Replay buffer:** FIFO with `--mpc_replay_capacity` (200k) and `--mpc_labels_per_refresh` (3000) random states per refresh. It is saved in the checkpoint and restored on resume unless `--mpc_reset_replay` is given.

**Held-out set:** `--mpc_holdout_games` (512), evaluated every `--mpc_holdout_eval_epochs` (5000). The games are played once, at full tMax with seed 12345 and **the run's current MPC settings**, and cached in `runs/<name>/training/mpc_holdout.pt`. **Later runs or resumes in that folder reuse the cache and ignore the current MPC flags.** Rename the file to regenerate it. Only states with time ≤ current t_max are scored.

---

## 4. Code map

| File | Contents |
|---|---|
| `dynamics/dynamics.py` (`CrazyflieInterception`) | `reach_fn`, `avoid_fn`, `boundary_fn`, `_box_margin`, `limit_state`, `value_upper_bound`, `_best_acceleration`, `hamiltonian`, `optimal_control` / `optimal_disturbance`, `sample_model_states` (velocities inside the speed disks) |
| `controllers/mpc.py` | `MPCConfig` (also domain and keep-out fields), `sample_control_sequences`, `rollout_joint_sequences`, `reach_avoid_suffix_values`, `optimize_joint_sequences` / `optimize_maxmin_sequences` / `optimize_mixed_sequences`, `solve_matrix_game`, `closed_loop_rollout` (`end_on_event`, `game_solver`, `replan_every`) |
| `utils/mpc_data.py` | `MPCReplayBuffer` (`sample_up_to_time`, without repeats), `sample_mpc_initial_states`, `sample_mpc_initial_times`, `mpc_domain_constraint`, `in_domain_mask`, `mpc_label_times`, `outcome_metrics`, `abs_error_by_time`, `TIME_BINS` |
| `utils/dataio.py` | `ReachabilityDataset`: batch composition, `POINT_GROUPS = ('random','geometric','capture','learned','mpc')`, curriculum |
| `utils/losses.py` | the brat loss; also returns per-point PDE residuals for logging |
| `experiments/experiments.py` | `_refresh_mpc_dataset` (`cap_labels`, `log_new_games`, `fixed_start_time`), `_network_values`, `_prepare_mpc_holdout`, `_evaluate_mpc_holdout`, `_batch_diagnostics`, `train()` (loss terms, bound metrics, LR schedule, holdout, refresh), `validate()` (plots; `--val_slice interception`) |
| `run_experiment.py` | all flags. **On resume, the current command's flags are used** (`mpc_options = opt`), so pass the full command again with `--resume`. `--minWith` and `--dynamics_class` are required. The W&B flags are rejected unless `--use_wandb` is given. |
| `utils/generate_mpc_scenarios.py` | standalone scenario generator |
| `utils/view_mpc_trajectories.py` | viewer for generated scenarios |
| `utils/benchmark_mpc.py` | paired MPC benchmark (exploitability), with a stand-in for the network |
| `simulate_nn.py` | single-game simulation: network controller vs MPC (`--controller attacker_mpc` / `defender_mpc` / `both_mpc` / `bang_bang`). Limitations: one hard-coded start state (`state0`), the MPC there uses the network's value at its horizon end (not independent), no box rule. |
| `tests/test_mpc.py`, `tests/test_mpc_data.py` | 57 tests, all passing: `python -m pytest -q tests` |

---

## 5. W&B metrics

| Metric | Meaning |
|---|---|
| `train_loss`, `pde_loss` | total loss and PDE residual |
| `pde_loss_random` / `_capture` / `_mpc` | PDE residual split by where the points came from |
| `mpc_data_loss` | \|V − label\| on MPC points × weight |
| `value_ceiling_loss`, `monotonic_loss` | the two penalties |
| `bounds/ceiling_violation_share`, `bounds/ceiling_excess` | how often and by how much V exceeds U; also `_<group>` per point group |
| `bounds/dvdt_positive_share` / `_mean` | how often V increases with t |
| `mpc_train/*` | on the batch's training labels: `mean_error` (V − label: positive leans defender), `balanced_accuracy` (winner by the sign of V, averaged over attacker-win and defender-win labels), `abs_error_t0.0-0.5` … `t1.5-max` (by time-to-go) |
| `mpc_new/*` | each refresh's new games, scored before training on them: `start_*` at the starting states, plus `abs_error_t*`. With `tmax`, values before the curriculum reaches tMax are meaningless (untrained times). |
| `holdout/start_*` | the 512 full-length starting states, the hardest case |
| `holdout/states_*` | all states of the held-out games, about 20k, mostly short time-to-go |
| `mpc_labels_capped_share` / `_mean_excess` | how many new labels the cap lowered |
| `mpc_replay_buffer_size`, `mpc_joint_mean_score` (average MPC game value at the starts), `lr` | |

**Validation plot** (`--val_slice interception`):
- rows are time-to-go; columns are the defender at rest at (px_d, 0);
- each panel shows the attacker at (px_a, py_a) flying at the target at `--val_attacker_speed`;
- blue means the defender wins, red the attacker, black is V = 0.

Known weaknesses of the plot (fix not done):
- px_d = 0 puts the defender inside its own keep-out zone, so that column is trivially red;
- the negative px_d columns mirror the positive ones;
- the colour scale (about ±0.8, from the t = 0 row) hides the long-time structure, which lives within ±0.2.

Suggested fix: px_d from about 0.3 to 1.0, and colours clipped to about ±0.25.

---

## 6. History and findings

- **v1** (60k batch, uniform start times, L2 then L1 labels): plateaued at train about 0.027 and PDE about 0.020. Labels were biased because games cut off at the event used the original time-to-go. Fixed by measuring label time to the event.
- **v2** (`current_max` start times, MPC share capped to the available labels, interception plot, held-out and new-game metrics, L1 labels weight 1, LR 2e-5 → 5e-6):
  - the long-time values (t ≥ 1.33) were far too high (0.6–0.8) where the exact ceiling says at most 0.2;
  - the network kept the shape of the t = 0 boundary;
  - `mpc_new/start_attacker_win_accuracy` was about 0.7.
- **v2_bounds** (v2 resumed with the ceiling and monotonicity penalties, constant LR 1e-5):
  - the ceiling was quickly satisfied, and the defender optimism disappeared (`mpc_new/start_mean_error` about −0.01);
  - held-out error dropped from 0.031 to 0.024 and held-out accuracy went to 0.93;
  - the PDE loss rose: the wrong-but-smooth long-time solution had been easy to fit. Judge runs on `holdout/*`, not the PDE loss.
- **v3** (from scratch, penalties from the start, `tmax`, batch 45k, LR 2e-5 → 1e-5):
  - matched v2_bounds with less compute: held-out states accuracy 0.94–0.95 and error 0.024;
  - held-out starts: about 0.88 accuracy, error 0.044, flat since about step 600 (the hard case);
  - about 2.7% of points were above the ceiling by about 0.15, now known to come almost entirely from MPC points and wrong labels (section 3).
- **MPC quality analysis** (this container, 32 interception starts at 2 s, end-on-event, compared against 128 samples × 5 iterations):

  | Setting | Time | Same winner as reference | Mean \|ΔV\| | Same winner across two seeds |
  |---|---|---|---|---|
  | 32×3 | 42 s | 0.81 | 0.055 | 0.88 |
  | 32×6 | 92 s | 0.88 | 0.050 | |
  | 32×10 | 127 s | 0.91 | 0.049 | 0.91 |
  | 64×3 | 172 s | 0.84 | 0.048 | |
  | 64×5 | 291 s | 0.84 | 0.048 | 0.78 |

  Conclusions:
  - iterations help more than samples (cost grows with samples²);
  - start-state labels carry 10–20% winner noise from close games;
  - the network's start accuracy (about 0.88) is at about the labels' own consistency, so better labels are needed to improve there;
  - each share is uncertain by about ±0.1 (32 games).

  The user now uses 15 iterations.
- **Per-group ceiling check:**
  - violations at MPC points about 7.3% (excess 0.16 per violator); random 0.1%; capture 0.2%;
  - the buffer check showed 7.5% of labels above U by 0.169. This led to the two fixes in section 3.

---

## 7. Current state / what the user is about to run

The user is stopping run v3 and resuming it with full-length games and capped labels:
1. Rename `runs\crazyflie_2d_mpc_2s_v3\training\mpc_holdout.pt` (so the held-out set is replayed with the new rules).
2. Resume with the full v3 command plus `--resume --additional_epochs 50000 --resume_lr 1e-5 --lr_final 1e-5 --mpc_end_on_event false --mpc_cap_labels true --mpc_reset_replay --mpc_iterations 15 --mpc_num_initial_states 128 --numpoints 45000 --wandb_name v3_fullgames`. The training ends at `num_epochs` + `additional_epochs`.

What to check when results come in:
- `mpc_labels_capped_share` should be far below 7.5% (that shows full-length games removed the mismatch);
- `bounds/ceiling_violation_share_mpc` should drop;
- `mpc_data_loss` should drop;
- the held-out scores should improve. The new held-out set is not comparable with earlier curves; its first evaluation is the new baseline.
- Refreshes will be much slower (full 2 s games × 15 iterations).

The full v3 command is in the conversation history. Its key flags are the ones above, plus:
- `--value_ceiling_weight 1 --monotonic_weight 1`;
- `--mpc_time_distribution tmax`;
- `--mpc_game_solver maxmin --mpc_num_samples 32 --mpc_replan_every 5 --mpc_control_hold_steps 10 --mpc_horizon_steps 100 --mpc_dt 0.02`;
- `--mpc_use_network false --mpc_defender_keep_out true --mpc_crop_to_domain true`;
- `--capture_fraction 0.1666667 --mpc_fraction 0.33 --pretrain_geometric_fraction 0.25 --learned_boundary_fraction 0 --geometric_boundary_fraction 0`;
- `--pretrain --pretrain_iters 15000 --counter_end 100000 --num_epochs 165000 --tMax 2.0 --minWith target`;
- the game parameters from section 2;
- `--val_slice interception --val_time_resolution 4`;
- `--mpc_holdout_games 512 --mpc_holdout_eval_epochs 5000`;
- `--mpc_loss_type l1 --mpc_loss_weight 1 --mpc_start_epoch 15000 --mpc_refresh_epochs 1000 --mpc_labels_per_refresh 3000 --mpc_replay_capacity 200000 --mpc_seed 1`;
- the W&B flags.

---

## 8. Open ideas (not implemented; ask before doing any)

1. **Plot fix:** defender at px_d of about 0.3–1.0, colours at about ±0.25.
2. **Drop labels after the event** in full-length games, to spend the label budget on undecided states.
3. **A "fly straight at the target at full acceleration" plan** among the MPC attacker's fixed candidates. Its random plans sit near "do nothing", so it rarely finds a full-strength dash.
4. **Batch evaluation of the network as a controller** against an independent MPC (no network, all rules including the box) over many interception starts, with win rates compared to an MPC-vs-MPC baseline. `simulate_nn.py` only does single games, and its MPC uses the network.
5. **A stronger-MPC held-out set** (for example 256 samples × 10 iterations), so evaluation is closer to the truth than agreement with the training MPC.
6. **One-sided label bound:** V(t, x) ≤ label for t > the label's time (labels used at later times with a hinge loss). Mostly superseded by full-length games.
7. **`--clip_grad`** for the occasional short loss spikes. Low priority.
8. **Seed variation** (`--seed`, `--mpc_seed`) to check whether differences between runs are real rather than luck.
