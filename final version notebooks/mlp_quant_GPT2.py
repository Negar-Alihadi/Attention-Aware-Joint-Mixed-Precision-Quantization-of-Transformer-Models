#!/usr/bin/env python3

# --- stdlib only -------------------------------------------
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime

FORBIDDEN_GPU = 0
DEFAULT_GPU = 1

ARM_ORDER = [
    "fp16",                      # sec 10  -- fp32 control (no quantization)
    "uniform4",                  # sec 11  -- uniform 4-bit GPTQ
    "adaptive",                  # sec 12  -- JAB adaptive allocation (needs the scoring pass)
    "sweep",                     # sec 12b -- budget sweep over JAB / JAB+prop / oracle criteria
    "joint",                     # sec 13  -- joint MLP-aware fine-tuning, uniform 4-bit
    "adaptive_joint",            # sec 13b -- adaptive allocation + joint fine-tuning
    "adaptive_joint_depth4",     # sec 13c -- depth-scaling diagnostic
    "adaptive_joint_kl0",        # sec 13d -- KL ablation (a no-op in the MLP setting)
]


def _csv_ints(s):
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def _csv_numbers(s):
    """
    Comma-separated numbers, keeping integral values as ints.

    This matters and is not cosmetic: the notebook's SWEEP_BITS is `[2.5, 3, 3.5, 4, 4.5, 6]`
    with genuine ints, and section 12b builds its result keys by f-stringing those values
    (`jab_3_gptq`). Parsing everything as float would produce `jab_3.0_gptq` while section 14's
    row lookup -- whose rows come from `sorted(set([2, 3] + SWEEP_BITS))` -- would still ask for
    `jab_3_gptq`, and every affected cell in the comparison table would silently print '--'.
    """
    out = []
    for x in str(s).split(","):
        x = x.strip()
        if not x:
            continue
        v = float(x)
        out.append(int(v) if v.is_integer() else v)
    return out


def _opt_csv_ints(s):
    """`--joint-layers`: a comma-separated layer list, or 'none' for the notebook's None."""
    if s is None or str(s).strip().lower() in ("", "none"):
        return None
    return _csv_ints(s)


def _bool_flag(s):
    v = str(s).strip().lower()
    if v in ("1", "true", "t", "yes", "y"):
        return True
    if v in ("0", "false", "f", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {s!r}")


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_mlp_quant.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="JAB-Hessian adaptive precision allocation for GPT-2's MLP (c_fc + c_proj).")

    # --- run control ---
    g = p.add_argument_group("run control")
    g.add_argument("--model-id", default="gpt2",
                   help="also: gpt2-medium, gpt2-large -- NOT gpt2-xl, see the GROUP_SIZE "
                        "assertion in section 9 (1600 is not a multiple of 128)")
    g.add_argument("--out-dir", default="runs",
                   help="log file and results.json are written here")
    g.add_argument("--seed", type=int, default=42,
                   help="seeds random / numpy / torch / torch.cuda. The Hutchinson estimator "
                        "draws random Rademacher probes, so scoring is not reproducible without "
                        "this")
    g.add_argument("--arms", default="all",
                   help="comma-separated subset of: " + ",".join(ARM_ORDER) + " (or 'all')")
    g.add_argument("--rerun", action="store_true",
                   help="re-run arms already present in results.json instead of skipping them")

    # --- GPU selection ---
    g = p.add_argument_group("GPU selection")
    g.add_argument("--gpu", type=int, default=DEFAULT_GPU,
                   help="physical GPU index; applied as CUDA_VISIBLE_DEVICES before torch is "
                        "imported, after which the card is always cuda:0. Ignored (with a "
                        "notice) if CUDA_VISIBLE_DEVICES is already set in the environment")
    g.add_argument("--allow-gpu0", action="store_true",
                   help=f"permit --gpu {FORBIDDEN_GPU}, which is otherwise refused because it is "
                        f"in heavy use by another user on this machine")

    # --- section 0 constants -----------------------------------------------------------------

    #
    # GPU_DTYPE (fp32) and GPTQ_DTYPE (float64) are deliberately absent: both are documented
    # bug fixes, not tuning knobs. See their comments in the section 0 block below.
    g = p.add_argument_group("section 0: calibration / scoring")
    g.add_argument("--calib-samples", type=int, default=128, metavar="CALIB_N_SAMPLES")
    g.add_argument("--calib-seq-len", type=int, default=512, metavar="CALIB_SEQ_LEN")
    g.add_argument("--hessian-batches", type=int, default=128, metavar="HESSIAN_N_BATCHES",
                   help="batches used for H = 2 X^T X -- ALL of them, not a subset")
    g.add_argument("--hutch-samples", type=int, default=10, metavar="HUTCH_SAMPLES")
    g.add_argument("--jab-batches", type=int, default=2, metavar="JAB_N_BATCHES")
    g.add_argument("--lambda-kl", type=float, default=0.0, metavar="LAMBDA_KL",
                   help="no-op in the MLP setting; a nonzero value is ignored with a warning")
    g.add_argument("--group-size", type=int, default=128, metavar="GROUP_SIZE")
    g.add_argument("--perturb-mode", choices=["rtn", "gptq"], default="rtn",
                   metavar="PERTURB_MODE")
    g.add_argument("--bit-widths", type=_csv_ints, default=[2, 3, 4, 8, 16], metavar="BIT_WIDTHS")

    g = p.add_argument_group("section 0: evaluation")
    g.add_argument("--eval-max-length", type=int, default=1024, metavar="EVAL_MAX_LENGTH",
                   help="GPT-2's full context (n_positions) -- do not shrink this")
    g.add_argument("--eval-stride", type=int, default=512, metavar="EVAL_STRIDE")

    g = p.add_argument_group("section 0: joint fine-tuning")
    g.add_argument("--steps-per-block", type=int, default=200, metavar="JOINT_STEPS_PER_BLOCK")
    g.add_argument("--joint-lr", type=float, default=1e-4, metavar="JOINT_LR")
    g.add_argument("--grad-clip", type=float, default=1.0, metavar="JOINT_GRAD_CLIP")
    g.add_argument("--joint-layers", type=_opt_csv_ints, default=None, metavar="JOINT_LAYERS",
                   help="restrict fine-tuning to these layers; 'none' for every layer")
    g.add_argument("--alpha-w", type=float, default=0.05, metavar="ALPHA_W")
    g.add_argument("--alpha-s", type=float, default=0.02, metavar="ALPHA_S")
    g.add_argument("--score-every", type=int, default=10, metavar="SCORE_EVERY")

    g = p.add_argument_group("section 0: allocation / sweep")
    g.add_argument("--target-avg-bits", type=float, default=4.3, metavar="TARGET_AVG_BITS")
    g.add_argument("--propagation-weight", type=_bool_flag, default=True,
                   metavar="PROPAGATION_WEIGHT")
    g.add_argument("--run-oracle-criterion", type=_bool_flag, default=True,
                   metavar="RUN_ORACLE_CRITERION",
                   help="O(layers*2*bits) model loads inside the sweep arm")
    g.add_argument("--sweep-bits", type=_csv_numbers, default=[2.5, 3, 3.5, 4, 4.5, 6],
                   metavar="SWEEP_BITS")
    g.add_argument("--depth-test-layers", type=int, default=4,
                   metavar="JOINT_LAYERS_DEPTH_TEST",
                   help="13c fine-tunes range(0, N) only")

    g = p.add_argument_group("section 0: diagnostics")
    g.add_argument("--log-loss-components", type=_bool_flag, default=False,
                   metavar="LOG_LOSS_COMPONENTS")
    return p


ARGS = build_parser().parse_args()

_T_START = time.time()


def _resolve_arms(spec):
    """--arms -> the subset of ARM_ORDER to run, always in ARM_ORDER order."""
    if str(spec).strip().lower() in ("", "all"):
        return list(ARM_ORDER)
    wanted = [a.strip() for a in str(spec).split(",") if a.strip()]
    unknown = [a for a in wanted if a not in ARM_ORDER]
    if unknown:
        raise SystemExit(f"unknown arm(s) {unknown}; choose from {ARM_ORDER} or 'all'")
    return [a for a in ARM_ORDER if a in wanted]


_SELECTED_ARMS = _resolve_arms(ARGS.arms)


# --- bootstrap: logging, GPU pinning, results, signals ---------------------------------------

class _Tee:
    """
    Mirror every write to stdout AND the run's log file.
    """
    def __init__(self, stream, fh):
        self._stream, self._fh = stream, fh

    def write(self, data):
        self._stream.write(data)
        self._fh.write(data)
        self._fh.flush()
        return len(data)

    def flush(self):
        self._stream.flush()
        self._fh.flush()

    def isatty(self):
        return self._stream.isatty()


def _assert_torch_not_yet_imported():
    """
    If torch is already in sys.modules by the time we get here, CUDA_VISIBLE_DEVICES is being set
    too late to have any effect and this process would run on whatever card torch picked. That is
    a correctness-of-the-shared-machine bug that produces no error on its own, so it is turned
    into one here.
    """
    late = sorted(m for m in ("torch", "transformers", "datasets") if m in sys.modules)
    if late:
        raise RuntimeError(
            f"{', '.join(late)} was imported before CUDA_VISIBLE_DEVICES was set. GPU pinning is "
            f"now a no-op and this process may initialize contexts on GPUs belonging to other "
            f"users. Move that import below the bootstrap block -- see the ORDERING CONTRACT in "
            f"this file's module docstring.")


def _select_gpu(args):
    """
    Returns (visible_devices_string, inherited_from_env).
    Honours a pre-set CUDA_VISIBLE_DEVICES rather than overriding it -- that is how a scheduler
    or a wrapper script pins jobs, and second-guessing it would make this script unusable under
    one.
    """
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES")
    if inherited is not None and inherited.strip() != "":
        return inherited, True
    if args.gpu == FORBIDDEN_GPU and not args.allow_gpu0:
        raise SystemExit(
            f"refusing --gpu {FORBIDDEN_GPU}: that card is in heavy use by another user on this "
            f"machine (it was at ~89% utilization and 24 GB when this script was written). Pick "
            f"another card with --gpu, or pass --allow-gpu0 if you really mean it.")
    if args.gpu < 0:
        raise SystemExit(f"--gpu must be a non-negative device index, got {args.gpu}")
    return str(args.gpu), False


RESULTS_PATH = None
RESULTS = {"meta": {}, "scoring": None, "arms": {}}


def _jsonable(obj):
    """Make the notebook's result objects JSON-safe: tuple keys, int keys, numpy/torch scalars."""
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, range):
        return list(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "item"):          # 0-d torch/numpy scalar
        try:
            return obj.item()
        except Exception:
            pass
    return str(obj)


def flush_results():
    """Write results.json. Called after every arm, and from the signal handlers."""
    if RESULTS_PATH is None:
        return
    tmp = RESULTS_PATH + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(_jsonable(RESULTS), fh, indent=2, sort_keys=False)
        fh.write("\n")
    os.replace(tmp, RESULTS_PATH)     # atomic, so a crash mid-write cannot corrupt the record


def _should_run(arm):
    """
    Single gate for every arm: --arms selection AND resume-from-results.json, with the reason
    printed. Section 0's RUN_* booleans are bound to this, so the notebook's own
    `if RUN_X:` arm blocks are carried over unchanged.
    """
    if arm not in _SELECTED_ARMS:
        print(f"  [skip] arm '{arm}': not in --arms")
        return False
    if arm in RESULTS.get("arms", {}) and not ARGS.rerun:
        rec = RESULTS["arms"][arm]
        ppl = rec.get("perplexity")
        got = f"perplexity {ppl:.3f}" if isinstance(ppl, (int, float)) else "already recorded"
        print(f"  [skip] arm '{arm}': {got} in results.json -- pass --rerun to redo it")
        return False
    return True


def record_arm(arm, *, perplexity=None, bits=None, notes=None, flip_rates=None,
               unit_flip_rates=None, elapsed_s=None, extra=None):
    """
    Checkpoint one arm into results.json, immediately. A crash in a later arm must not lose the
    ones already finished, and this is what makes the resume path above work.
    """
    rec = {
        "perplexity": perplexity,
        "bits": bits,
        "notes": notes,
        "flip_rates": flip_rates,
        "unit_flip_rates": unit_flip_rates,
        "elapsed_s": None if elapsed_s is None else round(elapsed_s, 2),
        "gpu": {"cuda_visible_devices": _VISIBLE_DEVICES,
                "inherited": _GPU_INHERITED,
                "name": RESULTS.get("meta", {}).get("gpu_name")},
        "finished": datetime.now().isoformat(timespec="seconds"),
    }
    if extra:
        rec.update(extra)
    RESULTS.setdefault("arms", {})[arm] = rec
    flush_results()
    if elapsed_s is not None:
        print(f"  [checkpoint] arm '{arm}' written to results.json "
              f"({elapsed_s / 60:.1f} min)")


def section(title):
    """Echo a notebook section header into the log, so a log file can be read against the
    notebook cell-for-cell."""
    print("\n" + "=" * 96)
    print(title)
    print("=" * 96)


def _install_signal_handlers():
    def _handler(signum, frame):
        name = signal.Signals(signum).name
        print(f"\n\n!! {name} received -- flushing results.json and exiting. "
              f"Completed arms are preserved; re-running without --rerun resumes from here.",
              flush=True)
        flush_results()
        sys.exit(130 if signum == signal.SIGINT else 143)

    signal.signal(signal.SIGINT, _handler)
    signal.signal(signal.SIGTERM, _handler)


_assert_torch_not_yet_imported()
_VISIBLE_DEVICES, _GPU_INHERITED = _select_gpu(ARGS)
os.environ["CUDA_VISIBLE_DEVICES"] = _VISIBLE_DEVICES

os.makedirs(ARGS.out_dir, exist_ok=True)
_RUN_STAMP = datetime.now().strftime("%Y%m%d-%H%M%S")
_LOG_PATH = os.path.join(ARGS.out_dir, f"run_{_RUN_STAMP}.log")
_LOG_FH = open(_LOG_PATH, "a", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.stdout, _LOG_FH)
sys.stderr = _Tee(sys.stderr, _LOG_FH)

RESULTS_PATH = os.path.join(ARGS.out_dir, "results.json")
if os.path.exists(RESULTS_PATH):
    with open(RESULTS_PATH, encoding="utf-8") as fh:
        RESULTS = json.load(fh)
    RESULTS.setdefault("meta", {})
    RESULTS.setdefault("scoring", None)
    RESULTS.setdefault("arms", {})
_install_signal_handlers()

print("=" * 96)
print(f"run_mlp_quant.py -- JAB-Hessian adaptive precision allocation, GPT-2 MLP "
      f"(mlp.c_fc + mlp.c_proj)")
print(f"started {datetime.now().isoformat(timespec='seconds')}")
print(f"log      {_LOG_PATH}")
print(f"results  {RESULTS_PATH}")
print(f"argv     {' '.join(sys.argv)}")
if _GPU_INHERITED:
    print(f"CUDA_VISIBLE_DEVICES={_VISIBLE_DEVICES} was INHERITED from the environment; "
          f"--gpu {ARGS.gpu} ignored.")
else:
    print(f"CUDA_VISIBLE_DEVICES={_VISIBLE_DEVICES} (from --gpu); after masking the selected "
          f"card is cuda:0.")
print("=" * 96)
print(f"arms     {', '.join(_SELECTED_ARMS)}"
      f"{'  (--rerun: ignoring any existing results.json entries)' if ARGS.rerun else ''}")
print("=" * 96)

# =============================================================================================
# torch imports -- everything below this line may import torch. See the ORDERING CONTRACT in the
# module docstring for why nothing above it does.
# =============================================================================================
import gc
import inspect
import itertools
import math
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

# --- Blackwell: TF32 off, before any CUDA tensor is created ----------------------------------
# These cards use TF32 for fp32 matmuls by default, and TF32's 10-bit mantissa is FEWER mantissa
# bits than the fp16 this notebook's own section 0 comments blame for non-monotone
# perplexity-vs-bit-width -- it would silently degrade every Hessian accumulation
# (collect_hessian_via_hook), every compute_mlp forward and every teacher target.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

# --- build / device report (printed, not asserted) -------------------------------------------
# Blackwell needs a cu128-or-newer PyTorch build; if sm_120 is missing from the arch list below,
# that is why the run failed, and it will be in this log.
print("\n--- environment ---")
print(f"  torch                : {torch.__version__}")
print(f"  torch.version.cuda   : {torch.version.cuda}")
print(f"  arch list            : {torch.cuda.get_arch_list()}")
if not torch.cuda.is_available():
    raise SystemExit(
        f"CUDA is not available with CUDA_VISIBLE_DEVICES={_VISIBLE_DEVICES}. The requested GPU "
        f"is unavailable; this script does not fall back to another card (GPU {FORBIDDEN_GPU} "
        f"belongs to another user). Check the index and that the driver is up.")
if torch.cuda.device_count() < 1:
    raise SystemExit(
        f"no visible CUDA device after masking to CUDA_VISIBLE_DEVICES={_VISIBLE_DEVICES} -- the "
        f"requested index almost certainly does not exist on this machine.")

DEVICE = "cuda:0"     # after CUDA_VISIBLE_DEVICES masking the selected card is always index 0
_free_b, _total_b = torch.cuda.mem_get_info(0)
print(f"  device name          : {torch.cuda.get_device_name(0)}")
print(f"  physical index       : {_VISIBLE_DEVICES}"
      f"{' (inherited from environment)' if _GPU_INHERITED else ''}  ->  {DEVICE}")
print(f"  compute capability   : sm_{''.join(str(x) for x in torch.cuda.get_device_capability(0))}")
print(f"  memory               : {_free_b / 1e9:.1f} GB free / {_total_b / 1e9:.1f} GB total")

# --- seeding ---------------------------------------------------------------------------------
# The Hutchinson estimator draws random Rademacher probes, so pass_score is not reproducible
# without this, and arm-to-arm comparisons would not be either.
random.seed(ARGS.seed)
np.random.seed(ARGS.seed)
torch.manual_seed(ARGS.seed)
torch.cuda.manual_seed_all(ARGS.seed)
print(f"  seed                 : {ARGS.seed} (random / numpy / torch / torch.cuda)")
print("--- environment ---\n")


# =============================================================================================
# Section 0 constants, verbatim from the notebook except that each is now overridable from the
# command line. No default differs from the notebook's value.
# =============================================================================================
MODEL_ID = ARGS.model_id    # also: "gpt2-medium", "gpt2-large" -- NOT "gpt2-xl", see the
                             # GROUP_SIZE assertion in section 9 (1600 is not a multiple of 128)

# The working reference notebook loads plain fp32 regardless of device, and does the entire GPTQ
# solve in float64 (see GPTQ_DTYPE below). An earlier version of this notebook inherited fp16
# compute and float32 GPTQ from the Mistral streaming notebook, where both were real memory/speed
# concessions for a 14.5 GB model on a free T4. GPT-2 (124M-774M) has none of those constraints --
# fp16 here bought nothing and cost real precision: it round-off the activations feeding every
# Hessian AND the quantized weight write-back, and float32 GPTQ divides by a Hessian-inverse entry
# 768 times sequentially per column loop, compounding fp32 rounding into the same degenerate
# behavior as round-to-nearest (non-monotone perplexity vs. bit-width, worse than uniform 4-bit
# GPTQ should ever be). Both are fixed below; do not reintroduce either as a "speed" shortcut.
#
# This matters MORE here than in the QKV notebook, not less: c_proj's Hessian is (3072, 3072) and
# its GPTQ column loop is 3072 sequential divisions deep, four times the QKV notebook's 768.
#
# NOT exposed as a command-line flag, for exactly that reason. Same for GPTQ_DTYPE below.
GPU_DTYPE = torch.float32

# --- calibration / scoring ---
# 128 batches of 512 tokens for every Hessian, matching the reference notebook's
# build_calibration_batches(n_samples=128, seq_len=512) and its unsliced use of the full list in
# collect_hessian_via_hook -- NOT a truncated subset. HESSIAN_N_BATCHES == CALIB_N_SAMPLES so the
# `calib_ids[:n_hessian_batches]` slice pass_quantize_eval takes is a no-op, using everything.
CALIB_N_SAMPLES   = ARGS.calib_samples
CALIB_SEQ_LEN     = ARGS.calib_seq_len
HESSIAN_N_BATCHES = ARGS.hessian_batches   # batches used for H = 2 X^T X -- ALL of them
HUTCH_SAMPLES     = ARGS.hutch_samples     # Hutchinson probes per (layer, batch)
JAB_N_BATCHES     = ARGS.jab_batches       # batches averaged into each layer's trace
# The MLP has no attention map, so there is no distribution to take a KL against: `kl_loss` and
# the `lambda_kl` term of `attention_loss` have NO analogue in this notebook and are gone (see
# `mlp_loss`, section 2). LAMBDA_KL is kept at 0.0 -- not deleted -- only so that pass_score /
# pass_quantize_eval can keep the QKV notebook's exact signatures; a nonzero value is ignored with
# a one-time warning. Deliberately NOT replaced with a substitute KL (e.g. on final logits): this
# experiment isolates the effect of changing the target module, so the objective must not change
# at the same time.
LAMBDA_KL         = ARGS.lambda_kl         # no-op in the MLP setting -- see above
GROUP_SIZE        = ARGS.group_size        # must divide hidden_size AND intermediate_size (sec 9)

# --- GPTQ compute choices ---
# float64, unconditionally -- this is what the working reference notebook's gptq_quantize_layer
# does (clone, Hessian damping, torch.linalg.inv(H), and the whole sequential column loop all run
# in float64; there is no float32 code path in it at all). GPT-2's MLP Hessians are at most
# 5120x5120 (gpt2-large's c_proj) -- fp64 is still cheap. See the GPU_DTYPE comment above for
# what float32 broke. NOT exposed as a command-line flag.
GPTQ_DTYPE   = torch.float64
PERTURB_MODE = ARGS.perturb_mode   # "rtn" | "gptq" -- the allocator's sensitivity-table heuristic,
                                    # not the GPTQ solve itself; kept at fp32 internally (see
                                    # grid_perturbation in section 1) since it's a fast
                                    # approximation feeding bit allocation, not the quantizer --
                                    # both notebooks were confirmed to land on the same
                                    # assignment, so this was not implicated in the regression.

# --- evaluation ---
# Full WikiText-2 test set, sliding window (max_length=1024, stride=512) -- matches the working
# reference notebook's evaluate_perplexity exactly, so every scored token has >=512 tokens of real
# context. An earlier version of this notebook capped evaluation at a small N_EVAL_WINDOWS subset;
# that subset's ppl swings by roughly +/-1 between runs, which is close to the entire gap between
# arms -- not nearly enough resolution to tell uniform-4-bit, adaptive, and joint apart.
EVAL_MAX_LENGTH = ARGS.eval_max_length   # GPT-2's full context -- do not shrink this
EVAL_STRIDE     = ARGS.eval_stride

# --- which passes to run ---
# In the notebook these were three standalone booleans. Here each is `_should_run(arm)`, which
# folds in BOTH --arms selection and resume-from-results.json, and prints why an arm is being
# skipped. The arm blocks in sections 10-13d are otherwise unchanged from the notebook.
RUN_FP16_BASELINE  = _should_run("fp16")            # unquantized control; also an end-to-end
                                                     # check on the pipeline code
RUN_UNIFORM4       = _should_run("uniform4")
RUN_ADAPTIVE       = _should_run("adaptive")
RUN_SWEEP          = _should_run("sweep")
RUN_JOINT_FINETUNE = _should_run("joint")
RUN_ADAPTIVE_JOINT = _should_run("adaptive_joint")  # adaptive bit allocation + MLP-aware
                                                     # fine-tuning, combined

# --- joint fine-tuning ---
JOINT_STEPS_PER_BLOCK = ARGS.steps_per_block   # was 8: GPTQ output sits exactly on grid points,
                                                # so 8 steps at a fixed absolute lr never crossed
                                                # a rounding boundary
JOINT_LR              = ARGS.joint_lr          # unused by the FT block now (see ALPHA_W/ALPHA_S)
                                                # -- kept as the default for any other caller of
                                                # pass_quantize_eval
JOINT_GRAD_CLIP       = ARGS.grad_clip
JOINT_LAYERS          = ARGS.joint_layers      # e.g. 0,2,4,6,8,10; others still get the GPTQ
                                                # warm start
ALPHA_W               = ARGS.alpha_w           # lr_w = ALPHA_W * scale_flat.mean();
                                                # cosine-decayed over JOINT_STEPS_PER_BLOCK steps
                                                # (was 0.2 -- with the grouped scale, 0.2 let
                                                # latent weights travel ~40 grid steps)
ALPHA_S               = ARGS.alpha_s           # lr_s = ALPHA_S * scale_flat.mean() -- LSQ scale lr
SCORE_EVERY           = ARGS.score_every       # score the held-out batch every N FT steps
PROPAGATION_WEIGHT    = ARGS.propagation_weight  # multiply per-layer sensitivity by
                                                  # (n_layers - layer_idx)
RUN_ORACLE_CRITERION  = ARGS.run_oracle_criterion  # gate measure_end_to_end_sensitivity -- now
                                                    # O(layers*2*bits) model loads, twice the QKV
                                                    # notebook's, since c_fc and c_proj are
                                                    # separate allocator units scored separately
SWEEP_BITS            = ARGS.sweep_bits

# --- diagnostics ---
LOG_LOSS_COMPONENTS   = ARGS.log_loss_components   # print the MSE term inside mlp_loss
RUN_DEPTH_SCALING_TEST = _should_run("adaptive_joint_depth4")   # 13c
JOINT_LAYERS_DEPTH_TEST = range(0, ARGS.depth_test_layers)
RUN_KL_ABLATION_TEST  = _should_run("adaptive_joint_kl0")            # 13d: a NO-OP here (the loss
                                                                      # is already MSE-only) --
                                                                      # kept so section numbering
                                                                      # matches the QKV notebook

BIT_WIDTHS_ARG  = ARGS.bit_widths          # consumed by section 4's BIT_WIDTHS
TARGET_AVG_BITS = ARGS.target_avg_bits     # consumed by section 12


def free(*objs):
    """
    Run a collection and release cached VRAM.

    CAREFUL about what this does and does not do. The arguments are only there for readability:
    deleting them inside this function drops *this* frame's references, NOT the caller's bindings,
    so `free(x)` alone never releases `x`.
    """
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def vram(tag=""):
    if torch.cuda.is_available():
        a = torch.cuda.memory_allocated() / 1e9
        p = torch.cuda.max_memory_allocated() / 1e9
        print(f"    [vram{' ' + tag if tag else ''}: {a:.2f} GB now, {p:.2f} GB peak this pass]")


def reset_vram_peak():
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


_LAMBDA_KL_WARNED = [False]


def warn_if_lambda_kl(lambda_kl, where=""):
    """
    One-time warning for a nonzero `lambda_kl`.

    `lambda_kl` survives in pass_score / pass_quantize_eval's signatures purely so callers written
    against the QKV notebook keep working; there is nothing in the MLP path for it to weight (see
    the LAMBDA_KL comment above). Silently ignoring it would be worse than either honoring it or
    erroring, so it is ignored LOUDLY, once.
    """
    if lambda_kl and not _LAMBDA_KL_WARNED[0]:
        _LAMBDA_KL_WARNED[0] = True
        print(f"WARNING: lambda_kl={lambda_kl} is IGNORED{' by ' + where if where else ''}. The "
              f"MLP produces no post-softmax attention distribution, so this notebook's joint "
              f"loss is MSE-only on the MLP output. Set LAMBDA_KL=0.0 to silence this.")


print("Device:", DEVICE)
print("Model:", MODEL_ID, "| compute dtype:", GPU_DTYPE, "| GPTQ dtype:", GPTQ_DTYPE)
print("Quantization target: layer.mlp.c_fc + layer.mlp.c_proj (attention stays fp32)")


# The section-12 scoring pass feeds the adaptive allocation, the sweep, and 13b/13c/13d -- those
# five arms are the ones that need it. Run (or reload from results.json's "scoring" cache) if ANY
# of them will actually execute; note that RUN_ADAPTIVE may be False because that one arm is
# already recorded while 13b still needs the assignment.
NEED_SCORING = (RUN_ADAPTIVE or RUN_SWEEP or RUN_ADAPTIVE_JOINT
                or RUN_DEPTH_SCALING_TEST or RUN_KL_ABLATION_TEST)


# ============================================================================================
# 0. GPT-2 architecture adapter
# ============================================================================================
class BlockConfig:
    """
    Shape bookkeeping pulled out of a HF GPT2Config.

    Renamed from the QKV notebook's `AttnConfig` because it now carries BOTH the attention shapes
    and the MLP shapes. The attention fields are deliberately kept even though nothing in this
    notebook quantizes attention: section 7's validation still uses them to assert that the
    attention path -- which this notebook leaves in fp32 and delegates wholesale to the real
    `layer.attn` -- is the plain-MHA, no-GQA, no-RoPE, no-sliding-window case it is assumed to be.

    MLP shapes:
      c_fc   : (hidden_size, intermediate_size)    input  ln_2(h)
      c_proj : (intermediate_size, hidden_size)    input  gelu(c_fc(ln_2(h)))
    Both are `Conv1D`, i.e. already the (d_in, d_out) "x @ W" convention GPTQ wants.
    """

    def __init__(self, config):
        self.hidden_size   = config.n_embd
        self.num_heads     = config.n_head
        self.num_kv_heads  = config.n_head          # no GQA in GPT-2
        self.head_dim      = config.n_embd // config.n_head
        self.q_out         = self.num_heads * self.head_dim       # == hidden_size
        self.kv_out        = self.num_kv_heads * self.head_dim    # == hidden_size (no GQA shrink)
        self.n_rep         = self.num_heads // self.num_kv_heads  # always 1
        self.scaling       = self.head_dim ** -0.5
        self.n_layers      = config.n_layer

        # Attention numels -- unused by this notebook's quantization path, kept so the two
        # notebooks' configs stay comparable and so section 14b can print the QKV-equivalent
        # budget without hard-coding shapes.
        self.q_numel  = self.hidden_size * self.q_out
        self.kv_numel = self.hidden_size * self.kv_out

        # HF leaves n_inner unset (None) for every public GPT-2 checkpoint and GPT2MLP then uses
        # 4 * n_embd; read it anyway so a config that DOES set it is honored rather than silently
        # overridden with the wrong width.
        self.intermediate_size = getattr(config, "n_inner", None) or 4 * config.n_embd
        self.fc_numel   = self.hidden_size * self.intermediate_size
        self.proj_numel = self.intermediate_size * self.hidden_size

    @property
    def qkv_numel(self):
        return self.q_numel + 2 * self.kv_numel

    @property
    def mlp_numel(self):
        return self.fc_numel + self.proj_numel

    def __repr__(self):
        return (f"BlockConfig(hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
                f"layers={self.n_layers}, heads={self.num_heads}, head_dim={self.head_dim})")


def build_attn_mask(seq_len, device):
    """
    Plain causal mask -- GPT-2 has no sliding window.

    Nothing in this notebook's MLP path needs a mask (the MLP is position-wise), so unlike the QKV
    notebook this is NOT threaded through the loss. It survives only for section 7's check that
    the attention path -- delegated to the real `layer.attn`, which builds its own causal bias
    internally -- is the plain causal case.
    """
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()


# ============================================================================================
# 1. GPTQ core
# ============================================================================================
def _quantize_to_grid(w_col, scale, bits):
    """
    Symmetric per-output-row fake quantization of a single input-column (shape: d_out) using a
    fixed per-row scale computed up front from the original weight statistics.
    """
    qmax = 2 ** (bits - 1) - 1
    return torch.clamp(torch.round(w_col / scale), -qmax, qmax) * scale


@torch.no_grad()
def gptq_quantize_layer(weight_in_out, H, bits=4, damp_percent=0.01, group_size=None,
                        act_order=True, return_scale=False, return_compact=False, work_dtype=None):
    """
    Quantizes a weight matrix in the (d_in, d_out) "x @ W" convention using GPTQ.
    UNCHANGED from the Mistral notebook -- this function is architecture-agnostic; it only ever
    sees a plain (d_in, d_out) matrix and a (d_in, d_in) Hessian.

    weight_in_out: (d_in, d_out) float tensor
    H: (d_in, d_in) Hessian
    damp_percent: Hessian damping for numerical stability (GPTQ default ~0.01)
    group_size: separate per-row scale per contiguous group of `group_size` input columns
    act_order: quantize columns in order of decreasing Hessian diagonal
    return_scale: ALSO return the exact per-(output channel, group) scale used, in the ORIGINAL
        (unpermuted) column order and the same (d_in, d_out) orientation -- so a caller can later
        fake-quantize the SAME weight onto the SAME grid (needed for the STE warm start).
    work_dtype: fp64 is GPTQ-canonical; GPT-2's Hessians are small enough (<=1280x1280) that this
        costs nothing, unlike the Mistral notebook's T4-driven fp32 compromise.

    Returns the fake-quantized weight (also written into weight_in_out in place), or
    (W_final, scale) if return_scale.
    """
    work_dtype = work_dtype or GPTQ_DTYPE
    device = weight_in_out.device
    W = weight_in_out.detach().clone().to(work_dtype).T.contiguous()  # (d_out, d_in)
    d_out, d_in = W.shape

    H = H.clone().to(work_dtype)
    mean_diag = H.diagonal().mean()
    H += damp_percent * mean_diag * torch.eye(d_in, dtype=work_dtype, device=device)

    if act_order:
        perm = torch.argsort(torch.diag(H), descending=True)
        invperm = torch.argsort(perm)
        W = W[:, perm]
        H = H[perm][:, perm]
    else:
        perm = invperm = torch.arange(d_in, device=device)

    H_inv = torch.linalg.inv(H)
    free(H)

    # Per-(row, group) scale, from the original weights in the permuted column order, before any
    # quantization.
    qmax = 2 ** (bits - 1) - 1
    gs = group_size if group_size is not None else d_in
    n_groups = (d_in + gs - 1) // gs
    scale = torch.zeros(d_out, n_groups, dtype=work_dtype, device=device)
    for g in range(n_groups):
        start, end = g * gs, min((g + 1) * gs, d_in)
        scale[:, g] = (W[:, start:end].abs().amax(dim=1) / qmax).clamp(min=1e-8)

    for i in range(d_in):
        row_scale = scale[:, i // gs]
        w_col = W[:, i]
        q_col = _quantize_to_grid(w_col, row_scale, bits)
        err = (w_col - q_col) / H_inv[i, i]
        if i + 1 < d_in:
            W[:, i + 1:] -= torch.outer(err, H_inv[i, i + 1:])
        W[:, i] = q_col
    free(H_inv)

    if act_order:
        W = W[:, invperm]

    W_final = W.T.contiguous().to(weight_in_out.dtype)  # back to (d_in, d_out)
    weight_in_out.copy_(W_final)

    if not return_scale:
        return W_final

    group_idx = torch.arange(d_in, device=device) // gs
    scale_cols = scale[:, group_idx]                                     # (d_out, d_in), permuted
    scale_cols = scale_cols[:, invperm] if act_order else scale_cols
    scale_expanded = scale_cols.T.contiguous().to(weight_in_out.dtype)

    if not return_compact:
        return W_final, scale_expanded

    # g_idx maps each ORIGINAL column to its group; groups are contiguous in PERMUTED order, so
    # this must go through invperm (original column c sits at permuted position invperm[c]).
    g_idx = (invperm // gs) if act_order else (torch.arange(d_in, device=device) // gs)
    return W_final, scale_expanded, scale.to(weight_in_out.dtype), g_idx


@torch.no_grad()
def gptq_quantize_conv1d(conv1d, H, bits=4, damp_percent=0.01, group_size=None,
                         act_order=True, return_scale=False, return_compact=False):
    """
    GPTQ for a HF `Conv1D` module (used by GPT-2's mlp.c_fc / mlp.c_proj / c_attn), whose weight is
    ALREADY `(d_in, d_out)` -- the forward is `x @ weight + bias`, unlike `nn.Linear`'s
    `(d_out, d_in)` / `x @ weight.T + bias`. This is the single most likely place to introduce a
    silent bug when porting GPTQ code between the two module conventions: get the orientation
    backwards and GPTQ quantizes along output channels instead of input channels, and nothing
    raises -- the shapes are square-ish enough (768x2304, 1024x4096, ...) that a transpose error
    would not even reliably crash, just silently degrade quantization quality.

    SMALL SAFETY IMPROVEMENT OVER THE QKV NOTEBOOK: `mlp.c_proj` is `(3072, 768)` -- NON-square,
    unlike the fused `(768, 2304)` c_attn (whose 2304 is a multiple of 768, so even a transposed
    weight still matched a 768-sized Hessian on one axis). For c_proj a transpose error makes the
    assertion below fire outright (768 != 3072) instead of silently degrading, so this notebook's
    second matrix catches exactly the bug class the QKV notebook could only warn about. `c_fc` is
    `(768, 3072)`, equally non-square in the other direction.

    Guarded here with an explicit shape assertion (`gptq_quantize_layer` expects `weight_in_out`'s
    ROW count to equal `H`'s size, i.e. the input dimension) and re-checked in section 7's
    validation.

    Returns W_io (d_in, d_out) [, scale (d_in, d_out)].
    """
    d_in = H.shape[0]
    assert conv1d.weight.shape[0] == d_in, (
        f"Conv1D weight shape {tuple(conv1d.weight.shape)} does not put the input dimension "
        f"({d_in}) first -- if this fires, the (d_in, d_out) orientation assumption below is "
        f"wrong and gptq_quantize_layer would quantize along the wrong axis")
    # NO premature dtype cast here: gptq_quantize_layer upcasts to GPTQ_DTYPE (float64) itself.
    # An earlier version of this function cast to float32 HERE, before that upcast -- harmless in
    # isolation, but combined with a float16 model load (see GPU_DTYPE in section 0) it meant the
    # weight had already been rounded to fp16, then fp32, before ever reaching the float64 solve,
    # so the float64 work below was solving with already-corrupted inputs.
    W_io = conv1d.weight.data.clone()   # (d_in, d_out), NO transpose
    out = gptq_quantize_layer(W_io, H, bits=bits, damp_percent=damp_percent,
                              group_size=group_size, act_order=act_order,
                              return_scale=return_scale, return_compact=return_compact)
    if return_compact:
        W_q, scale, scale_compact, g_idx = out
        conv1d.weight.data.copy_(W_q.to(conv1d.weight.dtype))
        return W_q, scale, scale_compact, g_idx
    W_q, scale = out if return_scale else (out, None)
    conv1d.weight.data.copy_(W_q.to(conv1d.weight.dtype))
    return (W_q, scale) if return_scale else W_q


@torch.no_grad()
def quantize_mlp(reader, layer, bits, calib_ids, H_fc=None, group_size=None, damp_percent=0.01,
                 return_scales=False, return_compact=False, device=None):
    """
    GPTQ for GPT-2's MLP: `c_fc` first, then `c_proj`. Replaces the QKV notebook's
    `quantize_qkv(c_attn, H, ...)`.

    THE HESSIAN ORDERING IS THE WHOLE POINT OF THIS FUNCTION'S SIGNATURE. `c_proj`'s input is
    `gelu(c_fc(ln_2(h)))` -- it is downstream of `c_fc` inside the same block. So `H_proj` cannot
    be a precomputed argument the way `H` was in `quantize_qkv`: collecting it before `c_fc` is
    quantized would describe activations `c_proj` will never see. Instead this takes `reader` and
    `calib_ids` and does:

        quantize c_fc  ->  collect H_proj from the live model  ->  quantize c_proj

    which is the in-block analogue of the across-layers sequential-GPTQ property that
    `collect_hessian_via_hook` (section 5) already gets for free by hooking the live model.

    bits: an int applied to both matrices, or a `(bits_fc, bits_proj)` pair -- the allocator
        (section 4) treats c_fc and c_proj as separate units and routinely assigns them different
        widths.
    H_fc: optional precomputed (hidden, hidden) Hessian for c_fc. Unlike H_proj this one CAN be
        passed in, because c_fc's input does not depend on either MLP matrix; collected here if
        omitted.
    calib_ids: batches for the Hessian hook -- already sliced by the caller.

    Returns None if not return_scales, else the per-matrix GPTQ outputs as
    ((W_fc, scale_fc[, scale_compact_fc, g_idx_fc]), (W_proj, ...)).

    Note this calls `mlp_of` / `collect_hessian_via_hook`, both defined in section 5 -- resolved at
    call time, exactly as the QKV notebook's section-1 helpers were.
    """
    device = DEVICE if device is None else device
    group_size = GROUP_SIZE if group_size is None else group_size
    bits_fc, bits_proj = (bits, bits) if isinstance(bits, int) else tuple(bits)
    c_fc, c_proj = mlp_of(layer)

    # --- c_fc: input is ln_2(h), independent of both MLP matrices ---
    if H_fc is None:
        H_fc = collect_hessian_via_hook(reader, c_fc, calib_ids, device)
    out_fc = gptq_quantize_conv1d(c_fc, H_fc, bits=bits_fc, damp_percent=damp_percent,
                                  group_size=group_size, act_order=True,
                                  return_scale=return_scales, return_compact=return_compact)
    free(H_fc)

    # --- c_proj: collected ONLY NOW, so the hook sees gelu(c_fc_QUANTIZED(ln_2(h))) ---
    H_proj = collect_hessian_via_hook(reader, c_proj, calib_ids, device)
    assert H_proj.shape[0] == c_proj.weight.shape[0] == layer.mlp.c_fc.weight.shape[1], (
        f"H_proj is {tuple(H_proj.shape)} but c_proj expects an input dimension of "
        f"{c_proj.weight.shape[0]} -- the c_proj hook is wired to the wrong tensor")
    out_proj = gptq_quantize_conv1d(c_proj, H_proj, bits=bits_proj, damp_percent=damp_percent,
                                    group_size=group_size, act_order=True,
                                    return_scale=return_scales, return_compact=return_compact)
    free(H_proj)

    if not return_scales:
        return None      # already applied in place; caller discards this when want_scales is False
    return out_fc, out_proj


@torch.no_grad()
def grid_perturbation(W_io, bits, group_size=None, H=None, mode=None):
    """
    HAWQ-V2's `||Q(W) - W||_F^2`, measured in weight space on GPTQ's own per-(output channel,
    group) symmetric grid. Unchanged from the Mistral notebook.

    mode="rtn"  : round-to-nearest on that grid. Milliseconds.
    mode="gptq" : the full GPTQ solve (requires H). Reproduces the GPT-2 notebook exactly, much
                  slower -- GPT-2's small matrices make this far more affordable than on Mistral,
                  but "rtn" stays the default for parity.
    """
    mode = PERTURB_MODE if mode is None else mode
    if mode == "gptq":
        if H is None:
            raise ValueError('mode="gptq" needs the Hessian H')
        W_q = gptq_quantize_layer(W_io.clone(), H, bits=bits, group_size=group_size, act_order=True)
        out = (W_q - W_io).pow(2).sum().item()
        free(W_q)
        return out

    W = W_io.T                                   # (d_out, d_in)
    d_out, d_in = W.shape
    gs = group_size if group_size is not None else d_in
    qmax = 2 ** (bits - 1) - 1
    total = 0.0
    for start in range(0, d_in, gs):
        blk = W[:, start:min(start + gs, d_in)].to(torch.float32)
        scale = (blk.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
        blk_q = torch.clamp(torch.round(blk / scale), -qmax, qmax) * scale
        total += (blk_q - blk).pow(2).sum().item()
    return total


# ============================================================================================
# 2. MLP-aware joint loss
# ============================================================================================
# GPT-2's activation is `gelu_new` (the tanh-based Hendrycks approximation with the 0.044715 cubic
# term), NOT `F.gelu` (the exact erf form) and NOT the plain `tanh` approximation. HF materializes
# it as `NewGELUActivation` and every GPT2MLP instance holds one as `self.act`.
#
# THIS IS THE HIGHEST-RISK LINE IN THE PORT. `F.gelu` and `gelu_new` agree to ~1e-3 relative --
# close enough that a perplexity run still "works", far enough that every teacher target, every
# Hutchinson trace and every STE gradient in this notebook would be computed against a model the
# real one is not: exactly the silent-degradation bug class this notebook's other comments warn
# about. `compute_mlp` therefore takes an `act` argument, every real call site passes the ACTUAL
# `layer.mlp.act` module off the model being quantized, and section 7 asserts the module-level
# default below matches it on random input.
from transformers.activations import NewGELUActivation

GPT2_ACT = NewGELUActivation()


def reshape_weights(w_flat, cfg):
    """
    Flat vector -> (W_fc, W_proj), both in the (d_in, d_out) "x @ W" convention `Conv1D` already
    stores them in.

      W_fc   : (hidden_size, intermediate_size)
      W_proj : (intermediate_size, hidden_size)

    Note the two halves are the same LENGTH (hidden*intermediate either way) but NOT the same
    shape -- so, unlike the QKV notebook's three equal (hidden, hidden) thirds, a swapped reshape
    here raises instead of silently producing a transposed matrix.
    """
    n = cfg.fc_numel
    return (w_flat[0:n].reshape(cfg.hidden_size, cfg.intermediate_size),
            w_flat[n:n + cfg.proj_numel].reshape(cfg.intermediate_size, cfg.hidden_size))


def flatten_weights(W_fc, W_proj):
    """Inverse of reshape_weights (both in (d_in, d_out) orientation)."""
    return torch.cat([W_fc.reshape(-1), W_proj.reshape(-1)])


def compute_mlp(W_fc, W_proj, X, cfg, b_fc=None, b_proj=None, act=None):
    """
    GPT-2's MLP output, matching HF's `GPT2MLP.forward` exactly:

        c_proj(act(c_fc(x)))   ==   act(X @ W_fc + b_fc) @ W_proj + b_proj

    W_fc: (hidden, intermediate); W_proj: (intermediate, hidden) -- both `Conv1D`-native
        (d_in, d_out), so there is NO transpose anywhere in this function.
    b_fc: (intermediate,); b_proj: (hidden,). GPT-2's MLP biases. Same rule as the QKV notebook's
        c_attn bias: snapshotted once per layer in fp32, NEVER a GPTQ or STE target, and passed
        identically into the teacher's and the student's call.
    act: the activation module. Pass the real `layer.mlp.act` whenever a real model is in scope;
        falls back to the module-level `GPT2_ACT` (see the comment at the top of this cell).
    X: (batch, seq_len, hidden) -- the tensor entering c_fc, i.e. ln_2(h).

    Returns (batch, seq_len, hidden): the MLP's contribution to the residual stream, pre-add.

    `GPT2MLP.forward` also applies `self.dropout` after c_proj. Every model in this notebook is in
    .eval() and every forward is under no_grad or an autograd-only STE graph, so that dropout is
    the identity -- it is deliberately omitted here rather than reproduced, and section 7 checks
    this hand-written forward against the real module to prove the omission is exact.
    """
    act = GPT2_ACT if act is None else act
    h = X @ W_fc
    if b_fc is not None:
        h = h + b_fc
    h = act(h)
    out = h @ W_proj
    if b_proj is not None:
        out = out + b_proj
    return out


def mlp_loss(w_flat, X, target_out, cfg, b_fc=None, b_proj=None, act=None, log_components=False):
    """
    L = ||MLP(X) - MLP_hat(X)||^2 -- the joint loss for this notebook.

    MSE-ONLY, AND DELIBERATELY SO. The QKV notebook's `attention_loss` added
    `lambda_kl * KL(attention maps)`; `kl_loss` operates on a post-softmax distribution over
    positions, and the MLP produces no such distribution, so that term has no analogue here. It is
    dropped rather than replaced: substituting a KL on some other quantity (final logits, say)
    would change the objective at the same time as the target module, and this experiment exists
    to isolate the effect of the target module alone. See LAMBDA_KL / warn_if_lambda_kl,
    section 0.

    Both matrices are covered by this one loss -- w_flat is the concatenated [W_fc | W_proj], and
    the reconstruction target is the output of the whole two-matrix MLP, so the Hutchinson trace
    taken against it (section 8) is a JOINT trace over both.

    log_components: diagnostic print only; does not change what is returned or optimized. Kept for
        structural parity with `attention_loss`, which used it to expose the MSE/KL imbalance --
        there is only one component left to print.
    """
    W_fc, W_proj = reshape_weights(w_flat, cfg)
    out_hat = compute_mlp(W_fc, W_proj, X, cfg, b_fc=b_fc, b_proj=b_proj, act=act)
    loss = F.mse_loss(out_hat, target_out)
    if log_components:
        print(f"    [mlp_loss] mse={loss.item():.6f} (no KL term in the MLP setting)")
    return loss


# ============================================================================================
# 3. Hutchinson trace estimator
# ============================================================================================
def hessian_vector_product(loss_fn, params, vector, retain_graph=True):
    # create_graph=True keeps the graph alive for the second grad -- do not release it
    grad = autograd.grad(loss_fn(params), params, create_graph=True, retain_graph=True)[0]
    hvp = autograd.grad(grad, params, grad_outputs=vector, retain_graph=True)[0]
    del grad
    return hvp


def hutchinson_trace_estimator(loss_fn, params, samples=50):
    # trace(H) ~= (1/n) * sum(v_i^T H v_i)
    if not params.requires_grad:
        params.requires_grad_(True)

    estimated_trace = 0.0
    for _ in range(samples):
        # Rademacher vector (as in HAWQ-V2)
        vec = (torch.randint(0, 2, params.shape, device=params.device) * 2 - 1).to(params.dtype)
        hvp = hessian_vector_product(loss_fn, params, vec)
        estimated_trace += torch.dot(vec.flatten(), hvp.flatten())
        del vec, hvp
    return estimated_trace / samples


# ============================================================================================
# 4. Greedy sensitivity-per-cost allocator
# ============================================================================================
BIT_WIDTHS = BIT_WIDTHS_ARG      # notebook value [2, 3, 4, 8, 16]; see --bit-widths

# The two allocator units. c_fc and c_proj are scored, allocated and quantized SEPARATELY -- one
# unit per matrix, not one per MLP -- which gives the allocator twice the granularity the QKV
# notebook had (24 units instead of 12 for gpt2) at no extra allocator cost.
UNIT_KINDS = ("c_fc", "c_proj")


def cost(bits, param_count=1.0):
    """
    Memory cost of one unit at a given bit-width.

    param_count=1.0 is still valid with two units per layer, and that is not a coincidence worth
    glossing over: c_fc is (hidden, 4*hidden) and c_proj is (4*hidden, hidden), so BOTH hold
    exactly 4 * hidden_size**2 parameters (2,359,296 each for gpt2). Every allocator unit in this
    notebook therefore has the same parameter count as every other -- just as every layer's fused
    QKV matrix did in the QKV notebook -- so an unweighted `bits` cost keeps `budget` directly
    readable as "average bits per MATRIX" and greedy_allocate's cost-saved-per-accuracy-lost ratio
    stays an apples-to-apples comparison across units.

    (Across the two notebooks the unit sizes differ: a QKV unit is 3*hidden**2 = 1.77 M params,
    an MLP unit is 4*hidden**2 = 2.36 M, and this notebook has twice as many of them. That is what
    section 14b's parameter-weighted table is for.)
    """
    return bits * param_count


def greedy_allocate(scores, budget):
    """
    scores: {unit_name: {bits: sensitivity}};  budget: max total cost.
      1. Start every unit at the HIGHEST bit-width.
      2. While over budget, apply the downgrade with the best cost-saved-per-accuracy-lost ratio.
      3. Spend leftover budget on the best affordable upgrades.
    """
    current_bits = {name: max(BIT_WIDTHS) for name in scores}

    def total_cost():
        return sum(cost(current_bits[n]) for n in scores)

    def total_sensitivity():
        return sum(scores[n][current_bits[n]] for n in scores)

    while total_cost() > budget:                     # downgrade loop
        best = best_bits = best_ratio = None
        for name in scores:
            lower = [b for b in BIT_WIDTHS if b < current_bits[name]]
            if not lower:
                continue                             # already at the lowest bit-width
            nb = max(lower)
            saved = cost(current_bits[name]) - cost(nb)
            added = scores[name][nb] - scores[name][current_bits[name]]
            ratio = added / saved
            if best_ratio is None or ratio < best_ratio:
                best_ratio, best, best_bits = ratio, name, nb
        if best is None:
            break                                    # nothing left to downgrade
        current_bits[best] = best_bits

    made_an_upgrade = True                           # spend any remaining budget
    while made_an_upgrade:
        made_an_upgrade = False
        best = best_bits = best_ratio = None
        for name in scores:
            higher = [b for b in BIT_WIDTHS if b > current_bits[name]]
            if not higher:
                continue
            nb = min(higher)
            extra = cost(nb) - cost(current_bits[name])
            if total_cost() + extra > budget:
                continue                             # can't afford it
            ratio = (scores[name][current_bits[name]] - scores[name][nb]) / extra
            if best_ratio is None or ratio > best_ratio:
                best_ratio, best, best_bits = ratio, name, nb
        if best is not None:
            current_bits[best] = best_bits
            made_an_upgrade = True

    return current_bits, total_cost(), total_sensitivity()


def brute_force_optimal(scores, budget, max_units=8):
    """
    Exhaustive search, for sanity-checking the greedy allocator on a small subset.
    Guarded: with 5 bit-widths this is 5^n_units combinations.
    """
    names = list(scores.keys())
    if len(names) > max_units:
        raise ValueError(f"brute_force_optimal over {len(names)} units = "
                         f"{len(BIT_WIDTHS)}^{len(names)} combinations. Restrict `scores` to at "
                         f"most {max_units} units, or raise max_units if you really mean it.")
    best_assignment = best_cost = None
    best_sensitivity = float("inf")
    for combo in itertools.product(*([BIT_WIDTHS] * len(names))):
        c = sum(cost(b) for b in combo)
        if c > budget:
            continue
        s = sum(scores[names[i]][combo[i]] for i in range(len(names)))
        if s < best_sensitivity:
            best_sensitivity, best_cost = s, c
            best_assignment = dict(zip(names, combo))
    return best_assignment, best_cost, best_sensitivity


def scores_to_allocator_format(jab_scores, bit_widths=None):
    """Heuristic table (trace / bits^1.5), kept as an alternative to the HAWQ-V2 measured form."""
    bit_widths = BIT_WIDTHS if bit_widths is None else bit_widths
    return {n: {b: t / (b ** 1.5) for b in bit_widths} for n, t in jab_scores.items()}


def unit_name(i, which):
    """
    Allocator unit name. Replaces the QKV notebook's `layer_name(i)` -> "layer_{i}_QKV".

    `which` is one of UNIT_KINDS ("c_fc" / "c_proj"), so names look like "layer_3_c_proj". The
    "layer_{idx}_..." prefix is preserved deliberately -- `layer_idx_of` below and the propagation
    weighting in section 12b parse it the same way they always did.
    """
    assert which in UNIT_KINDS, f"unknown unit kind {which!r} (expected one of {UNIT_KINDS})"
    return f"layer_{i}_{which}"


def layer_idx_of(name):
    """Layer index out of a unit name -- unchanged, the index is still field 1."""
    return int(name.split("_")[1])


def unit_kind_of(name):
    """"c_fc" / "c_proj" out of a unit name (the kind itself contains an underscore)."""
    return "_".join(name.split("_")[2:])


# ============================================================================================
# 5. Model loading & per-layer helpers
# ============================================================================================
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, AutoConfig
from datasets import load_dataset


def load_model_and_tokenizer(model_id, device):
    """
    Loads a full, resident GPT-2 -- fits whole in VRAM/RAM (124 MB to 1.5 GB depending on
    `MODEL_ID`), so there is no need to stream shards off disk one tensor at a time.

    Every arm below (sections 10-15) calls this itself, once per arm, and gets back a BRAND NEW
    model. This matters: `pass_quantize_eval` quantizes `mlp.c_fc.weight` and
    `mlp.c_proj.weight` IN PLACE. Reusing one resident model across arms -- as an earlier
    version of this notebook did, loading `reader` once in section 9 and never again -- means
    each arm's damage compounds into the next: the adaptive-allocation scoring pass would run
    its Hessians against an already-4-bit model instead of the intended float baseline, and
    every arm after the first quantizing one would be GPTQ-quantizing already-quantized
    matrices. The working reference notebook avoids this by calling
    `AutoModelForCausalLM.from_pretrained("gpt2")` fresh before every single arm; this function
    is that same fresh-load, just factored out.

    `pass_score`/`pass_quantize_eval`'s `reader` parameter is simply bound to whichever resident
    model this call returns -- `load_decoder_layer` indexes into it instead of allocating from a
    shard, which is what lets those two functions' signatures and per-layer loop structure stay
    unchanged regardless of which arm is calling them.
    """
    tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    # Plain fp32, unconditionally -- see the GPU_DTYPE comment in section 0 for why.
    model = GPT2LMHeadModel.from_pretrained(model_id).to(device).to(GPU_DTYPE)
    model.config._attn_implementation = "eager"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def load_decoder_layer(reader, config, idx):
    """
    No allocation, no meta-device trick -- `reader` is the resident model, so this is just an
    index into `reader.transformer.h`. Kept as a function purely so `pass_score`/
    `pass_quantize_eval`'s per-layer loop bodies read identically regardless of what `reader` is
    bound to.
    """
    return reader.transformer.h[idx]


def mlp_of(layer):
    """
    GPT-2's MLP as the two `Conv1D` modules this notebook quantizes, in forward order.

    Replaces the QKV notebook's `qkv_of(layer) -> layer.attn.c_attn`. Returning a PAIR (rather
    than the `layer.mlp` container) is deliberate: every caller has to decide which of the two it
    means, and the two are not interchangeable -- different shapes, different Hessians, and
    c_proj's Hessian is only valid once c_fc is already quantized (see quantize_mlp, section 1).

    NOTE what is NOT here: `layer.attn.c_proj`. GPT-2 has two modules called `c_proj` per block --
    the attention output projection and the MLP output projection. This notebook quantizes ONLY
    `mlp.c_proj`; `attn.c_attn` and `attn.c_proj` both stay fp32.
    """
    return layer.mlp.c_fc, layer.mlp.c_proj


def mlp_weights_io(layer, dtype=None):
    """
    (W_fc, W_proj) in the (d_in, d_out) convention.

    `Conv1D.weight` is ALREADY (d_in, d_out) -- (hidden, intermediate) for c_fc and
    (intermediate, hidden) for c_proj -- so there is NO transpose here, same as the QKV notebook's
    `qkv_weights_io`. Unlike that one there is also no splitting: these are two separate modules,
    not one fused matrix.
    """
    W_fc, W_proj = layer.mlp.c_fc.weight, layer.mlp.c_proj.weight
    if dtype:
        return W_fc.to(dtype), W_proj.to(dtype)
    return W_fc, W_proj


def mlp_bias_io(layer, dtype=None):
    """
    (b_fc, b_proj) -- shapes (intermediate,) and (hidden,).

    Biases are NEVER a GPTQ/STE target -- callers snapshot them once per layer (alongside the
    float weight snapshot) and pass the same fixed b_fc/b_proj into every compute_mlp call for that
    layer, teacher and student alike.
    """
    b_fc, b_proj = layer.mlp.c_fc.bias, layer.mlp.c_proj.bias
    if dtype:
        return b_fc.to(dtype), b_proj.to(dtype)
    return b_fc, b_proj


@torch.no_grad()
def write_mlp_flat(layer, w_flat, cfg):
    """Scatter a flat [W_fc|W_proj] vector back into the two `.weight.data` tensors, in place, in
    the same (d_in, d_out) orientation they are already stored in -- no transpose."""
    W_fc, W_proj = reshape_weights(w_flat, cfg)
    layer.mlp.c_fc.weight.data.copy_(W_fc.to(layer.mlp.c_fc.weight.dtype))
    layer.mlp.c_proj.weight.data.copy_(W_proj.to(layer.mlp.c_proj.weight.dtype))


def block_forward(layer, h, cfg, attn_mask=None, w_mlp=None, act=None, return_attn=False):
    """
    One GPT-2 block, run manually for the MLP piece (the part that needs to accept a swapped-in
    STE-quantized weight) but delegating everything else -- both LayerNorms and the whole attention
    sub-block, c_attn and attn.c_proj included -- to the real resident modules.

    THIS IS THE EXACT INVERSE OF THE QKV NOTEBOOK, where attention was hand-written and
    `layer.mlp(...)` was called directly. Same rule in both: hand-write only what gets quantized.

    Used by section 7's validation (to prove this hand-written piece matches the real model) and
    nowhere else: the main pipeline (section 8) gets its activations from real forward passes via
    hooks (`collect_hessian_via_hook`, `capture_mlp_input` below), not from a hand-propagated
    hidden-state cache, so there is no second place this logic could silently drift from the real
    model's own forward.

    w_mlp: optional (W_fc, W_proj) in (d_in, d_out) form, used INSTEAD of the module's own weights.
    act: optional activation override; defaults to this layer's own `mlp.act`.
    attn_mask: boolean causal mask (T, T), as built by `build_attn_mask`. Converted here into
        the ADDITIVE float mask HF's attention expects and handed to the real `layer.attn`. This
        is LOAD-BEARING and must not be dropped: as of transformers 5.x `GPT2Attention` no longer
        builds its own causal mask -- `GPT2Model` builds it and passes it down -- so calling
        `layer.attn(x)` with no mask yields fully BIDIRECTIONAL attention and silently wrong
        hidden states (rel. error ~0.28 against the real model, caught by section 7a). In
        transformers 4.x the module masked internally and adding this mask is a harmless no-op,
        so it is passed unconditionally rather than version-sniffed.
    return_attn: accepted for signature parity; always yields None. Attention is delegated, so
        there are no hand-computed attention maps to hand back (and nothing in this notebook
        consumes them -- the MLP loss has no KL term; see mlp_loss, section 2).
    """
    T = h.shape[1]
    if attn_mask is None:
        attn_mask = build_attn_mask(T, h.device)
    # bool "may attend" -> additive float mask. finfo.min rather than -inf so that transformers
    # 4.x, which ALSO applies its own finfo.min causal mask before adding this one, does not end
    # up with a NaN-producing (-inf) + (-inf) row.
    add_mask = torch.zeros(T, T, dtype=h.dtype, device=h.device)
    add_mask = add_mask.masked_fill(~attn_mask.to(h.device), torch.finfo(h.dtype).min)[None, None]

    attn_out = layer.attn(layer.ln_1(h), attention_mask=add_mask)
    # HF has returned both `(attn_output, present, attentions)` (transformers 4.x) and
    # `(attn_output, attn_weights)` (5.x) here; index 0 is the attention output in every version.
    attn_out = attn_out[0] if isinstance(attn_out, tuple) else attn_out
    h = h + attn_out

    x = layer.ln_2(h)
    b_fc, b_proj = mlp_bias_io(layer)
    W_fc, W_proj = w_mlp if w_mlp is not None else mlp_weights_io(layer)
    h = h + compute_mlp(W_fc, W_proj, x, cfg, b_fc=b_fc, b_proj=b_proj,
                        act=layer.mlp.act if act is None else act)
    return (h, None)


@torch.no_grad()
def embed_cache(reader, ids_list, device=None, dtype=None):
    """Hidden states entering layer 0, for every sequence -- used only by section 7's validation.
    `wte(ids) + wpe(positions)`: GPT-2 bakes ALL position information in once, here; there is no
    per-layer RoPE cache anywhere in this notebook."""
    device = device or DEVICE
    dtype = dtype or GPU_DTYPE
    wte = reader.transformer.wte.weight.to(device=device, dtype=dtype)
    wpe = reader.transformer.wpe.weight.to(device=device, dtype=dtype)
    out = []
    for ids in ids_list:
        ids = ids.to(device)
        T = ids.shape[1]
        pos = torch.arange(T, device=device)
        out.append(F.embedding(ids, wte) + F.embedding(pos, wpe)[None])
    free(wte, wpe)
    return out


def collect_hessian_via_hook(model, module, calibration_batches, device):
    """
    Registers a forward PRE-hook on `module` (a block's `mlp.c_fc` or `mlp.c_proj`), runs
    `calibration_batches` through the WHOLE model in no_grad mode, and returns the accumulated
    Hessian H = 2 X^T X for that module.

    UNCHANGED from the QKV notebook except for which module it is pointed at -- it never looked at
    anything but `module.weight.shape[0]` and the tensor arriving at the module, so it handles
    c_fc's (768, 768) and c_proj's (3072, 3072) Hessians identically.

    This replaces hand-computing `X = layer.ln_2(h)` against a manually-propagated hidden-state
    cache. The two should be mathematically identical -- section 7's validation confirms the
    hand-written forward matches the real model to within float precision -- but this is the
    version actually exercised end to end by the real model's own forward machinery. It is ALSO
    what makes the sequential-GPTQ property automatic, and in this notebook that property is needed
    at TWO scales:

      across layers -- `model` is the SAME object `pass_quantize_eval` is quantizing layer by
        layer, so by the time this runs for layer i, blocks 0..i-1 already hold their final
        weights and this hook sees activations produced by them;
      within a block -- `mlp.c_proj`'s input IS `gelu(mlp.c_fc(...))`, so calling this for c_proj
        AFTER c_fc has been quantized is what makes H_proj describe the activations c_proj will
        actually see. `quantize_mlp` (section 1) enforces that ordering; collecting both Hessians
        up front would quietly break it.

    `calibration_batches` should be an iterable of input_ids tensors of shape (1, seq_len).
    Returns H, a (d_in, d_in) float64 tensor -- d_in = 768 for c_fc, 3072 for c_proj (gpt2).
    """
    d_in = module.weight.shape[0]           # Conv1D weight is (in_features, out_features)
    H = torch.zeros(d_in, d_in, dtype=torch.float64, device=device)
    n_samples = [0]

    def _hook(mod, inputs):
        x = inputs[0].detach()
        x = x.reshape(-1, x.shape[-1]).to(torch.float64)   # (tokens, d_in)
        H.add_(2.0 * x.T @ x)
        n_samples[0] += x.shape[0]

    handle = module.register_forward_pre_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            for input_ids in calibration_batches:
                model(input_ids.to(device))
    finally:
        handle.remove()

    if n_samples[0] > 0:
        H /= n_samples[0]
    return H


def capture_mlp_input(model, layer, batch, device, which="c_fc"):
    """
    One-shot capture of the tensor entering `layer.mlp.c_fc` (i.e. ln_2(h)) or `layer.mlp.c_proj`
    (i.e. gelu(c_fc(ln_2(h)))) from a REAL forward pass of `model` on `batch`, via a temporary
    forward pre-hook. The single-batch analogue of `collect_hessian_via_hook` above; replaces the
    QKV notebook's `capture_c_attn_input`, with `which` selecting the hook target.

    The pipeline (section 8) only ever asks for `which="c_fc"`: that is the input to the WHOLE
    two-matrix MLP, which is what both the JAB-Hessian trace and the STE fine-tune need, and
    c_proj's input is then produced inside `compute_mlp` from whatever weights the student
    currently holds. `which="c_proj"` exists for diagnostics and is exercised by section 7.

    Note that ln_2(h) does NOT depend on either MLP weight of layer i -- only on layers 0..i-1 and
    this layer's (never-quantized) attention -- so the captured X is frozen for the whole of layer
    i's fine-tune, which is what lets pass_quantize_eval cache it per batch. It still reflects the
    CURRENT state of every upstream block, for the same reason `collect_hessian_via_hook` does:
    it's a real forward pass through the live, possibly-partially-quantized model.
    """
    assert which in UNIT_KINDS, f"unknown capture target {which!r} (expected one of {UNIT_KINDS})"
    module = layer.mlp.c_fc if which == "c_fc" else layer.mlp.c_proj
    captured = {}

    def _hook(mod, inputs):
        captured["X"] = inputs[0].detach().to(torch.float32)

    handle = module.register_forward_pre_hook(_hook)
    try:
        with torch.no_grad():
            model(batch.to(device))
    finally:
        handle.remove()
    return captured["X"]


@torch.no_grad()
def _sliding_window_nll(model, ids, max_length, stride):
    """
    Sliding-window NLL sum and scored-token count over a single 1D token-id tensor. The shared
    core of `evaluate_perplexity` below (full WikiText-2 test) and section 7's tiny-model
    validation (synthetic ids) -- factored out so the tiny-model check exercises the SAME
    windowing/masking logic as the real evaluation without needing network access or a real
    tokenizer's vocabulary.
    """
    device = model.device
    ids = ids.to(device)
    seq_len = ids.shape[0]
    model.eval()
    nll_sum, n_tokens, prev_end = 0.0, 0, 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        input_ids = ids[begin:end].unsqueeze(0)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        out = model(input_ids, labels=target_ids)
        nll_sum += out.loss.item() * trg_len
        n_tokens += trg_len
        prev_end = end
        if end == seq_len:
            break
    return nll_sum, n_tokens


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, max_length=None, stride=None):
    """
    Sliding-window perplexity on the FULL WikiText-2 test set, using the model's OWN forward
    (`model(input_ids, labels=target_ids)`, i.e. HF's own cross-entropy) rather than a hand-rolled
    logit-chunking loop. every scored token gets
    `stride` tokens of real context (the standard sliding-window scheme), and scoring the full test
    set rather than a small window subset is what gives arm-to-arm differences enough resolution to
    be distinguishable from run-to-run noise.
    """
    max_length = EVAL_MAX_LENGTH if max_length is None else max_length
    stride = EVAL_STRIDE if stride is None else stride
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    nll_sum, n_tokens = _sliding_window_nll(model, ids, max_length, stride)
    return math.exp(nll_sum / n_tokens)


# ============================================================================================
# 6. Calibration and evaluation data
# ============================================================================================
from datasets import load_dataset


def build_calibration_ids(tokenizer, n_samples=None, seq_len=None):
    """`n_samples` chunks of `seq_len` tokens from WikiText-2 train, as (1, seq_len) id tensors.
    Same dataset as the Mistral notebook, unchanged."""
    n_samples = CALIB_N_SAMPLES if n_samples is None else n_samples
    seq_len = CALIB_SEQ_LEN if seq_len is None else seq_len
    assert seq_len <= 1024, f"seq_len={seq_len} exceeds GPT-2's context length (1024)"

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    # GPT-2's tokenizer does not prepend a BOS token by default (unlike Mistral's, which prepends
    # <s>), so add_special_tokens=False is a no-op here -- kept anyway, for parity and because it
    # is the correct call regardless of tokenizer behavior when splicing chunks out of a
    # mid-corpus string.
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    out = []
    for i in range(n_samples):
        start = i * seq_len
        if start + seq_len > ids.shape[0]:
            break
        out.append(ids[start:start + seq_len].unsqueeze(0))
    return out

# Full-test-set evaluation (evaluate_perplexity, section 5) tokenizes WikiText-2 test itself and
# scores every window via the model's own forward -- there is no separate "build the eval windows"
# step any more. An earlier version of this notebook built a small N_EVAL_WINDOWS-capped subset
# here, which is exactly the noise-floor problem section 5's evaluate_perplexity docstring
# describes.


# ============================================================================================
# 7. Validation
# ============================================================================================
def _validate_gpt2_forward(verbose=True):
    """
    7a: hand-written block_forward -- MLP hand-written, attention delegated -- vs. a real
        GPT2LMHeadModel, on a tiny random model. This is the inverse of the QKV notebook's 7a,
        which hand-wrote attention and delegated the MLP.
    7a-act: `compute_mlp`'s default activation vs. the real `layer.mlp.act`. Separated out because
        it is the single highest-risk new line in the port: `F.gelu` (erf) and `gelu_new` (tanh
        approximation) agree to ~1e-3, which is small enough to survive a perplexity run and large
        enough to corrupt every teacher target, Hutchinson trace and STE gradient in the notebook.
    7a-H: `collect_hessian_via_hook` on `mlp.c_proj` must produce an (intermediate, intermediate)
        Hessian -- the cheapest possible guard against wiring c_proj's hook to the wrong tensor
        (ln_2(h) is (hidden,)-wide, so a mis-wired hook yields (hidden, hidden) and is caught here
        rather than 3000 GPTQ columns later).
    7b: _sliding_window_nll (the core of evaluate_perplexity) vs. HF's own labels=... loss, on
        the same tiny model and the same single window.
    Plus: the Conv1D orientation guard on BOTH MLP matrices, and the
    STE-grid-is-a-no-op-on-GPTQ's-own-output check.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(0)
    tiny = GPT2Config(vocab_size=256, n_embd=32, n_head=4, n_layer=3, n_positions=64,
                      bos_token_id=0, eos_token_id=0)
    ref = GPT2LMHeadModel(tiny)
    ref.config._attn_implementation = "eager"
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = BlockConfig(ref.config)
    # Attention shape assumptions. Nothing here quantizes attention -- block_forward delegates the
    # whole sub-block to the real `layer.attn` -- but these are what make that delegation safe to
    # reason about, and they are the reason BlockConfig keeps the attention fields at all.
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.n_rep) == (4, 4, 8, 1), (
        "no GQA in GPT-2 -- num_kv_heads must equal num_heads and n_rep must be 1")
    assert cfg.q_out == cfg.kv_out == 32, "Q/K/V must be equal-sized thirds (no GQA shrink)"
    # MLP shape assumptions -- the ones this notebook actually quantizes against.
    assert cfg.intermediate_size == 4 * cfg.hidden_size == 128, (
        f"intermediate_size {cfg.intermediate_size} != 4 * hidden {cfg.hidden_size}")
    assert cfg.fc_numel == cfg.proj_numel == cfg.mlp_numel // 2, (
        "c_fc and c_proj must hold equal parameter counts -- cost(bits, param_count=1.0) in "
        "section 4 depends on it")
    assert tuple(ref.transformer.h[0].mlp.c_fc.weight.shape) == (32, 128), \
        "c_fc must be (d_in, d_out) = (hidden, intermediate) -- Conv1D, not nn.Linear"
    assert tuple(ref.transformer.h[0].mlp.c_proj.weight.shape) == (128, 32), \
        "c_proj must be (d_in, d_out) = (intermediate, hidden) -- Conv1D, not nn.Linear"

    T = 24
    ids = torch.randint(0, 256, (1, T))
    with torch.no_grad():
        want = ref(ids).logits

    mask = build_attn_mask(T, "cpu")
    assert int(mask[T - 1].sum()) == T, "plain causal mask: the last row must attend to everything"

    # --- 7a-act: the activation, checked on its own before anything depends on it ---
    act_probe = torch.randn(64, 128) * 3.0     # wide enough to cover the tails, where the erf and
                                                # tanh GELUs disagree most
    real_act = ref.transformer.h[0].mlp.act
    with torch.no_grad():
        got_act, want_act = GPT2_ACT(act_probe), real_act(act_probe)
        act_err = (got_act - want_act).abs().max().item()
        wrong_act = (F.gelu(act_probe) - want_act).abs().max().item()
    assert type(GPT2_ACT) is type(real_act), (
        f"compute_mlp's default activation is {type(GPT2_ACT).__name__} but the model's is "
        f"{type(real_act).__name__}")
    assert act_err == 0.0, (
        f"compute_mlp's activation does not match layer.mlp.act (max abs diff {act_err:.3e})")
    assert wrong_act > 1e-4, (
        "F.gelu and gelu_new agree to machine precision on this probe -- this check cannot "
        "distinguish them, so it is vacuous. Widen act_probe.")
    if verbose:
        print(f"  7a-act: compute_mlp's activation == layer.mlp.act exactly (max abs diff "
              f"{act_err:.1e}); the wrong choice F.gelu would be off by {wrong_act:.3e}")

    # embed_cache reads DEVICE/GPU_DTYPE as globals (not parameters), so on a GPU runtime it would
    # move hidden states onto CUDA while `ref` -- deliberately never touched by .to(DEVICE) --
    # stays on CPU, and ln_1/ln_2/ln_f would mismatch devices. This tiny check is meant to run in
    # seconds regardless of accelerator, so force it onto CPU/fp32 for its duration and restore
    # afterward, the same guard `_validate_adaptive_joint_pass` already uses.
    old_device, old_dtype = globals()["DEVICE"], globals()["GPU_DTYPE"]
    globals()["DEVICE"], globals()["GPU_DTYPE"] = "cpu", torch.float32
    try:
        # --- 7a: hand-written MLP + delegated attention, stacked over every block ---
        with torch.no_grad():
            h = embed_cache(ref, [ids])[0]
            for i in range(cfg.n_layers):
                h, _ = block_forward(ref.transformer.h[i], h, cfg, mask)
            got = ref.lm_head(ref.transformer.ln_f(h))

        rel = ((got - want).norm() / want.norm()).item()
        if verbose:
            print(f"  7a: hand-written MLP forward vs. GPT2LMHeadModel.forward: relative error "
                  f"{rel:.3e}")
        assert rel < 1e-3, (
            f"hand-written forward disagrees with the real model (rel err {rel:.3e}). Check the "
            f"Conv1D orientation of c_fc/c_proj, the activation, and the bias split.")

        # A tighter, more localized version of the same claim: compute_mlp vs. the real GPT2MLP
        # module on one block, on its own. The stacked check above could in principle hide a small
        # MLP error behind the residual stream; this one cannot.
        with torch.no_grad():
            x_probe = torch.randn(1, T, cfg.hidden_size)
            mlp_mod = ref.transformer.h[0].mlp
            want_mlp = mlp_mod(x_probe)
            W_fc, W_proj = mlp_weights_io(ref.transformer.h[0])
            b_fc, b_proj = mlp_bias_io(ref.transformer.h[0])
            got_mlp = compute_mlp(W_fc, W_proj, x_probe, cfg, b_fc=b_fc, b_proj=b_proj,
                                  act=mlp_mod.act)
        mlp_abs = (got_mlp - want_mlp).abs().max().item()
        assert mlp_abs < 1e-5, (
            f"compute_mlp disagrees with the real GPT2MLP (max abs diff {mlp_abs:.3e})")
        if verbose:
            print(f"  7a: compute_mlp vs. the real GPT2MLP module: max abs diff {mlp_abs:.3e}")

        # Prove the check is not vacuous: perturb W_fc and confirm the block output moves. (The
        # QKV notebook's analogue disabled the causal mask; attention is delegated here, so the
        # load-bearing thing to perturb is the weight this notebook actually quantizes.)
        with torch.no_grad():
            h0 = embed_cache(ref, [ids])[0]
            out_ok, _ = block_forward(ref.transformer.h[0], h0, cfg, mask)
            W_fc_bad = W_fc + 0.1 * torch.randn_like(W_fc)
            out_bad, _ = block_forward(ref.transformer.h[0], h0, cfg, mask,
                                       w_mlp=(W_fc_bad, W_proj))
        rel_broken = ((out_bad - out_ok).norm() / out_ok.norm()).item()
        assert rel_broken > 1e-4, (
            "perturbing W_fc changed nothing -- block_forward is ignoring w_mlp and this check "
            "is vacuous!")
        if verbose:
            print(f"  block_forward's w_mlp override is load-bearing (perturbing W_fc moves the "
                  f"block output by {rel_broken:.3e})")

        # ...and the causal mask this notebook hands to the delegated attention is load-bearing
        # too: transformers 5.x's GPT2Attention no longer masks internally (see block_forward's
        # attn_mask docstring), so dropping it must visibly change the block output.
        with torch.no_grad():
            full_mask = torch.ones(T, T, dtype=torch.bool)
            out_nomask, _ = block_forward(ref.transformer.h[0], h0, cfg, full_mask)
        rel_mask = ((out_nomask - out_ok).norm() / out_ok.norm()).item()
        assert rel_mask > 1e-4, (
            "disabling the causal mask changed nothing -- either block_forward is ignoring "
            "attn_mask or the installed transformers masks internally; either way this check is "
            "vacuous and 7a would not catch a dropped mask")
        if verbose:
            print(f"  the causal mask passed to the delegated attention is load-bearing "
                  f"(disabling it moves the block output by {rel_mask:.3e})")

        # --- 7a-H: c_proj's Hessian must be (intermediate, intermediate) ---
        h_batches = [torch.randint(0, 256, (1, T)) for _ in range(2)]
        H_fc = collect_hessian_via_hook(ref, ref.transformer.h[0].mlp.c_fc, h_batches, "cpu")
        H_proj = collect_hessian_via_hook(ref, ref.transformer.h[0].mlp.c_proj, h_batches, "cpu")
        assert tuple(H_fc.shape) == (cfg.hidden_size, cfg.hidden_size), (
            f"c_fc Hessian is {tuple(H_fc.shape)}, expected "
            f"({cfg.hidden_size}, {cfg.hidden_size})")
        assert tuple(H_proj.shape) == (cfg.intermediate_size, cfg.intermediate_size), (
            f"c_proj Hessian is {tuple(H_proj.shape)}, expected "
            f"({cfg.intermediate_size}, {cfg.intermediate_size}) -- if this is "
            f"({cfg.hidden_size}, {cfg.hidden_size}) the c_proj hook is on the wrong module and "
            f"is seeing ln_2(h) instead of gelu(c_fc(ln_2(h)))")
        # ...and capture_mlp_input(which=...) must agree with those two widths.
        X_fc = capture_mlp_input(ref, ref.transformer.h[0], ids, "cpu", which="c_fc")
        X_proj = capture_mlp_input(ref, ref.transformer.h[0], ids, "cpu", which="c_proj")
        assert X_fc.shape[-1] == cfg.hidden_size and X_proj.shape[-1] == cfg.intermediate_size, (
            f"capture_mlp_input widths wrong: c_fc {tuple(X_fc.shape)}, "
            f"c_proj {tuple(X_proj.shape)}")
        # and c_proj's captured input really is the activated c_fc output, not something else
        with torch.no_grad():
            X_proj_expected = mlp_mod.act(X_fc @ W_fc + b_fc)
        assert (X_proj - X_proj_expected).abs().max().item() < 1e-5, (
            "capture_mlp_input('c_proj') is not gelu(c_fc(ln_2(h)))")
        if verbose:
            print(f"  7a-H: Hessians are c_fc {tuple(H_fc.shape)}, c_proj {tuple(H_proj.shape)}; "
                  f"c_proj's captured input == gelu(c_fc(ln_2(h)))")
        free(H_fc, H_proj, X_fc, X_proj)

        # 7b: _sliding_window_nll (section 5 -- the shared core of evaluate_perplexity) against
        # HF's own labels=... loss, on the SAME single window (stride == max_length == T).
        nll_ours, n_ours = _sliding_window_nll(ref, ids[0], max_length=T, stride=T)
        ppl_ours = math.exp(nll_ours / n_ours)
        with torch.no_grad():
            ppl_hf = math.exp(ref(ids, labels=ids).loss.item())
        rel_ppl = abs(ppl_ours - ppl_hf) / ppl_hf
        if verbose:
            print(f"  7b: _sliding_window_nll vs. HF's labels=... loss: ours={ppl_ours:.4f} "
                  f"hf={ppl_hf:.4f} (rel err {rel_ppl:.3e})")
        assert rel_ppl < 1e-3, (
            f"_sliding_window_nll disagrees with HF's own loss (rel err {rel_ppl:.3e})")
    finally:
        globals()["DEVICE"], globals()["GPU_DTYPE"] = old_device, old_dtype

    # The STE grid must be a no-op on GPTQ's own output (the section-13 warm-start guarantee).
    # CPU-only, no model involved -- no device guard needed.
    W0 = torch.randn(32, 32) * 0.05
    Xd = torch.randn(2000, 32)
    Hd = (2.0 * Xd.T @ Xd / Xd.shape[0]).double()
    Wq, sc = gptq_quantize_layer(W0.clone(), Hd, bits=4, group_size=8, act_order=True,
                                 return_scale=True, work_dtype=torch.float64)
    qmax = 7
    rel_ste = ((torch.clamp(torch.round(Wq / sc), -qmax, qmax) * sc - Wq).norm() / Wq.norm()).item()
    assert rel_ste < 1e-6, f"return_scale is not the grid GPTQ used (rel err {rel_ste:.3e})"
    if verbose:
        print(f"  GPTQ return_scale is exact (re-quantization relative error {rel_ste:.3e})")

    # Explicit Conv1D orientation guard, on BOTH MLP shapes: quantize a (d_in, d_out) module and
    # confirm the shape survives and x @ W still type-checks -- the check for the bug class
    # described in gptq_quantize_conv1d's docstring (section 1). Both directions are covered
    # (d_in < d_out like c_fc, and d_in > d_out like c_proj) because the two matrices are
    # transposes of each other in shape, and only running one of them would leave the other's
    # orientation untested.
    class _FakeConv1D:
        def __init__(self, w):
            self.weight = torch.nn.Parameter(w.clone())

    for tag, (d_in, d_out) in (("c_fc-like", (16, 64)), ("c_proj-like", (64, 16))):
        fake = _FakeConv1D(torch.randn(d_in, d_out) * 0.05)
        Xc = torch.randn(500, d_in)
        Hc = (2.0 * Xc.T @ Xc / Xc.shape[0]).double()
        W_before_shape = tuple(fake.weight.data.shape)
        W_after = gptq_quantize_conv1d(fake, Hc, bits=4, group_size=8)
        assert tuple(W_after.shape) == W_before_shape == (d_in, d_out), (
            f"gptq_quantize_conv1d changed the {tag} weight orientation: {W_before_shape} -> "
            f"{tuple(W_after.shape)}")
        _ = Xc @ fake.weight.data     # must not raise a shape error
        # and a TRANSPOSED weight must be rejected outright, not silently quantized along the
        # wrong axis -- non-square shapes are what makes this catchable at all
        bad = _FakeConv1D(fake.weight.data.T.contiguous())
        try:
            gptq_quantize_conv1d(bad, Hc, bits=4, group_size=8)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"gptq_quantize_conv1d accepted a transposed {tag} weight "
                f"{tuple(bad.weight.shape)} against a {tuple(Hc.shape)} Hessian -- the "
                f"orientation guard is not firing")
        if verbose:
            print(f"  gptq_quantize_conv1d preserves the {tag} (d_in, d_out) orientation "
                  f"{W_before_shape}, and rejects its transpose")

    free(ref)
    return rel


print("Validation 7a/7b: hand-written MLP forward + perplexity path vs. real GPT2LMHeadModel...")
_validate_gpt2_forward()
print("PASSED -- the hand-written MLP forward and the perplexity path reproduce GPT2LMHeadModel.\n")


def _validate_adaptive_joint_pass(verbose=True):
    """
    7c-warm: ste_quantize_mlp must be an exact no-op on GPTQ's own output for BOTH matrices --
        the section-13 warm-start guarantee, extended to two independently permuted grids.
    7c: pass_quantize_eval(bits_fn=..., finetune=True) on a hand-made mixed 2/3/4/8-bit
    assignment, tiny random GPT-2, CPU-only, seconds not minutes. Exercises exactly the
    section-13b code path (adaptive allocation + joint fine-tuning together) that sections 7a/7b,
    11, 12 and 13 each individually miss.

    EXTENDED FROM THE QKV NOTEBOOK: `bits_fn` is now `(layer_idx, which) -> bits`, and the
    assignment below gives c_fc and c_proj DIFFERENT widths within the same layer. That is the
    property the QKV notebook could not test and this one must: the STE block carries two
    scale_compact/g_idx pairs and two qmax values through one concatenated weight vector, and a
    mix-up there would be invisible if both matrices always shared a bit-width.

    `reader` in this notebook IS the resident model, so a tiny in-memory `GPT2LMHeadModel` can be
    passed directly to pass_quantize_eval -- no CheckpointReader-shaped stand-in needed. The one
    thing that DOES need a stand-in is `eval_fn`: the real pipeline's eval_fn closes over
    evaluate_perplexity, which downloads real WikiText-2 -- this test instead builds a synthetic
    "test set" from random ids and scores it with the SAME _sliding_window_nll evaluate_perplexity
    itself uses, keeping this test network-free and CPU-only.
    """
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(1)
    tiny = GPT2Config(vocab_size=256, n_embd=32, n_head=4, n_layer=4, n_positions=64,
                      bos_token_id=0, eos_token_id=0)
    ref = GPT2LMHeadModel(tiny)
    ref.config._attn_implementation = "eager"
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = BlockConfig(ref.config)

    # every bit-width this notebook supports except 16 (kept out only for speed); 2-bit is
    # included deliberately -- the allocator can and does emit it, and it must not be silently
    # clamped up to some higher floor. c_fc and c_proj differ in every layer, and each layer's
    # PAIR is distinct, so a swapped fc/proj bit-width cannot pass unnoticed.
    mixed_bits = {(0, "c_fc"): 2, (0, "c_proj"): 4,
                  (1, "c_fc"): 3, (1, "c_proj"): 8,
                  (2, "c_fc"): 4, (2, "c_proj"): 2,
                  (3, "c_fc"): 8, (3, "c_proj"): 3}
    assert {i for i, _ in mixed_bits} == set(range(cfg.n_layers)), \
        "mixed_bits must cover every tiny layer"
    assert {w for _, w in mixed_bits} == set(UNIT_KINDS), \
        "mixed_bits must cover both matrices"

    calib_ids = [torch.randint(0, 256, (1, 16)) for _ in range(4)]
    eval_ids = torch.randint(0, 256, (48,))          # synthetic "test set" stand-in

    # --- 7c-warm: the STE grid must be a NO-OP on GPTQ's own output, for BOTH matrices at once ---
    # The section-13 warm start depends on ste_quantize_mlp(w_flat_0, ...) == w_flat_0 exactly at
    # step 0. Section 7a proves `gptq_quantize_layer(return_scale=...)` hands back the grid it
    # actually used, but that is a single-matrix check; what is untested until here is the
    # compact->per-weight expansion for TWO matrices with DIFFERENT shapes, DIFFERENT group counts
    # and INDEPENDENT act_order permutations. Mis-map either g_idx and the warm start silently
    # starts off-grid: init_loss is wrong, so the commit-only-if-better safety net compares against
    # the wrong baseline and every flip-rate number is measured from the wrong origin. Nothing
    # would raise.
    warm = GPT2LMHeadModel(tiny)
    warm.config._attn_implementation = "eager"
    warm.eval()
    for p in warm.parameters():
        p.requires_grad_(False)
    old_device_w, old_dtype_w = globals()["DEVICE"], globals()["GPU_DTYPE"]
    globals()["DEVICE"], globals()["GPU_DTYPE"] = "cpu", torch.float32
    try:
        warm_bits = (4, 3)      # different widths, so a swapped qmax cannot cancel out
        (W_fc_q, s_fc, sc_fc, g_fc), (W_proj_q, s_proj, sc_proj, g_proj) = quantize_mlp(
            warm, warm.transformer.h[0], warm_bits, calib_ids[:2], group_size=8,
            return_scales=True, return_compact=True)
    finally:
        globals()["DEVICE"], globals()["GPU_DTYPE"] = old_device_w, old_dtype_w

    w0 = flatten_weights(W_fc_q, W_proj_q).to(torch.float32)
    # the expanded per-weight scale must BE the grid GPTQ reported, in flatten_weights order
    scale_err = (expand_scale_mlp(sc_fc, g_fc, sc_proj, g_proj, cfg).to(torch.float32)
                 - flatten_weights(s_fc, s_proj).to(torch.float32)).abs().max().item()
    assert scale_err == 0.0, (
        f"expand_scale_mlp does not reproduce GPTQ's own per-weight scale (max abs diff "
        f"{scale_err:.3e}) -- a g_idx is mis-mapped for one of the two matrices")
    scale_w0 = expand_scale_mlp(sc_fc, g_fc, sc_proj, g_proj, cfg).to(torch.float32)
    with torch.no_grad():
        w0_q = ste_quantize_mlp(w0, sc_fc, g_fc, sc_proj, g_proj, warm_bits[0], warm_bits[1],
                                8, cfg)
        # The INTEGER grid positions are what has to be identical, and they are exactly what
        # `flip_rate` measures in pass_quantize_eval -- so assert on those, not on the float
        # values. gptq_quantize_layer does its solve in float64 and casts the result to float32,
        # so `q * s` recomputed in float32 differs from the stored weight by about one ULP; the
        # QKV notebook's own return_scale check is a relative-norm check for the same reason.
        warm_flips = (torch.round(w0 / scale_w0) != torch.round(w0_q / scale_w0)).sum().item()
    warm_rel = ((w0_q - w0).norm() / w0.norm()).item()
    assert warm_flips == 0, (
        f"ste_quantize_mlp moved {warm_flips} of {w0.numel()} grid positions on GPTQ's own "
        f"output -- the STE warm start starts off-grid, so init_loss and every flip rate are "
        f"measured from the wrong origin")
    assert warm_rel < 1e-6, (
        f"ste_quantize_mlp is not a no-op on GPTQ's own output (rel err {warm_rel:.3e})")
    # and prove the check is not vacuous: the WRONG bit-width must move the grid
    with torch.no_grad():
        w0_wrong = ste_quantize_mlp(w0, sc_fc, g_fc, sc_proj, g_proj, 2, 2, 8, cfg)
        wrong_flips = (torch.round(w0 / scale_w0) != torch.round(w0_wrong / scale_w0)).sum().item()
    assert wrong_flips > 0, (
        "re-quantizing at 2 bits moved no grid positions -- the warm-start check is vacuous")
    if verbose:
        print(f"  7c-warm: ste_quantize_mlp is a no-op on GPTQ's c_fc {warm_bits[0]}b / c_proj "
              f"{warm_bits[1]}b output -- 0 grid flips, rel err {warm_rel:.1e} (both g_idx maps "
              f"correct); re-quantizing at 2b instead moves {wrong_flips:,} positions")
    free(warm, w0, w0_q, W_fc_q, W_proj_q, s_fc, s_proj, sc_fc, sc_proj)

    def synthetic_eval_fn():
        nll, n = _sliding_window_nll(ref, eval_ids, max_length=16, stride=16)
        return math.exp(nll / n)

    # pass_quantize_eval reads DEVICE/GPU_DTYPE as globals, not as parameters -- swap them to
    # CPU/fp32 for this check, then restore.
    old_device, old_dtype = globals()["DEVICE"], globals()["GPU_DTYPE"]
    globals()["DEVICE"], globals()["GPU_DTYPE"] = "cpu", torch.float32
    try:
        ppl, notes = pass_quantize_eval(
            ref, ref.config, cfg, calib_ids, synthetic_eval_fn,
            bits_fn=lambda i, which: mixed_bits[(i, which)], finetune=True,
            steps_per_block=2, n_hessian_batches=2, group_size=8,
            label="7c: mixed-bit adaptive+joint (tiny CPU model)", verbose=False,
            return_notes=True)
    finally:
        globals()["DEVICE"], globals()["GPU_DTYPE"] = old_device, old_dtype

    assert math.isfinite(ppl) and ppl > 0, f"non-finite/degenerate perplexity: {ppl}"
    assert set(notes) == {i for i, _ in mixed_bits}, (
        f"notes missing layers: {{i for i, _ in mixed_bits}} - {set(notes)} -- return_notes must "
        f"record every bits_fn'd layer, regardless of bit-width")
    for i in sorted(notes):
        got_bits, committed = notes[i]
        want_bits = (mixed_bits[(i, "c_fc")], mixed_bits[(i, "c_proj")])
        assert tuple(got_bits) == want_bits, (
            f"layer {i}: note recorded {got_bits} bits, assigned {want_bits} -- if these are "
            f"swapped, bits_fn's `which` argument is being ignored or mis-routed")
        assert committed in (True, False), (
            f"layer {i} ({got_bits} bits): finetune=True but committed={committed!r} -- "
            f"want_scales must not be gated on bit-width, so even the 2-bit matrix needs a real "
            f"fine-tune attempt and a True/False outcome, not None")

    # The weights must actually be ON the assigned grids afterwards, not merely recorded in the
    # notes -- this is what catches a `bits` that was threaded into the notes but not into the
    # quantizer, or a c_fc/c_proj mix-up inside the STE block.
    #
    # Symmetric quantization puts every weight sharing one scale on at most 2*qmax+1 = 2**bits - 1
    # levels. Scales are per (OUTPUT ROW, group of input columns) -- so the set that shares a scale
    # is a run of ROWS within one COLUMN of the (d_in, d_out) weight, and a whole column spans
    # n_groups of them. act_order permutes which rows form a group, so the bound is taken per
    # column rather than per group: at most n_groups * (2**bits - 1) distinct values per column.
    levels = {}
    for i in sorted(notes):
        for which in UNIT_KINDS:
            bits = mixed_bits[(i, which)]
            W = (ref.transformer.h[i].mlp.c_fc if which == "c_fc"
                 else ref.transformer.h[i].mlp.c_proj).weight.data
            n_groups = (W.shape[0] + 7) // 8              # group_size=8 in this test
            max_levels = n_groups * (2 ** bits - 1)
            worst = max(torch.unique(W[:, j]).numel() for j in range(W.shape[1]))
            assert worst <= max_levels, (
                f"layer {i} {which}: a column holds {worst} distinct values, more than the "
                f"{max_levels} a {bits}-bit grid over {n_groups} groups allows -- this matrix was "
                f"not quantized at the width it was assigned")
            levels[(i, which)] = worst

    # ...and the bound above is only tight at low bit-widths, so also require that, for each
    # matrix kind, more assigned bits really does mean more distinct values. A swapped or ignored
    # `which` would break this ordering even where the per-column bound is slack.
    for which in UNIT_KINDS:
        by_bits = sorted((mixed_bits[(i, which)], levels[(i, which)]) for i in sorted(notes))
        assert all(a[1] <= b[1] for a, b in zip(by_bits, by_bits[1:])), (
            f"{which}: distinct-value count is not monotone in assigned bits {by_bits} -- "
            f"bits_fn's `which` argument is being ignored or mis-routed")
        assert by_bits[0][1] < by_bits[-1][1], (
            f"{which}: the {by_bits[0][0]}-bit and {by_bits[-1][0]}-bit matrices are equally "
            f"fine-grained ({by_bits[0][1]} levels) -- this check is vacuous")

    # Backwards compatibility: a plain 1-arg bits_fn (the uniform-4-bit arm's `lambda i: 4`) must
    # still work and must apply to BOTH matrices.
    torch.manual_seed(1)
    ref2 = GPT2LMHeadModel(tiny)
    ref2.config._attn_implementation = "eager"
    ref2.eval()
    for p in ref2.parameters():
        p.requires_grad_(False)
    globals()["DEVICE"], globals()["GPU_DTYPE"] = "cpu", torch.float32
    try:
        _, notes2 = pass_quantize_eval(
            ref2, ref2.config, cfg, calib_ids,
            lambda: math.exp(_sliding_window_nll(ref2, eval_ids, 16, 16)[0] / 48),
            bits_fn=lambda i: 4, finetune=False, n_hessian_batches=2, group_size=8,
            label="7c: legacy 1-arg bits_fn", verbose=False, return_notes=True)
    finally:
        globals()["DEVICE"], globals()["GPU_DTYPE"] = old_device, old_dtype
    assert all(tuple(b) == (4, 4) for b, _ in notes2.values()), (
        f"a legacy bits_fn(layer_idx) must be applied to both matrices; got "
        f"{[b for b, _ in notes2.values()]}")

    if verbose:
        detail = ", ".join(f"L{i}={b[0]}/{b[1]}b/{'FT' if c else 'GPTQ'}"
                           for i, (b, c) in sorted(notes.items()))
        print(f"  mixed 2/3/4/8-bit adaptive+joint pass ran end-to-end on a tiny CPU GPT-2 model "
              f"(ppl={ppl:.3f}), c_fc/c_proj widths differing in every layer: {detail}")
        print(f"  legacy 1-arg bits_fn(layer_idx) still applies to both matrices")

    free(ref, ref2)
    return ppl


# ============================================================================================
# 8. The pipeline passes
# ============================================================================================
class STEQuantize(torch.autograd.Function):
    """Straight-through on w; LSQ gradient on scale (Esser et al.), scaled by 1/sqrt(n_weights*qmax)
    since each group's scale now accumulates gradient from n_weights shared weights."""

    @staticmethod
    def forward(ctx, w, scale, bits, n_weights):
        qmax = 2 ** (bits - 1) - 1
        v = w / scale
        q = torch.clamp(torch.round(v), -qmax, qmax)
        ctx.save_for_backward(v, q)
        ctx.qmax = qmax
        ctx.grad_scale_factor = (n_weights * qmax) ** -0.5
        return q * scale

    @staticmethod
    def backward(ctx, grad_output):
        v, q = ctx.saved_tensors
        qmax = ctx.qmax
        grad_scale_local = torch.where(v.abs() <= qmax, q - v, torch.sign(v) * qmax)
        return grad_output, grad_output * grad_scale_local * ctx.grad_scale_factor, None, None


def ste_quantize(w, scale, bits, n_weights):
    return STEQuantize.apply(w, scale, bits, n_weights)


def expand_scale(scale_compact, g_idx):
    """
    ONE matrix's (d_out, n_groups) scale -- one entry per (output row, group), with a shared
    per-input-column group map g_idx (d_in,) -- expanded to a per-weight tensor flattened in the
    same (d_in, d_out) row-major order `W.reshape(-1)` uses.

    Now per-MATRIX, not per-notebook: the QKV notebook's version took `cfg` and hard-coded the
    [Q|K|V] thirds of one fused matrix. Here c_fc and c_proj have different shapes, different
    group counts and their own g_idx (act_order permutes each independently), so each gets its own
    call and `expand_scale_mlp` below concatenates the two in [W_fc | W_proj] order.
    """
    return scale_compact[:, g_idx].T.reshape(-1)


def expand_scale_mlp(sc_fc, g_fc, sc_proj, g_proj, cfg):
    """Per-weight scale for the concatenated [W_fc | W_proj] vector -- the flatten_weights order."""
    return torch.cat([expand_scale(sc_fc, g_fc), expand_scale(sc_proj, g_proj)])


def ste_quantize_mlp(w_flat, sc_fc, g_fc, sc_proj, g_proj, bits_fc, bits_proj, n_weights, cfg):
    """
    STE-quantize the concatenated [W_fc | W_proj] vector, each half on ITS OWN grid and at ITS OWN
    bit-width.

    The two halves cannot share a call: c_fc and c_proj are separate allocator units (section 4)
    and routinely get different widths, and `qmax = 2**(bits-1) - 1` lives inside STEQuantize. The
    slice-and-cat is differentiable, so w_flat stays a single leaf parameter with a single
    optimizer group, exactly as in the QKV notebook.
    """
    n = cfg.fc_numel
    return torch.cat([
        ste_quantize(w_flat[:n], expand_scale(sc_fc, g_fc), bits_fc, n_weights),
        ste_quantize(w_flat[n:], expand_scale(sc_proj, g_proj), bits_proj, n_weights),
    ])


def _resolve_bits_fn(bits_fn):
    """
    Normalize `bits_fn` to the 2-argument form `(layer_idx, which) -> bits`.

    This notebook's allocator has two units per layer, so the natural signature is
    `bits_fn(layer_idx, which)`. A 1-argument callable -- the QKV notebook's form, and what the
    uniform-4-bit arm (`lambda i: 4`) still passes -- is accepted and applied to BOTH matrices, so
    no caller written against the old signature breaks. Section 7c tests both forms.

    Arity is read off the signature rather than probed with try/except, so a TypeError raised from
    INSIDE a correct 2-arg bits_fn is never mistaken for a 1-arg one.
    """
    if bits_fn is None:
        return None
    try:
        params = [p for p in inspect.signature(bits_fn).parameters.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        n_positional = len(params)
        takes_varargs = any(p.kind is p.VAR_POSITIONAL
                            for p in inspect.signature(bits_fn).parameters.values())
    except (TypeError, ValueError):
        return bits_fn                      # not introspectable -- assume it is already 2-arg
    if n_positional >= 2 or takes_varargs:
        return bits_fn
    return lambda i, which: bits_fn(i)


def pass_score(reader, config, cfg, calib_ids, *, samples=None, n_score_batches=None,
               n_hessian_batches=None, lambda_kl=None, group_size=None, bit_widths=None,
               perturb_mode=None, verbose=True):
    """
    Scoring pass over an all-float model. SIGNATURE UNCHANGED from the QKV notebook (`lambda_kl`
    included -- it is accepted, ignored, and warned about once; see warn_if_lambda_kl, section 0).

    `reader` MUST be a freshly-loaded, unquantized model -- every caller in section 12 gets one
    from `load_model_and_tokenizer` for exactly this reason (see that function's docstring,
    section 5): scoring against an already-quantized model would measure "how sensitive is the
    already-degraded model", not "where should bits go", defeating the point of adaptive
    allocation.

    TWO UNITS PER LAYER. Each layer contributes `layer_{i}_c_fc` AND `layer_{i}_c_proj`:

      trace        -- ONE Hutchinson trace per layer, taken over the CONCATENATED [W_fc | W_proj]
                      flat vector against the joint MLP-output MSE. Both units share it.
      perturbation -- per UNIT: `||Q(W)-W||_F^2` measured on that matrix alone.

    So Omega(unit, bits) = trace_layer * ||Q(W_unit) - W_unit||_F^2. This is the natural extension
    of the QKV notebook's structure, which also took one trace over the whole fused [Q|K|V] vector
    and then summed the perturbation over the three sub-matrices -- the only change is that the
    perturbation is now reported per sub-matrix instead of summed over them, which is what gives
    the allocator its extra granularity. Splitting the trace as well would mean two double-backward
    passes per layer for a quantity that is not separable anyway: the loss is the output of the
    two matrices IN SERIES, so its curvature genuinely couples them.

    Returns (jab_scores, allocator_input):
      jab_scores      -- {unit_name: Hessian trace of the MLP-aware loss} (both units of a layer
                         carry that layer's shared trace)
      allocator_input -- HAWQ-V2 style {unit_name: {bits: trace * ||Q(W)-W||_F^2}}
    """
    samples = HUTCH_SAMPLES if samples is None else samples
    n_score_batches = JAB_N_BATCHES if n_score_batches is None else n_score_batches
    n_hessian_batches = HESSIAN_N_BATCHES if n_hessian_batches is None else n_hessian_batches
    lambda_kl = LAMBDA_KL if lambda_kl is None else lambda_kl
    group_size = GROUP_SIZE if group_size is None else group_size
    bit_widths = BIT_WIDTHS if bit_widths is None else bit_widths
    perturb_mode = PERTURB_MODE if perturb_mode is None else perturb_mode
    warn_if_lambda_kl(lambda_kl, "pass_score")

    reset_vram_peak()

    jab_scores, allocator_input = {}, {}
    print(f"Scoring {cfg.n_layers} layers x {len(UNIT_KINDS)} matrices = "
          f"{cfg.n_layers * len(UNIT_KINDS)} units: {samples} Hutchinson probes x "
          f"{n_score_batches} batches, loss = MSE on the MLP output (no KL term), "
          f"perturbation mode '{perturb_mode}'")

    for i in range(cfg.n_layers):
        t_layer = time.time()                     # unattended-run instrumentation, see run.sh
        layer = load_decoder_layer(reader, config, i)
        names = {which: unit_name(i, which) for which in UNIT_KINDS}

        # --- JAB-Hessian trace of the MLP-aware loss, JOINT over [W_fc | W_proj] ---
        W32 = mlp_weights_io(layer, dtype=torch.float32)      # (W_fc, W_proj)
        b32 = mlp_bias_io(layer, dtype=torch.float32)         # (b_fc, b_proj) -- never a target
        act = layer.mlp.act                                    # the REAL gelu_new module
        w_flat = flatten_weights(*W32).clone().requires_grad_(True)
        traces = []
        for b in calib_ids[:n_score_batches]:
            # X: the tensor entering this layer's c_fc (ln_2(h)), from a real forward pass of
            # `reader` on batch `b` (section 5's capture_mlp_input) -- not a hand-computed ln_2(h).
            # c_proj's input is NOT captured: it is produced inside compute_mlp from whatever W_fc
            # the Hutchinson probe currently holds, which is exactly what makes the trace joint.
            X = capture_mlp_input(reader, layer, b, DEVICE, which="c_fc")
            with torch.no_grad():
                # Target comes from this layer's own float weights, so the loss is exactly 0 at
                # w = w_true and the trace is the curvature of the reconstruction loss there.
                t_out = compute_mlp(*W32, X, cfg, b_fc=b32[0], b_proj=b32[1], act=act)

            def loss_fn(p):
                return mlp_loss(p, X, t_out, cfg, b_fc=b32[0], b_proj=b32[1], act=act)

            traces.append(hutchinson_trace_estimator(loss_fn, w_flat, samples=samples).item())
            free(X, t_out)
        trace = sum(traces) / len(traces)
        free(w_flat)

        # --- Hessians, ONLY if the perturbation heuristic actually needs them ---
        # grid_perturbation ignores H in the default "rtn" mode. The QKV notebook collected its
        # single Hessian unconditionally; doing that here would mean TWO full 128-batch collection
        # passes per layer (c_fc's and c_proj's) for a tensor nothing reads, roughly doubling this
        # pass's wall clock. Numerically identical either way -- collect_hessian_via_hook only
        # reads.
        H_fc = H_proj = None
        if perturb_mode == "gptq":
            H_fc = collect_hessian_via_hook(reader, layer.mlp.c_fc,
                                            calib_ids[:n_hessian_batches], DEVICE)
            H_proj = collect_hessian_via_hook(reader, layer.mlp.c_proj,
                                              calib_ids[:n_hessian_batches], DEVICE)

        # --- HAWQ-V2 table: Omega_u(bits) = trace_layer * ||Q(W_u) - W_u||_F^2, PER UNIT ---
        for which, W, H in (("c_fc", W32[0], H_fc), ("c_proj", W32[1], H_proj)):
            name = names[which]
            jab_scores[name] = trace
            table = {bits: trace * grid_perturbation(W, bits, group_size=group_size, H=H,
                                                     mode=perturb_mode)
                     for bits in bit_widths}
            allocator_input[name] = table
            if verbose:
                level = "HIGH" if trace > 100 else "MEDIUM" if trace > 10 else "LOW"
                print(f"  {name}: trace={trace:.4f} [{level}]  " +
                      ", ".join(f"{b}b={v:.3e}" for b, v in table.items()))

        del H_fc, H_proj, W32          # del, not free(...) -- see free()'s docstring
        free()

        if verbose:
            print(f"    [layer {i} scored in {time.time() - t_layer:.1f}s]")

        del layer
        free()

    vram("after scoring pass")
    return jab_scores, allocator_input


def pass_quantize_eval(reader, config, cfg, calib_ids, eval_fn, *, bits_fn=None,
                       finetune=False, n_hessian_batches=None, group_size=None, lambda_kl=None,
                       steps_per_block=None, lr=None, grad_clip_norm=None, layer_indices=None,
                       label="", verbose=True, return_notes=False):
    """
    Quantize-and-evaluate pass. SIGNATURE UNCHANGED from the QKV notebook. `reader` MUST be a
    freshly-loaded model whenever `bits_fn` is not None -- quantization mutates `mlp.c_fc.weight`
    and `mlp.c_proj.weight` in place, so an already-quantized `reader` would have this arm
    quantize already-quantized matrices (see load_model_and_tokenizer's docstring).

    eval_fn: zero-argument callable returning the perplexity to report once every layer has been
        processed, e.g. `lambda: evaluate_perplexity(reader, tokenizer, ...)`. Factored out (rather
        than calling evaluate_perplexity directly in here) purely so section 7's tiny-model
        validation can inject a synthetic evaluator that needs no network access -- the real
        pipeline always passes a closure over evaluate_perplexity.

    bits_fn: None -> leave weights untouched (the control); else `(layer_idx, which) -> bit-width`,
        where `which` is "c_fc" or "c_proj". A legacy 1-argument `layer_idx -> bits` callable is
        also accepted and applied to both matrices (see _resolve_bits_fn above).
    lambda_kl: accepted for signature compatibility, IGNORED (warned about once) -- the MLP has no
        attention distribution to take a KL against. See mlp_loss, section 2.
    finetune: additionally run the STE joint fine-tune of Objective 1 on each layer, against a
        float-weight teacher applied to the SAME input the student sees (GPTQ's own convention:
        target = MLP_float(X_quant) -- there is no separate float hidden-state trajectory here).
        The fine-tune is JOINT over both matrices: one w_flat, one loss, one commit decision.
    layer_indices: restrict fine-tuning to these layers; the rest still get the GPTQ warm start.
    return_notes: if True, also return {layer_idx: ((bits_fc, bits_proj), committed)} where
        `committed` is True if a fine-tuned weight beat the GPTQ-only starting point and was
        written back, False if GPTQ-only was kept, None if the layer had no bits_fn/finetune
        outcome to record. (The QKV notebook's `bits` scalar becomes a PAIR here -- c_fc and
        c_proj are separate allocator units and need not share a width.)

    Returns perplexity, or (perplexity, notes) if return_notes.
    """
    n_hessian_batches = HESSIAN_N_BATCHES if n_hessian_batches is None else n_hessian_batches
    group_size = GROUP_SIZE if group_size is None else group_size
    lambda_kl = LAMBDA_KL if lambda_kl is None else lambda_kl
    steps_per_block = JOINT_STEPS_PER_BLOCK if steps_per_block is None else steps_per_block
    lr = JOINT_LR if lr is None else lr
    grad_clip_norm = JOINT_GRAD_CLIP if grad_clip_norm is None else grad_clip_norm
    tune = None if layer_indices is None else set(layer_indices)
    warn_if_lambda_kl(lambda_kl, "pass_quantize_eval")
    bits_of = _resolve_bits_fn(bits_fn)

    print(f"\n=== pass: {label} ===")
    reset_vram_peak()

    notes = {}
    flip_rates = {}
    flip_rates_by_unit = {}
    for i in range(cfg.n_layers):
        t_layer = time.time()                     # unattended-run instrumentation, see run.sh
        layer = load_decoder_layer(reader, config, i)
        note = ""
        note_committed = None

        # (1) float snapshot, BEFORE either module is quantized. The biases are fixed (never a
        # GPTQ/STE target) but still snapshotted alongside the weights, since both feed the
        # teacher's compute_mlp call below. `act` is the layer's own module, not a global -- see
        # the comment at the top of section 2 for why that matters.
        W_float32 = None
        if finetune:
            W_float32 = tuple(W.clone() for W in mlp_weights_io(layer, dtype=torch.float32))
            b_fc, b_proj = mlp_bias_io(layer, dtype=torch.float32)
            act = layer.mlp.act

        if bits_of is not None:
            bits_fc, bits_proj = bits_of(i, "c_fc"), bits_of(i, "c_proj")
            want_scales = finetune and (tune is None or i in tune)
            # (2) GPTQ warm start, c_fc then c_proj. `reader` is quantized layer by layer in place,
            # and collect_hessian_via_hook runs a REAL forward pass through the current (possibly
            # partially-quantized) `reader` -- so layer i's Hessians automatically reflect layers
            # 0..i-1 already being quantized, and, inside quantize_mlp, c_proj's Hessian reflects
            # c_fc already being quantized too. Both orderings are the sequential-GPTQ behaviour.
            out = quantize_mlp(reader, layer, (bits_fc, bits_proj),
                               calib_ids[:n_hessian_batches], group_size=group_size,
                               return_scales=want_scales, return_compact=want_scales)
            note = f"c_fc {bits_fc}b / c_proj {bits_proj}b"

            # (3) STE joint fine-tune against the float reference
            if want_scales:
                (W_fc_q, s_fc, sc_fc, g_fc), (W_proj_q, s_proj, sc_proj, g_proj) = out
                # w_flat is seeded from GPTQ's own fp32 output and scale_flat is GPTQ's own grid,
                # so ste_quantize_mlp is a no-op at step 0 (checked in section 7).
                w_flat = flatten_weights(W_fc_q, W_proj_q).to(torch.float32)
                w_flat = w_flat.clone().requires_grad_(True)
                gptq_w = w_flat.detach().clone()          # diagnostic B: pre-fine-tune grid point
                scale_flat = flatten_weights(s_fc, s_proj).to(w_flat.dtype)   # frozen GPTQ grid
                # one learnable scale per (output row, group) per MATRIX -- NOT one per weight: a
                # per-weight scale makes q_i*s_i free precision, not bits-bit quantization. Two
                # separate parameters because c_fc's is (intermediate, n_groups_768) and c_proj's
                # is (hidden, n_groups_3072) -- different shapes, and act_order permutes each
                # matrix's columns independently so they cannot share a g_idx either.
                sp_fc = sc_fc.clone().to(torch.float32).requires_grad_(True)
                sp_proj = sc_proj.clone().to(torch.float32).requires_grad_(True)
                n_weights = group_size                    # weights sharing each group's scale
                del out, W_fc_q, W_proj_q, s_fc, s_proj
                free()
                lr_w = ALPHA_W * scale_flat.mean().item()
                lr_s = ALPHA_S * scale_flat.mean().item()
                opt = torch.optim.Adam([{"params": [w_flat], "lr": lr_w},
                                        {"params": [sp_fc, sp_proj], "lr": lr_s}])
                sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_block)

                def quantized(w, s_fc_p, s_proj_p):
                    return ste_quantize_mlp(w, s_fc_p, g_fc, s_proj_p, g_proj,
                                            bits_fc, bits_proj, n_weights, cfg)

                # held_out is reserved out of the cycling pool entirely (not just the next index
                # after `order`) so it stays held out even when steps_per_block > len(calib_ids)-1
                # and `order` wraps around the whole pool via modulo.
                held_out = len(calib_ids) - 1
                pool = list(range(len(calib_ids) - 1))
                start = (i * steps_per_block) % len(pool)
                order = [pool[(start + s) % len(pool)] for s in range(steps_per_block)]

                def targets(j):
                    # Teacher and student see the SAME input (GPTQ's own convention: target =
                    # MLP_float(X_quant)). A separate float-hidden-state trajectory would let the
                    # residual include upstream quantization error this layer cannot fix. X comes
                    # from a real forward pass of `reader` (capture_mlp_input, section 5), so it
                    # reflects layers 0..i-1's CURRENT (already-quantized) state, same as (2)
                    # above. Only c_fc's input is captured -- the teacher runs its OWN float W_fc
                    # to get c_proj's input, which is the whole point of a joint two-matrix target.
                    X = capture_mlp_input(reader, layer, calib_ids[j], DEVICE, which="c_fc")
                    with torch.no_grad():
                        t_out = compute_mlp(*W_float32, X, cfg, b_fc=b_fc, b_proj=b_proj, act=act)
                    return X, t_out

                # X depends only on layers 0..i-1 and this layer's (unquantized) attention, never
                # on either MLP weight, so it is frozen for layer i's whole FT -- compute each
                # distinct batch's targets once (capture_mlp_input is a full model forward) and
                # reuse them instead of recomputing every step.
                batch_cache = {j: targets(j) for j in set(order) | {held_out}}

                def score(w, s_fc_p, s_proj_p):
                    with torch.no_grad():
                        X_ho, t_ho = batch_cache[held_out]
                        return mlp_loss(quantized(w, s_fc_p, s_proj_p), X_ho, t_ho, cfg,
                                        b_fc=b_fc, b_proj=b_proj, act=act).item()

                init_loss = score(w_flat.detach(), sp_fc.detach(), sp_proj.detach())
                best_loss, best_w = init_loss, w_flat.detach().clone()
                for step_idx, j in enumerate(order):
                    X, t_out = batch_cache[j]
                    opt.zero_grad()
                    loss = mlp_loss(quantized(w_flat, sp_fc, sp_proj), X, t_out, cfg,
                                    b_fc=b_fc, b_proj=b_proj, act=act,
                                    log_components=LOG_LOSS_COMPONENTS)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([w_flat, sp_fc, sp_proj], grad_clip_norm)
                    opt.step()
                    sched.step()
                    # score every SCORE_EVERY steps (always on the last) instead of every step --
                    # held-out targets are cached above, so this is just the loss/quantize forward.
                    if step_idx == steps_per_block - 1 or (step_idx + 1) % SCORE_EVERY == 0:
                        step_loss = score(w_flat.detach(), sp_fc.detach(), sp_proj.detach())
                        if step_loss < best_loss:
                            best_loss = step_loss
                            with torch.no_grad():
                                best_w = quantized(w_flat, sp_fc, sp_proj).detach().clone()
                    free(loss)
                del batch_cache
                free()

                # diagnostic B: fraction of grid positions fine-tuning actually moved, regardless
                # of whether the result was committed -- against the FROZEN GPTQ grid (scale_flat).
                # Reported jointly AND per matrix: c_fc and c_proj sit at different depths in the
                # block, see different activation statistics (pre- vs. post-GELU) and routinely get
                # different bit-widths, so a single pooled number can hide one of them being
                # completely inert.
                with torch.no_grad():
                    flips = (torch.round(gptq_w / scale_flat) !=
                             torch.round(best_w / scale_flat)).float()
                    flip_rate = flips.mean().item()
                    fr_fc = flips[:cfg.fc_numel].mean().item()
                    fr_proj = flips[cfg.fc_numel:].mean().item()
                    del flips

                # Safety net: commit only if fine-tuning beat the GPTQ-only starting point.
                # Otherwise the modules already hold that GPTQ-only solution -- leave them alone.
                rates = f"flip rate {flip_rate:.3f} (c_fc {fr_fc:.3f}, c_proj {fr_proj:.3f})"
                if best_loss <= init_loss:
                    write_mlp_flat(layer, best_w, cfg)
                    note += f", FT {init_loss:.6f} -> {best_loss:.6f}, {rates}"
                    note_committed = True
                else:
                    note += f", kept GPTQ-only ({init_loss:.6f} vs FT {best_loss:.6f}), {rates}"
                    note_committed = False
                flip_rates[i] = flip_rate
                flip_rates_by_unit[unit_name(i, "c_fc")] = fr_fc
                flip_rates_by_unit[unit_name(i, "c_proj")] = fr_proj
                del w_flat, gptq_w, scale_flat, sp_fc, sp_proj, best_w, opt, sched
                del targets, score, quantized
                free()
            elif finetune:
                note += ", GPTQ-only (not in layer_indices)"
        del W_float32
        free()

        del layer
        free()

        if bits_of is not None:
            notes[i] = ((bits_fc, bits_proj), note_committed)
        if verbose:
            print(f"  layer {i:>2}/{cfg.n_layers} [{time.time() - t_layer:6.1f}s]: "
                  f"{note or 'fp32 (no quantization)'}")

    globals()["LAST_FLIP_RATES"] = flip_rates
    globals()["LAST_FLIP_RATES_BY_UNIT"] = flip_rates_by_unit
    ppl = eval_fn()
    vram(f"after '{label}'")
    print(f"=== {label}: perplexity {ppl:.3f} ===")
    return (ppl, notes) if return_notes else ppl


print("Validation 7c (deferred from section 7 -- needs pass_quantize_eval, defined just above): "
      "adaptive allocation + joint fine-tuning, mixed 2/3/4/8-bit (tiny CPU model)...")
_validate_adaptive_joint_pass()
print("PASSED -- pass_quantize_eval(bits_fn=..., finetune=True) is bit-width-agnostic end to "
      "end, and routes c_fc/c_proj bit-widths independently.")


# =============================================================================================
# 9. Setup
# =============================================================================================
section("## 9. Setup")

_t_setup = time.time()
print(f"Loading {MODEL_ID}'s tokenizer and config (cached after the first run)...")
# Only the tokenizer and config are loaded here, NOT a model -- every arm in sections 10-15 loads
# its OWN fresh model (see load_model_and_tokenizer's docstring, section 5) right before it
# quantizes anything, so there is no shared, mutable `reader` for one arm's damage to leak into
# the next. The tokenizer is safe to share across every arm: it has no weights, nothing here ever
# mutates it.
tokenizer = GPT2TokenizerFast.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(MODEL_ID)
config._attn_implementation = "eager"

# Renamed from the QKV notebook's `ATTN` -- this notebook's config carries the MLP shapes too, and
# the name should not suggest the quantization target is attention. It is not; attention stays
# fp32 throughout.
BLOCK = BlockConfig(config)
print(BLOCK)
assert BLOCK.hidden_size % GROUP_SIZE == 0, (
    f"GROUP_SIZE={GROUP_SIZE} must divide hidden_size={BLOCK.hidden_size} -- true for gpt2 (768), "
    f"gpt2-medium (1024) and gpt2-large (1280), but NOT gpt2-xl (1600)")
# c_proj's input dimension is intermediate_size, not hidden_size, so GROUP_SIZE has to divide BOTH.
# It does for every public GPT-2 (3072 / 4096 / 5120 are all multiples of 128), but the assertion
# is what makes that a checked fact rather than an assumption -- a group size that divides 768 but
# not 3072 would leave c_proj with a ragged final group, which gptq_quantize_layer handles but
# grid_perturbation's allocator heuristic would then score on a differently-shaped grid.
assert BLOCK.intermediate_size % GROUP_SIZE == 0, (
    f"GROUP_SIZE={GROUP_SIZE} must ALSO divide intermediate_size={BLOCK.intermediate_size} -- "
    f"c_proj's input dimension is the intermediate size, not the hidden size")
print(f"MLP parameters per layer: {BLOCK.mlp_numel:,} "
      f"(c_fc {BLOCK.fc_numel:,} + c_proj {BLOCK.proj_numel:,})")
print(f"  for reference, the QKV notebook's target was {BLOCK.qkv_numel:,} per layer "
      f"({BLOCK.mlp_numel / BLOCK.qkv_numel:.2f}x smaller) -- see section 14b")
print(f"Allocator units: {BLOCK.n_layers} layers x {len(UNIT_KINDS)} matrices = "
      f"{BLOCK.n_layers * len(UNIT_KINDS)}")

print("\nBuilding the shared calibration set (used for every arm below)...")
calib_ids = build_calibration_ids(tokenizer)
print(f"{len(calib_ids)} calibration chunks of {CALIB_SEQ_LEN} tokens")
print("Evaluation is the FULL WikiText-2 test set, scored per-arm by evaluate_perplexity "
      f"(max_length={EVAL_MAX_LENGTH}, stride={EVAL_STRIDE}) -- not a precomputed window list.")

# `results` is the notebook's plain {arm: perplexity} dict that sections 13c/13d/14/14b read.
# On a resumed run it is pre-populated from results.json, so the comparison tables at the end
# still show arms that were completed by an earlier invocation.
results = {}
for _name, _rec in RESULTS.get("arms", {}).items():
    if isinstance(_rec.get("perplexity"), (int, float)):
        results[_name] = _rec["perplexity"]
if results:
    print(f"\nResumed from results.json: {', '.join(sorted(results))}")

RESULTS["meta"] = {
    "model_id": MODEL_ID,
    "seed": ARGS.seed,
    "started": datetime.now().isoformat(timespec="seconds"),
    "cuda_visible_devices": _VISIBLE_DEVICES,
    "gpu_inherited_from_env": _GPU_INHERITED,
    "gpu_name": torch.cuda.get_device_name(0),
    "gpu_capability": list(torch.cuda.get_device_capability(0)),
    "torch": torch.__version__,
    "torch_cuda": torch.version.cuda,
    "tf32_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
    "argv": sys.argv,
    "config": {
        "CALIB_N_SAMPLES": CALIB_N_SAMPLES, "CALIB_SEQ_LEN": CALIB_SEQ_LEN,
        "HESSIAN_N_BATCHES": HESSIAN_N_BATCHES, "HUTCH_SAMPLES": HUTCH_SAMPLES,
        "JAB_N_BATCHES": JAB_N_BATCHES, "LAMBDA_KL": LAMBDA_KL, "GROUP_SIZE": GROUP_SIZE,
        "PERTURB_MODE": PERTURB_MODE, "BIT_WIDTHS": BIT_WIDTHS,
        "EVAL_MAX_LENGTH": EVAL_MAX_LENGTH, "EVAL_STRIDE": EVAL_STRIDE,
        "JOINT_STEPS_PER_BLOCK": JOINT_STEPS_PER_BLOCK, "JOINT_LR": JOINT_LR,
        "JOINT_GRAD_CLIP": JOINT_GRAD_CLIP, "JOINT_LAYERS": JOINT_LAYERS,
        "ALPHA_W": ALPHA_W, "ALPHA_S": ALPHA_S, "SCORE_EVERY": SCORE_EVERY,
        "PROPAGATION_WEIGHT": PROPAGATION_WEIGHT,
        "RUN_ORACLE_CRITERION": RUN_ORACLE_CRITERION, "SWEEP_BITS": SWEEP_BITS,
        "TARGET_AVG_BITS": TARGET_AVG_BITS,
        "JOINT_LAYERS_DEPTH_TEST": list(JOINT_LAYERS_DEPTH_TEST),
        "GPU_DTYPE": str(GPU_DTYPE), "GPTQ_DTYPE": str(GPTQ_DTYPE),
    },
}
flush_results()
print(f"\n[section 9 setup: {time.time() - _t_setup:.1f}s]")


# =============================================================================================
# 10. fp32 control (validation 7d)
# =============================================================================================
section("## 10. fp32 control (validation 7d)")

if RUN_FP16_BASELINE:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # fresh, unmutated by any prior arm
    results["fp16"] = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=None, label="fp32 control (no quantization)")
    # Full WikiText-2 test, gpt2/-medium/-large: expect roughly 18-30. Wide range because this
    # also has to tolerate MODEL_ID changing (gpt2-large scores noticeably lower than gpt2) --
    # this just catches a broken pipeline, not a precise calibration. The control is identical to
    # the QKV notebook's (nothing is quantized in either), so the two notebooks' fp32 numbers must
    # agree; if they do not, the difference is in the pipeline, not in the target module.
    if not (15.0 < results["fp16"] < 60.0):
        print(f"\nWARNING: fp32 control perplexity {results['fp16']:.3f} is outside the expected "
              f"~18-30 range for {MODEL_ID} on WikiText-2. Investigate the pipeline before "
              f"reading anything into the quantization results below.")
    record_arm("fp16", perplexity=results["fp16"], elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("RUN_FP16_BASELINE=False -- skipping the control pass.")


# =============================================================================================
# 11. Uniform 4-bit GPTQ baseline
# =============================================================================================
section("## 11. Uniform 4-bit GPTQ baseline")

if RUN_UNIFORM4:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # fresh, unmutated by any prior arm
    # `lambda i: 4` is the QKV notebook's bits_fn verbatim -- a 1-argument callable.
    # _resolve_bits_fn (section 8) accepts it and applies 4 bits to both c_fc and c_proj, so this
    # arm needed no edit.
    results["uniform4"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=lambda i: 4, label="uniform 4-bit GPTQ (MLP)", return_notes=True)
    record_arm("uniform4", perplexity=results["uniform4"], notes=_notes,
               bits={unit_name(i, w): 4 for i in range(BLOCK.n_layers) for w in UNIT_KINDS},
               elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("uniform4 not selected -- skipping.")


# =============================================================================================
# 12. JAB-Hessian adaptive allocation
# =============================================================================================
section("## 12. JAB-Hessian adaptive allocation")

assignment = None
jab_scores = allocator_input = None

if NEED_SCORING:
    _cached = None if ARGS.rerun else RESULTS.get("scoring")
    if _cached:
        # Reload rather than re-score. JSON turns the allocator table's integer bit-width keys
        # into strings, so they are converted back here; nothing else about the table changes.
        print("Reusing the cached section-12 scoring pass from results.json "
              f"(scored in {_cached.get('elapsed_s', float('nan')):.1f}s by an earlier run). "
              "Pass --rerun to re-score.")
        jab_scores = {n: float(v) for n, v in _cached["jab_scores"].items()}
        allocator_input = {n: {int(b): float(v) for b, v in t.items()}
                           for n, t in _cached["allocator_input"].items()}
        assignment = {n: int(b) for n, b in _cached["assignment"].items()}
        TARGET_AVG_BITS = float(_cached.get("target_avg_bits", TARGET_AVG_BITS))
    else:
        _t_score = time.time()
        reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # fresh, unquantized -- see
                                                                  # pass_score's docstring for why
                                                                  # scoring needs this
        torch.manual_seed(ARGS.seed)     # notebook had a hardcoded 42; now driven by --seed, so
                                          # the Hutchinson probes are reproducible run to run
        jab_scores, allocator_input = pass_score(reader, config, BLOCK, calib_ids)
        del reader
        free()

        # len(allocator_input) is now 2 * n_layers (24 for gpt2), and `budget` is still
        # "average bits per MATRIX" -- valid because c_fc and c_proj hold identical parameter
        # counts (4 * hidden**2 each). See `cost`'s docstring, section 4.
        budget = TARGET_AVG_BITS * len(allocator_input)
        assignment, cost_used, sensitivity = greedy_allocate(allocator_input, budget)
        print(f"\nBudget {budget:.1f} bits ({TARGET_AVG_BITS} avg x {len(allocator_input)} units)")
        print(f"Allocated: {cost_used:.1f} bits total, {cost_used / len(allocator_input):.2f} "
              f"avg, total sensitivity {sensitivity:.4e}")
        RESULTS["scoring"] = {
            "jab_scores": jab_scores,
            "allocator_input": allocator_input,
            "assignment": assignment,
            "target_avg_bits": TARGET_AVG_BITS,
            "budget": budget,
            "cost_used": cost_used,
            "total_sensitivity": sensitivity,
            "elapsed_s": round(time.time() - _t_score, 2),
        }
        flush_results()
        print(f"[section 12 scoring pass: {(time.time() - _t_score) / 60:.1f} min, "
              f"checkpointed to results.json]")

    # diagnostic A: assignment histogram, overall and split by matrix kind -- c_fc and c_proj sit
    # at different depths in the block and may attract systematically different budgets, which a
    # single pooled histogram would hide.
    hist = Counter(assignment.values())
    print("Assignment histogram (bits -> unit count):", dict(sorted(hist.items())))
    for which in UNIT_KINDS:
        h = Counter(b for n, b in assignment.items() if unit_kind_of(n) == which)
        mean_b = sum(b for n, b in assignment.items() if unit_kind_of(n) == which) / BLOCK.n_layers
        print(f"  {which:>7}: {dict(sorted(h.items()))}  (mean {mean_b:.2f} bits)")

    print("Per-layer assignment (layer-index order):")
    for i in range(BLOCK.n_layers):
        print(f"  layer {i:>2}: c_fc {assignment[unit_name(i, 'c_fc')]:>2} bits, "
              f"c_proj {assignment[unit_name(i, 'c_proj')]:>2} bits")

    print("\nBudget sensitivity sweep (free -- reuses the scores above):")
    for avg_bits in [3.5, 4.0, 4.3, 4.5, 6.0]:
        alloc, c, _ = greedy_allocate(allocator_input, avg_bits * len(allocator_input))
        print(f"  avg_bits={avg_bits}: cost {c:.1f}, distinct widths {sorted(set(alloc.values()))}")
else:
    print("no scoring-dependent arm selected -- skipping the scoring pass.")

if RUN_ADAPTIVE:
    _t_arm = time.time()
    # The notebook reused the SAME `reader` the scoring pass ran against, noting that pass_score
    # never mutates weights. A fresh load is used here instead so this arm works identically
    # whether the scoring pass just ran or was reloaded from results.json on a resume;
    # from_pretrained is deterministic, so the two are the same model.
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)
    results["adaptive"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=lambda i, which: assignment[unit_name(i, which)],
        label=f"JAB adaptive ({TARGET_AVG_BITS} avg bits/matrix)", return_notes=True)
    record_arm("adaptive", perplexity=results["adaptive"], notes=_notes, bits=assignment,
               elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("adaptive not selected -- skipping.")


# =============================================================================================
# 12b. Budget sweep: JAB / JAB+propagation / oracle criteria
# =============================================================================================
section("## 12b. Budget sweep: JAB / JAB+propagation / oracle criteria")


@torch.no_grad()
def measure_end_to_end_sensitivity(model_loader, calib_ids, bit_widths, n_chunks=8):
    """Oracle criterion: quantize ONLY (unit, bits) on a fresh model, mean KL(fp32 || quantized)
    logits over n_chunks. Same {unit_name: {bits: score}} shape as the JAB table.

    TWICE the QKV notebook's cost -- O(n_layers * 2 * len(bit_widths)) model loads -- because c_fc
    and c_proj are separate allocator units and each has to be measured with the OTHER left in
    fp32. That isolation is the point of an oracle: quantizing both at once would measure the pair,
    not the unit. Gated behind RUN_ORACLE_CRITERION (section 0).

    Note that `H` here is collected from an otherwise-FLOAT model for both units, including
    c_proj's. That is correct for this criterion and deliberately different from `quantize_mlp`'s
    sequential rule: the oracle asks "what does quantizing this one matrix alone cost?", so the
    rest of the model -- c_fc included -- must stay float.
    """
    fp32_model, _ = model_loader(MODEL_ID, DEVICE)
    chunks = calib_ids[:n_chunks]
    fp32_probs = [F.softmax(fp32_model(c.to(DEVICE)).logits, dim=-1).cpu() for c in chunks]
    n_layers = fp32_model.config.n_layer
    del fp32_model
    free()

    scores = {}
    for i in range(n_layers):
        for which in UNIT_KINDS:
            name = unit_name(i, which)
            scores[name] = {}
            h_model, _ = model_loader(MODEL_ID, DEVICE)      # H doesn't depend on bits -- one
            target = mlp_of(h_model.transformer.h[i])[UNIT_KINDS.index(which)]   # collection per
            H = collect_hessian_via_hook(h_model, target,                        # unit
                                         calib_ids[:HESSIAN_N_BATCHES], DEVICE)
            del h_model, target
            free()
            for bits in bit_widths:
                m, _ = model_loader(MODEL_ID, DEVICE)
                mod = mlp_of(m.transformer.h[i])[UNIT_KINDS.index(which)]
                gptq_quantize_conv1d(mod, H, bits=bits, group_size=GROUP_SIZE)
                kls = []
                for c, p_fp32 in zip(chunks, fp32_probs):
                    log_q = F.log_softmax(m(c.to(DEVICE)).logits, dim=-1)
                    kls.append(F.kl_div(log_q, p_fp32.to(DEVICE), reduction="batchmean").item())
                scores[name][bits] = sum(kls) / len(kls)
                del m, mod
                free()
            del H
            free()
    return scores


def apply_propagation_weight(allocator_input, n_layers):
    """Unchanged -- `layer_idx_of` still parses the layer index out of a unit name, so both of a
    layer's units get that layer's depth weight."""
    if not PROPAGATION_WEIGHT:
        return allocator_input
    return {n: {b: v * (n_layers - layer_idx_of(n)) for b, v in t.items()}
            for n, t in allocator_input.items()}


sweep_results, sweep_notes, sweep_flip_rates, sweep_unit_flip_rates = {}, {}, {}, {}
sweep_allocs = {}     # key -> {unit_name: bits}, kept for section 14b's compression table

# The sweep is one arm but 40-ish full quantize+evaluate passes. It is therefore checkpointed
# after EVERY sub-pass rather than only at the end: an interrupted sweep resumes from the last
# completed key instead of restarting. This is the only place the port adds checkpointing inside
# an arm rather than around it.
_sweep_done = {} if ARGS.rerun else dict((RESULTS.get("arms", {}).get("sweep") or {})
                                         .get("sub_results", {}) or {})
for _k, _v in _sweep_done.items():
    sweep_results[_k] = _v
_sweep_prev = (RESULTS.get("arms", {}).get("sweep") or {}) if not ARGS.rerun else {}
sweep_allocs.update(_sweep_prev.get("allocs", {}) or {})
# JSON stringifies the per-layer flip-rate dicts' integer keys; put them back so section 14's
# `sorted(rates.items())` orders layers numerically rather than lexicographically on a resume.
sweep_flip_rates.update({k: {int(i): r for i, r in v.items()}
                         for k, v in (_sweep_prev.get("flip_rates", {}) or {}).items()})
sweep_unit_flip_rates.update(_sweep_prev.get("unit_flip_rates", {}) or {})
sweep_notes.update(_sweep_prev.get("notes", {}) or {})


def _checkpoint_sweep(elapsed):
    record_arm("sweep", perplexity=None, elapsed_s=elapsed, extra={
        "sub_results": sweep_results,
        "allocs": sweep_allocs,
        "notes": sweep_notes,
        "flip_rates": sweep_flip_rates,
        "unit_flip_rates": sweep_unit_flip_rates,
    })


if RUN_SWEEP and assignment is not None:
    _t_arm = time.time()
    if _sweep_done:
        print(f"Resuming the sweep: {len(_sweep_done)} sub-pass(es) already in results.json "
              f"({', '.join(sorted(_sweep_done))})")

    criteria = {"jab": allocator_input,
                "jab_prop": apply_propagation_weight(allocator_input, BLOCK.n_layers)}
    if RUN_ORACLE_CRITERION:
        criteria["oracle"] = measure_end_to_end_sensitivity(load_model_and_tokenizer, calib_ids,
                                                            BIT_WIDTHS)

    for crit_name, table in criteria.items():
        for avg_bits in SWEEP_BITS:
            alloc, _, _ = greedy_allocate(table, avg_bits * len(table))
            for ft in (False, True):
                key = f"{crit_name}_{avg_bits}_{'ft' if ft else 'gptq'}"
                if key in sweep_results:
                    print(f"  [skip] sweep sub-pass '{key}': already in results.json")
                    continue
                reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)
                sweep_allocs[key] = dict(alloc)
                out = pass_quantize_eval(
                    reader, config, BLOCK, calib_ids,
                    lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                                stride=EVAL_STRIDE),
                    bits_fn=lambda i, which: alloc[unit_name(i, which)], finetune=ft,
                    layer_indices=JOINT_LAYERS, return_notes=ft,
                    label=f"{crit_name} avg={avg_bits} {'FT' if ft else 'GPTQ-only'}")
                if ft:
                    sweep_results[key], sweep_notes[key] = out
                    sweep_flip_rates[key] = dict(globals().get("LAST_FLIP_RATES", {}))
                    sweep_unit_flip_rates[key] = dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {}))
                else:
                    sweep_results[key] = out
                del reader
                free()
                _checkpoint_sweep(time.time() - _t_arm)

    for bits in (2, 3):
        for ft in (False, True):
            key = f"uniform_{bits}_{'ft' if ft else 'gptq'}"
            if key in sweep_results:
                print(f"  [skip] sweep sub-pass '{key}': already in results.json")
                continue
            reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)
            sweep_allocs[key] = {unit_name(i, w): bits
                                 for i in range(BLOCK.n_layers) for w in UNIT_KINDS}
            out = pass_quantize_eval(
                reader, config, BLOCK, calib_ids,
                lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                            stride=EVAL_STRIDE),
                bits_fn=lambda i: bits, finetune=ft, layer_indices=JOINT_LAYERS, return_notes=ft,
                label=f"uniform {bits}-bit {'FT' if ft else 'GPTQ-only'}")
            if ft:
                sweep_results[key], sweep_notes[key] = out
                sweep_flip_rates[key] = dict(globals().get("LAST_FLIP_RATES", {}))
                sweep_unit_flip_rates[key] = dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {}))
            else:
                sweep_results[key] = out
            del reader
            free()
            _checkpoint_sweep(time.time() - _t_arm)

    _checkpoint_sweep(time.time() - _t_arm)
elif RUN_SWEEP:
    print("sweep selected but no `assignment` (the scoring pass did not run) -- skipping.")
else:
    print("sweep not selected -- skipping.")


# =============================================================================================
# 13. Joint MLP-aware fine-tuning (Objective 1)
#
# STE fine-tuning of `c_fc` and `c_proj` JOINTLY -- one concatenated [W_fc | W_proj] parameter
# vector, one MSE loss on the MLP's output, one commit decision -- against a float-weight teacher
# applied to the SAME input the student sees (see `pass_quantize_eval`'s docstring, section 8).
# Every checkpoint -- the untrained starting point and the weights after each optimizer step -- is
# scored on one fixed held-out calibration batch, evaluated strictly after `opt.step()`; a
# fine-tuned result is written back only if it beats the untrained starting point on that batch.
#
# Two differences from the QKV notebook's Objective 1, both consequences of the target module and
# neither a change of method:
#   - The loss is MSE only. `attention_loss`'s lambda_kl * KL(attention maps) term has no
#     analogue -- the MLP emits no distribution (see `mlp_loss`, section 2).
#   - The gradient has to travel through GELU and through `c_fc` to reach `W_fc`. `c_proj`'s error
#     signal is immediate; `c_fc`'s passes through the activation's saturating tails, so the two
#     matrices can have very different flip rates under the same schedule. That is why the
#     flip-rate diagnostic is reported per matrix as well as pooled.
# =============================================================================================
section("## 13. Joint MLP-aware fine-tuning (Objective 1)")

if RUN_JOINT_FINETUNE:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # own fresh model
    results["joint"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=lambda i: 4, finetune=True, layer_indices=JOINT_LAYERS,
        label="joint MLP-aware fine-tuned (4-bit)", return_notes=True)
    print("\nPer-matrix flip rate for this arm (c_fc must travel through GELU to get a gradient, "
          "c_proj does not):")
    for name, rate in sorted(globals().get("LAST_FLIP_RATES_BY_UNIT", {}).items(),
                             key=lambda kv: (layer_idx_of(kv[0]), unit_kind_of(kv[0]))):
        print(f"  {name:<20}: {rate:.3f}")
    record_arm("joint", perplexity=results["joint"], notes=_notes,
               bits={unit_name(i, w): 4 for i in range(BLOCK.n_layers) for w in UNIT_KINDS},
               flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {})),
               elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("RUN_JOINT_FINETUNE=False -- skipping Objective 1.")


# =============================================================================================
# 13b. Adaptive allocation + joint fine-tuning
# =============================================================================================
section("## 13b. Adaptive allocation + joint fine-tuning")

if RUN_ADAPTIVE_JOINT and assignment is not None:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # own fresh model
    results["adaptive_joint"], adaptive_joint_notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=lambda i, which: assignment[unit_name(i, which)], finetune=True,
        layer_indices=JOINT_LAYERS,
        label=f"JAB adaptive + joint fine-tuned ({TARGET_AVG_BITS} avg bits/matrix)",
        return_notes=True)

    # notes now carry a (bits_fc, bits_proj) PAIR per layer, and the commit decision is joint over
    # both matrices -- so the outcome is tallied per PAIR, not per bit-width.
    per_bits = {}   # (bits_fc, bits_proj) -> [fine-tuned count, GPTQ-only count]
    for bits_pair, committed in adaptive_joint_notes.values():
        counts = per_bits.setdefault(tuple(bits_pair), [0, 0])
        counts[0 if committed else 1] += 1
    print()
    print("Per-(c_fc, c_proj) fine-tune outcome (adaptive_joint):")
    for bits_pair in sorted(per_bits):
        ft, gptq_only = per_bits[bits_pair]
        print(f"  c_fc {bits_pair[0]}b / c_proj {bits_pair[1]}b: {ft} fine-tuned, "
              f"{gptq_only} GPTQ-only ({ft + gptq_only} layers)")

    unit_rates = globals().get("LAST_FLIP_RATES_BY_UNIT", {})
    if unit_rates:
        print("\nMean flip rate by matrix:")
        for which in UNIT_KINDS:
            rs = [r for n, r in unit_rates.items() if unit_kind_of(n) == which]
            print(f"  {which:>7}: {sum(rs) / len(rs):.3f}")
    record_arm("adaptive_joint", perplexity=results["adaptive_joint"],
               notes=adaptive_joint_notes, bits=assignment,
               flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(unit_rates), elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("RUN_ADAPTIVE_JOINT=False or no `assignment` (section 12 skipped) -- "
          "skipping the adaptive+joint pass.")


# =============================================================================================
# 13c. Depth-scaling diagnostic (Objective 1D)
#
# If damage from joint fine-tuning compounds with the number of fine-tuned layers, restricting
# fine-tuning to a handful of layers (JOINT_LAYERS_DEPTH_TEST, the first four) should land closer
# to the GPTQ-only `adaptive` perplexity than the full-depth `adaptive_joint` run does. Reuses the
# same adaptive `assignment` from section 12; does not re-run allocation or scoring.
#
# Worth watching more closely here than in the QKV notebook: the MLP carries 2.67x the parameters,
# so each fine-tuned layer is a larger intervention, and any compounding should show up
# correspondingly sooner.
# =============================================================================================
section("## 13c. Depth-scaling diagnostic (Objective 1D)")

if RUN_DEPTH_SCALING_TEST and assignment is not None:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # own fresh model
    n_ft_layers = len(list(JOINT_LAYERS_DEPTH_TEST))
    results["adaptive_joint_depth4"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=lambda i, which: assignment[unit_name(i, which)], finetune=True,
        layer_indices=JOINT_LAYERS_DEPTH_TEST,
        label=f"JAB adaptive + joint fine-tuned, first {n_ft_layers} layers only "
              f"({TARGET_AVG_BITS} avg bits/matrix)", return_notes=True)
    print(f"\nDepth-scaling check ({n_ft_layers}-layer fine-tune vs. full-depth vs. GPTQ-only):")
    for key, desc in (("adaptive_joint_depth4", f"{n_ft_layers}-layer FT"),
                      ("adaptive_joint", "full-depth FT"),
                      ("adaptive", "GPTQ-only")):
        if key in results:
            print(f"  {desc:<16}: {results[key]:.3f}")
    record_arm("adaptive_joint_depth4", perplexity=results["adaptive_joint_depth4"],
               notes=_notes, bits=assignment,
               flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {})),
               elapsed_s=time.time() - _t_arm,
               extra={"ft_layers": list(JOINT_LAYERS_DEPTH_TEST)})
    del reader
    free()
else:
    print("RUN_DEPTH_SCALING_TEST=False or no `assignment` -- skipping depth-scaling diagnostic.")


# =============================================================================================
# 13d. KL ablation (lambda_kl=0.0) -- A NO-OP IN THE MLP SETTING
#
# This arm is expected to reproduce 13b exactly, and is kept only so the section numbering matches
# the QKV notebook. Read it as a regression check, not as an experiment.
#
# In the QKV notebook this arm mattered: attention_loss was MSE + lambda_kl * KL(attention maps),
# and because F.kl_div(reduction="batchmean") divides only by batch size while F.mse_loss averages
# over every element, the raw KL term ran several orders of magnitude above its nominal lambda_kl
# weight. Setting lambda_kl=0.0 there genuinely changed the objective.
#
# Here there is no KL term to ablate. The MLP emits no post-softmax distribution, so mlp_loss is
# MSE-only by construction and LAMBDA_KL is already 0.0 (section 0). Passing lambda_kl=0.0
# explicitly below therefore changes nothing, and this arm should land on the same perplexity as
# 13b -- which is exactly what makes it worth keeping: if the two disagree, something in
# pass_quantize_eval is reading lambda_kl when it should not, or the fine-tune is nondeterministic
# in a way the other diagnostics assume it is not.
#
# Nothing was substituted for the dropped KL. A KL on some other quantity -- final logits, say --
# would have changed the objective at the same time as the target module, and this experiment
# exists to isolate the target module alone. See `mlp_loss`, section 2.
# =============================================================================================
section("## 13d. KL ablation (lambda_kl=0.0) -- a no-op in the MLP setting")

if RUN_KL_ABLATION_TEST and assignment is not None:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID, DEVICE)   # own fresh model
    results["adaptive_joint_kl0"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids,
        lambda: evaluate_perplexity(reader, tokenizer, max_length=EVAL_MAX_LENGTH,
                                    stride=EVAL_STRIDE),
        bits_fn=lambda i, which: assignment[unit_name(i, which)], finetune=True,
        layer_indices=JOINT_LAYERS, lambda_kl=0.0,
        label=f"JAB adaptive + joint fine-tuned, MSE-only ({TARGET_AVG_BITS} avg bits/matrix)",
        return_notes=True)
    print("\nKL ablation (expected to be a NO-OP -- the MLP loss is MSE-only already):")
    # Both FT arms carry lambda_kl=0.0 here -- 13b inherits it from LAMBDA_KL, 13d passes it
    # explicitly -- so label them by section rather than by a value that is the same either way.
    for key, desc in (("adaptive_joint_kl0", "13d explicit 0.0"),
                      ("adaptive_joint", f"13b default {LAMBDA_KL}"),
                      ("adaptive", "GPTQ-only")):
        if key in results:
            print(f"  {desc:<16}: {results[key]:.3f}")
    if "adaptive_joint" in results:
        delta = abs(results["adaptive_joint_kl0"] - results["adaptive_joint"])
        # 1e-3 rather than exact equality: the two arms are the same computation on the same
        # inputs, but they are two separate GPU runs and non-deterministic reduction order
        # can move perplexity in the last few digits. Anything larger is a real discrepancy.
        if delta > 1e-3:
            print(f"\nWARNING: 13d differs from 13b by {delta:.3e}. In this notebook the two arms "
                  f"are the SAME objective (mlp_loss ignores lambda_kl), so they should be "
                  f"identical. Either pass_quantize_eval is reading lambda_kl somewhere it should "
                  f"not, or the fine-tune is nondeterministic -- both worth knowing before "
                  f"reading anything into the other diagnostics.")
        else:
            print(f"\n13d matches 13b to {delta:.1e} -- confirms lambda_kl is inert, as "
                  f"intended.")
    record_arm("adaptive_joint_kl0", perplexity=results["adaptive_joint_kl0"], notes=_notes,
               bits=assignment, flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {})),
               elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("RUN_KL_ABLATION_TEST=False or no `assignment` -- skipping KL ablation diagnostic.")


# =============================================================================================
# 14. Compare results
# =============================================================================================
section("## 14. Compare results")

CRIT_LABELS = {"uniform": "uniform", "jab": "JAB", "jab_prop": "JAB+prop", "oracle": "oracle"}

print(f"{MODEL_ID}  ({BLOCK.n_layers} layers, hidden {BLOCK.hidden_size}, "
      f"intermediate {BLOCK.intermediate_size}, {BLOCK.num_heads} heads)")
print("Quantized: mlp.c_fc + mlp.c_proj.  fp32: attn.c_attn, attn.c_proj, LayerNorms, embeddings.")
print(f"'avg bits' below is per MATRIX ({BLOCK.n_layers * len(UNIT_KINDS)} units), not per "
      f"layer -- see section 14b for the parameter-weighted view and the QKV-notebook-comparable "
      f"numbers.")
if "fp16" in results:
    print(f"fp32 control: {results['fp16']:.3f}\n")


def lookup(crit, avg_bits, ft):
    if crit == "uniform" and avg_bits == 4:
        return results.get("joint" if ft else "uniform4")
    key = f"{crit}_{avg_bits}_{'ft' if ft else 'gptq'}"
    return sweep_results.get(key)


col_keys = [(c, ft) for c in ("uniform", "jab", "jab_prop", "oracle") for ft in (False, True)]
rows = sorted(set([2, 3] + SWEEP_BITS))
header = f"{'avg bits':>10}" + "".join(f"{CRIT_LABELS[c] + ('/ft' if ft else '/gptq'):>16}"
                                       for c, ft in col_keys)
print(header)
for avg_bits in rows:
    row = f"{avg_bits:>10}"
    for c, ft in col_keys:
        v = lookup(c, avg_bits, ft)
        row += f"{v:>16.3f}" if v is not None else f"{'--':>16}"
    print(row)

print("\nPer-layer flip rate (fraction of grid positions moved by FT, pooled over both matrices):")
for key, rates in sweep_flip_rates.items():
    if rates:
        mean_rate = sum(rates.values()) / len(rates)
        print(f"  {key:<24}: mean={mean_rate:.3f}  " +
              ", ".join(f"L{i}={r:.3f}" for i, r in sorted(rates.items())))

print("\nFlip rate by matrix (mean over layers) -- c_fc's gradient crosses GELU, c_proj's does "
      "not, so these can differ a lot:")
for key, rates in sweep_unit_flip_rates.items():
    if rates:
        parts = []
        for which in UNIT_KINDS:
            rs = [r for n, r in rates.items() if unit_kind_of(n) == which]
            if rs:
                parts.append(f"{which}={sum(rs) / len(rs):.3f}")
        print(f"  {key:<24}: " + ", ".join(parts))


# =============================================================================================
# 14b. Compression, and how to read this against the QKV notebook
#
# "4 bits" does not mean the same thing in the two notebooks. The MLP holds 2.67x the parameters
# the fused QKV matrix does -- 56.6 M vs. 21.2 M for gpt2, or 66.6% vs. 25.0% of all
# non-embedding weight parameters -- so an arm labelled "4 avg bits" here compresses the model far
# more aggressively than the identically-labelled arm there. Comparing the two perplexity tables
# row by row, as though the row label were a shared x-axis, would systematically flatter the QKV
# notebook.
#
# The fp16 reference is a convention, not a claim about this run: everything here computes in
# fp32 (see GPU_DTYPE, section 0). It is used because fp16 is the standard storage baseline in the
# quantization literature and in the QKV notebook's framing.
# =============================================================================================
section("## 14b. Compression, and how to read this against the QKV notebook")

REF_BITS = 16      # storage baseline for the compression ratio -- see the section header above

# Non-embedding weight parameters per layer, counted from the shapes rather than hard-coded so
# this stays correct for gpt2-medium / -large.
_PER_LAYER = {
    "attn.c_attn": BLOCK.qkv_numel,                        # 768 x 2304
    "attn.c_proj": BLOCK.hidden_size * BLOCK.hidden_size,  # 768 x 768
    "mlp.c_fc":    BLOCK.fc_numel,                         # 768 x 3072
    "mlp.c_proj":  BLOCK.proj_numel,                       # 3072 x 768
}
NONEMB_PARAMS = sum(_PER_LAYER.values()) * BLOCK.n_layers
MLP_PARAMS    = (_PER_LAYER["mlp.c_fc"] + _PER_LAYER["mlp.c_proj"]) * BLOCK.n_layers
QKV_PARAMS    = _PER_LAYER["attn.c_attn"] * BLOCK.n_layers
UNIT_PARAMS   = {"c_fc": _PER_LAYER["mlp.c_fc"], "c_proj": _PER_LAYER["mlp.c_proj"]}

print(f"{MODEL_ID}: {NONEMB_PARAMS:,} non-embedding WEIGHT parameters "
      f"(bias vectors, LayerNorm gains/biases and the wte/wpe embeddings all excluded -- none of "
      f"them is a quantization target in either notebook)")
print(f"  this notebook quantizes the MLP:    {MLP_PARAMS:>12,} "
      f"({100 * MLP_PARAMS / NONEMB_PARAMS:.1f}%)")
print(f"  the QKV notebook quantizes c_attn: {QKV_PARAMS:>12,} "
      f"({100 * QKV_PARAMS / NONEMB_PARAMS:.1f}%)  -- "
      f"{MLP_PARAMS / QKV_PARAMS:.2f}x smaller\n")


def arm_compression(bits_map):
    """
    bits_map: {unit_name: bits} for every quantized unit, or a plain int for a uniform arm.

    Returns (quantized_params, param_weighted_avg_bits, compression_ratio_vs_fp16,
             qkv_equivalent_bits).

    qkv_equivalent_bits is the average bit-width the QKV notebook would need on its c_attn
    matrices to reach the SAME total model size. It can come out below zero, which is not a bug:
    the MLP is 2.67x larger, so past a fairly shallow budget no QKV allocation -- not even a
    hypothetical 0-bit one -- saves as many bits. `mlp_bits_matching_qkv` below reads the same
    relationship in the other direction, which is the direction that is always defined.
    """
    if isinstance(bits_map, int):
        bits_map = {unit_name(i, w): bits_map
                    for i in range(BLOCK.n_layers) for w in UNIT_KINDS}
    q_params = sum(UNIT_PARAMS[unit_kind_of(n)] for n in bits_map)
    q_bits = sum(UNIT_PARAMS[unit_kind_of(n)] * b for n, b in bits_map.items())
    avg_bits = q_bits / q_params
    total_bits = q_bits + (NONEMB_PARAMS - q_params) * REF_BITS
    ratio = (NONEMB_PARAMS * REF_BITS) / total_bits
    saved = q_params * (REF_BITS - avg_bits)
    qkv_equiv = REF_BITS - saved / QKV_PARAMS
    return q_params, avg_bits, ratio, qkv_equiv


def mlp_bits_matching_qkv(qkv_bits):
    """The MLP average bit-width that gives the same total model size as `qkv_bits` on c_attn.
    Always defined and always above `qkv_bits`, since the MLP is the larger target."""
    return REF_BITS - QKV_PARAMS * (REF_BITS - qkv_bits) / MLP_PARAMS


arms = []
if "uniform4" in results:
    arms.append(("uniform 4-bit (sec 11)", 4, results["uniform4"]))
if "joint" in results:
    arms.append(("uniform 4-bit + FT (sec 13)", 4, results["joint"]))
if assignment is not None:
    for key, label in (("adaptive", f"JAB adaptive {TARGET_AVG_BITS} (sec 12)"),
                       ("adaptive_joint", f"JAB adaptive {TARGET_AVG_BITS} + FT (sec 13b)"),
                       ("adaptive_joint_depth4", f"JAB adaptive {TARGET_AVG_BITS} + FT x4 (13c)"),
                       ("adaptive_joint_kl0", f"JAB adaptive {TARGET_AVG_BITS} + FT, kl0 (13d)")):
        if key in results:
            arms.append((label, assignment, results[key]))
for key in sorted(sweep_allocs):
    if key in sweep_results:
        arms.append((f"sweep: {key}", sweep_allocs[key], sweep_results[key]))

hdr = (f"{'arm':<40}{'q-params':>12}{'avg bits':>10}{'compress':>10}"
       f"{'QKV-equiv':>13}{'ppl':>9}")
print(hdr)
print("-" * len(hdr))
for label, bits_map, ppl in arms:
    q_params, avg_bits, ratio, qkv_equiv = arm_compression(bits_map)
    equiv = f"{qkv_equiv:.2f}b" if qkv_equiv >= 0 else "unreachable"
    print(f"{label:<40}{q_params:>12,}{avg_bits:>10.2f}{ratio:>9.2f}x{equiv:>13}{ppl:>9.3f}")

print(f"\n'QKV-equiv': the average bit-width the QKV notebook would have to run c_attn at to reach")
print(f"this arm's total model size. 'unreachable' means it cannot -- c_attn is only "
      f"{100 * QKV_PARAMS / NONEMB_PARAMS:.0f}% of the")
print(f"non-embedding weights against the MLP's {100 * MLP_PARAMS / NONEMB_PARAMS:.0f}%, so even "
      f"0-bit QKV leaves the model larger than a")
print(f"moderately quantized MLP does. Expect this for most rows; that IS the headline.")

# The same relationship read the other way round, which is always defined and is what you actually
# need to line the two tables up: given a row label in the QKV notebook, which row HERE is the
# size-matched comparison?
print(f"\nRow-matching guide -- a QKV-notebook arm at N bits costs the same total model size as an")
print(f"MLP arm at:")
print(f"    {'QKV notebook':>14} {'->':^4} {'this notebook':<14}")
for b in sorted(set([2, 3, 4] + [x for x in SWEEP_BITS])):
    print(f"    {b:>13}b {'->':^4} {mlp_bits_matching_qkv(b):>12.2f}b")
print(f"  i.e. this notebook's rows are NOT comparable to the QKV notebook's at the same label;")
print(f"  its 4-bit arm is size-matched to an MLP arm at roughly "
      f"{mlp_bits_matching_qkv(4):.1f} bits, far above")
print(f"  anything in the sweep. Every arm in this notebook is the more aggressive of the two.")


# =============================================================================================
# done
# =============================================================================================
RESULTS["meta"]["finished"] = datetime.now().isoformat(timespec="seconds")
RESULTS["meta"]["total_elapsed_s"] = round(time.time() - _T_START, 2)
flush_results()
section(f"## run complete -- {(time.time() - _T_START) / 60:.1f} min total")
print(f"results  {RESULTS_PATH}")
print(f"log      {_LOG_PATH}")
for _name in ARM_ORDER:
    _rec = RESULTS.get("arms", {}).get(_name)
    if not _rec:
        print(f"  {_name:<24} --")
        continue
    _ppl = _rec.get("perplexity")
    _el = _rec.get("elapsed_s")
    _ppl_s = f"{_ppl:.3f}" if isinstance(_ppl, (int, float)) else "(sub-results)"
    _el_s = f"{_el / 60:.1f} min" if isinstance(_el, (int, float)) else "--"
    print(f"  {_name:<24} ppl {_ppl_s:>14}   {_el_s:>10}")
