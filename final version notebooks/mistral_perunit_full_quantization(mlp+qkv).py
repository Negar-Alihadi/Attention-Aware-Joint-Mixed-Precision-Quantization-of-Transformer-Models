#!/usr/bin/env python3
"""
run_mistral_perunit.py -- JAB-Hessian sensitivity estimation & adaptive precision allocation for
Mistral-7B-v0.1, quantizing ALL SEVEN linear modules of every decoder block
(`self_attn.{q,k,v,o}_proj` + `mlp.{gate,up,down}_proj`).

Direct descendant of run_mlp_quant.py (GPT-2, mlp.c_fc + mlp.c_proj). The GPTQ core, the
Hutchinson estimator, the greedy allocator, the checkpoint/resume/signal machinery and the CLI are
carried over; what does NOT carry over is everything model-specific, and the differences are
load-bearing rather than cosmetic:

  * WEIGHT ORIENTATION IS INVERTED. GPT-2's `Conv1D` stores `(d_in, d_out)`; Mistral's `nn.Linear`
    stores `(d_out, d_in)`. run_mlp_quant.py's `gptq_quantize_conv1d` ASSERTED the GPT-2
    orientation and rejected its transpose -- that assertion is exactly backwards here. Orientation
    is now an explicit per-model flag (`WEIGHT_LAYOUT`, threaded into `gptq_quantize_module` as a
    required keyword), never an assumption. Getting it backwards quantizes along the output axis
    instead of the input axis and raises nothing.
  * SEVEN UNITS, FOUR HESSIANS. q/k/v share one input (`input_layernorm(h)`), gate/up share another
    (`post_attention_layernorm(h)`); `o_proj` and `down_proj` have their own. See `quantize_block`.
  * PARAMETER-WEIGHTED COST. Unit sizes span 14x (k_proj 4.2 M vs. gate_proj 58.7 M), so `cost`
    takes a real parameter count and every budget in this file is BITS PER PARAMETER, never bits
    per matrix.
  * NO ORACLE CRITERION. `measure_end_to_end_sensitivity` cost ~73 full model loads in the GPT-2
    script; at 7B that is infeasible and it is gone, not merely defaulted off.
  * PER-UNIT HESSIAN TRACES. `hutchinson_block_traces` splits each Rademacher probe's
    `v .* Hv` product along the seven units' coordinate ranges instead of summing it, so every
    unit gets its own unbiased Tr(H_uu) at the SAME cost as the single pooled trace
    run_mistral_quant.py computed (one HVP per probe). Summing the seven recovers that pooled
    number exactly, which is the one assertion the scoring pass keeps.
  * HARD BIT FLOOR. `--bit-floor` (default 3) removes every candidate width below the floor from
    BIT_WIDTHS before any sensitivity table is built, so no allocator, clamp or uniform arm can
    reach a 2-bit assignment. Arms and sweep budgets that are infeasible under the floor are
    skipped with a printed line rather than rounded up silently.
  * PER-BLOCK OFFLOADING. The model stays on CPU and exactly one decoder layer is resident on the
    GPU at a time. run_mlp_quant.py's hooks-on-the-whole-live-model Hessian collection assumed the
    whole model fits on the card; 28 GB of fp32 weights plus a 1.6 GB fp64 `down_proj` Hessian does
    not.
"""

# --- stdlib only -------------------------------------------
import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime


# GPU 0 is in heavy use by another user on this machine. It is never selected, not even as a
# fallback: if the requested card is unavailable the run aborts rather than moving to a
# neighbour's GPU.
FORBIDDEN_GPU = 0
DEFAULT_GPU = 1

ARM_ORDER = [
    "fp16",                      # sec 10  -- fp32 control (no quantization)
    "uniform4",                  # sec 11  -- uniform 4-bit GPTQ, all seven units
    "adaptive",                  # sec 12  -- JAB adaptive allocation (needs the scoring pass)
    "sweep",                     # sec 12b -- budget sweep: jab / jab_norm / type_offset / uniform
    "joint",                     # sec 13  -- joint block-aware fine-tuning, uniform 4-bit
    "adaptive_joint",            # sec 13b -- adaptive allocation + joint fine-tuning
    "adaptive_joint_depth4",     # sec 13c -- depth-scaling diagnostic
    "adaptive_joint_kl0",        # sec 13d -- KL ablation (a REAL ablation here; see 13d)
]

# The seven quantized units per decoder block, in the order they are quantized. This order is the
# sequential-GPTQ order and is not a free choice: o_proj's input is the merged attention output
# (so q/k/v must already be quantized), and down_proj's input is silu(gate(x))*up(x) (so gate/up
# must already be quantized). See `quantize_block`, section 1.
UNIT_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
ATTN_KINDS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_KINDS = ("gate_proj", "up_proj", "down_proj")
KIND_SHORT = {"q_proj": "q", "k_proj": "k", "v_proj": "v", "o_proj": "o",
              "gate_proj": "gate", "up_proj": "up", "down_proj": "down"}
SHORT_KIND = {v: k for k, v in KIND_SHORT.items()}

# The four Hessians per block, and which units read each one.
#   x1 : input_layernorm(h)                -> q_proj, k_proj, v_proj   (4096)
#   A  : merged attention output           -> o_proj                   (4096)
#   x2 : post_attention_layernorm(h)       -> gate_proj, up_proj       (4096)
#   M  : silu(gate(x2)) * up(x2)           -> down_proj                (14336)
HESSIAN_TAGS = ("x1", "A", "x2", "M")
KIND_HESSIAN = {"q_proj": "x1", "k_proj": "x1", "v_proj": "x1", "o_proj": "A",
                "gate_proj": "x2", "up_proj": "x2", "down_proj": "M"}

DEFAULT_TYPE_OFFSETS = {"q_proj": 0, "k_proj": 0, "v_proj": 0, "o_proj": 0,
                        "gate_proj": 1, "up_proj": 1, "down_proj": -1}


def _csv_ints(s):
    return [int(x) for x in str(s).split(",") if x.strip() != ""]


def _csv_numbers(s):
    """
    Comma-separated numbers, keeping integral values as ints.

    This matters and is not cosmetic: section 12b builds its result keys by f-stringing the sweep
    values (`jab_3_gptq`). Parsing everything as float would produce `jab_3.0_gptq` while section
    14's row lookup would still ask for `jab_3_gptq`, and every affected cell in the comparison
    table would silently print '--'.
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
    """`--joint-layers`: a comma-separated layer list, or 'none' for every layer."""
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


def _type_offsets(s):
    """
    '--type-offsets q:0,k:0,v:0,o:0,gate:+1,up:+1,down:-1' -> {unit_kind: int offset}.

    Short names (the KIND_SHORT keys) are accepted as well as full module names, because the full
    ones are what appear in unit names and the short ones are what anybody types. Unspecified kinds
    keep their DEFAULT_TYPE_OFFSETS value rather than silently becoming 0.
    """
    out = dict(DEFAULT_TYPE_OFFSETS)
    for item in str(s).split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise argparse.ArgumentTypeError(f"expected kind:offset pairs, got {item!r}")
        k, v = item.split(":", 1)
        k = SHORT_KIND.get(k.strip(), k.strip())
        if k not in UNIT_KINDS:
            raise argparse.ArgumentTypeError(
                f"unknown unit kind {k!r}; expected one of {sorted(SHORT_KIND)} or {UNIT_KINDS}")
        out[k] = int(v)
    return out


def build_parser():
    p = argparse.ArgumentParser(
        prog="run_mistral_perunit.py",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="JAB-Hessian adaptive precision allocation for Mistral-7B, all seven linear "
                    "modules per decoder block.")

    # --- run control ---
    g = p.add_argument_group("run control")
    g.add_argument("--model-id", default="mistralai/Mistral-7B-v0.1",
                   help="any Mistral-architecture checkpoint; the section 9 assertions pin the "
                        "shapes this script was written against (4096 / 14336 / 32 layers)")
    g.add_argument("--out-dir", default="runs-mistral",
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
                   help="permit --gpu %d, which is otherwise refused because it is in heavy use "
                        "by another user on this machine" % FORBIDDEN_GPU)
    g.add_argument("--min-free-gb", type=float, default=40.0,
                   help="startup guard: abort if the card has less free memory than this. Raised "
                        "far above the GPT-2 script's footprint -- one resident decoder block, its "
                        "float teacher copy, Adam state and a 1.6 GB fp64 down_proj Hessian plus "
                        "its inverse workspace all have to fit at once")

    # --- fp64 solve placement (section 0a) ---
    g = p.add_argument_group("section 0a: fp64 benchmark / GPTQ solve placement")
    g.add_argument("--gptq-device", choices=["auto", "cuda", "cpu"], default="auto",
                   help="where the float64 GPTQ solve runs. 'auto' picks whichever of GPU/CPU wins "
                        "the startup benchmark. The DTYPE is never negotiable -- only the device")
    g.add_argument("--bench-dim", type=int, default=14336,
                   help="size of the startup float64 solve benchmark; defaults to down_proj's "
                        "input dimension, the largest solve this run will do")
    g.add_argument("--skip-fp64-bench", action="store_true",
                   help="skip the startup benchmark. Requires an explicit --gptq-device, since "
                        "'auto' has nothing to decide on without it")

    # --- section 0 constants -----------------------------------------------------------------
    # GPU_DTYPE (fp32) and GPTQ_DTYPE (float64) are deliberately absent: both are documented bug
    # fixes inherited from the GPT-2 script, not tuning knobs.
    g = p.add_argument_group("section 0: calibration / scoring")
    g.add_argument("--calib-samples", type=int, default=128, metavar="CALIB_N_SAMPLES")
    g.add_argument("--calib-seq-len", type=int, default=512, metavar="CALIB_SEQ_LEN")
    g.add_argument("--hessian-batches", type=int, default=128, metavar="HESSIAN_N_BATCHES",
                   help="batches used for H = 2 X^T X -- ALL of them, not a subset")
    g.add_argument("--hutch-samples", type=int, default=30, metavar="HUTCH_SAMPLES",
                   help="Hutchinson probes per (layer, batch). Raised from the pooled estimator's "
                        "10: the per-unit traces carry the cross-block variance that cancels when "
                        "the seven partial sums are added back together, so each individual "
                        "Tr(H_uu) needs more probes than the pooled trace did for the same "
                        "stability. Still one HVP per probe")
    g.add_argument("--jab-batches", type=int, default=2, metavar="JAB_N_BATCHES")
    g.add_argument("--err-batches", type=int, default=8, metavar="ERR_N_BATCHES",
                   help="batches averaged into eps_A / eps_M (section 4 of the brief). A subset: "
                        "the float teacher block has to be run alongside the student for each one")
    g.add_argument("--lambda-kl", type=float, default=0.1, metavar="LAMBDA_KL",
                   help="weight on the attention-map KL term of the joint loss. LIVE here, unlike "
                        "the GPT-2 MLP script where it had no analogue")
    g.add_argument("--group-size", type=int, default=128, metavar="GROUP_SIZE")
    g.add_argument("--perturb-mode", choices=["rtn", "gptq"], default="rtn",
                   metavar="PERTURB_MODE")
    g.add_argument("--bit-widths", type=_csv_ints, default=[2, 3, 4, 8, 16], metavar="BIT_WIDTHS")
    g.add_argument("--bit-floor", type=int, default=3, metavar="BIT_FLOOR",
                   help="hard minimum width per unit: every candidate below it is dropped from "
                        "BIT_WIDTHS before any sensitivity table is built, so the greedy MCKP "
                        "walk, the type_offset clamp and the uniform arms cannot reach it. A "
                        "uniform arm or a sweep budget below the floor is INFEASIBLE and is "
                        "skipped with a printed line, never rounded up. 0 disables the floor")

    g = p.add_argument_group("section 0: evaluation")
    g.add_argument("--eval-windows", type=int, default=32, metavar="N_EVAL_WINDOWS",
                   help="WikiText-2 test windows scored per arm -- a SUBSET, labelled as such "
                        "everywhere it is reported")
    g.add_argument("--eval-seq-len", type=int, default=512, metavar="EVAL_SEQ_LEN")

    g = p.add_argument_group("section 0: joint fine-tuning")
    g.add_argument("--steps-per-block", type=int, default=200, metavar="JOINT_STEPS_PER_BLOCK")
    g.add_argument("--joint-lr", type=float, default=1e-4, metavar="JOINT_LR")
    g.add_argument("--grad-clip", type=float, default=1.0, metavar="JOINT_GRAD_CLIP")
    g.add_argument("--joint-layers", type=_opt_csv_ints, default=None, metavar="JOINT_LAYERS",
                   help="restrict fine-tuning to these layers; 'none' for every layer")
    g.add_argument("--alpha-w", type=float, default=0.05, metavar="ALPHA_W",
                   help="eta_w = ALPHA_W * s, with s the mean grid step of THAT UNIT -- never "
                        "pooled across units; see the per-unit-step discussion in section 8")
    g.add_argument("--alpha-s", type=float, default=0.02, metavar="ALPHA_S")
    g.add_argument("--score-every", type=int, default=10, metavar="SCORE_EVERY")
    g.add_argument("--ft-pool", type=int, default=8, metavar="FT_POOL",
                   help="distinct calibration batches cycled through the fine-tune, plus one "
                        "held-out. Each cached batch costs a block-output AND an attention-map "
                        "teacher target (~41 MB at 512 tokens), so the GPT-2 script's "
                        "cache-every-batch cannot be carried over unchanged")

    g = p.add_argument_group("section 0: allocation / sweep")
    g.add_argument("--target-avg-bits", type=float, default=4.3, metavar="TARGET_AVG_BITS",
                   help="BITS PER PARAMETER, not per matrix -- unit sizes span 14x here")
    g.add_argument("--propagation-weight", type=_bool_flag, default=True,
                   metavar="PROPAGATION_WEIGHT")
    g.add_argument("--sweep-bits", type=_csv_numbers, default=[3.0, 3.5, 4.5],
                   metavar="SWEEP_BITS",
                   help="integer budgets collapse every criterion onto the same uniform "
                        "assignment and yielded nothing on GPT-2, so the default is the half-bit "
                        "grid where the criteria can actually differ. 2.5 is gone from the "
                        "default because it is unreachable under --bit-floor 3: the minimum "
                        "width already costs 3 bits/param")
    g.add_argument("--type-offsets", type=_type_offsets, default=dict(DEFAULT_TYPE_OFFSETS),
                   metavar="TYPE_OFFSETS",
                   help="per-unit-kind bit offsets for the 'type_offset' criterion, e.g. "
                        "'gate:+1,up:+1,down:-1'")
    g.add_argument("--type-offset-ft", type=_bool_flag, default=False,
                   help="also run the fine-tuned variant of the type_offset criterion in the "
                        "sweep. Off by default: type_offset is specified as ONE evaluation per "
                        "budget (it has no scoring pass to amortize)")
    g.add_argument("--depth-test-layers", type=int, default=4,
                   metavar="JOINT_LAYERS_DEPTH_TEST",
                   help="13c fine-tunes range(0, N) only")

    g = p.add_argument_group("section 0: diagnostics")
    g.add_argument("--log-loss-components", type=_bool_flag, default=False,
                   metavar="LOG_LOSS_COMPONENTS")
    return p


ARGS = build_parser().parse_args()

if ARGS.skip_fp64_bench and ARGS.gptq_device == "auto":
    raise SystemExit("--skip-fp64-bench needs an explicit --gptq-device cuda|cpu: 'auto' is "
                     "decided BY the benchmark, so skipping it leaves nothing to decide on.")

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
# All stdlib. Runs before torch is imported; see the ORDERING CONTRACT in the module docstring.

class _Tee:
    """
    Mirror every write to stdout AND the run's log file.

    Deliberately a stdout tee rather than a conversion of the several hundred `print(...)` calls
    into logging calls: the printed output IS the result record, including its section banners and
    per-layer lines.
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
    Guard for the ORDERING CONTRACT in the module docstring.

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
            f"machine. Pick another card with --gpu, or pass --allow-gpu0 if you really mean it.")
    if args.gpu < 0:
        raise SystemExit(f"--gpu must be a non-negative device index, got {args.gpu}")
    return str(args.gpu), False


RESULTS_PATH = None
RESULTS = {"meta": {}, "scoring": None, "arms": {}}


def _jsonable(obj):
    """Make the result objects JSON-safe: tuple keys, int keys, numpy/torch scalars."""
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
    """Single gate for every arm: --arms selection AND resume-from-results.json, with the reason
    printed."""
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
               unit_flip_rates=None, elapsed_s=None, metrics=None, extra=None):
    """
    Checkpoint one arm into results.json, immediately. A crash in a later arm must not lose the
    ones already finished, and this is what makes the resume path above work.

    `metrics` is new in this script: the eps_W / eps_A / eps_M / rho_attn / rho_mlp record built by
    `pass_quantize_eval` (section 8).
    """
    rec = {
        "perplexity": perplexity,
        "bits": bits,
        # Realized minimum assigned width, so --bit-floor is auditable from results.json alone
        # rather than by re-deriving it from `bits`. `bit_floor` is what was asked for; `min_bits`
        # is what the arm actually came out at.
        "bit_floor": BIT_FLOOR,
        "min_bits": None if not bits else min(bits.values()),
        "notes": notes,
        "flip_rates": flip_rates,
        "unit_flip_rates": unit_flip_rates,
        "metrics": metrics,
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
        print(f"  [checkpoint] arm '{arm}' written to results.json ({elapsed_s / 60:.1f} min)")


def section(title):
    """Echo a section header into the log."""
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
print("run_mistral_perunit.py -- JAB-Hessian adaptive precision allocation, Mistral-7B, "
      "all 7 linear modules")
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
_FLOOR_BANNER = [b for b in ARGS.bit_widths if ARGS.bit_floor <= 0 or b >= ARGS.bit_floor]
print(f"bit floor{'':<1} {ARGS.bit_floor} bits/unit"
      f"{' (DISABLED)' if ARGS.bit_floor <= 0 else ''};  candidate widths "
      f"{_FLOOR_BANNER} (from --bit-widths {ARGS.bit_widths})")
print("scoring  per-unit Hutchinson traces (per-unit-v2), "
      f"{ARGS.hutch_samples} probes x {ARGS.jab_batches} batches")
print("=" * 96)

# =============================================================================================
# torch imports -- everything below this line may import torch. See the ORDERING CONTRACT in the
# module docstring for why nothing above it does.
# =============================================================================================
import copy
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

# --- TF32 off, before any CUDA tensor is created ---------------------------------------------
# Modern cards use TF32 for fp32 matmuls by default, and TF32's 10-bit mantissa is FEWER mantissa
# bits than the fp16 the GPT-2 script's section 0 blames for non-monotone perplexity-vs-bit-width.
# It would silently degrade every Hessian accumulation, every teacher target and every eps_A /
# eps_M measurement in this file.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")

# --- build / device report (printed, not asserted) -------------------------------------------
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
CPU = "cpu"
_free_b, _total_b = torch.cuda.mem_get_info(0)
print(f"  device name          : {torch.cuda.get_device_name(0)}")
print(f"  physical index       : {_VISIBLE_DEVICES}"
      f"{' (inherited from environment)' if _GPU_INHERITED else ''}  ->  {DEVICE}")
print(f"  compute capability   : sm_{''.join(str(x) for x in torch.cuda.get_device_capability(0))}")
print(f"  memory               : {_free_b / 1e9:.1f} GB free / {_total_b / 1e9:.1f} GB total")

# --- startup free-memory guard ---------------------------------------------------------------
# The GPT-2 script had no such guard: a 124 M-parameter model and a 3072x3072 fp64 Hessian fit
# anywhere. Here one resident fp32 decoder block (0.87 GB), its float teacher copy, the STE latent
# weights plus Adam moments (~2.6 GB), a 14336x14336 fp64 down_proj Hessian (1.64 GB) and the
# inverse's workspace are live simultaneously, and an OOM 20 hours into a sweep is expensive.
if _free_b / 1e9 < ARGS.min_free_gb:
    raise SystemExit(
        f"only {_free_b / 1e9:.1f} GB free on {torch.cuda.get_device_name(0)}, need "
        f"{ARGS.min_free_gb:.0f} GB. Either the card is busy or --min-free-gb is set too high for "
        f"it; this run would OOM partway through a block rather than at startup.")

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
# 0a. float64 solve benchmark -- RUNS BEFORE ANYTHING ELSE
#
# `down_proj`'s GPTQ column loop is 14336 iterations deep in float64, against GPT-2's 3072, and
# every iteration divides by a float64 Hessian-inverse entry. Before that the whole (14336, 14336)
# Hessian has to be inverted in float64. On a card whose fp64 throughput is a small fraction of its
# fp32 throughput -- which is every consumer and most workstation parts -- that single operation
# can be slower on the GPU than on the host CPU, and it is not obvious in advance which way it
# falls on any particular machine.
#
# So it is measured rather than assumed. The DTYPE IS NOT NEGOTIABLE: float64 is a documented bug
# fix inherited from the GPT-2 script (float32 turned the sequential column loop into something
# numerically indistinguishable from round-to-nearest). Only the DEVICE is chosen here.
# =============================================================================================
section("## 0a. float64 solve benchmark (before anything else)")


def _bench_fp64_solve(n, device):
    """
    Time ONE float64 (n, n) inverse -- the operation `gptq_quantize_layer` does once per unit, and
    the one that dominates `down_proj`. Returns seconds, or None if it could not be run (out of
    memory is the expected reason on a busy card, and is not fatal: it just means that device
    loses).

    The matrix is made diagonally dominant rather than merely random so the inverse is
    well-conditioned and LAPACK/cuSOLVER does not spend its time on pivoting pathologies that the
    real (damped, symmetric positive-definite) Hessians never present.
    """
    try:
        g = torch.Generator(device="cpu").manual_seed(0)
        A = (torch.randn(n, n, generator=g, dtype=torch.float64) * 0.01)
        A.diagonal().add_(float(n))
        A = A.to(device)
        if device != CPU:
            torch.cuda.synchronize()
        t0 = time.time()
        Ainv = torch.linalg.inv(A)
        if device != CPU:
            torch.cuda.synchronize()
        dt = time.time() - t0
        del A, Ainv
        gc.collect()
        torch.cuda.empty_cache()
        return dt
    except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
        print(f"  {device}: float64 solve failed ({type(e).__name__}: {e})")
        gc.collect()
        torch.cuda.empty_cache()
        return None


_BENCH = {"dim": ARGS.bench_dim, "gpu_s": None, "cpu_s": None, "skipped": ARGS.skip_fp64_bench}
if ARGS.skip_fp64_bench:
    print(f"--skip-fp64-bench: benchmark skipped, --gptq-device={ARGS.gptq_device} taken as given.")
    GPTQ_SOLVE_DEVICE = DEVICE if ARGS.gptq_device == "cuda" else CPU
else:
    _n = ARGS.bench_dim
    print(f"Timing one float64 ({_n}, {_n}) solve on each device "
          f"({_n * _n * 8 / 1e9:.2f} GB per matrix)...")
    _BENCH["gpu_s"] = _bench_fp64_solve(_n, DEVICE)
    _BENCH["cpu_s"] = _bench_fp64_solve(_n, CPU)
    _gpu_s = "FAILED" if _BENCH["gpu_s"] is None else f"{_BENCH['gpu_s']:.2f} s"
    _cpu_s = "FAILED" if _BENCH["cpu_s"] is None else f"{_BENCH['cpu_s']:.2f} s"
    print(f"  GPU  {torch.cuda.get_device_name(0)}: {_gpu_s}")
    print(f"  CPU  host: {_cpu_s}")
    if ARGS.gptq_device == "auto":
        gs = math.inf if _BENCH["gpu_s"] is None else _BENCH["gpu_s"]
        cs = math.inf if _BENCH["cpu_s"] is None else _BENCH["cpu_s"]
        if gs is math.inf and cs is math.inf:
            raise SystemExit("neither device could run a float64 solve of the benchmark size -- "
                             "lower --bench-dim to diagnose, but note that down_proj needs this "
                             "exact size to run at all.")
        GPTQ_SOLVE_DEVICE = CPU if cs < gs else DEVICE
        print(f"  --gptq-device=auto -> {GPTQ_SOLVE_DEVICE}"
              + ("  (the GPU is SLOWER at fp64 here; this is the weak-fp64 case the benchmark "
                 "exists to catch)" if GPTQ_SOLVE_DEVICE == CPU else ""))
    else:
        GPTQ_SOLVE_DEVICE = DEVICE if ARGS.gptq_device == "cuda" else CPU
        print(f"  --gptq-device={ARGS.gptq_device} -> {GPTQ_SOLVE_DEVICE} (benchmark reported "
              f"above but not used to decide)")
_BENCH["solve_device"] = GPTQ_SOLVE_DEVICE
print(f"\nfloat64 GPTQ solves will run on: {GPTQ_SOLVE_DEVICE}")
print("(the dtype stays float64 on either device -- only the placement was decided here)")


# =============================================================================================
# Section 0 constants. Every one is overridable from the command line; the values here are the
# defaults this script was designed around.
# =============================================================================================
MODEL_ID = ARGS.model_id

# Plain fp32, unconditionally -- inherited from the GPT-2 script, where fp16 compute was shown to
# round off the activations feeding every Hessian AND the quantized weight write-back. At 7B this
# costs 28 GB of host RAM, which is why the model lives on the CPU and is streamed a block at a
# time (see `move_block`, section 5) rather than being downcast.
GPU_DTYPE = torch.float32

# float64 for the GPTQ solve, unconditionally. `down_proj` divides by a Hessian-inverse diagonal
# entry 14336 times sequentially -- four and a half times GPT-2's c_proj depth -- so if anything
# this matters MORE here than it did there. NOT exposed as a flag; only its DEVICE is (section 0a).
GPTQ_DTYPE = torch.float64

# --- weight orientation: the single most dangerous line in this port -------------------------
# GPT-2's `Conv1D` stores (d_in, d_out) and computes `x @ W`. Mistral's `nn.Linear` stores
# (d_out, d_in) and computes `x @ W.T`. The GPTQ core wants (d_in, d_out). An explicit flag rather
# than an assumption, checked against the real modules in section 7 and asserted on every call.
WEIGHT_LAYOUT = "out_in"          # "out_in" = nn.Linear (d_out, d_in);  "in_out" = Conv1D

# --- calibration / scoring ---
CALIB_N_SAMPLES = ARGS.calib_samples
CALIB_SEQ_LEN = ARGS.calib_seq_len
HESSIAN_N_BATCHES = ARGS.hessian_batches   # batches used for H = 2 X^T X -- ALL of them
HUTCH_SAMPLES = ARGS.hutch_samples         # Hutchinson probes per (layer, batch)
JAB_N_BATCHES = ARGS.jab_batches           # batches averaged into each layer's trace
ERR_N_BATCHES = ARGS.err_batches           # batches averaged into eps_A / eps_M

# LIVE in this script, unlike the GPT-2 MLP port where it was pinned to 0.0 and warned about.
# Mistral's blocks contain the attention sub-block, so there IS a post-softmax distribution to take
# a KL against, and the joint loss is MSE(block output) + LAMBDA_KL * KL(attention maps).
LAMBDA_KL = ARGS.lambda_kl
GROUP_SIZE = ARGS.group_size               # must divide 4096 AND 14336 (asserted in section 9)
PERTURB_MODE = ARGS.perturb_mode           # "rtn" | "gptq" -- the allocator's sensitivity-table
                                            # heuristic, not the GPTQ solve itself

# --- evaluation ---
# A 32-window x 512-token SUBSET of WikiText-2 test, not the full test set: a 7B forward pass is
# streamed block-by-block off the CPU (see evaluate_perplexity, section 5) and the full test set
# would dominate the wall clock of every one of the ~30 quantize-and-evaluate passes below. Every
# place this number is printed says "subset" for the reason the GPT-2 script documents: a small
# window count has a noise floor comparable to the gaps between arms.
N_EVAL_WINDOWS = ARGS.eval_windows
EVAL_SEQ_LEN = ARGS.eval_seq_len

# --- which passes to run ---
RUN_FP16_BASELINE = _should_run("fp16")
RUN_UNIFORM4 = _should_run("uniform4")
RUN_ADAPTIVE = _should_run("adaptive")
RUN_SWEEP = _should_run("sweep")
RUN_JOINT_FINETUNE = _should_run("joint")
RUN_ADAPTIVE_JOINT = _should_run("adaptive_joint")

# --- joint fine-tuning ---
JOINT_STEPS_PER_BLOCK = ARGS.steps_per_block
JOINT_LR = ARGS.joint_lr                   # unused by the FT block (see ALPHA_W/ALPHA_S)
JOINT_GRAD_CLIP = ARGS.grad_clip
JOINT_LAYERS = ARGS.joint_layers
ALPHA_W = ARGS.alpha_w                     # eta_w = ALPHA_W * s, s = mean grid step PER UNIT
ALPHA_S = ARGS.alpha_s                     # eta_s = ALPHA_S * s, likewise per unit
SCORE_EVERY = ARGS.score_every
FT_POOL = ARGS.ft_pool
PROPAGATION_WEIGHT = ARGS.propagation_weight
SWEEP_BITS = ARGS.sweep_bits
TYPE_OFFSETS = ARGS.type_offsets
TYPE_OFFSET_FT = ARGS.type_offset_ft

# --- diagnostics ---
LOG_LOSS_COMPONENTS = ARGS.log_loss_components
RUN_DEPTH_SCALING_TEST = _should_run("adaptive_joint_depth4")     # 13c
JOINT_LAYERS_DEPTH_TEST = range(0, ARGS.depth_test_layers)
RUN_KL_ABLATION_TEST = _should_run("adaptive_joint_kl0")          # 13d: a REAL ablation here

BIT_WIDTHS_ARG = ARGS.bit_widths
BIT_FLOOR = ARGS.bit_floor                 # hard minimum width per unit; 0 disables

# Tag written into RESULTS["scoring"] and demanded back on resume. run_mistral_quant.py's cached
# scoring blocks hold ONE pooled trace per layer replicated across its seven units; this script's
# hold seven per-unit Tr(H_uu). They are different quantities on the same JSON keys, so a cache
# without this tag is refused and re-scored rather than reused -- see section 12.
SCORING_VERSION = "per-unit-v2"
TARGET_AVG_BITS = ARGS.target_avg_bits     # BITS PER PARAMETER -- see `cost`, section 4


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


print("Device:", DEVICE, "| model resident on:", CPU, "(one block at a time on the GPU)")
print("Model:", MODEL_ID, "| compute dtype:", GPU_DTYPE, "| GPTQ dtype:", GPTQ_DTYPE,
      "| GPTQ solve device:", GPTQ_SOLVE_DEVICE)
print("Quantization target: all 7 linear modules per block -- " + ", ".join(UNIT_KINDS))
print("fp32 throughout: RMSNorm gains, the embedding, the lm_head, and every bias "
      "(Mistral has none on these projections).")

# The section-12 scoring pass feeds the adaptive allocation, the jab/jab_norm sweep criteria, and
# 13b/13c/13d. Run (or reload from results.json's "scoring" cache) if ANY of them will execute.
NEED_SCORING = (RUN_ADAPTIVE or RUN_SWEEP or RUN_ADAPTIVE_JOINT
                or RUN_DEPTH_SCALING_TEST or RUN_KL_ABLATION_TEST)


# ============================================================================================
# 0. Mistral architecture adapter
# ============================================================================================
class BlockConfig:
    """
    Shape bookkeeping pulled out of a HF MistralConfig.

    Every field the hand-written block forward (`compute_block`, section 2) needs, plus the
    per-unit parameter counts the allocator needs. Unlike the GPT-2 adapters, the attention fields
    here are NOT decoration: attention is quantized, hand-written, and GQA means q and k/v have
    genuinely different output widths.

      q_proj    : (4096, 4096)   input input_layernorm(h)
      k_proj    : (1024, 4096)   input input_layernorm(h)        <- 8 KV heads, not 32
      v_proj    : (1024, 4096)   input input_layernorm(h)
      o_proj    : (4096, 4096)   input the merged attention output
      gate_proj : (14336, 4096)  input post_attention_layernorm(h)
      up_proj   : (14336, 4096)  input post_attention_layernorm(h)
      down_proj : (4096, 14336)  input silu(gate(x)) * up(x)
    All seven are `nn.Linear`, i.e. (d_out, d_in) -- the OPPOSITE of GPT-2's Conv1D. See
    WEIGHT_LAYOUT, section 0.
    """

    def __init__(self, config):
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", config.num_attention_heads)
        self.head_dim = (getattr(config, "head_dim", None)
                         or config.hidden_size // config.num_attention_heads)
        self.q_out = self.num_heads * self.head_dim
        self.kv_out = self.num_kv_heads * self.head_dim
        assert self.num_heads % self.num_kv_heads == 0, (
            f"GQA needs num_attention_heads ({self.num_heads}) to be a multiple of "
            f"num_key_value_heads ({self.num_kv_heads})")
        self.n_rep = self.num_heads // self.num_kv_heads          # 4 for Mistral-7B
        self.scaling = self.head_dim ** -0.5
        self.n_layers = config.num_hidden_layers
        self.intermediate_size = config.intermediate_size
        self.rms_eps = getattr(config, "rms_norm_eps", 1e-5)
        self.rope_theta = float(getattr(config, "rope_theta", 10000.0))
        self.sliding_window = getattr(config, "sliding_window", None)

        h, i = self.hidden_size, self.intermediate_size
        # (d_out, d_in), nn.Linear-native.
        self.unit_shapes = {
            "q_proj": (self.q_out, h),
            "k_proj": (self.kv_out, h),
            "v_proj": (self.kv_out, h),
            "o_proj": (h, self.q_out),
            "gate_proj": (i, h),
            "up_proj": (i, h),
            "down_proj": (h, i),
        }
        self.unit_params = {k: v[0] * v[1] for k, v in self.unit_shapes.items()}
        # Input dimension each unit's Hessian is sized by.
        self.unit_d_in = {k: v[1] for k, v in self.unit_shapes.items()}
        self.hessian_dims = {"x1": h, "A": self.q_out, "x2": h, "M": i}

    @property
    def attn_numel(self):
        return sum(self.unit_params[k] for k in ATTN_KINDS)

    @property
    def mlp_numel(self):
        return sum(self.unit_params[k] for k in MLP_KINDS)

    @property
    def block_numel(self):
        return sum(self.unit_params.values())

    def __repr__(self):
        return (f"BlockConfig(hidden={self.hidden_size}, intermediate={self.intermediate_size}, "
                f"layers={self.n_layers}, heads={self.num_heads}, kv_heads={self.num_kv_heads}, "
                f"head_dim={self.head_dim}, n_rep={self.n_rep})")


def build_attn_mask(seq_len, device, sliding_window=None):
    """
    Boolean "may attend" mask.

    Mistral-7B-v0.1's config carries `sliding_window=4096`. Every sequence in this script is
    CALIB_SEQ_LEN / EVAL_SEQ_LEN = 512 tokens, far inside that window, so the sliding-window mask
    and a plain causal mask are the same matrix -- but only because of that inequality, which
    section 9 asserts rather than assumes. If it were ever violated, this function would silently
    produce a mask the real model does not use.
    """
    m = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    if sliding_window is not None and seq_len > sliding_window:
        m = m & torch.ones_like(m).triu(-(sliding_window - 1))
    return m


def additive_mask(bool_mask, dtype, device):
    """bool 'may attend' (T, T) -> the additive (1, 1, T, T) float mask attention wants."""
    T = bool_mask.shape[-1]
    add = torch.zeros(T, T, dtype=dtype, device=device)
    return add.masked_fill(~bool_mask.to(device), torch.finfo(dtype).min)[None, None]


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
                        act_order=True, return_scale=False, return_compact=False, work_dtype=None,
                        solve_device=None):
    """
    Quantizes a weight matrix in the (d_in, d_out) "x @ W" convention using GPTQ.
    UNCHANGED from the GPT-2 script apart from `solve_device` -- this function is
    architecture-agnostic; it only ever sees a plain (d_in, d_out) matrix and a (d_in, d_in)
    Hessian. Callers are responsible for handing it that orientation; `gptq_quantize_module` below
    is what converts from whichever layout a given model's modules store.

    weight_in_out: (d_in, d_out) float tensor
    H: (d_in, d_in) Hessian
    damp_percent: Hessian damping for numerical stability (GPTQ default ~0.01)
    group_size: separate per-row scale per contiguous group of `group_size` input columns
    act_order: quantize columns in order of decreasing Hessian diagonal
    return_scale: ALSO return the exact per-(output channel, group) scale used, in the ORIGINAL
        (unpermuted) column order and the same (d_in, d_out) orientation -- so a caller can later
        fake-quantize the SAME weight onto the SAME grid (needed for the STE warm start).
    work_dtype: float64, always. See GPTQ_DTYPE, section 0.
    solve_device: WHERE the float64 work happens (section 0a). The result is always returned on
        `weight_in_out`'s own device, so callers never have to think about it. down_proj's solve is
        a 14336-deep sequential column loop in float64; on a weak-fp64 card the host CPU wins, and
        which one wins is measured at startup rather than assumed.

    Returns the fake-quantized weight (also written into weight_in_out in place), or
    (W_final, scale) if return_scale.
    """
    work_dtype = work_dtype or GPTQ_DTYPE
    out_device = weight_in_out.device
    device = out_device if solve_device is None else torch.device(solve_device)
    W = weight_in_out.detach().to(device=device, dtype=work_dtype).T.contiguous()  # (d_out, d_in)
    d_out, d_in = W.shape

    H = H.to(device=device, dtype=work_dtype).clone()
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

    W_final = W.T.contiguous().to(device=out_device, dtype=weight_in_out.dtype)  # (d_in, d_out)
    weight_in_out.copy_(W_final)

    if not return_scale:
        return W_final

    group_idx = torch.arange(d_in, device=device) // gs
    scale_cols = scale[:, group_idx]                                     # (d_out, d_in), permuted
    scale_cols = scale_cols[:, invperm] if act_order else scale_cols
    scale_expanded = scale_cols.T.contiguous().to(device=out_device, dtype=weight_in_out.dtype)

    if not return_compact:
        return W_final, scale_expanded

    # g_idx maps each ORIGINAL column to its group; groups are contiguous in PERMUTED order, so
    # this must go through invperm (original column c sits at permuted position invperm[c]).
    g_idx = (invperm // gs) if act_order else (torch.arange(d_in, device=device) // gs)
    return (W_final, scale_expanded,
            scale.to(device=out_device, dtype=weight_in_out.dtype), g_idx.to(out_device))


def _as_in_out(W, layout):
    """A weight in the model's native layout -> the (d_in, d_out) view the GPTQ core wants."""
    if layout == "in_out":
        return W                  # Conv1D (GPT-2): already x @ W
    if layout == "out_in":
        return W.T                # nn.Linear (Mistral/Llama): stored (d_out, d_in), computes x@W.T
    raise ValueError(f"unknown weight layout {layout!r} (expected 'in_out' or 'out_in')")


def _from_in_out(W_io, layout):
    """Inverse of `_as_in_out`."""
    return W_io if layout == "in_out" else W_io.T


@torch.no_grad()
def gptq_quantize_module(module, H, *, layout, bits=4, damp_percent=0.01, group_size=None,
                         act_order=True, return_compact=False, solve_device=None):
    """
    GPTQ for one linear module, with the weight ORIENTATION PASSED IN EXPLICITLY.

    THIS IS THE SINGLE MOST DANGEROUS FUNCTION IN THE PORT. Its GPT-2 ancestor
    (`gptq_quantize_conv1d`) hard-coded `weight.shape[0] == d_in` and asserted it -- correct for
    `Conv1D`, and EXACTLY BACKWARDS for `nn.Linear`, which stores (d_out, d_in). Mistral's q_proj
    and o_proj are square (4096, 4096), so for four of the seven units a transpose error would not
    even change a shape: GPTQ would happily quantize along output channels instead of input
    channels, the Hessian would be paired with the wrong axis, and NOTHING WOULD RAISE. The result
    is a model that loads, runs, and is quietly much worse than it should be.

    So `layout` is a required keyword with no default. Callers must say which convention the module
    stores, and the assertion below then checks the CORRESPONDING axis against the Hessian rather
    than a fixed one. The non-square units (k/v_proj 1024x4096, gate/up_proj 14336x4096, down_proj
    4096x14336) are what make the check bite: with `layout` wrong, five of seven units raise
    immediately instead of degrading silently, and section 7 asserts both directions.

    Returns a dict:
      W        -- the fake-quantized weight in the module's NATIVE layout (also written in place)
      scale    -- per-weight scale in the module's native layout, or None
      compact  -- (d_out, n_groups) scale, or None
      g_idx    -- (d_in,) group map, or None
      eps_W    -- ||W - W_hat||_F / ||W||_F for this unit (section 4's weight-error metric)
    """
    d_in = H.shape[0]
    axis = 0 if layout == "in_out" else 1
    assert module.weight.shape[axis] == d_in, (
        f"weight shape {tuple(module.weight.shape)} with layout={layout!r} puts the input "
        f"dimension at axis {axis} = {module.weight.shape[axis]}, but the Hessian is {d_in}-wide. "
        f"Either the layout flag is wrong for this model (GPT-2's Conv1D is 'in_out'; "
        f"nn.Linear is 'out_in') or this module is wired to the wrong Hessian -- in the first case "
        f"GPTQ would quantize along the output axis and nothing else would complain.")
    assert module.bias is None, (
        "this module has a bias; Mistral's projections do not. A bias would have to be excluded "
        "from the GPTQ solve and snapshotted for the teacher, as the GPT-2 script did -- it is "
        "not handled here because it should not exist.")

    # NO premature dtype cast here: gptq_quantize_layer upcasts to GPTQ_DTYPE (float64) itself. An
    # earlier version of the GPT-2 ancestor cast to float32 HERE, before that upcast, which
    # combined with an fp16 load meant the float64 solve was fed already-rounded inputs.
    W_native0 = module.weight.data.clone()
    W_io = _as_in_out(W_native0, layout).contiguous()    # (d_in, d_out), a real copy after .clone()
    out = gptq_quantize_layer(W_io, H, bits=bits, damp_percent=damp_percent,
                              group_size=group_size, act_order=act_order,
                              return_scale=return_compact, return_compact=return_compact,
                              solve_device=solve_device)

    if return_compact:
        W_q_io, scale_io, scale_compact, g_idx = out
    else:
        W_q_io, scale_io, scale_compact, g_idx = out, None, None, None

    W_q = _from_in_out(W_q_io, layout).contiguous()
    denom = W_native0.norm()
    eps_W = ((W_q - W_native0).norm() / denom).item() if denom > 0 else 0.0
    module.weight.data.copy_(W_q.to(module.weight.dtype))
    res = {"W": W_q, "scale": None, "compact": scale_compact, "g_idx": g_idx, "eps_W": eps_W}
    if scale_io is not None:
        res["scale"] = _from_in_out(scale_io, layout).contiguous()
    del W_native0, W_io
    return res


@torch.no_grad()
def quantize_block(layer, Xs, bits_of, cfg, ctx, *, teacher=None, group_size=None,
                   damp_percent=0.01, return_compact=False, n_hessian_batches=None,
                   n_err_batches=None, solve_device=None, verbose=False):
    """
    GPTQ for one whole Mistral decoder block: all seven units, FOUR Hessians, in sequential order.

    THE HESSIAN STRUCTURE IS THE WHOLE POINT OF THIS FUNCTION. The GPT-2 script had two units whose
    inputs were in series (c_fc then c_proj) and therefore two Hessians. Here there are seven units
    but only four distinct inputs, and they are in series in a specific order:

        x1 = input_layernorm(h)             -> q_proj, k_proj, v_proj   (ONE Hessian for all three)
        A  = merged attention output        -> o_proj
        x2 = post_attention_layernorm(h)    -> gate_proj, up_proj       (ONE Hessian for both)
        M  = silu(gate(x2)) * up(x2)        -> down_proj

    Collecting each shared Hessian ONCE is not just an optimization -- collecting it three times
    would give three numerically different estimates of the same quantity (the hook sees a live
    model that changes between passes) and silently break the "q, k and v were scored against the
    same input distribution" premise the allocator rests on.

    The ORDER is equally load-bearing: A depends on the quantized q/k/v, x2 depends on the
    quantized o_proj, M depends on the quantized gate/up. Collecting all four Hessians up front --
    which is what a naive port of the GPT-2 two-Hessian code does -- would describe activations
    that three of the four groups will never see. Each Hessian is therefore collected immediately
    before the units that read it, from the block as it stands at that moment.

    Each Hessian is freed the instant its units are solved. down_proj's is (14336, 14336) float64 =
    1.64 GB, and its inverse needs the same again; holding two of those at once is the difference
    between fitting and not.

    bits_of: `kind -> bits` callable.
    teacher: an unquantized float copy of THIS block, on the same device. If given, eps_A and eps_M
        are measured against it (see the metrics discussion in section 8). May be None.
    Xs: the cached inputs to this block, one (1, T, hidden) tensor per calibration batch.
    ctx: the position/mask context for a block forward (see `block_context`, section 5).

    Returns a dict with per-unit GPTQ outputs (if return_compact), per-unit eps_W, the eps_A /
    eps_M activation errors, and each Hessian's mean diagonal (which is what the `jab_norm`
    criterion divides by -- see section 12b).
    """
    group_size = GROUP_SIZE if group_size is None else group_size
    n_hessian_batches = HESSIAN_N_BATCHES if n_hessian_batches is None else n_hessian_batches
    n_err_batches = ERR_N_BATCHES if n_err_batches is None else n_err_batches
    solve_device = GPTQ_SOLVE_DEVICE if solve_device is None else solve_device

    groups = (("x1", ("q_proj", "k_proj", "v_proj")),
              ("A", ("o_proj",)),
              ("x2", ("gate_proj", "up_proj")),
              ("M", ("down_proj",)))

    gptq_out, eps_W, hdiag = {}, {}, {}
    eps_act = {"A": None, "M": None}
    for tag, kinds in groups:
        # eps_A / eps_M are the teacher-vs-student errors of exactly the two tensors that ARE
        # Hessian inputs here (A and M), so they are measured during the same collection pass
        # rather than in a separate forward: same batches, same block state, no second traversal.
        want_err = teacher is not None and tag in ("A", "M")
        H, err = collect_block_hessian(
            layer, tag, Xs[:n_hessian_batches], cfg, ctx,
            teacher=teacher if want_err else None,
            n_err=n_err_batches if want_err else 0)
        hdiag[tag] = H.diagonal().mean().item()
        if want_err:
            eps_act[tag] = err
        assert H.shape[0] == cfg.hessian_dims[tag], (
            f"Hessian for '{tag}' is {tuple(H.shape)}, expected "
            f"({cfg.hessian_dims[tag]}, {cfg.hessian_dims[tag]})")

        for kind in kinds:
            module = module_of(layer, kind)
            res = gptq_quantize_module(module, H, layout=WEIGHT_LAYOUT, bits=bits_of(kind),
                                       damp_percent=damp_percent, group_size=group_size,
                                       act_order=True, return_compact=return_compact,
                                       solve_device=solve_device)
            eps_W[kind] = res["eps_W"]
            if return_compact:
                gptq_out[kind] = res
            else:
                del res
            if verbose:
                print(f"      {kind:<10} {bits_of(kind)}b  eps_W={eps_W[kind]:.4e}")
        del H
        free()

    return {"gptq": gptq_out if return_compact else None, "eps_W": eps_W,
            "eps_A": eps_act["A"], "eps_M": eps_act["M"], "hdiag": hdiag}


@torch.no_grad()
def grid_perturbation(W, bits, *, layout, group_size=None, H=None, mode=None):
    """
    HAWQ-V2's `||Q(W) - W||_F^2`, measured in weight space on GPTQ's own per-(output channel,
    group) symmetric grid.

    `layout` again: the grid is per (OUTPUT ROW, group of INPUT columns), so which axis is which
    decides what shares a scale. With the layout wrong, the groups run along output channels and
    every number this returns is a different quantity -- and since it feeds the allocator rather
    than the quantizer, the result would be a plausible-looking but meaningless bit assignment.

    mode="rtn"  : round-to-nearest on that grid. Milliseconds.
    mode="gptq" : the full GPTQ solve (requires H). Much slower; at 7B, considerably more so than
                  the GPT-2 script's version, because this runs once per unit per bit-width.
    """
    mode = PERTURB_MODE if mode is None else mode
    W_oi = W if layout == "out_in" else W.T          # (d_out, d_in)
    if mode == "gptq":
        if H is None:
            raise ValueError('mode="gptq" needs the Hessian H')
        W_io = W_oi.T.contiguous()
        W_q = gptq_quantize_layer(W_io.clone(), H, bits=bits, group_size=group_size,
                                  act_order=True, solve_device=GPTQ_SOLVE_DEVICE)
        out = (W_q - W_io).pow(2).sum().item()
        free(W_q)
        return out

    d_out, d_in = W_oi.shape
    gs = group_size if group_size is not None else d_in
    qmax = 2 ** (bits - 1) - 1
    total = 0.0
    for start in range(0, d_in, gs):
        blk = W_oi[:, start:min(start + gs, d_in)].to(torch.float32)
        scale = (blk.abs().amax(dim=1, keepdim=True) / qmax).clamp(min=1e-8)
        blk_q = torch.clamp(torch.round(blk / scale), -qmax, qmax) * scale
        total += (blk_q - blk).pow(2).sum().item()
    return total


# ============================================================================================
# 2. Block-aware joint loss
#
# The GPT-2 MLP script's `compute_mlp` hand-wrote two matmuls and a GELU and delegated everything
# else to the real modules. Here the quantization target is the ENTIRE block, so the hand-written
# forward has to reproduce the entire block: RMSNorm (not LayerNorm), RoPE, GQA head repetition,
# SwiGLU, and the two residual adds. Section 7a checks it against the real MistralDecoderLayer and
# that check is load-bearing -- every teacher target, every Hutchinson trace and every STE gradient
# below is computed through this function.
#
# Mistral has NO BIAS on any of the seven projections (asserted in `gptq_quantize_module`), so
# unlike the GPT-2 script there is no bias to snapshot and thread through. The RMSNorm gains are
# parameters but never quantization targets; they are read off the layer, exactly as the GPT-2
# script read `layer.mlp.act` off the layer rather than using a module-level default.
# ============================================================================================
def rms_norm(x, weight, eps):
    """
    MistralRMSNorm, reproduced exactly INCLUDING its float32 upcast.

    HF computes the variance in float32 and casts back to the input dtype before multiplying by the
    gain. Everything in this script is already float32 (GPU_DTYPE), so the upcast is a no-op -- it
    is written anyway so that this function stays correct if the dtype ever changes, and so the
    section 7a comparison is against the same arithmetic rather than a coincidence.
    """
    dtype = x.dtype
    x32 = x.to(torch.float32)
    var = x32.pow(2).mean(-1, keepdim=True)
    x32 = x32 * torch.rsqrt(var + eps)
    return weight * x32.to(dtype)


def rotate_half(x):
    """The RoPE half-rotation, in HF's interleaving convention (first half / second half, NOT
    even/odd pairs). Getting the convention wrong rotates by a different angle per dimension and
    degrades attention smoothly rather than loudly -- 7a is what catches it."""
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def apply_rope(q, k, cos, sin):
    """
    cos/sin: (batch, seq, head_dim) as returned by HF's rotary embedding module.
    q: (b, n_heads, seq, head_dim); k: (b, n_kv_heads, seq, head_dim).
    """
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


def repeat_kv(x, n_rep):
    """
    GQA head repetition: (b, n_kv_heads, s, d) -> (b, n_kv_heads * n_rep, s, d).

    Mistral-7B has 32 query heads and 8 KV heads, so n_rep = 4. The GPT-2 adapters carried an
    n_rep field that was always 1 and therefore never exercised this path at all; here it is
    always 4 and a wrong expansion order (interleave vs. block-repeat) pairs each query head with
    the wrong KV head -- which, again, degrades rather than raises.
    """
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def unit_weights(layer, dtype=None, detach=True):
    """{kind: weight} in the module's native (d_out, d_in) layout."""
    out = {}
    for kind in UNIT_KINDS:
        w = module_of(layer, kind).weight
        w = w.detach() if detach else w
        out[kind] = w.to(dtype) if dtype is not None else w
    return out


def compute_block(layer, h, cfg, cos, sin, add_mask, w=None, upto="out", return_attn=False):
    """
    One Mistral decoder block, hand-written, with every quantizable weight substitutable.

        x1   = input_layernorm(h)
        q,k,v = x1 @ Wq.T, x1 @ Wk.T, x1 @ Wv.T          (GQA: k/v are 1024-wide, q is 4096)
        RoPE(q, k); k, v repeated n_rep times
        A    = merge_heads(softmax(q k^T / sqrt(d) + mask) v)      <- the pre-o_proj output
        h    = h + A @ Wo.T
        x2   = post_attention_layernorm(h)
        M    = silu(x2 @ Wgate.T) * (x2 @ Wup.T)                   <- down_proj's input
        h    = h + M @ Wdown.T

    w: optional {kind: (d_out, d_in) weight} used INSTEAD of the module's own weights. This is what
       the STE fine-tune substitutes; the RMSNorm gains always come from the module.
    upto: stop early and return only what has been computed so far -- "x1", "A", "x2", "M" or
       "out". The four early stops are exactly the four Hessian inputs, so a caller that only needs
       `x1` does not pay for an attention forward.
    return_attn: also return the post-softmax attention probabilities (b, n_heads, T, T), which is
       what the joint loss takes its KL against. Unlike the GPT-2 MLP script -- where this argument
       existed only for signature parity and always returned None -- this one is live.

    Returns a dict with whichever of {x1, A, x2, M, out, probs} were computed.
    """
    w = unit_weights(layer) if w is None else w
    res = {}
    x1 = rms_norm(h, layer.input_layernorm.weight, cfg.rms_eps)
    res["x1"] = x1
    if upto == "x1":
        return res

    b, T, _ = h.shape
    q = F.linear(x1, w["q_proj"]).view(b, T, cfg.num_heads, cfg.head_dim).transpose(1, 2)
    k = F.linear(x1, w["k_proj"]).view(b, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
    v = F.linear(x1, w["v_proj"]).view(b, T, cfg.num_kv_heads, cfg.head_dim).transpose(1, 2)
    q, k = apply_rope(q, k, cos, sin)
    k = repeat_kv(k, cfg.n_rep)
    v = repeat_kv(v, cfg.n_rep)

    scores = torch.matmul(q, k.transpose(2, 3)) * cfg.scaling
    if add_mask is not None:
        scores = scores + add_mask[..., :T, :T]
    # softmax in float32 then back, matching HF's eager attention exactly.
    probs = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
    ctxv = torch.matmul(probs, v)                                   # (b, heads, T, head_dim)
    A = ctxv.transpose(1, 2).contiguous().view(b, T, cfg.q_out)     # the merged attention output
    res["A"] = A
    if return_attn:
        res["probs"] = probs
    if upto == "A":
        return res

    h = h + F.linear(A, w["o_proj"])
    x2 = rms_norm(h, layer.post_attention_layernorm.weight, cfg.rms_eps)
    res["x2"] = x2
    if upto == "x2":
        return res

    M = F.silu(F.linear(x2, w["gate_proj"])) * F.linear(x2, w["up_proj"])
    res["M"] = M
    if upto == "M":
        return res

    res["out"] = h + F.linear(M, w["down_proj"])
    return res


def kl_loss(student_probs, teacher_probs, eps=1e-9):
    """
    Mean KL(teacher || student) per attention row.

    REDUCTION CHOICE, STATED EXPLICITLY because it is the thing that made the ancestor notebook's
    KL term unusable: `F.kl_div(..., reduction="batchmean")` divides only by the batch dimension,
    so on a (1, 32, 512, 512) attention tensor it sums 8.4 M terms and divides by 1, running
    several orders of magnitude above its nominal LAMBDA_KL weight while `F.mse_loss` averages over
    every element. That imbalance is documented in the GPT-2 script's section 13d as the reason the
    KL ablation there was worth running at all. Here the sum is divided by the number of attention
    DISTRIBUTIONS (batch * heads * queries), making this the mean KL between two attention rows --
    a per-row quantity comparable in magnitude to the per-element MSE, so LAMBDA_KL=0.1 means what
    it looks like it means.
    """
    n_rows = student_probs.shape[0] * student_probs.shape[1] * student_probs.shape[2]
    return F.kl_div((student_probs + eps).log(), teacher_probs, reduction="sum") / n_rows


def block_loss(w, h, target_out, target_probs, layer, cfg, cos, sin, add_mask,
               lambda_kl=None, log_components=False):
    """
    L = ||block(h) - block_hat(h)||^2 + lambda_kl * KL(attention maps)

    The joint loss over all seven units at once: one forward, one scalar, one commit decision. The
    GPT-2 MLP script's `mlp_loss` was MSE-only because the MLP emits no distribution; the KL term
    is back here because the block contains attention, and it is weighted at LAMBDA_KL=0.1 rather
    than 0.0.
    """
    lambda_kl = LAMBDA_KL if lambda_kl is None else lambda_kl
    want_attn = bool(lambda_kl) and target_probs is not None
    out = compute_block(layer, h, cfg, cos, sin, add_mask, w=w, return_attn=want_attn)
    mse = F.mse_loss(out["out"], target_out)
    loss = mse
    kl = None
    if want_attn:
        kl = kl_loss(out["probs"], target_probs)
        loss = loss + lambda_kl * kl
    if log_components:
        kl_s = "n/a" if kl is None else f"{kl.item():.6f}"
        print(f"    [block_loss] mse={mse.item():.6f}  kl={kl_s}  lambda_kl={lambda_kl}")
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


def hutchinson_block_traces(loss_fn, params, cfg, samples=50):
    """
    Per-unit diagonal-block traces: {kind: Tr(H_uu)} for all seven units, at ONE HVP per probe --
    exactly the cost of the single pooled trace `hutchinson_trace_estimator` returns.

    WHY THIS IS UNBIASED, and why it is free. The pooled estimator forms v^T H v with a Rademacher
    probe v over the CONCATENATED seven-unit vector and sums the elementwise product v .* Hv over
    every coordinate. Written out over the unit blocks that sum is

        v^T H v = sum_u v_u^T H_uu v_u  +  sum_{u != u'} v_u^T H_uu' v_u'

    and the partial sum over unit u's coordinate range is precisely v_u^T H_uu v_u, whose
    expectation is Tr(H_uu) because the Rademacher signs are independent across coordinates
    (E[v_i v_j] = delta_ij). Every cross term has zero mean for the same reason. So the per-unit
    estimates need no extra Hessian-vector products at all -- only the final dot product has to be
    PARTITIONED instead of summed, which is what `prod[a:b].sum()` below does.

    What is NOT free is variance: the cross-block noise that cancels in the pooled sum survives in
    each individual partial sum, which is why HUTCH_SAMPLES is raised here (see --hutch-samples).

    Returns (traces, pooled) where `pooled` is the plain v.Hv average over the same probes, kept so
    the caller can assert sum(traces.values()) == pooled -- the one free check on the partition.
    """
    segs = unit_offsets(cfg)[0]                       # [(kind, a, b), ...]
    if not params.requires_grad:
        params.requires_grad_(True)
    acc = {k: 0.0 for k, _, _ in segs}
    pooled = 0.0
    for _ in range(samples):
        vec = (torch.randint(0, 2, params.shape, device=params.device) * 2 - 1).to(params.dtype)
        hvp = hessian_vector_product(loss_fn, params, vec)
        pooled += torch.dot(vec.flatten(), hvp.flatten()).item()
        prod = vec.flatten() * hvp.flatten()          # elementwise, NOT summed
        for k, a, b in segs:
            acc[k] += prod[a:b].sum().item()
        del vec, hvp, prod
        free()
    return {k: v / samples for k, v in acc.items()}, pooled / samples


# ============================================================================================
# 4. Greedy sensitivity-per-cost allocator
# ============================================================================================
# THE CANDIDATE SET, FLOORED ONCE, HERE. Every width in this file is chosen from BIT_WIDTHS --
# `greedy_allocate`'s downgrade and upgrade walks, `_nearest_bits`, `type_offset_allocate`'s clamp,
# `brute_force_optimal`'s product, `scores_to_allocator_format` and the sensitivity tables built in
# `pass_score`. Filtering it at the single point of definition is therefore what makes the floor a
# hard constraint rather than a post-hoc clamp: a width below the floor is never a candidate, so no
# allocator can walk to it and there is nothing to clamp back afterwards. Arms and budgets that
# cannot be expressed under the floor are infeasible and are skipped with a printed line (sections
# 12b and 14), never rounded up.
BIT_WIDTHS = [b for b in BIT_WIDTHS_ARG if BIT_FLOOR <= 0 or b >= BIT_FLOOR]
if not BIT_WIDTHS:
    raise SystemExit(f"--bit-floor {BIT_FLOOR} removes every width in --bit-widths "
                     f"{BIT_WIDTHS_ARG}: there is nothing left to allocate.")


def uniform_feasible(bits, label):
    """A uniform arm pinned to `bits` is INFEASIBLE when `bits` is below the floor. Say so once and
    let the caller skip it; rounding it up to the floor would silently relabel the arm and put a
    number in the comparison table's 2-bit row that was not measured at 2 bits."""
    if BIT_FLOOR > 0 and bits < BIT_FLOOR:
        print(f"  [skip] {label}: {bits} bits is below --bit-floor {BIT_FLOOR}, so this uniform "
              f"arm is infeasible. Not rounded up -- skipped.")
        return False
    return True


def cost(bits, param_count):
    """
    Memory cost of one unit at a given bit-width, in BITS -- parameter-weighted, mandatorily.

    The GPT-2 scripts could get away with `param_count=1.0`: every allocator unit there held
    exactly the same number of parameters (c_fc and c_proj are both 4*hidden^2; the QKV notebook's
    units were all one fused c_attn), so an unweighted `bits` cost was already proportional to
    memory and `budget` could be read as "average bits per matrix".

    That is FALSE here and the error would be large. Mistral's seven units span 14x:

        k_proj / v_proj   4.2 M      gate_proj / up_proj / down_proj   58.7 M
        q_proj / o_proj  16.8 M

    An unweighted cost would let the allocator buy a bit-width upgrade on `gate_proj` -- 58.7 M
    parameters -- for the same "price" as one on `k_proj`, which is fourteen times smaller, and
    greedy_allocate's cost-saved-per-accuracy-lost ratio would stop being a memory ratio at all.
    So `param_count` is REQUIRED (no default), and every `budget` in this file is
    BITS PER PARAMETER x total parameters, never bits per matrix.
    """
    return bits * param_count


def unit_name(i, which):
    """Allocator unit name, e.g. "layer_3_down_proj". The "layer_{idx}_" prefix is preserved from
    the GPT-2 scripts because `layer_idx_of` and the propagation weighting parse it."""
    assert which in UNIT_KINDS, f"unknown unit kind {which!r} (expected one of {UNIT_KINDS})"
    return f"layer_{i}_{which}"


def layer_idx_of(name):
    """Layer index out of a unit name -- the index is still field 1."""
    return int(name.split("_")[1])


def unit_kind_of(name):
    """"q_proj" / "down_proj" / ... out of a unit name (the kind itself contains an underscore)."""
    return "_".join(name.split("_")[2:])


def unit_params_map(cfg, names=None):
    """{unit_name: parameter count} -- what `cost` weights by and what every budget is scaled by."""
    if names is None:
        names = [unit_name(i, k) for i in range(cfg.n_layers) for k in UNIT_KINDS]
    return {n: cfg.unit_params[unit_kind_of(n)] for n in names}


def greedy_allocate(scores, budget, params):
    """
    scores: {unit_name: {bits: sensitivity}};  budget: max total cost IN BITS;
    params: {unit_name: parameter count} -- REQUIRED, see `cost`.
      1. Start every unit at the HIGHEST bit-width.
      2. While over budget, apply the downgrade with the best cost-saved-per-accuracy-lost ratio.
      3. Spend leftover budget on the best affordable upgrades.

    Structurally identical to the GPT-2 version; the only change is that every `cost(...)` call now
    carries that unit's real parameter count, which turns the ratio in step 2 into
    accuracy-lost-per-BIT-OF-MEMORY-saved rather than per-bit-of-width.
    """
    current_bits = {name: max(BIT_WIDTHS) for name in scores}

    def total_cost():
        return sum(cost(current_bits[n], params[n]) for n in scores)

    def total_sensitivity():
        return sum(scores[n][current_bits[n]] for n in scores)

    while total_cost() > budget:                     # downgrade loop
        best = best_bits = best_ratio = None
        for name in scores:
            lower = [b for b in BIT_WIDTHS if b < current_bits[name]]
            if not lower:
                continue                             # already at the lowest bit-width
            nb = max(lower)
            saved = cost(current_bits[name], params[name]) - cost(nb, params[name])
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
            extra = cost(nb, params[name]) - cost(current_bits[name], params[name])
            if total_cost() + extra > budget:
                continue                             # can't afford it
            ratio = (scores[name][current_bits[name]] - scores[name][nb]) / extra
            if best_ratio is None or ratio > best_ratio:
                best_ratio, best, best_bits = ratio, name, nb
        if best is not None:
            current_bits[best] = best_bits
            made_an_upgrade = True

    return current_bits, total_cost(), total_sensitivity()


def brute_force_optimal(scores, budget, params, max_units=8):
    """Exhaustive search, for sanity-checking the greedy allocator on a small subset.
    Guarded: with 5 bit-widths this is 5^n_units combinations."""
    names = list(scores.keys())
    if len(names) > max_units:
        raise ValueError(f"brute_force_optimal over {len(names)} units = "
                         f"{len(BIT_WIDTHS)}^{len(names)} combinations. Restrict `scores` to at "
                         f"most {max_units} units, or raise max_units if you really mean it.")
    best_assignment = best_cost = None
    best_sensitivity = float("inf")
    for combo in itertools.product(*([BIT_WIDTHS] * len(names))):
        c = sum(cost(b, params[names[i]]) for i, b in enumerate(combo))
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


def _nearest_bits(x, bit_widths=None):
    """Nearest available bit-width, clamped into the BIT_WIDTHS range. Ties go to the lower width,
    so the clamp never silently spends memory it was not asked to."""
    bit_widths = BIT_WIDTHS if bit_widths is None else bit_widths
    return min(sorted(bit_widths), key=lambda b: (abs(b - x), b))


def type_offset_allocate(budget_bpp, offsets, cfg, bit_widths=None):
    """
    The `type_offset` criterion: NO SCORING PASS AT ALL.

    Every unit's bit-width is a hardcoded per-unit-KIND offset from a single base width, clamped
    onto BIT_WIDTHS, with the base chosen so the parameter-weighted average lands on the requested
    bits-per-parameter budget. There is no Hutchinson trace, no Hessian, no oracle: the entire
    criterion is a table of seven integers (`--type-offsets`), so it costs one quantize-and-evaluate
    per budget and nothing else. It is in the sweep as the "does the expensive part buy anything?"
    control -- if jab / jab_norm cannot beat a fixed per-module-type prior, the scoring pass is not
    earning its wall clock.

    Two steps, because the bit-width grid is coarse:
      1. Scan a continuous base `t`; each kind takes `nearest(t + offset[kind])`. Keep the `t`
         whose parameter-weighted cost is closest to the budget.
      2. Close the remaining gap by moving INDIVIDUAL units (not whole kinds) one grid step at a
         time, always taking the single change that most reduces |cost - budget|. Ties are broken
         by a fixed order -- larger offset first, then layer index -- so the result is
         deterministic and does not depend on dict iteration order.

    One property of BIT_WIDTHS worth knowing before reading the results: the grid is [2, 3, 4, 8,
    16], which is unit-spaced only up to 4. A +-1 offset therefore bites at the 2.5 and 3.5 budgets
    and is swallowed by the clamp at 4.5 (4+1 = 5 rounds back to 4), where step 2 expresses the
    prior by promoting individual high-offset units to 8 instead. That is a property of the
    bit-width grid, not of the criterion, and it is why the default sweep sits on the half-bit
    points rather than on integers.
    """
    bit_widths = BIT_WIDTHS if bit_widths is None else bit_widths
    # Clamp the candidate set against the floor explicitly, even though BIT_WIDTHS is already
    # filtered: `bit_widths` is an overridable keyword, and step 1's `_nearest_bits` scan plus
    # step 2's individual moves would otherwise be free to land a unit below the floor whenever a
    # caller passes an unfiltered list.
    bit_widths = [b for b in bit_widths if BIT_FLOOR <= 0 or b >= BIT_FLOOR]
    if not bit_widths:
        raise ValueError(f"type_offset_allocate: --bit-floor {BIT_FLOOR} leaves no candidate "
                         f"widths.")
    names = [unit_name(i, k) for i in range(cfg.n_layers) for k in UNIT_KINDS]
    params = unit_params_map(cfg, names)
    budget = budget_bpp * sum(params.values())

    lo = min(bit_widths) - max(offsets.values()) - 1
    hi = max(bit_widths) - min(offsets.values()) + 1
    best = None
    steps = int(round((hi - lo) / 0.01)) + 1
    for s in range(steps):
        t = lo + 0.01 * s
        assign = {n: _nearest_bits(t + offsets[unit_kind_of(n)], bit_widths) for n in names}
        c = sum(cost(assign[n], params[n]) for n in names)
        if best is None or abs(c - budget) < abs(best[0] - budget):
            best = (c, assign)
    cost_used, assign = best[0], dict(best[1])

    order = sorted(names, key=lambda n: (-offsets[unit_kind_of(n)], layer_idx_of(n),
                                         UNIT_KINDS.index(unit_kind_of(n))))
    while True:
        gap = abs(cost_used - budget)
        move = None
        for n in order:
            cur = assign[n]
            for nb in bit_widths:
                if nb == cur:
                    continue
                new_cost = cost_used + cost(nb - cur, params[n])
                gain = gap - abs(new_cost - budget)
                if gain > 1e-9 and (move is None or gain > move[0]):
                    move = (gain, n, nb, new_cost)
        if move is None:
            break
        _, n, nb, cost_used = move
        assign[n] = nb

    return assign, cost_used, sum(params.values())


# ============================================================================================
# 5. Model loading & per-block helpers
#
# THE BIG STRUCTURAL DEPARTURE FROM THE GPT-2 SCRIPTS. There, `reader` was a fully resident model
# on the GPU and every Hessian came from a hook fired during a forward pass of the WHOLE model, so
# layer i's Hessians automatically reflected layers 0..i-1 already being quantized. At 7B that is
# not available: fp32 weights alone are 28 GB, before a 1.64 GB fp64 down_proj Hessian and its
# inverse workspace.
#
# The replacement keeps the same sequential-GPTQ property by a different route. The model stays on
# the CPU; the hidden states entering the CURRENT block are cached (one tensor per calibration
# batch); exactly one decoder layer at a time is moved to the GPU, has its Hessians collected and
# its seven units quantized, is run once more to produce the NEXT block's inputs, and is moved
# back. Because the cache is regenerated from the quantized block each time, the inputs layer i
# sees are exactly the ones the quantized layers 0..i-1 produce -- the same property, obtained by
# propagation instead of by re-running the whole model.
# ============================================================================================
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Which module's INPUT each of the four Hessians is. Hooks are placed on the real modules of the
# real block (not on `compute_block`) for the reason the GPT-2 script gives: the activations that
# feed a Hessian should come from the model's own forward machinery, so there is no second
# implementation for them to drift from. `compute_block` is used only where a substituted weight is
# required -- the STE fine-tune and the Hutchinson trace -- and section 7a is what ties the two
# together.
HESSIAN_HOOK_MODULE = {"x1": ("self_attn", "q_proj"), "A": ("self_attn", "o_proj"),
                       "x2": ("mlp", "gate_proj"), "M": ("mlp", "down_proj")}


def load_model_and_tokenizer(model_id, device=None):
    """
    Loads a full Mistral-7B ON THE CPU, in fp32.

    `device` is accepted and IGNORED for signature parity with the GPT-2 scripts -- it is not a
    placement decision any caller gets to make here. 28 GB of fp32 weights do not belong on the
    card; blocks are streamed to it one at a time by `move_block`. Passing something other than
    CPU would silently reintroduce the OOM this whole section exists to avoid, so it is asserted
    against rather than honoured.

    Every arm below loads its OWN fresh model, exactly as in the GPT-2 scripts and for the same
    reason: `pass_quantize_eval` quantizes weights IN PLACE, so reusing one model across arms
    compounds each arm's damage into the next.
    """
    assert device in (None, CPU, "cpu"), (
        f"load_model_and_tokenizer was asked for device={device!r}. This model stays on the CPU "
        f"and is streamed to the GPU one block at a time (see move_block); a resident 28 GB fp32 "
        f"7B model is the OOM this entire section exists to avoid.")
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # fp32 EXPLICITLY, never the checkpoint's own dtype. Mistral-7B-v0.1 ships a bfloat16
    # checkpoint, and a from_pretrained that honours it would hand back a bf16 model -- FEWER
    # mantissa bits than the fp16 the GPT-2 script's section 0 blames for non-monotone
    # perplexity-vs-bit-width. The keyword was spelled `torch_dtype` before transformers 5.x and
    # `dtype` after, so both are tried rather than pinning a version.
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=GPU_DTYPE, low_cpu_mem_usage=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=GPU_DTYPE, low_cpu_mem_usage=True)
    bad = next((n for n, p in model.named_parameters() if p.dtype != GPU_DTYPE), None)
    assert bad is None, (
        f"{bad} loaded as {dict(model.named_parameters())[bad].dtype}, not {GPU_DTYPE} -- the "
        f"dtype keyword did not take effect and every Hessian below would be accumulated from "
        f"already-rounded activations")
    model.config._attn_implementation = "eager"
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tokenizer


def blocks_of(model):
    """The decoder layers. Mistral puts them at `model.model.layers`, not GPT-2's
    `transformer.h`."""
    return model.model.layers


def load_decoder_layer(reader, config, idx):
    """Index into the resident (CPU) model. Kept as a function so the per-layer loop bodies read
    the same as the GPT-2 scripts'."""
    return blocks_of(reader)[idx]


def module_of(layer, kind):
    """The `nn.Linear` for one unit kind. The four attention projections live under `self_attn`,
    the three MLP ones under `mlp`; `o_proj` and `down_proj` are the two output projections and
    must not be confused with each other (the GPT-2 scripts had the same trap with two modules
    named `c_proj`)."""
    assert kind in UNIT_KINDS, f"unknown unit kind {kind!r}"
    return getattr(layer.self_attn if kind in ATTN_KINDS else layer.mlp, kind)


def d_in_of(module, layout=None):
    """The input dimension of a linear module under the declared weight layout."""
    layout = WEIGHT_LAYOUT if layout is None else layout
    return module.weight.shape[1] if layout == "out_in" else module.weight.shape[0]


def move_block(layer, device):
    """Move one decoder layer between CPU and GPU. The only place model weights change device."""
    layer.to(device)
    return layer


def teacher_copy(layer):
    """
    A float copy of the CURRENT block, for the eps_A / eps_M teacher and the fine-tune target.

    ONE BLOCK ONLY, never the whole model: a second resident 7B copy would be another 28 GB, and
    the teacher is only ever asked about the block being quantized right now. Freed at the end of
    each layer iteration.
    """
    t = copy.deepcopy(layer)
    t.eval()
    for p in t.parameters():
        p.requires_grad_(False)
    return t


def rope_cos_sin(model, cfg, T, device, dtype):
    """
    (cos, sin) for `T` positions, taken from the MODEL's own rotary embedding module wherever one
    exists, and computed from the config only as a fallback.

    Preferring the model's module is deliberate: RoPE scaling variants (linear, dynamic NTK, yarn)
    live in that module, and a hand-rolled inv_freq would silently ignore them. The fallback
    reproduces plain theta-based RoPE, which is what Mistral-7B-v0.1 uses; section 7a compares the
    whole hand-written block against the real one either way, so a mismatch is caught, not assumed
    away.
    """
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    rot = getattr(getattr(model, "model", model), "rotary_emb", None)
    if rot is not None:
        rot = rot.to(device)
        probe = torch.zeros(1, 1, T, cfg.head_dim, dtype=dtype, device=device)
        cos, sin = rot(probe, position_ids)
        return cos.to(dtype), sin.to(dtype), position_ids
    inv_freq = 1.0 / (cfg.rope_theta ** (
        torch.arange(0, cfg.head_dim, 2, dtype=torch.float32, device=device) / cfg.head_dim))
    freqs = position_ids.to(torch.float32)[..., None] * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype), position_ids


def block_context(model, cfg, T, device, dtype=None):
    """
    Everything a decoder block needs besides its input: RoPE tables, position ids and the additive
    causal mask. Built once per (T, device) and reused for every block and every batch.

    The mask is LOAD-BEARING and must not be dropped: as of transformers 5.x the attention modules
    no longer build their own causal mask -- the top-level model does, and passes it down -- so
    calling `layer(x)` with no mask yields fully BIDIRECTIONAL attention and silently wrong hidden
    states. The GPT-2 script hit exactly this and documents the ~0.28 relative error it produced.
    Section 7a asserts the mask still moves the output, so a version that masks internally cannot
    make this check vacuous without being noticed.
    """
    dtype = GPU_DTYPE if dtype is None else dtype
    cos, sin, position_ids = rope_cos_sin(model, cfg, T, device, dtype)
    add_mask = additive_mask(build_attn_mask(T, device, cfg.sliding_window), dtype, device)
    return {"cos": cos, "sin": sin, "position_ids": position_ids, "add_mask": add_mask,
            "seq_len": T, "device": device}


def run_real_block(layer, h, ctx):
    """
    Call the REAL MistralDecoderLayer, passing whichever of the mask / position_ids /
    position_embeddings arguments the installed transformers version actually accepts.

    Version-tolerant by signature inspection rather than by version sniffing: 4.3x takes
    `position_ids` and builds cos/sin internally, 4.4x+ takes a precomputed `position_embeddings`
    tuple, and the cache argument has been spelled both `past_key_value` and `past_key_values`.
    Passing the same cos/sin this script's own `compute_block` uses is what makes the two
    comparable in section 7a.
    """
    kw = {"attention_mask": ctx["add_mask"],
          "position_ids": ctx["position_ids"],
          "position_embeddings": (ctx["cos"], ctx["sin"]),
          "past_key_value": None,
          "past_key_values": None,
          "use_cache": False,
          "output_attentions": False}
    accepted = inspect.signature(layer.forward).parameters
    kw = {k: v for k, v in kw.items() if k in accepted}
    out = layer(h, **kw)
    return out[0] if isinstance(out, tuple) else out


@torch.no_grad()
def build_block_inputs(model, ids_list, device=None, dtype=None):
    """
    The hidden states entering block 0, one per calibration batch, kept ON THE CPU.

    Mistral feeds `embed_tokens(ids)` straight into block 0 -- no learned position embedding to add
    (RoPE is applied inside attention, per layer) and no embedding scale factor. 128 batches of 512
    tokens is 1.07 GB in fp32, which is why this list lives in host RAM and is moved to the card
    one batch at a time.
    """
    device = CPU if device is None else device
    dtype = GPU_DTYPE if dtype is None else dtype
    emb = model.model.embed_tokens
    return [emb(ids.to(emb.weight.device)).to(device=device, dtype=dtype) for ids in ids_list]


@torch.no_grad()
def collect_block_hessian(layer, tag, Xs, cfg, ctx, teacher=None, n_err=0):
    """
    H = 2 X^T X for one of the four block inputs, accumulated over `Xs` by hooking the REAL module
    whose input it is, and -- if a float `teacher` block is supplied -- the relative Frobenius
    error of that same tensor against the teacher, on the first `n_err` batches.

    Folding the error measurement into this pass is what makes eps_A / eps_M honest: they are
    measured on the SAME batches and the SAME block state that the Hessian describes, with the
    student's input coming from the live, partially-quantized model. Running them separately would
    invite exactly the mismatch that makes such numbers uninterpretable.

    Returns (H, eps) with eps None when no teacher was given.
    """
    grp, name = HESSIAN_HOOK_MODULE[tag]
    module = getattr(getattr(layer, grp), name)
    device = module.weight.device
    d_in = d_in_of(module)

    H = torch.zeros(d_in, d_in, dtype=torch.float64, device=device)
    n_samples = [0]
    cap = {}

    def _hook(mod, inputs):
        cap["x"] = inputs[0].detach()

    handle = module.register_forward_pre_hook(_hook)
    t_handle = None
    t_cap = {}
    if teacher is not None and n_err > 0:
        t_module = getattr(getattr(teacher, grp), name)

        def _t_hook(mod, inputs):
            t_cap["x"] = inputs[0].detach()

        t_handle = t_module.register_forward_pre_hook(_t_hook)

    num = den = 0.0
    try:
        for bi, X in enumerate(Xs):
            x_in = X.to(device=device, non_blocking=True)
            run_real_block(layer, x_in, ctx)
            x = cap.pop("x")
            flat = x.reshape(-1, x.shape[-1]).to(torch.float64)
            H.add_(2.0 * flat.T @ flat)
            n_samples[0] += flat.shape[0]
            del flat
            if t_handle is not None and bi < n_err:
                run_real_block(teacher, x_in, ctx)
                xt = t_cap.pop("x")
                num += (x - xt).pow(2).sum().item()
                den += xt.pow(2).sum().item()
                del xt
            del x, x_in
    finally:
        handle.remove()
        if t_handle is not None:
            t_handle.remove()

    if n_samples[0] > 0:
        H /= n_samples[0]
    eps = None if t_handle is None else (math.sqrt(num / den) if den > 0 else 0.0)
    return H, eps


@torch.no_grad()
def propagate_block(layer, Xs, ctx, inplace=True):
    """
    Replace each cached input with this block's OUTPUT, so the next block sees what the (now
    quantized) current block actually produces.

    This is the step that carries the sequential-GPTQ property across layers in the offloaded
    setting -- the GPT-2 scripts got it for free from re-running the whole live model. Run AFTER
    the block is fully quantized, never before.
    """
    device = next(layer.parameters()).device
    out = Xs if inplace else [None] * len(Xs)
    for i, X in enumerate(Xs):
        y = run_real_block(layer, X.to(device=device, non_blocking=True), ctx)
        out[i] = y.to(X.device)
        del y
    return out


@torch.no_grad()
def capture_block_input(layer, X, cfg, ctx, tag):
    """One-shot capture of one of the four block inputs from a REAL forward of `layer`. The
    single-batch analogue of `collect_block_hessian`; used by section 7's validation."""
    grp, name = HESSIAN_HOOK_MODULE[tag]
    module = getattr(getattr(layer, grp), name)
    cap = {}

    def _hook(mod, inputs):
        cap["x"] = inputs[0].detach()

    handle = module.register_forward_pre_hook(_hook)
    try:
        run_real_block(layer, X.to(module.weight.device), ctx)
    finally:
        handle.remove()
    return cap["x"]


@torch.no_grad()
def evaluate_perplexity(model, eval_windows, cfg, device=None, verbose=False):
    """
    Perplexity on a SUBSET of WikiText-2 test: N_EVAL_WINDOWS independent windows of EVAL_SEQ_LEN
    tokens, each scored with its own context and no sliding-window overlap.

    A SUBSET, and labelled as one everywhere it is printed. The GPT-2 script scored the full test
    set and its docstring explains why that mattered there: a small window count has a noise floor
    comparable to the gaps between arms. That argument has not stopped being true -- what has
    changed is the cost. A 7B forward is streamed block-by-block off the CPU here, and this
    function is called roughly thirty times across the arms and the sweep. Read the absolute
    numbers as a subset estimate and the between-arm gaps with that noise floor in mind.

    Streamed exactly like the quantization pass: embed on the CPU, then move ONE decoder layer to
    the GPU at a time and push all windows through it. Peak VRAM is one block plus the window
    cache, not a 28 GB model.
    """
    device = DEVICE if device is None else device
    hs = build_block_inputs(model, eval_windows, device=CPU)
    T = hs[0].shape[1]
    ctx = block_context(model, cfg, T, device)

    for layer in blocks_of(model):
        move_block(layer, device)
        for wi in range(len(hs)):
            hs[wi] = run_real_block(layer, hs[wi].to(device), ctx).to(CPU)
        move_block(layer, CPU)
        free()

    norm = model.model.norm.to(device)
    head = model.lm_head.to(device)
    nll_sum, n_tokens = 0.0, 0
    for wi, ids in enumerate(eval_windows):
        logits = head(norm(hs[wi].to(device))).float()
        tgt = ids.to(device)
        nll_sum += F.cross_entropy(logits[0, :-1], tgt[0, 1:], reduction="sum").item()
        n_tokens += tgt.shape[1] - 1
        del logits
    model.model.norm.to(CPU)
    model.lm_head.to(CPU)
    del hs, norm, head
    free()
    if verbose:
        print(f"    [eval: {len(eval_windows)} windows x {T} tokens = {n_tokens:,} scored tokens "
              f"(WikiText-2 test SUBSET)]")
    return math.exp(nll_sum / n_tokens)


# ============================================================================================
# 6. Calibration and evaluation data
# ============================================================================================
def build_calibration_ids(tokenizer, n_samples=None, seq_len=None):
    """`n_samples` chunks of `seq_len` tokens from WikiText-2 train, as (1, seq_len) id tensors."""
    n_samples = CALIB_N_SAMPLES if n_samples is None else n_samples
    seq_len = CALIB_SEQ_LEN if seq_len is None else seq_len

    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    # add_special_tokens=False is NOT cosmetic for Mistral, unlike GPT-2: this tokenizer prepends
    # <s>, and splicing a BOS into the middle of every mid-corpus chunk would put a token there
    # that the calibration text never contains.
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]

    out = []
    for i in range(n_samples):
        start = i * seq_len
        if start + seq_len > ids.shape[0]:
            break
        out.append(ids[start:start + seq_len].unsqueeze(0))
    return out


def build_eval_windows(tokenizer, n_windows=None, seq_len=None):
    """The evaluation subset: `n_windows` non-overlapping windows of `seq_len` tokens from
    WikiText-2 TEST. Disjoint from the calibration set, which comes from train."""
    n_windows = N_EVAL_WINDOWS if n_windows is None else n_windows
    seq_len = EVAL_SEQ_LEN if seq_len is None else seq_len
    raw = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in raw["text"] if t.strip())
    ids = tokenizer(text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    out = []
    for i in range(n_windows):
        start = i * seq_len
        if start + seq_len > ids.shape[0]:
            break
        out.append(ids[start:start + seq_len].unsqueeze(0))
    return out


# ============================================================================================
# 7. Validation
# ============================================================================================
def _validate_mistral_forward(verbose=True):
    """
    7a: hand-written `compute_block` -- RMSNorm, RoPE, GQA head repetition, SwiGLU, both residual
        adds, ALL of it -- against a real `MistralDecoderLayer`, stacked into a real
        MistralForCausalLM, on a tiny random model.

        This check is MORE load-bearing than its GPT-2 ancestor, which hand-wrote two matmuls and
        a GELU and delegated the rest. Four things here fail SILENTLY rather than loudly if they
        are wrong, and each of them is checked separately below so a passing total is not a
        coincidence of cancelling errors:
          RMSNorm  -- no mean subtraction and no bias, unlike LayerNorm. A LayerNorm here is a
                      small, smooth, everywhere-wrong perturbation.
          RoPE     -- the rotation convention (first-half/second-half vs. even/odd interleave)
                      changes the angle per dimension; attention degrades, nothing raises.
          GQA      -- 32 query heads against 8 KV heads. Expanding in the wrong order pairs each
                      query head with the wrong KV head. The GPT-2 adapters carried n_rep == 1 and
                      never exercised this path at all.
          SwiGLU   -- silu(gate(x)) * up(x), not silu(gate(x) * up(x)) and not gelu.
    7a-H: the four block Hessians must have the four expected widths -- three (hidden, hidden) and
        one (intermediate, intermediate). The cheapest possible guard against a hook wired to the
        wrong module: down_proj's input is the only 14336-wide tensor in the block, so a mis-wired
        M hook produces a 4096-wide Hessian and is caught here rather than 14336 GPTQ columns
        later. The real-model widths (4096 x3, 14336) are asserted again in section 9.
    7a-layout: `gptq_quantize_module` must accept an nn.Linear at layout="out_in" and REJECT the
        same module at layout="in_out" -- the inverted-transpose bug class this whole port turns
        on.
    7b: `evaluate_perplexity`'s streamed block-by-block forward against HF's own `labels=...` loss
        on the same window.
    """
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(0)
    tiny_cfg = MistralConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, rms_norm_eps=1e-6, sliding_window=4096,
                             bos_token_id=1, eos_token_id=2, attn_implementation="eager")
    ref = MistralForCausalLM(tiny_cfg)
    ref.config._attn_implementation = "eager"
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)

    cfg = BlockConfig(ref.config)
    # Architecture assumptions, asserted rather than assumed -- these are what the rest of this
    # function's comparisons mean.
    assert (cfg.num_heads, cfg.num_kv_heads, cfg.head_dim, cfg.n_rep) == (4, 2, 8, 2), (
        f"tiny GQA setup is wrong: {cfg!r}. This check exists to exercise n_rep > 1; with "
        f"n_rep == 1 the head-repetition path is never entered and 7a would not cover it.")
    assert cfg.q_out == 32 and cfg.kv_out == 16, (
        "k/v must be NARROWER than q under GQA -- if they are equal this model has no GQA and the "
        "repeat_kv path is untested")
    for kind in UNIT_KINDS:
        mod = module_of(ref.model.layers[0], kind)
        assert isinstance(mod, nn.Linear) and mod.bias is None, (
            f"{kind} is {type(mod).__name__} with bias={mod.bias is not None}; this script assumes "
            f"bias-free nn.Linear everywhere (see WEIGHT_LAYOUT and gptq_quantize_module)")
        assert tuple(mod.weight.shape) == cfg.unit_shapes[kind], (
            f"{kind} weight is {tuple(mod.weight.shape)}, BlockConfig says "
            f"{cfg.unit_shapes[kind]} -- nn.Linear stores (d_out, d_in), the OPPOSITE of the "
            f"Conv1D layout the GPT-2 scripts assumed")

    T = 24
    ids = torch.randint(0, 256, (1, T))
    with torch.no_grad():
        want = ref(ids).logits

    ctx = block_context(ref, cfg, T, "cpu")
    mask_bool = build_attn_mask(T, "cpu", cfg.sliding_window)
    assert int(mask_bool[T - 1].sum()) == T, "plain causal mask: the last row must attend to all"

    # --- 7a: the hand-written block, stacked over every layer ---
    with torch.no_grad():
        h = build_block_inputs(ref, [ids], device="cpu")[0]
        for layer in blocks_of(ref):
            h = compute_block(layer, h, cfg, ctx["cos"], ctx["sin"], ctx["add_mask"])["out"]
        got = ref.lm_head(ref.model.norm(h))
    rel = ((got - want).norm() / want.norm()).item()
    if verbose:
        print(f"  7a: hand-written compute_block forward vs. MistralForCausalLM.forward: "
              f"relative error {rel:.3e}")
    assert rel < 1e-3, (
        f"hand-written block forward disagrees with the real model (rel err {rel:.3e}). Check "
        f"RMSNorm (no mean subtraction), the RoPE convention, repeat_kv's expansion order, the "
        f"SwiGLU bracketing, and the nn.Linear (d_out, d_in) orientation.")

    # A tighter, localized version of the same claim: one block, hand-written vs. the real module.
    # The stacked check could in principle hide a small error behind two residual streams.
    with torch.no_grad():
        h0 = build_block_inputs(ref, [ids], device="cpu")[0]
        want_blk = run_real_block(blocks_of(ref)[0], h0, ctx)
        got_blk = compute_block(blocks_of(ref)[0], h0, cfg, ctx["cos"], ctx["sin"],
                                ctx["add_mask"])["out"]
    blk_abs = (got_blk - want_blk).abs().max().item()
    assert blk_abs < 1e-5, (
        f"compute_block disagrees with the real MistralDecoderLayer (max abs diff {blk_abs:.3e})")
    if verbose:
        print(f"  7a: compute_block vs. the real MistralDecoderLayer: max abs diff {blk_abs:.3e}")

    # Each of the four risky pieces, shown to be load-bearing on its own. A check that passes
    # because the thing it tests does not matter is worse than no check.
    probe = torch.randn(1, T, cfg.hidden_size)
    gain = blocks_of(ref)[0].input_layernorm.weight
    ln_wrong = F.layer_norm(probe, (cfg.hidden_size,), eps=cfg.rms_eps) * gain
    rms_gap = ((rms_norm(probe, gain, cfg.rms_eps) - ln_wrong).norm() / ln_wrong.norm()).item()
    assert rms_gap > 1e-3, ("RMSNorm and LayerNorm agree on this probe, so 7a cannot tell them "
                            "apart -- the check is vacuous")

    with torch.no_grad():
        no_rope = compute_block(blocks_of(ref)[0], h0, cfg, torch.ones_like(ctx["cos"]),
                                torch.zeros_like(ctx["sin"]), ctx["add_mask"])["out"]
    rope_gap = ((no_rope - got_blk).norm() / got_blk.norm()).item()
    assert rope_gap > 1e-4, "disabling RoPE changed nothing -- cos/sin are not reaching attention"

    kv_probe = torch.arange(1 * cfg.num_kv_heads * T * cfg.head_dim, dtype=torch.float32).reshape(
        1, cfg.num_kv_heads, T, cfg.head_dim)
    rep = repeat_kv(kv_probe, cfg.n_rep)
    assert rep.shape[1] == cfg.num_heads, f"repeat_kv produced {rep.shape[1]} heads"
    assert torch.equal(rep[0, 0], rep[0, 1]) and not torch.equal(rep[0, 0], rep[0, cfg.n_rep]), (
        "repeat_kv must BLOCK-repeat each KV head n_rep times (head j serves queries "
        "j*n_rep ... (j+1)*n_rep-1), not interleave them -- interleaving pairs every query head "
        "with the wrong KV head and only degrades quality")

    x2p = torch.randn(1, T, cfg.hidden_size)
    mlp0 = blocks_of(ref)[0].mlp
    with torch.no_grad():
        swiglu_ours = F.silu(mlp0.gate_proj(x2p)) * mlp0.up_proj(x2p)
        swiglu_wrong = F.silu(mlp0.gate_proj(x2p) * mlp0.up_proj(x2p))
    sg_gap = ((swiglu_ours - swiglu_wrong).norm() / swiglu_ours.norm()).item()
    assert sg_gap > 1e-3, "the two SwiGLU bracketings agree on this probe -- check is vacuous"
    if verbose:
        print(f"  7a: the risky pieces are all load-bearing -- LayerNorm-instead-of-RMSNorm would "
              f"be off by {rms_gap:.3e}, no-RoPE by {rope_gap:.3e}, mis-bracketed SwiGLU by "
              f"{sg_gap:.3e}; repeat_kv block-repeats")

    # The weight override must actually be used, and the causal mask must actually mask.
    with torch.no_grad():
        w_bad = unit_weights(blocks_of(ref)[0], dtype=torch.float32)
        w_bad = dict(w_bad)
        w_bad["q_proj"] = w_bad["q_proj"] + 0.1 * torch.randn_like(w_bad["q_proj"])
        out_bad = compute_block(blocks_of(ref)[0], h0, cfg, ctx["cos"], ctx["sin"],
                                ctx["add_mask"], w=w_bad)["out"]
        out_nomask = compute_block(blocks_of(ref)[0], h0, cfg, ctx["cos"], ctx["sin"],
                                   None)["out"]
    rel_broken = ((out_bad - got_blk).norm() / got_blk.norm()).item()
    rel_mask = ((out_nomask - got_blk).norm() / got_blk.norm()).item()
    assert rel_broken > 1e-4, ("perturbing W_q changed nothing -- compute_block is ignoring `w` "
                               "and every STE gradient below would be against the wrong weights")
    assert rel_mask > 1e-4, ("dropping the causal mask changed nothing -- either the mask is not "
                             "reaching attention or attention masks internally; either way 7a "
                             "would not catch a dropped mask, which transformers 5.x makes a real "
                             "possibility")
    if verbose:
        print(f"  compute_block's `w` override is load-bearing ({rel_broken:.3e}) and so is the "
              f"causal mask ({rel_mask:.3e})")

    # --- 7a-H: the four Hessians, and the tensors they are built from ---
    h_batches = [build_block_inputs(ref, [torch.randint(0, 256, (1, T))], device="cpu")[0]
                 for _ in range(2)]
    shapes = {}
    for tag in HESSIAN_TAGS:
        H, _ = collect_block_hessian(blocks_of(ref)[0], tag, h_batches, cfg, ctx)
        shapes[tag] = tuple(H.shape)
        want_d = cfg.hessian_dims[tag]
        assert shapes[tag] == (want_d, want_d), (
            f"Hessian '{tag}' is {shapes[tag]}, expected ({want_d}, {want_d}) -- the hook for this "
            f"tag is on the wrong module")
        free(H)
    assert shapes["x1"] == shapes["A"] == shapes["x2"] == (cfg.hidden_size, cfg.hidden_size), (
        f"three of the four Hessians must be (hidden, hidden): {shapes}")
    assert shapes["M"] == (cfg.intermediate_size, cfg.intermediate_size), (
        f"down_proj's Hessian must be (intermediate, intermediate), got {shapes['M']} -- if it is "
        f"(hidden, hidden) the M hook is seeing post_attention_layernorm(h) instead of "
        f"silu(gate(x)) * up(x)")

    # ...and each captured tensor really is the thing it is supposed to be, checked against
    # compute_block's own intermediates (which 7a just proved match the real module).
    with torch.no_grad():
        inter = compute_block(blocks_of(ref)[0], h0, cfg, ctx["cos"], ctx["sin"], ctx["add_mask"])
    for tag in HESSIAN_TAGS:
        cap = capture_block_input(blocks_of(ref)[0], h0, cfg, ctx, tag)
        gap = (cap - inter[tag]).abs().max().item()
        assert gap < 1e-5, (
            f"the '{tag}' hook captured a tensor that differs from compute_block's own {tag} by "
            f"{gap:.3e} -- one of the two is wired wrong")
    if verbose:
        print(f"  7a-H: Hessian widths {shapes}; all four hooked tensors match compute_block's "
              f"own intermediates")

    # --- 7a-layout: the inverted-transpose guard, both directions ---
    class _FakeLinear:
        def __init__(self, w):
            self.weight = torch.nn.Parameter(w.clone())
            self.bias = None

    for tag, (d_out, d_in) in (("gate_proj-like", (64, 16)), ("down_proj-like", (16, 64))):
        fake = _FakeLinear(torch.randn(d_out, d_in) * 0.05)
        Xc = torch.randn(500, d_in)
        Hc = (2.0 * Xc.T @ Xc / Xc.shape[0]).double()
        shape_before = tuple(fake.weight.data.shape)
        res = gptq_quantize_module(fake, Hc, layout="out_in", bits=4, group_size=8)
        assert tuple(res["W"].shape) == shape_before == (d_out, d_in), (
            f"gptq_quantize_module changed the {tag} weight orientation: {shape_before} -> "
            f"{tuple(res['W'].shape)}")
        _ = Xc @ fake.weight.data.T          # must not raise
        assert 0.0 < res["eps_W"] < 1.0, (
            f"eps_W for {tag} is {res['eps_W']} -- a 4-bit quantization must move the weight, but "
            f"not to the point of exceeding its own norm")
        # the WRONG layout flag must be rejected outright, not silently quantized along the output
        # axis. This is the whole reason `layout` is a required keyword.
        try:
            gptq_quantize_module(fake, Hc, layout="in_out", bits=4, group_size=8)
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"gptq_quantize_module accepted layout='in_out' for a {tag} nn.Linear weight "
                f"{shape_before} against a {tuple(Hc.shape)} Hessian -- the orientation guard is "
                f"not firing, and for the SQUARE units (q_proj, o_proj) nothing else would ever "
                f"notice")
        if verbose:
            print(f"  7a-layout: {tag} keeps its (d_out, d_in) orientation {shape_before} under "
                  f"layout='out_in', and layout='in_out' is refused")

    # The STE grid must be a no-op on GPTQ's own output (the section-13 warm-start guarantee).
    W0 = torch.randn(32, 32) * 0.05
    Xd = torch.randn(2000, 32)
    Hd = (2.0 * Xd.T @ Xd / Xd.shape[0]).double()
    Wq, sc = gptq_quantize_layer(W0.clone(), Hd, bits=4, group_size=8, act_order=True,
                                 return_scale=True, work_dtype=torch.float64, solve_device="cpu")
    qmax = 7
    rel_ste = ((torch.clamp(torch.round(Wq / sc), -qmax, qmax) * sc - Wq).norm() / Wq.norm()).item()
    assert rel_ste < 1e-6, f"return_scale is not the grid GPTQ used (rel err {rel_ste:.3e})"
    if verbose:
        print(f"  GPTQ return_scale is exact (re-quantization relative error {rel_ste:.3e})")

    # --- 7b: the streamed perplexity path vs. HF's own labels=... loss ---
    ppl_ours = evaluate_perplexity(ref, [ids], cfg, device="cpu")
    with torch.no_grad():
        ppl_hf = math.exp(ref(ids, labels=ids).loss.item())
    rel_ppl = abs(ppl_ours - ppl_hf) / ppl_hf
    if verbose:
        print(f"  7b: streamed evaluate_perplexity vs. HF's labels=... loss: ours={ppl_ours:.4f} "
              f"hf={ppl_hf:.4f} (rel err {rel_ppl:.3e})")
    assert rel_ppl < 1e-3, (
        f"the streamed block-by-block evaluation disagrees with HF's own loss (rel err "
        f"{rel_ppl:.3e}) -- the offloaded forward is not reproducing the resident one")

    free(ref)
    return rel


print("Validation 7a/7b: hand-written Mistral block forward (RMSNorm/RoPE/GQA/SwiGLU) + the "
      "streamed perplexity path vs. a real MistralForCausalLM...")
_validate_mistral_forward()
print("PASSED -- compute_block reproduces MistralDecoderLayer, the four Hessians have the right "
      "widths, and the nn.Linear orientation guard fires on its transpose.\n")


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


def expand_scale(scale_compact, g_idx, layout=None):
    """
    One unit's (d_out, n_groups) scale -- one entry per (output row, group of input columns), with
    the group map g_idx (d_in,) -- expanded to a per-weight tensor in the module's NATIVE layout.

    The layout appears here too, and for the same reason it appears in `gptq_quantize_module`:
    `scale_compact[:, g_idx]` is naturally (d_out, d_in). For `nn.Linear` that IS the native layout
    and there is nothing to do; the GPT-2 ancestor's `.T.reshape(-1)` was undoing the Conv1D
    convention and transposing here would now be the bug rather than the fix.
    """
    layout = WEIGHT_LAYOUT if layout is None else layout
    s = scale_compact[:, g_idx]                 # (d_out, d_in)
    return s if layout == "out_in" else s.T


def ste_quantize_unit(w, scale_compact, g_idx, bits, n_weights, layout=None):
    """STE-quantize ONE unit on ITS OWN grid at ITS OWN bit-width.

    Seven separate calls rather than one over a concatenated vector: the seven units are separate
    allocator units with different bit-widths, different shapes, different group counts and
    independent act_order permutations. They are also separate OPTIMIZER PARAMETERS here -- see the
    per-unit learning rate discussion in `pass_quantize_eval`, which is the reason this script does
    not reuse the GPT-2 version's single flat parameter vector."""
    return ste_quantize(w, expand_scale(scale_compact, g_idx, layout), bits, n_weights)


def unit_offsets(cfg):
    """[(kind, start, stop)] into the concatenated all-units weight vector, in UNIT_KINDS order."""
    out, o = [], 0
    for k in UNIT_KINDS:
        n = cfg.unit_params[k]
        out.append((k, o, o + n))
        o += n
    return out, o


def flatten_units(wd, cfg):
    """{kind: weight} -> one flat vector, UNIT_KINDS order. Used ONLY by the Hutchinson trace,
    which needs a single leaf tensor to differentiate twice through."""
    return torch.cat([wd[k].reshape(-1) for k in UNIT_KINDS])


def reshape_units(w_flat, cfg):
    """Inverse of `flatten_units`; differentiable, so the trace's probe flows into all seven."""
    out = {}
    for k, a, b in unit_offsets(cfg)[0]:
        out[k] = w_flat[a:b].reshape(cfg.unit_shapes[k])
    return out


def _resolve_bits_fn(bits_fn):
    """
    Normalize `bits_fn` to the 2-argument form `(layer_idx, kind) -> bits`.

    A 1-argument callable -- the uniform arms' `lambda i: 4` -- is accepted and applied to all
    seven units, so no caller written against the GPT-2 signature breaks. Arity is read off the
    signature rather than probed with try/except, so a TypeError raised from INSIDE a correct
    2-arg bits_fn is never mistaken for a 1-arg one.
    """
    if bits_fn is None:
        return None
    try:
        sig = inspect.signature(bits_fn).parameters
        params = [p for p in sig.values()
                  if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        takes_varargs = any(p.kind is p.VAR_POSITIONAL for p in sig.values())
    except (TypeError, ValueError):
        return bits_fn                      # not introspectable -- assume it is already 2-arg
    if len(params) >= 2 or takes_varargs:
        return bits_fn
    return lambda i, which: bits_fn(i)


def _bits_note(bits_by_kind):
    """Compact per-layer bit-width string, e.g. 'q4/k4/v4/o4/gate5/up5/down3'."""
    return "/".join(f"{KIND_SHORT[k]}{bits_by_kind[k]}" for k in UNIT_KINDS)


def summarize_metrics(eps_W_by_unit, eps_A_by_layer, eps_M_by_layer, cfg):
    """
    The five numbers of section 4 of the brief, assembled from one pass's per-layer records.

    rho_attn and rho_mlp are reported SEPARATELY and are never pooled into a single rho. The reason
    is not presentational: with the MLP quantized too, a block's eps_A already contains error
    propagated from the quantized MLPs of every lower layer, so eps_A and eps_M are not measuring
    the same experiment as each other and neither is comparable to a QKV-only measurement in which
    the MLP stayed in fp32.
    """
    by_kind = {k: [] for k in UNIT_KINDS}
    for name, v in eps_W_by_unit.items():
        by_kind[unit_kind_of(name)].append(v)
    eps_W_by_kind = {k: (sum(v) / len(v) if v else None) for k, v in by_kind.items()}

    def _mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    eps_W_attn = _mean([eps_W_by_kind[k] for k in ATTN_KINDS])
    eps_W_mlp = _mean([eps_W_by_kind[k] for k in MLP_KINDS])
    eps_A = _mean(list(eps_A_by_layer.values()))
    eps_M = _mean(list(eps_M_by_layer.values()))
    rho_attn = (eps_A / eps_W_attn) if (eps_A is not None and eps_W_attn) else None
    rho_mlp = (eps_M / eps_W_mlp) if (eps_M is not None and eps_W_mlp) else None
    return {
        "eps_W_by_unit": dict(eps_W_by_unit),
        "eps_W_by_kind": eps_W_by_kind,
        "eps_W_attn_mean": eps_W_attn,
        "eps_W_mlp_mean": eps_W_mlp,
        "eps_A_by_layer": dict(eps_A_by_layer),
        "eps_M_by_layer": dict(eps_M_by_layer),
        "eps_A_mean": eps_A,
        "eps_M_mean": eps_M,
        "rho_attn": rho_attn,
        "rho_mlp": rho_mlp,
    }


def print_metrics(m, indent="  "):
    """One compact block of the five metrics, per arm."""
    if not m:
        return

    def f(x):
        return "--" if x is None else f"{x:.4e}"

    print(f"{indent}eps_W by unit kind: " +
          ", ".join(f"{KIND_SHORT[k]}={f(m['eps_W_by_kind'][k])}" for k in UNIT_KINDS))
    print(f"{indent}eps_W attn={f(m['eps_W_attn_mean'])}  eps_W mlp={f(m['eps_W_mlp_mean'])}  "
          f"eps_A={f(m['eps_A_mean'])}  eps_M={f(m['eps_M_mean'])}")
    ra = "--" if m["rho_attn"] is None else f"{m['rho_attn']:.3f}"
    rm = "--" if m["rho_mlp"] is None else f"{m['rho_mlp']:.3f}"
    print(f"{indent}rho_attn={ra}   rho_mlp={rm}   (reported separately, never pooled -- eps_A "
          f"here includes error propagated from quantized MLPs in lower layers, so it is NOT "
          f"comparable to a QKV-only measurement)")


def pass_score(reader, config, cfg, calib_ids, *, samples=None, n_score_batches=None,
               n_hessian_batches=None, lambda_kl=None, group_size=None, bit_widths=None,
               perturb_mode=None, verbose=True):
    """
    Scoring pass over an all-float model, one block at a time.

    `reader` MUST be a freshly-loaded, unquantized model: scoring against an already-quantized
    model would measure "how sensitive is the already-degraded model", not "where should bits go".

    SEVEN UNITS PER LAYER, SEVEN TRACES. Each layer contributes `layer_{i}_{kind}` for all seven
    kinds:

      trace        -- per UNIT: an unbiased estimate of Tr(H_uu), the diagonal BLOCK of the
                      curvature of the block-output MSE + lambda_kl * attention-map KL restricted
                      to that unit's own coordinates. `hutchinson_block_traces` gets all seven out
                      of the SAME probes the pooled estimator used, at one HVP per probe, by
                      partitioning the final `v .* Hv` product along the seven coordinate ranges
                      instead of summing it -- the Rademacher signs are independent across
                      coordinates, so each partial sum is already an unbiased Tr(H_uu) and every
                      cross term has zero mean. Summing the seven recovers the old pooled trace
                      exactly, which is asserted below against the pooled v.Hv from the same
                      probes.
      perturbation -- per UNIT: `||Q(W)-W||_F^2` on that matrix alone.
      mean diag H  -- per unit, accumulated for free from the same forward. This is what the
                      `jab_norm` criterion divides the trace by; see section 12b for why raw traces
                      are not comparable across module kinds here.

    So Omega(unit, bits) = trace_unit * ||Q(W_unit) - W_unit||_F^2. This is the one substantive
    change from run_mistral_quant.py, where the single pooled trace was assigned to all seven units
    and Omega could only vary WITHIN a layer through the perturbation term; here the curvature term
    varies per unit too.

    Returns (jab_scores, allocator_input, hdiag_means).
    """
    samples = HUTCH_SAMPLES if samples is None else samples
    n_score_batches = JAB_N_BATCHES if n_score_batches is None else n_score_batches
    lambda_kl = LAMBDA_KL if lambda_kl is None else lambda_kl
    group_size = GROUP_SIZE if group_size is None else group_size
    bit_widths = BIT_WIDTHS if bit_widths is None else bit_widths
    perturb_mode = PERTURB_MODE if perturb_mode is None else perturb_mode

    reset_vram_peak()
    jab_scores, allocator_input, hdiag_means = {}, {}, {}

    # The scoring pass propagates its own hidden-state cache down the stack, same as
    # pass_quantize_eval -- but over an all-float model, so every layer's inputs are the float
    # model's own activations. Only n_score_batches batches are needed: nothing here collects a
    # full Hessian (perturb_mode='rtn' does not need one), and the mean-diagonal statistic that
    # jab_norm uses converges on far fewer tokens than a full 4096x4096 matrix does.
    Xs = build_block_inputs(reader, calib_ids[:n_score_batches], device=CPU)
    ctx = block_context(reader, cfg, Xs[0].shape[1], DEVICE)

    print(f"Scoring {cfg.n_layers} layers x {len(UNIT_KINDS)} units = "
          f"{cfg.n_layers * len(UNIT_KINDS)} units: {samples} Hutchinson probes x "
          f"{n_score_batches} batches, loss = MSE(block output) + {lambda_kl} * KL(attention "
          f"maps), perturbation mode '{perturb_mode}'")
    print(f"Trace estimator: PER-UNIT Tr(H_uu) from the partitioned v.Hv product "
          f"(one HVP per probe, {len(UNIT_KINDS)} traces per layer); candidate widths "
          f"{sorted(bit_widths)} under bit floor {BIT_FLOOR}")

    for i in range(cfg.n_layers):
        t_layer = time.time()
        layer = move_block(load_decoder_layer(reader, config, i), DEVICE)
        names = {k: unit_name(i, k) for k in UNIT_KINDS}

        W32 = {k: v.detach().clone().to(torch.float32) for k, v in unit_weights(layer).items()}
        w_flat = flatten_units(W32, cfg).clone().requires_grad_(True)

        traces = []
        sumsq = {t: 0.0 for t in HESSIAN_TAGS}
        ntok = {t: 0 for t in HESSIAN_TAGS}
        for X_cpu in Xs:
            X = X_cpu.to(DEVICE)
            with torch.no_grad():
                # Target from this layer's own float weights, so the loss is exactly 0 at
                # w = w_true and the trace is the curvature of the reconstruction loss there.
                ref_out = compute_block(layer, X, cfg, ctx["cos"], ctx["sin"], ctx["add_mask"],
                                        return_attn=bool(lambda_kl))
                t_out = ref_out["out"]
                t_probs = ref_out.get("probs")
                # mean(diag(H)) for each of the four inputs, from the SAME forward:
                # mean_j 2*sum_i x_ij^2 / n  ==  2 * ||X||_F^2 / (n * d). No d x d matrix needed.
                for tag in HESSIAN_TAGS:
                    tens = ref_out[tag]
                    sumsq[tag] += tens.to(torch.float64).pow(2).sum().item()
                    ntok[tag] += tens.numel() // tens.shape[-1]
                del ref_out

            def loss_fn(p):
                return block_loss(reshape_units(p, cfg), X, t_out, t_probs, layer, cfg,
                                  ctx["cos"], ctx["sin"], ctx["add_mask"], lambda_kl=lambda_kl)

            t_unit, t_pooled = hutchinson_block_traces(loss_fn, w_flat, cfg, samples=samples)
            # THE PARTITION CHECK, and the only assertion in this loop: the seven per-unit traces
            # are the pooled trace's own summands, so they must add back up to it over the same
            # probes. It costs nothing (`t_pooled` was accumulated alongside them inside
            # `hutchinson_block_traces`, from the same HVPs -- no second estimator is run) and it
            # is exactly the invariant a mis-specified coordinate range would break.
            _summed = sum(t_unit.values())
            assert abs(_summed - t_pooled) <= 1e-6 * max(abs(t_pooled), 1e-30), (
                f"layer {i}: per-unit traces sum to {_summed!r} but the pooled v.Hv over the same "
                f"probes is {t_pooled!r} (relative error "
                f"{abs(_summed - t_pooled) / max(abs(t_pooled), 1e-30):.3e} > 1e-6). The seven "
                f"coordinate ranges from `unit_offsets` do not partition the concatenated vector.")
            traces.append(t_unit)
            del t_out, t_probs, X
            free()
        # Average the per-unit dicts across batches exactly as the scalar was averaged.
        trace = {k: sum(t[k] for t in traces) / len(traces) for k in UNIT_KINDS}
        del w_flat
        free()

        hdiag = {tag: (2.0 * sumsq[tag] / (ntok[tag] * cfg.hessian_dims[tag]))
                 for tag in HESSIAN_TAGS}

        for kind in UNIT_KINDS:
            name = names[kind]
            jab_scores[name] = trace[kind]
            # `hdiag_means` is untouched: still one mean diagonal per unit, keyed by which of the
            # four Hessians that unit's input is. The `jab_norm` path therefore now divides a
            # PER-UNIT trace by a per-unit mean diagonal instead of a layer-wide trace by one.
            hdiag_means[name] = hdiag[KIND_HESSIAN[kind]]
            table = {bits: trace[kind] * grid_perturbation(W32[kind], bits, layout=WEIGHT_LAYOUT,
                                                           group_size=group_size,
                                                           mode=perturb_mode)
                     for bits in bit_widths}
            allocator_input[name] = table
        if verbose:
            pooled = sum(trace.values())
            level = "HIGH" if pooled > 100 else "MEDIUM" if pooled > 10 else "LOW"
            print(f"  layer {i:>2}: trace(sum)={pooled:.4f} [{level}]  per unit: " +
                  ", ".join(f"{KIND_SHORT[k]}={trace[k]:.4f}" for k in UNIT_KINDS))
            print("           mean diag H: " +
                  ", ".join(f"{t}={hdiag[t]:.3e}" for t in HESSIAN_TAGS))
            print("           Omega(4b): " +
                  ", ".join(f"{KIND_SHORT[k]}={allocator_input[names[k]][4]:.3e}"
                            for k in UNIT_KINDS if 4 in bit_widths))

        del W32
        free()
        # advance the cache through the (still float) block, then send the block home
        propagate_block(layer, Xs, ctx)
        move_block(layer, CPU)
        del layer
        free()

        if verbose:
            print(f"    [layer {i} scored in {time.time() - t_layer:.1f}s]")

    del Xs
    free()
    vram("after scoring pass")
    return jab_scores, allocator_input, hdiag_means


def pass_quantize_eval(reader, config, cfg, calib_ids, eval_fn, *, bits_fn=None,
                       finetune=False, n_hessian_batches=None, group_size=None, lambda_kl=None,
                       steps_per_block=None, lr=None, grad_clip_norm=None, layer_indices=None,
                       label="", verbose=True, return_notes=False, collect_metrics=True,
                       n_err_batches=None):
    """
    Quantize-and-evaluate pass, one block at a time, with the weight/activation error metrics
    computed in the SAME pass.

    `reader` MUST be a freshly-loaded model whenever `bits_fn` is not None -- quantization mutates
    the seven `.weight` tensors in place.

    bits_fn: None -> leave weights untouched (the control); else `(layer_idx, kind) -> bit-width`.
        A legacy 1-argument `layer_idx -> bits` callable is accepted and applied to all seven units.
    finetune: additionally run the STE joint fine-tune on each layer, against a float-weight
        teacher applied to the SAME input the student sees (GPTQ's own convention). The fine-tune
        is JOINT over all seven units: seven parameters, one loss, one commit decision.
    layer_indices: restrict fine-tuning to these layers; the rest still get the GPTQ warm start.
    collect_metrics: compute eps_W / eps_A / eps_M / rho_attn / rho_mlp. Needs a float teacher copy
        of the current block, which is one extra 0.87 GB resident tensor set and one extra block
        forward per error batch -- cheap next to the GPTQ solves, but switchable.

    Returns perplexity, or (perplexity, notes) if return_notes. The metrics record is left in the
    module-level LAST_METRICS (alongside LAST_FLIP_RATES / LAST_FLIP_RATES_BY_UNIT), which is what
    the arms hand to `record_arm`.
    """
    n_hessian_batches = HESSIAN_N_BATCHES if n_hessian_batches is None else n_hessian_batches
    n_err_batches = ERR_N_BATCHES if n_err_batches is None else n_err_batches
    group_size = GROUP_SIZE if group_size is None else group_size
    lambda_kl = LAMBDA_KL if lambda_kl is None else lambda_kl
    steps_per_block = JOINT_STEPS_PER_BLOCK if steps_per_block is None else steps_per_block
    lr = JOINT_LR if lr is None else lr
    grad_clip_norm = JOINT_GRAD_CLIP if grad_clip_norm is None else grad_clip_norm
    tune = None if layer_indices is None else set(layer_indices)
    bits_of = _resolve_bits_fn(bits_fn)
    if finetune and bits_of is None:
        raise ValueError("finetune=True needs a bits_fn: the STE fine-tune is warm-started from "
                         "GPTQ's own grid (scale_compact / g_idx), which only exists once the "
                         "units have been quantized. There is nothing to fine-tune against a "
                         "float weight.")

    print(f"\n=== pass: {label} ===")
    reset_vram_peak()

    notes, flip_rates, flip_rates_by_unit = {}, {}, {}
    eps_W_by_unit, eps_A_by_layer, eps_M_by_layer = {}, {}, {}

    need_inputs = bits_of is not None
    Xs = ctx = None
    if need_inputs:
        Xs = build_block_inputs(reader, calib_ids[:n_hessian_batches], device=CPU)
        ctx = block_context(reader, cfg, Xs[0].shape[1], DEVICE)

    for i in range(cfg.n_layers):
        t_layer = time.time()
        layer = load_decoder_layer(reader, config, i)
        note, note_committed, extra_lines = "", None, []

        if not need_inputs:
            if verbose:
                print(f"  layer {i:>2}/{cfg.n_layers} [{time.time() - t_layer:6.1f}s]: "
                      f"fp32 (no quantization)")
            continue

        move_block(layer, DEVICE)
        want_scales = finetune and (tune is None or i in tune)
        # (1) float snapshot of THIS BLOCK ONLY, before anything is quantized. Never the whole
        # model: a second resident 7B copy is 28 GB. Used as the eps_A/eps_M teacher and, when
        # fine-tuning, as the reconstruction target.
        teacher = teacher_copy(layer) if (collect_metrics or want_scales) else None

        bits_by_kind = {k: bits_of(i, k) for k in UNIT_KINDS}
        # (2) GPTQ warm start: four Hessians, seven units, in sequential order. Each Hessian
        # reflects the units quantized before it, and -- because Xs was propagated through the
        # already-quantized layers 0..i-1 -- also every earlier layer's quantization.
        qb = quantize_block(layer, Xs, lambda k: bits_by_kind[k], cfg, ctx, teacher=teacher,
                            group_size=group_size, return_compact=want_scales,
                            n_hessian_batches=n_hessian_batches,
                            n_err_batches=n_err_batches if collect_metrics else 0)
        for k, v in qb["eps_W"].items():
            eps_W_by_unit[unit_name(i, k)] = v
        if qb["eps_A"] is not None:
            eps_A_by_layer[i] = qb["eps_A"]
        if qb["eps_M"] is not None:
            eps_M_by_layer[i] = qb["eps_M"]
        note = _bits_note(bits_by_kind)
        mean_eps_W = sum(qb["eps_W"].values()) / len(qb["eps_W"])
        note += f", eps_W {mean_eps_W:.3e}"
        if qb["eps_A"] is not None:
            note += f", eps_A {qb['eps_A']:.3e}, eps_M {qb['eps_M']:.3e}"

        # (3) STE joint fine-tune of all seven units against the float-block teacher
        if want_scales:
            gp = qb["gptq"]
            w = {k: gp[k]["W"].detach().clone().to(torch.float32).requires_grad_(True)
                 for k in UNIT_KINDS}
            gptq_w = {k: w[k].detach().clone() for k in UNIT_KINDS}
            scale_w = {k: gp[k]["scale"].detach().to(torch.float32) for k in UNIT_KINDS}
            g_idx = {k: gp[k]["g_idx"] for k in UNIT_KINDS}
            sp = {k: gp[k]["compact"].detach().clone().to(torch.float32).requires_grad_(True)
                  for k in UNIT_KINDS}

            # ---------------------------------------------------------------------------------
            # PER-UNIT grid step. The GPT-2 MLP script computed ONE learning rate from
            # `scale_flat.mean()` over the CONCATENATED [c_fc | c_proj] vector -- pooled across two
            # matrices with different input dimensionality. That is the fixed-absolute-rate failure
            # its own logs show: deep layers reported flip rate exactly 0.000 with the FT loss
            # unchanged to six decimals, because a step sized for the pooled mean was far below the
            # grid step of the finer-grained matrix and no latent weight ever crossed a rounding
            # boundary.
            #
            # Here s is the mean grid step of THAT UNIT, and each unit gets its own Adam parameter
            # group at eta_w = ALPHA_W * s_unit. The seven units span a 14x size range and four
            # different input distributions (post-RMSNorm hidden, merged attention output,
            # post-RMSNorm hidden again, and the SwiGLU product), so their grid steps differ by
            # more than the two GPT-2 matrices' did -- pooling would be a worse mistake here, not a
            # smaller one. Both s and the resulting flip rate are logged per unit on every layer
            # line so a repeat of the 0.000 failure is visible in the log rather than inferred
            # afterwards.
            # ---------------------------------------------------------------------------------
            s_unit = {k: scale_w[k].mean().item() for k in UNIT_KINDS}
            n_weights = group_size
            opt = torch.optim.Adam(
                [{"params": [w[k]], "lr": ALPHA_W * s_unit[k]} for k in UNIT_KINDS] +
                [{"params": [sp[k]], "lr": ALPHA_S * s_unit[k]} for k in UNIT_KINDS])
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps_per_block)
            del gp, qb
            free()

            def quantized(wd, spd):
                return {k: ste_quantize_unit(wd[k], spd[k], g_idx[k], bits_by_kind[k], n_weights)
                        for k in UNIT_KINDS}

            # The FT batch pool is CAPPED, unlike the GPT-2 script's cache-every-calibration-batch.
            # Each cached batch holds a block-output target AND an attention-map target
            # (1, 32, 512, 512) = 33 MB; caching all 128 would be 5 GB of teacher targets alone.
            pool_n = max(1, min(FT_POOL, len(Xs) - 1))
            held_out = pool_n                       # reserved out of the cycling pool entirely
            pool = list(range(pool_n))
            start = (i * steps_per_block) % len(pool)
            order = [pool[(start + s) % len(pool)] for s in range(steps_per_block)]

            def targets(j):
                # Teacher and student see the SAME input (GPTQ's own convention:
                # target = block_float(X_quant)). X comes from the propagated cache, so it already
                # reflects layers 0..i-1's quantized state; a separate float trajectory would let
                # the residual carry upstream error this layer cannot fix.
                X = Xs[j].to(DEVICE)
                with torch.no_grad():
                    ref_out = compute_block(teacher, X, cfg, ctx["cos"], ctx["sin"],
                                            ctx["add_mask"], return_attn=bool(lambda_kl))
                return X, ref_out["out"].detach(), (ref_out.get("probs").detach()
                                                    if lambda_kl else None)

            batch_cache = {j: targets(j) for j in set(order) | {held_out}}

            def score(wd, spd):
                with torch.no_grad():
                    X_ho, t_ho, p_ho = batch_cache[held_out]
                    return block_loss(quantized(wd, spd), X_ho, t_ho, p_ho, layer, cfg,
                                      ctx["cos"], ctx["sin"], ctx["add_mask"],
                                      lambda_kl=lambda_kl).item()

            det_w = {k: w[k].detach() for k in UNIT_KINDS}
            det_s = {k: sp[k].detach() for k in UNIT_KINDS}
            init_loss = score(det_w, det_s)
            best_loss = init_loss
            best_w = {k: w[k].detach().clone() for k in UNIT_KINDS}
            for step_idx, j in enumerate(order):
                X, t_out, t_probs = batch_cache[j]
                opt.zero_grad()
                loss = block_loss(quantized(w, sp), X, t_out, t_probs, layer, cfg,
                                  ctx["cos"], ctx["sin"], ctx["add_mask"], lambda_kl=lambda_kl,
                                  log_components=LOG_LOSS_COMPONENTS)
                loss.backward()
                torch.nn.utils.clip_grad_norm_([w[k] for k in UNIT_KINDS] +
                                               [sp[k] for k in UNIT_KINDS], grad_clip_norm)
                opt.step()
                sched.step()
                if step_idx == steps_per_block - 1 or (step_idx + 1) % SCORE_EVERY == 0:
                    step_loss = score(det_w, det_s)
                    if step_loss < best_loss:
                        best_loss = step_loss
                        with torch.no_grad():
                            q = quantized(det_w, det_s)
                            best_w = {k: q[k].detach().clone() for k in UNIT_KINDS}
                            del q
                free(loss)
            del batch_cache
            free()

            # diagnostic B: fraction of grid positions fine-tuning actually moved, regardless of
            # whether the result was committed -- against the FROZEN GPTQ grid. Per unit AND
            # pooled: the seven units sit at four different points in the block and routinely get
            # different bit-widths, so one pooled number can hide a unit being completely inert
            # (which is exactly the failure the per-unit learning rate above is meant to fix).
            fr = {}
            moved = total = 0
            with torch.no_grad():
                for k in UNIT_KINDS:
                    flips = (torch.round(gptq_w[k] / scale_w[k]) !=
                             torch.round(best_w[k] / scale_w[k]))
                    fr[k] = flips.float().mean().item()
                    moved += int(flips.sum().item())
                    total += flips.numel()
                    del flips
            flip_rate = moved / total

            # Safety net: commit only if fine-tuning beat the GPTQ-only starting point. Otherwise
            # the modules already hold that GPTQ-only solution -- leave them alone.
            if best_loss <= init_loss:
                with torch.no_grad():
                    for k in UNIT_KINDS:
                        module_of(layer, k).weight.data.copy_(
                            best_w[k].to(module_of(layer, k).weight.dtype))
                note += f", FT {init_loss:.6f} -> {best_loss:.6f}, flip {flip_rate:.3f}"
                note_committed = True
            else:
                note += (f", kept GPTQ-only ({init_loss:.6f} vs FT {best_loss:.6f}), "
                         f"flip {flip_rate:.3f}")
                note_committed = False
            flip_rates[i] = flip_rate
            for k in UNIT_KINDS:
                flip_rates_by_unit[unit_name(i, k)] = fr[k]
            extra_lines.append("    per-unit s (mean grid step) and flip rate: " +
                               ", ".join(f"{KIND_SHORT[k]} s={s_unit[k]:.3e} f={fr[k]:.3f}"
                                         for k in UNIT_KINDS))
            del w, gptq_w, scale_w, sp, best_w, opt, sched, det_w, det_s
            del targets, score, quantized
            free()
        elif finetune:
            note += ", GPTQ-only (not in layer_indices)"

        del teacher
        free()

        # (4) advance the cached hidden states through the now-quantized block -- this is what
        # carries the sequential-GPTQ property to layer i+1 in the offloaded setting.
        propagate_block(layer, Xs, ctx)
        move_block(layer, CPU)
        del layer
        free()

        notes[i] = (tuple(bits_by_kind[k] for k in UNIT_KINDS), note_committed)
        if verbose:
            print(f"  layer {i:>2}/{cfg.n_layers} [{time.time() - t_layer:6.1f}s]: {note}")
            for line in extra_lines:
                print(line)

    del Xs, ctx
    free()

    metrics = None
    if collect_metrics and eps_W_by_unit:
        metrics = summarize_metrics(eps_W_by_unit, eps_A_by_layer, eps_M_by_layer, cfg)
    globals()["LAST_FLIP_RATES"] = flip_rates
    globals()["LAST_FLIP_RATES_BY_UNIT"] = flip_rates_by_unit
    globals()["LAST_METRICS"] = metrics

    ppl = eval_fn()
    vram(f"after '{label}'")
    print(f"=== {label}: perplexity {ppl:.3f} (WikiText-2 test SUBSET, "
          f"{N_EVAL_WINDOWS} x {EVAL_SEQ_LEN} tokens) ===")
    if metrics:
        print_metrics(metrics, indent="    ")
    return (ppl, notes) if return_notes else ppl


def _validate_adaptive_joint_pass(verbose=True):
    """
    7c-warm: `ste_quantize_unit` must be an exact no-op on GPTQ's own output for ALL SEVEN units --
        the warm-start guarantee, now across seven independently permuted grids with different
        shapes, group counts and bit-widths. Mis-map one g_idx and the warm start silently starts
        off-grid: init_loss is wrong, so the commit-only-if-better safety net compares against the
        wrong baseline and every flip rate is measured from the wrong origin. Nothing would raise.
    7c: `pass_quantize_eval(bits_fn=..., finetune=True)` on a hand-made mixed 2/4/8-bit assignment,
        tiny random Mistral, CPU-only, seconds not minutes. Exercises the section-13b code path
        (adaptive allocation + joint fine-tuning + the eps metrics) end to end, which 7a/7b and the
        real arms each individually miss.

    The tiny model is a real MistralForCausalLM, so this runs the same offloading, the same four
    Hessians and the same seven-unit fine-tune the 7B run does -- just at hidden=32.
    """
    from transformers import MistralConfig, MistralForCausalLM

    torch.manual_seed(1)
    tiny_cfg = MistralConfig(vocab_size=256, hidden_size=32, intermediate_size=64,
                             num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                             max_position_embeddings=64, rms_norm_eps=1e-6, sliding_window=4096,
                             bos_token_id=1, eos_token_id=2, attn_implementation="eager")
    ref = MistralForCausalLM(tiny_cfg)
    ref.config._attn_implementation = "eager"
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    cfg = BlockConfig(ref.config)

    # Every kind gets a DIFFERENT (layer 0, layer 1) pair of widths, so a mis-routed `kind`
    # argument cannot pass unnoticed; 2-bit is included deliberately -- the allocator emits it and
    # it must not be silently clamped up to some higher floor.
    layer0 = {k: 2 for k in UNIT_KINDS}
    layer1 = {"q_proj": 4, "k_proj": 8, "v_proj": 4, "o_proj": 8,
              "gate_proj": 4, "up_proj": 8, "down_proj": 4}
    mixed_bits = {(0, k): layer0[k] for k in UNIT_KINDS}
    mixed_bits.update({(1, k): layer1[k] for k in UNIT_KINDS})

    calib_ids = [torch.randint(0, 256, (1, 16)) for _ in range(4)]
    eval_windows = [torch.randint(0, 256, (1, 16)) for _ in range(2)]

    old = (globals()["DEVICE"], globals()["GPTQ_SOLVE_DEVICE"])
    globals()["DEVICE"], globals()["GPTQ_SOLVE_DEVICE"] = "cpu", "cpu"
    try:
        # --- 7c-warm ---
        warm = MistralForCausalLM(tiny_cfg)
        warm.config._attn_implementation = "eager"
        warm.eval()
        for p in warm.parameters():
            p.requires_grad_(False)
        Xw = build_block_inputs(warm, calib_ids[:2], device="cpu")
        ctxw = block_context(warm, cfg, Xw[0].shape[1], "cpu")
        warm_bits = dict(layer1)          # differing widths, so a swapped qmax cannot cancel out
        qb = quantize_block(blocks_of(warm)[0], Xw, lambda k: warm_bits[k], cfg, ctxw,
                            group_size=8, return_compact=True, n_hessian_batches=2,
                            n_err_batches=0)
        total_flips = 0
        for k in UNIT_KINDS:
            g = qb["gptq"][k]
            w0 = g["W"].detach().to(torch.float32)
            scale = g["scale"].detach().to(torch.float32)
            # the expanded per-weight scale must BE the grid GPTQ reported, in the native layout
            gap = (expand_scale(g["compact"].to(torch.float32), g["g_idx"]) - scale).abs()
            assert gap.max().item() == 0.0, (
                f"{k}: expand_scale does not reproduce GPTQ's own per-weight scale (max abs diff "
                f"{gap.max().item():.3e}) -- this unit's g_idx is mis-mapped")
            with torch.no_grad():
                w0_q = ste_quantize_unit(w0, g["compact"].to(torch.float32), g["g_idx"],
                                         warm_bits[k], 8)
                # The INTEGER grid positions are what has to be identical, and they are exactly
                # what `flip_rate` measures. gptq_quantize_layer solves in float64 and casts to
                # float32, so `q * s` recomputed in float32 differs by about one ULP.
                flips = (torch.round(w0 / scale) != torch.round(w0_q / scale)).sum().item()
                wrong = ste_quantize_unit(w0, g["compact"].to(torch.float32), g["g_idx"], 2, 8)
                wrong_flips = (torch.round(w0 / scale) != torch.round(wrong / scale)).sum().item()
            assert flips == 0, (
                f"{k}: ste_quantize_unit moved {flips} of {w0.numel()} grid positions on GPTQ's "
                f"own {warm_bits[k]}-bit output -- the STE warm start starts off-grid")
            rel = ((w0_q - w0).norm() / w0.norm()).item()
            assert rel < 1e-6, f"{k}: ste_quantize_unit is not a no-op (rel err {rel:.3e})"
            if warm_bits[k] != 2:
                assert wrong_flips > 0, (
                    f"{k}: re-quantizing at 2 bits moved no grid positions -- the warm-start "
                    f"check is vacuous for this unit")
                total_flips += wrong_flips
            assert 0.0 < g["eps_W"] < 1.0, f"{k}: eps_W={g['eps_W']} is not a sane relative error"
        if verbose:
            print(f"  7c-warm: ste_quantize_unit is a no-op on GPTQ's own output for all "
                  f"{len(UNIT_KINDS)} units (0 grid flips each, all g_idx maps correct); "
                  f"re-quantizing at 2b instead moves {total_flips:,} positions")
        del warm, qb, Xw
        free()

        # --- 7c ---
        def synthetic_eval_fn():
            return evaluate_perplexity(ref, eval_windows, cfg, device="cpu")

        ppl, notes = pass_quantize_eval(
            ref, ref.config, cfg, calib_ids, synthetic_eval_fn,
            bits_fn=lambda i, which: mixed_bits[(i, which)], finetune=True,
            steps_per_block=2, n_hessian_batches=3, n_err_batches=2, group_size=8,
            label="7c: mixed 2/4/8-bit adaptive+joint (tiny CPU Mistral)", verbose=False,
            return_notes=True)
        metrics = globals().get("LAST_METRICS")
    finally:
        globals()["DEVICE"], globals()["GPTQ_SOLVE_DEVICE"] = old

    assert math.isfinite(ppl) and ppl > 0, f"non-finite/degenerate perplexity: {ppl}"
    assert set(notes) == {0, 1}, f"return_notes must record every bits_fn'd layer, got {set(notes)}"
    for i in sorted(notes):
        got_bits, committed = notes[i]
        want_bits = tuple(mixed_bits[(i, k)] for k in UNIT_KINDS)
        assert tuple(got_bits) == want_bits, (
            f"layer {i}: note recorded {got_bits}, assigned {want_bits} -- bits_fn's `kind` "
            f"argument is being ignored or mis-routed")
        assert committed in (True, False), (
            f"layer {i}: finetune=True but committed={committed!r} -- every unit needs a real "
            f"fine-tune attempt and a True/False outcome, not None")

    # The weights must actually be ON the assigned grids afterwards, not merely recorded in the
    # notes. Symmetric quantization puts every weight sharing one scale on at most 2**bits - 1
    # levels; scales are per (OUTPUT ROW, group of INPUT columns), so for an nn.Linear weight
    # (d_out, d_in) the set sharing a scale is a run of COLUMNS within one ROW, and a whole row
    # spans n_groups of them. NOTE THE ORIENTATION: for the GPT-2 Conv1D ancestor this bound was
    # taken per COLUMN. Taking it along the wrong axis here would make the check vacuous.
    levels = {}
    for i in sorted(notes):
        for kind in UNIT_KINDS:
            bits = mixed_bits[(i, kind)]
            W = module_of(blocks_of(ref)[i], kind).weight.data
            n_groups = (W.shape[1] + 7) // 8               # group_size=8 in this test
            max_levels = n_groups * (2 ** bits - 1)
            worst = max(torch.unique(W[j, :]).numel() for j in range(W.shape[0]))
            assert worst <= max_levels, (
                f"layer {i} {kind}: a row holds {worst} distinct values, more than the "
                f"{max_levels} a {bits}-bit grid over {n_groups} groups allows -- this unit was "
                f"not quantized at the width it was assigned")
            levels[(i, kind)] = worst

    for kind in UNIT_KINDS:
        by_bits = sorted((mixed_bits[(i, kind)], levels[(i, kind)]) for i in sorted(notes))
        assert all(a[1] <= b[1] for a, b in zip(by_bits, by_bits[1:])), (
            f"{kind}: distinct-value count is not monotone in assigned bits {by_bits} -- bits_fn's "
            f"`kind` argument is being ignored or mis-routed")
        assert by_bits[0][1] < by_bits[-1][1], (
            f"{kind}: the {by_bits[0][0]}-bit and {by_bits[-1][0]}-bit weights are equally "
            f"fine-grained ({by_bits[0][1]} levels) -- this check is vacuous")

    # --- the metrics, on the same pass ---
    assert metrics is not None, "collect_metrics defaulted on but no metrics record was produced"
    assert set(metrics["eps_W_by_unit"]) == {unit_name(i, k) for i in (0, 1) for k in UNIT_KINDS}, \
        "eps_W must be recorded for every unit of every quantized layer"
    for key in ("eps_A_mean", "eps_M_mean", "rho_attn", "rho_mlp"):
        v = metrics[key]
        assert v is not None and math.isfinite(v) and v > 0, f"{key} is {v!r}"
    assert metrics["rho_attn"] != metrics["rho_mlp"], (
        "rho_attn and rho_mlp came out identical -- they are different ratios over different "
        "units and one of them is almost certainly reading the other's numbers")
    if verbose:
        print(f"  7c: mixed 2/4/8-bit adaptive+joint ran end-to-end on a tiny CPU Mistral "
              f"(ppl={ppl:.3f}); eps_A={metrics['eps_A_mean']:.3e}, "
              f"eps_M={metrics['eps_M_mean']:.3e}, rho_attn={metrics['rho_attn']:.3f}, "
              f"rho_mlp={metrics['rho_mlp']:.3f}")

    # Backwards compatibility: a plain 1-arg bits_fn (the uniform arms' `lambda i: 4`) must still
    # work and must apply to all seven units.
    torch.manual_seed(1)
    ref2 = MistralForCausalLM(tiny_cfg)
    ref2.config._attn_implementation = "eager"
    ref2.eval()
    for p in ref2.parameters():
        p.requires_grad_(False)
    globals()["DEVICE"], globals()["GPTQ_SOLVE_DEVICE"] = "cpu", "cpu"
    try:
        _, notes2 = pass_quantize_eval(
            ref2, ref2.config, cfg, calib_ids,
            lambda: evaluate_perplexity(ref2, eval_windows, cfg, device="cpu"),
            bits_fn=lambda i: 4, finetune=False, n_hessian_batches=2, n_err_batches=1,
            group_size=8, label="7c: legacy 1-arg bits_fn", verbose=False, return_notes=True)
    finally:
        globals()["DEVICE"], globals()["GPTQ_SOLVE_DEVICE"] = old
    assert all(tuple(b) == (4,) * len(UNIT_KINDS) for b, _ in notes2.values()), (
        f"a legacy bits_fn(layer_idx) must apply to all seven units; got "
        f"{[b for b, _ in notes2.values()]}")
    if verbose:
        print("  legacy 1-arg bits_fn(layer_idx) still applies to all seven units")

    free(ref, ref2)
    return ppl


print("Validation 7c (deferred from section 7 -- needs pass_quantize_eval, defined just above): "
      "adaptive allocation + joint fine-tuning + the eps metrics, mixed 2/4/8-bit (tiny CPU "
      "Mistral)...")
_validate_adaptive_joint_pass()
print("PASSED -- pass_quantize_eval(bits_fn=..., finetune=True) is bit-width-agnostic end to end, "
      "routes all seven units independently, and produces the eps/rho metrics.\n")


# =============================================================================================
# 9. Setup
# =============================================================================================
section("## 9. Setup")

_t_setup = time.time()
print(f"Loading {MODEL_ID}'s tokenizer and config (cached after the first run)...")
# Only the tokenizer and config are loaded here, NOT a model -- every arm below loads its OWN
# fresh model right before it quantizes anything, so there is no shared, mutable `reader` for one
# arm's damage to leak into the next.
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
config = AutoConfig.from_pretrained(MODEL_ID)
config._attn_implementation = "eager"

BLOCK = BlockConfig(config)
print(BLOCK)

# --- shape assertions, all of them checked rather than assumed ------------------------------
assert BLOCK.intermediate_size == 14336, (
    f"intermediate_size is {BLOCK.intermediate_size}, expected 14336. Every memory estimate in "
    f"this file -- the 1.64 GB fp64 down_proj Hessian, the 40 GB free-memory guard, the fp64 "
    f"benchmark dimension -- is sized against 14336.")
assert BLOCK.hidden_size % GROUP_SIZE == 0, (
    f"GROUP_SIZE={GROUP_SIZE} must divide hidden_size={BLOCK.hidden_size}")
assert BLOCK.intermediate_size % GROUP_SIZE == 0, (
    f"GROUP_SIZE={GROUP_SIZE} must ALSO divide intermediate_size={BLOCK.intermediate_size} -- "
    f"down_proj's input dimension is the intermediate size, not the hidden size. A group size "
    f"that divides 4096 but not 14336 leaves down_proj with a ragged final group, which "
    f"gptq_quantize_layer handles but grid_perturbation would then score on a differently-shaped "
    f"grid.")
_want_h = (BLOCK.hidden_size, BLOCK.hidden_size)
assert (BLOCK.hessian_dims["x1"], BLOCK.hessian_dims["A"], BLOCK.hessian_dims["x2"]) == \
       (BLOCK.hidden_size,) * 3 and BLOCK.hessian_dims["M"] == BLOCK.intermediate_size, (
    f"the four block Hessians must be {_want_h} x3 and "
    f"({BLOCK.intermediate_size}, {BLOCK.intermediate_size}); BlockConfig says "
    f"{BLOCK.hessian_dims}")
print(f"Four Hessians per block: x1 {_want_h}, A ({BLOCK.q_out}, {BLOCK.q_out}), x2 {_want_h}, "
      f"M ({BLOCK.intermediate_size}, {BLOCK.intermediate_size}) "
      f"[{BLOCK.intermediate_size ** 2 * 8 / 1e9:.2f} GB in float64]")
assert BLOCK.num_kv_heads < BLOCK.num_heads, (
    f"expected GQA ({BLOCK.num_heads} query heads vs {BLOCK.num_kv_heads} KV heads); with "
    f"n_rep == 1 the repeat_kv path in compute_block is never exercised by the real model")
if BLOCK.sliding_window is not None:
    assert max(CALIB_SEQ_LEN, EVAL_SEQ_LEN) <= BLOCK.sliding_window, (
        f"sequence length {max(CALIB_SEQ_LEN, EVAL_SEQ_LEN)} exceeds the sliding window "
        f"{BLOCK.sliding_window}; build_attn_mask's plain-causal fast path would then differ from "
        f"what the real model does")

print(f"Quantizable parameters per layer: {BLOCK.block_numel:,} "
      f"(attention {BLOCK.attn_numel:,} + MLP {BLOCK.mlp_numel:,})")
print("Per-unit parameter counts (this is what `cost` weights by -- a 14x spread):")
for k in UNIT_KINDS:
    print(f"  {k:<10} {str(BLOCK.unit_shapes[k]):>16}  {BLOCK.unit_params[k]:>12,}  "
          f"({100 * BLOCK.unit_params[k] / BLOCK.block_numel:5.1f}% of the block)")
print(f"Allocator units: {BLOCK.n_layers} layers x {len(UNIT_KINDS)} units = "
      f"{BLOCK.n_layers * len(UNIT_KINDS)}")

print("\nBuilding the shared calibration set (used for every arm below)...")
calib_ids = build_calibration_ids(tokenizer)
print(f"{len(calib_ids)} calibration chunks of {CALIB_SEQ_LEN} tokens (WikiText-2 train)")
eval_windows = build_eval_windows(tokenizer)
print(f"Evaluation: {len(eval_windows)} windows x {EVAL_SEQ_LEN} tokens of WikiText-2 test -- a "
      f"SUBSET, not the full test set (see evaluate_perplexity, section 5, for the noise-floor "
      f"caveat that comes with it)")


def eval_fn_for(model):
    """The per-arm evaluation closure. Factored out so every arm below is identical."""
    return lambda: evaluate_perplexity(model, eval_windows, BLOCK)


# `results` is the plain {arm: perplexity} dict the comparison tables read. On a resumed run it is
# pre-populated from results.json, so those tables still show arms finished by an earlier run.
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
    "fp64_benchmark": _BENCH,
    "unit_params": BLOCK.unit_params,
    "config": {
        "CALIB_N_SAMPLES": CALIB_N_SAMPLES, "CALIB_SEQ_LEN": CALIB_SEQ_LEN,
        "HESSIAN_N_BATCHES": HESSIAN_N_BATCHES, "HUTCH_SAMPLES": HUTCH_SAMPLES,
        "JAB_N_BATCHES": JAB_N_BATCHES, "ERR_N_BATCHES": ERR_N_BATCHES,
        "LAMBDA_KL": LAMBDA_KL, "GROUP_SIZE": GROUP_SIZE,
        "PERTURB_MODE": PERTURB_MODE, "BIT_WIDTHS": BIT_WIDTHS,
        "BIT_WIDTHS_ARG": BIT_WIDTHS_ARG, "BIT_FLOOR": BIT_FLOOR,
        "SCORING_VERSION": SCORING_VERSION,
        "N_EVAL_WINDOWS": N_EVAL_WINDOWS, "EVAL_SEQ_LEN": EVAL_SEQ_LEN,
        "JOINT_STEPS_PER_BLOCK": JOINT_STEPS_PER_BLOCK, "JOINT_LR": JOINT_LR,
        "JOINT_GRAD_CLIP": JOINT_GRAD_CLIP, "JOINT_LAYERS": JOINT_LAYERS,
        "ALPHA_W": ALPHA_W, "ALPHA_S": ALPHA_S, "SCORE_EVERY": SCORE_EVERY, "FT_POOL": FT_POOL,
        "PROPAGATION_WEIGHT": PROPAGATION_WEIGHT, "SWEEP_BITS": SWEEP_BITS,
        "TYPE_OFFSETS": TYPE_OFFSETS, "TYPE_OFFSET_FT": TYPE_OFFSET_FT,
        "TARGET_AVG_BITS": TARGET_AVG_BITS,
        "JOINT_LAYERS_DEPTH_TEST": list(JOINT_LAYERS_DEPTH_TEST),
        "GPU_DTYPE": str(GPU_DTYPE), "GPTQ_DTYPE": str(GPTQ_DTYPE),
        "GPTQ_SOLVE_DEVICE": str(GPTQ_SOLVE_DEVICE), "WEIGHT_LAYOUT": WEIGHT_LAYOUT,
    },
}
flush_results()
print(f"\n[section 9 setup: {time.time() - _t_setup:.1f}s]")


# =============================================================================================
# 10. fp32 control
# =============================================================================================
section("## 10. fp32 control")

if RUN_FP16_BASELINE:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID)   # fresh, unmutated by any prior arm
    results["fp16"] = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=None, label="fp32 control (no quantization)")
    # Mistral-7B-v0.1 on WikiText-2 sits around 5-6 for the full test set; a 32-window subset can
    # sit anywhere in a wider band depending on which windows it caught, so the guard is loose --
    # it exists to catch a broken pipeline, not to calibrate anything.
    if not (3.0 < results["fp16"] < 15.0):
        print(f"\nWARNING: fp32 control perplexity {results['fp16']:.3f} is outside the expected "
              f"band for {MODEL_ID} on a WikiText-2 subset. Investigate the pipeline before "
              f"reading anything into the quantization results below.")
    record_arm("fp16", perplexity=results["fp16"], elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("RUN_FP16_BASELINE=False -- skipping the control pass.")


# =============================================================================================
# 11. Uniform 4-bit GPTQ baseline
# =============================================================================================
section("## 11. Uniform 4-bit GPTQ baseline (all seven units)")

if RUN_UNIFORM4 and not uniform_feasible(4, "uniform4 (sec 11)"):
    RUN_UNIFORM4 = False

if RUN_UNIFORM4:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    # `lambda i: 4` is the GPT-2 scripts' bits_fn verbatim -- a 1-argument callable.
    # _resolve_bits_fn accepts it and applies 4 bits to all seven units, so this arm needed no
    # edit beyond the unit count.
    results["uniform4"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i: 4, label="uniform 4-bit GPTQ (all 7 units)", return_notes=True)
    record_arm("uniform4", perplexity=results["uniform4"], notes=_notes,
               bits={unit_name(i, k): 4 for i in range(BLOCK.n_layers) for k in UNIT_KINDS},
               metrics=globals().get("LAST_METRICS"), elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("uniform4 not selected -- skipping.")


# =============================================================================================
# 12. JAB-Hessian adaptive allocation
# =============================================================================================
section("## 12. JAB-Hessian adaptive allocation")

PARAMS = unit_params_map(BLOCK)
TOTAL_PARAMS = sum(PARAMS.values())

assignment = None
jab_scores = allocator_input = hdiag_means = None

if NEED_SCORING:
    _cached = None if ARGS.rerun else RESULTS.get("scoring")
    if _cached and _cached.get("scoring_version") != SCORING_VERSION:
        print(f"Ignoring the cached section-12 scoring block in results.json: tagged "
              f"{_cached.get('scoring_version')!r}, not {SCORING_VERSION!r} -- old per-block "
              f"traces are not interchangeable with per-unit ones. Re-scoring.")
        _cached = None
    if _cached and "hdiag_means" in _cached:
        # Reload rather than re-score. JSON turns the allocator table's integer bit-width keys into
        # strings, so they are converted back here; nothing else about the table changes.
        print("Reusing the cached section-12 scoring pass from results.json "
              f"(scored in {_cached.get('elapsed_s', float('nan')):.1f}s by an earlier run). "
              "Pass --rerun to re-score.")
        jab_scores = {n: float(v) for n, v in _cached["jab_scores"].items()}
        allocator_input = {n: {int(b): float(v) for b, v in t.items()}
                           for n, t in _cached["allocator_input"].items()}
        hdiag_means = {n: float(v) for n, v in _cached["hdiag_means"].items()}
        assignment = {n: int(b) for n, b in _cached["assignment"].items()}
        TARGET_AVG_BITS = float(_cached.get("target_avg_bits", TARGET_AVG_BITS))
    else:
        _t_score = time.time()
        reader, _ = load_model_and_tokenizer(MODEL_ID)   # fresh, unquantized -- see pass_score
        torch.manual_seed(ARGS.seed)     # the Hutchinson probes are reproducible run to run
        jab_scores, allocator_input, hdiag_means = pass_score(reader, config, BLOCK, calib_ids)
        del reader
        free()

        # `budget` is BITS PER PARAMETER x total parameters. This is the parameter-weighted form
        # mandated by the unit-size spread (see `cost`, section 4); reading it as "bits per matrix"
        # -- which is what the GPT-2 scripts could get away with -- would be wrong by up to 14x per
        # unit here.
        budget = TARGET_AVG_BITS * TOTAL_PARAMS
        assignment, cost_used, sensitivity = greedy_allocate(allocator_input, budget, PARAMS)
        print(f"\nBudget {budget / 1e9:.2f} Gbit ({TARGET_AVG_BITS} bits/param x "
              f"{TOTAL_PARAMS:,} params over {len(allocator_input)} units)")
        print(f"Allocated: {cost_used / 1e9:.2f} Gbit, {cost_used / TOTAL_PARAMS:.3f} bits/param, "
              f"total sensitivity {sensitivity:.4e}")
        RESULTS["scoring"] = {
            "scoring_version": SCORING_VERSION,
            "bit_floor": BIT_FLOOR,
            "bit_widths": BIT_WIDTHS,
            "min_bits": min(assignment.values()),
            "jab_scores": jab_scores,
            "allocator_input": allocator_input,
            "hdiag_means": hdiag_means,
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

    # diagnostic A: assignment histogram, overall and split by unit kind -- the seven kinds sit at
    # different points in the block, see different input distributions and differ 14x in size, so
    # a single pooled histogram would hide almost everything interesting.
    hist = Counter(assignment.values())
    print("Assignment histogram (bits -> unit count):", dict(sorted(hist.items())))
    for kind in UNIT_KINDS:
        sel = {n: b for n, b in assignment.items() if unit_kind_of(n) == kind}
        h = Counter(sel.values())
        print(f"  {kind:>10}: {dict(sorted(h.items()))}  "
              f"(mean {sum(sel.values()) / len(sel):.2f} bits)")

    print("Per-layer assignment (layer-index order):")
    for i in range(BLOCK.n_layers):
        print(f"  layer {i:>2}: " +
              "  ".join(f"{KIND_SHORT[k]}={assignment[unit_name(i, k)]:>2}" for k in UNIT_KINDS))

    print("\nBudget sensitivity sweep (free -- reuses the scores above):")
    for avg_bits in [3.0, 3.5, 4.0, 4.3, 4.5, 6.0]:
        alloc, c, _ = greedy_allocate(allocator_input, avg_bits * TOTAL_PARAMS, PARAMS)
        print(f"  bits/param={avg_bits}: used {c / TOTAL_PARAMS:.3f}, "
              f"distinct widths {sorted(set(alloc.values()))}")
else:
    print("no scoring-dependent arm selected -- skipping the scoring pass.")

if RUN_ADAPTIVE and assignment is not None:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    results["adaptive"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i, which: assignment[unit_name(i, which)],
        label=f"JAB adaptive ({TARGET_AVG_BITS} bits/param)", return_notes=True)
    record_arm("adaptive", perplexity=results["adaptive"], notes=_notes, bits=assignment,
               metrics=globals().get("LAST_METRICS"), elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("adaptive not selected (or no assignment) -- skipping.")


# =============================================================================================
# 12b. Budget sweep: jab / jab_norm / type_offset  (+ uniform reference rows)
#
# FOUR CRITERIA, AND NO ORACLE. The GPT-2 scripts' `oracle` criterion quantized one unit at a time
# on a freshly loaded model and measured end-to-end KL, at O(n_units * n_bit_widths) full model
# loads. With 224 units that is thousands of 7B loads; it is not "expensive", it is impossible, and
# it is deleted rather than defaulted off so nobody re-enables it by accident.
#
# `jab_prop` (the GPT-2 scripts' depth-weighted variant) is likewise not in the criteria list. The
# helper `apply_propagation_weight` and the PROPAGATION_WEIGHT flag are kept below because they
# carry over unchanged and are cheap to apply to any table by hand, but the sweep runs exactly the
# four criteria specified for this experiment.
# =============================================================================================
section("## 12b. Budget sweep: jab / jab_norm / type_offset / uniform")


def apply_propagation_weight(table, n_layers):
    """Depth weighting: multiply each unit's sensitivities by (n_layers - layer_idx). Carried over
    unchanged from the GPT-2 scripts; `layer_idx_of` still parses the layer index out of a unit
    name, so all seven of a layer's units get that layer's depth weight. NOT part of this
    experiment's four criteria -- see the section header."""
    if not PROPAGATION_WEIGHT:
        return table
    return {n: {b: v * (n_layers - layer_idx_of(n)) for b, v in t.items()}
            for n, t in table.items()}


def normalize_by_hdiag(table, hdiag):
    """
    The `jab_norm` criterion: divide each unit's trace by mean(diag(H_unit)) before it multiplies
    the perturbation.

    WHY THIS EXISTS, and why it has no GPT-2 ancestor. The allocator compares Omega values ACROSS
    unit kinds, and Omega is trace * ||Q(W)-W||_F^2 where the trace is a property of the LAYER, not
    of the unit -- all seven units of a layer share it. What differs between kinds is the scale of
    the input distribution their Hessian is built from, and here those four distributions are not
    remotely comparable:

        x1  post-RMSNorm hidden states          (gain-scaled, roughly unit-variance per channel)
        A   the merged attention output          (a convex combination of value vectors)
        x2  post-RMSNorm hidden states again     (a different gain, a different residual)
        M   silu(gate(x)) * up(x)                (a PRODUCT of two projections -- much heavier
                                                  tailed, and 14336-wide)

    A raw trace multiplied by a raw weight-space perturbation therefore mixes units whose natural
    scales differ by orders of magnitude, and the allocator's ranking partly reflects that scale
    rather than sensitivity. Dividing by mean(diag(H)) -- the mean per-input-channel second moment
    of that unit's own input -- puts the seven kinds on a common footing before they are ranked.
    Whether that helps is exactly what the sweep is for.
    """
    return {n: {b: v / hdiag[n] for b, v in t.items()} for n, t in table.items()}


sweep_results, sweep_notes, sweep_flip_rates, sweep_unit_flip_rates = {}, {}, {}, {}
sweep_allocs = {}       # key -> {unit_name: bits}, kept for section 14b's compression table
sweep_metrics = {}      # key -> the eps/rho record for that sub-pass

# The sweep is one arm but ~20 full quantize+evaluate passes over a 7B model. It is therefore
# checkpointed after EVERY sub-pass rather than only at the end: an interrupted sweep resumes from
# the last completed key instead of restarting.
_sweep_done = {} if ARGS.rerun else dict((RESULTS.get("arms", {}).get("sweep") or {})
                                         .get("sub_results", {}) or {})
for _k, _v in _sweep_done.items():
    sweep_results[_k] = _v
_sweep_prev = (RESULTS.get("arms", {}).get("sweep") or {}) if not ARGS.rerun else {}
sweep_allocs.update(_sweep_prev.get("allocs", {}) or {})
sweep_metrics.update(_sweep_prev.get("metrics_by_key", {}) or {})
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
        "metrics_by_key": sweep_metrics,
        "bit_floor": BIT_FLOOR,
        "min_bits_by_key": {k: min(v.values()) for k, v in sweep_allocs.items() if v},
    })


def _run_sweep_subpass(key, alloc, ft, label):
    """One quantize-and-evaluate sub-pass of the sweep, checkpointed by the caller."""
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    sweep_allocs[key] = dict(alloc)
    out = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i, which: alloc[unit_name(i, which)], finetune=ft,
        layer_indices=JOINT_LAYERS, return_notes=ft, label=label)
    if ft:
        sweep_results[key], sweep_notes[key] = out
        sweep_flip_rates[key] = dict(globals().get("LAST_FLIP_RATES", {}))
        sweep_unit_flip_rates[key] = dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {}))
    else:
        sweep_results[key] = out
    sweep_metrics[key] = globals().get("LAST_METRICS")
    del reader
    free()


if RUN_SWEEP:
    # Note that type_offset and the uniform rows need NO scoring pass, so the sweep still has work
    # to do even when section 12 was skipped -- only the jab / jab_norm criteria drop out.
    _t_arm = time.time()
    if _sweep_done:
        print(f"Resuming the sweep: {len(_sweep_done)} sub-pass(es) already in results.json "
              f"({', '.join(sorted(_sweep_done))})")

    tables = {}
    if allocator_input is not None:
        tables["jab"] = allocator_input
        tables["jab_norm"] = normalize_by_hdiag(allocator_input, hdiag_means)
    print("Criteria: " + ", ".join(list(tables) + ["type_offset"]) +
          "  (+ uniform reference rows; NO oracle -- see the section header)")
    print("type_offset offsets (bits relative to the budget): " +
          ", ".join(f"{KIND_SHORT[k]}{TYPE_OFFSETS[k]:+d}" for k in UNIT_KINDS))

    # A budget below the floor cannot be met by ANY assignment: if the cheapest available width is
    # 3 bits/param then 2.5 bits/param is not a tight budget, it is an impossible one, and every
    # allocator would simply return the all-floor assignment and realize 3.0. Filtered once, here,
    # with one line per dropped budget -- so the sweep never reports an assignment under a budget
    # label it does not satisfy.
    _sweep_bits = []
    for _b in SWEEP_BITS:
        if BIT_FLOOR > 0 and _b < BIT_FLOOR:
            print(f"  [skip] sweep budget {_b} bits/param: below --bit-floor {BIT_FLOOR}, so no "
                  f"assignment can meet it (the minimum width already costs {BIT_FLOOR}). "
                  f"Infeasible -- skipped for every criterion.")
            continue
        _sweep_bits.append(_b)

    for crit_name in ("jab", "jab_norm", "type_offset"):
        if crit_name != "type_offset" and crit_name not in tables:
            print(f"  [skip] criterion '{crit_name}': the scoring pass did not run")
            continue
        for avg_bits in _sweep_bits:
            if crit_name == "type_offset":
                # No scoring pass, no Hessian, no trace: the assignment is a table of seven
                # integers offset from the budget and rescaled back onto it.
                alloc, c, _ = type_offset_allocate(avg_bits, TYPE_OFFSETS, BLOCK)
                fts = (False, True) if TYPE_OFFSET_FT else (False,)
            else:
                alloc, c, _ = greedy_allocate(tables[crit_name], avg_bits * TOTAL_PARAMS, PARAMS)
                fts = (False, True)
            print(f"\n[{crit_name} @ {avg_bits} bits/param] realized "
                  f"{c / TOTAL_PARAMS:.3f} bits/param; mean bits by kind: " +
                  ", ".join(f"{KIND_SHORT[k]}="
                            f"{sum(alloc[unit_name(i, k)] for i in range(BLOCK.n_layers)) / BLOCK.n_layers:.2f}"
                            for k in UNIT_KINDS))
            for ft in fts:
                key = f"{crit_name}_{avg_bits}_{'ft' if ft else 'gptq'}"
                if key in sweep_results:
                    print(f"  [skip] sweep sub-pass '{key}': already in results.json")
                    continue
                _run_sweep_subpass(key, alloc, ft,
                                   f"{crit_name} {avg_bits} bits/param "
                                   f"{'FT' if ft else 'GPTQ-only'}")
                _checkpoint_sweep(time.time() - _t_arm)

    # Uniform reference rows. A uniform assignment cannot sit on a fractional budget at all, so it
    # is swept over integer widths instead -- which is also why the comparison table's uniform
    # column is only populated on integer rows.
    for bits in (2, 3):
        if not uniform_feasible(bits, f"sweep uniform {bits}-bit rows"):
            continue
        for ft in (False, True):
            key = f"uniform_{bits}_{'ft' if ft else 'gptq'}"
            if key in sweep_results:
                print(f"  [skip] sweep sub-pass '{key}': already in results.json")
                continue
            alloc = {unit_name(i, k): bits for i in range(BLOCK.n_layers) for k in UNIT_KINDS}
            _run_sweep_subpass(key, alloc, ft,
                               f"uniform {bits}-bit {'FT' if ft else 'GPTQ-only'}")
            _checkpoint_sweep(time.time() - _t_arm)

    _checkpoint_sweep(time.time() - _t_arm)
else:
    print("sweep not selected -- skipping.")


# =============================================================================================
# 13. Joint block-aware fine-tuning (Objective 1)
#
# STE fine-tuning of all seven units JOINTLY -- seven parameters, one loss on the block's output
# plus the attention-map KL, one commit decision -- against a float-weight teacher applied to the
# SAME input the student sees. Every checkpoint is scored on one fixed held-out calibration batch
# evaluated strictly after `opt.step()`; a fine-tuned result is written back only if it beats the
# untrained starting point on that batch.
#
# Two differences from the GPT-2 MLP script's Objective 1, both consequences of the target module:
#   - The KL term is BACK. `mlp_loss` was MSE-only because an MLP emits no distribution; a decoder
#     block contains attention, so LAMBDA_KL=0.1 weights a real KL on the attention maps.
#   - The learning rate is PER UNIT. eta_w = ALPHA_W * s with s the mean grid step of that unit --
#     see the long comment in `pass_quantize_eval`, and the per-unit s/flip-rate line printed for
#     every layer.
# =============================================================================================
section("## 13. Joint block-aware fine-tuning (Objective 1)")

if RUN_JOINT_FINETUNE and not uniform_feasible(4, "joint uniform 4-bit + FT (sec 13)"):
    RUN_JOINT_FINETUNE = False

if RUN_JOINT_FINETUNE:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    results["joint"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i: 4, finetune=True, layer_indices=JOINT_LAYERS,
        label="joint block-aware fine-tuned (4-bit)", return_notes=True)
    print("\nMean flip rate by unit kind for this arm (a kind stuck at 0.000 is the "
          "fixed-absolute-rate failure the per-unit learning rate exists to prevent):")
    _unit_rates = globals().get("LAST_FLIP_RATES_BY_UNIT", {})
    for kind in UNIT_KINDS:
        rs = [r for n, r in _unit_rates.items() if unit_kind_of(n) == kind]
        if rs:
            print(f"  {kind:>10}: {sum(rs) / len(rs):.3f}")
    record_arm("joint", perplexity=results["joint"], notes=_notes,
               bits={unit_name(i, k): 4 for i in range(BLOCK.n_layers) for k in UNIT_KINDS},
               flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(_unit_rates), metrics=globals().get("LAST_METRICS"),
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
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    results["adaptive_joint"], adaptive_joint_notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i, which: assignment[unit_name(i, which)], finetune=True,
        layer_indices=JOINT_LAYERS,
        label=f"JAB adaptive + joint fine-tuned ({TARGET_AVG_BITS} bits/param)",
        return_notes=True)

    # notes carry a seven-tuple per layer and the commit decision is joint over all seven units,
    # so the outcome is tallied per TUPLE rather than per bit-width.
    per_bits = {}
    for bits_tuple, committed in adaptive_joint_notes.values():
        counts = per_bits.setdefault(tuple(bits_tuple), [0, 0])
        counts[0 if committed else 1] += 1
    print("\nPer-assignment fine-tune outcome (adaptive_joint):")
    for bits_tuple in sorted(per_bits):
        ft, gptq_only = per_bits[bits_tuple]
        shown = "/".join(f"{KIND_SHORT[k]}{b}" for k, b in zip(UNIT_KINDS, bits_tuple))
        print(f"  {shown}: {ft} fine-tuned, {gptq_only} GPTQ-only ({ft + gptq_only} layers)")

    unit_rates = globals().get("LAST_FLIP_RATES_BY_UNIT", {})
    if unit_rates:
        print("\nMean flip rate by unit kind:")
        for kind in UNIT_KINDS:
            rs = [r for n, r in unit_rates.items() if unit_kind_of(n) == kind]
            if rs:
                print(f"  {kind:>10}: {sum(rs) / len(rs):.3f}")
    record_arm("adaptive_joint", perplexity=results["adaptive_joint"],
               notes=adaptive_joint_notes, bits=assignment,
               flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(unit_rates), metrics=globals().get("LAST_METRICS"),
               elapsed_s=time.time() - _t_arm)
    del reader
    free()
else:
    print("RUN_ADAPTIVE_JOINT=False or no `assignment` (section 12 skipped) -- "
          "skipping the adaptive+joint pass.")


# =============================================================================================
# 13c. Depth-scaling diagnostic (Objective 1D)
#
# If damage from joint fine-tuning compounds with the number of fine-tuned layers, restricting
# fine-tuning to a handful of layers should land closer to the GPTQ-only `adaptive` perplexity than
# the full-depth `adaptive_joint` run does. Reuses the same assignment; does not re-score.
#
# Worth watching more closely here than in either GPT-2 script: each fine-tuned layer now carries
# all seven units -- 218 M parameters against the GPT-2 MLP's 4.7 M -- so any compounding should
# show up sooner.
# =============================================================================================
section("## 13c. Depth-scaling diagnostic (Objective 1D)")

if RUN_DEPTH_SCALING_TEST and assignment is not None:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    n_ft_layers = len(list(JOINT_LAYERS_DEPTH_TEST))
    results["adaptive_joint_depth4"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i, which: assignment[unit_name(i, which)], finetune=True,
        layer_indices=JOINT_LAYERS_DEPTH_TEST,
        label=f"JAB adaptive + joint fine-tuned, first {n_ft_layers} layers only "
              f"({TARGET_AVG_BITS} bits/param)", return_notes=True)
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
               metrics=globals().get("LAST_METRICS"), elapsed_s=time.time() - _t_arm,
               extra={"ft_layers": list(JOINT_LAYERS_DEPTH_TEST)})
    del reader
    free()
else:
    print("RUN_DEPTH_SCALING_TEST=False or no `assignment` -- skipping depth-scaling diagnostic.")


# =============================================================================================
# 13d. KL ablation (lambda_kl=0.0) -- A REAL ABLATION HERE
#
# In the GPT-2 MLP script this arm was a documented NO-OP kept only for section numbering: the MLP
# emits no post-softmax distribution, so `mlp_loss` was MSE-only by construction and setting
# lambda_kl=0.0 changed nothing.
#
# That is no longer true. The quantization target is the whole decoder block, attention included,
# so `block_loss` really does carry `LAMBDA_KL * KL(attention maps)` and setting it to 0.0 really
# does change the objective -- it drops the only term that constrains where attention LOOKS, as
# opposed to what the block outputs. The comparison against 13b is therefore an experiment again,
# not a regression check, and a difference between the two arms is a result rather than a bug.
# =============================================================================================
section("## 13d. KL ablation (lambda_kl=0.0)")

if RUN_KL_ABLATION_TEST and assignment is not None:
    _t_arm = time.time()
    reader, _ = load_model_and_tokenizer(MODEL_ID)
    results["adaptive_joint_kl0"], _notes = pass_quantize_eval(
        reader, config, BLOCK, calib_ids, eval_fn_for(reader),
        bits_fn=lambda i, which: assignment[unit_name(i, which)], finetune=True,
        layer_indices=JOINT_LAYERS, lambda_kl=0.0,
        label=f"JAB adaptive + joint fine-tuned, MSE-only ({TARGET_AVG_BITS} bits/param)",
        return_notes=True)
    print("\nKL ablation (a REAL ablation in this script -- see the section header):")
    for key, desc in (("adaptive_joint_kl0", "13d lambda_kl=0.0"),
                      ("adaptive_joint", f"13b lambda_kl={LAMBDA_KL}"),
                      ("adaptive", "GPTQ-only")):
        if key in results:
            print(f"  {desc:<20}: {results[key]:.3f}")
    if "adaptive_joint" in results:
        delta = results["adaptive_joint_kl0"] - results["adaptive_joint"]
        print(f"\n  13d - 13b = {delta:+.3f} perplexity. Positive means the attention-map KL term "
              f"was helping; negative means it was costing. Unlike the GPT-2 MLP script, where "
              f"these two arms were the same objective and any difference was a bug, a difference "
              f"here is the measurement.")
    record_arm("adaptive_joint_kl0", perplexity=results["adaptive_joint_kl0"], notes=_notes,
               bits=assignment, flip_rates=dict(globals().get("LAST_FLIP_RATES", {})),
               unit_flip_rates=dict(globals().get("LAST_FLIP_RATES_BY_UNIT", {})),
               metrics=globals().get("LAST_METRICS"), elapsed_s=time.time() - _t_arm,
               extra={"lambda_kl": 0.0})
    del reader
    free()
else:
    print("RUN_KL_ABLATION_TEST=False or no `assignment` -- skipping KL ablation diagnostic.")


# =============================================================================================
# 14. Compare results
# =============================================================================================
section("## 14. Compare results")

CRIT_LABELS = {"uniform": "uniform", "jab": "JAB", "jab_norm": "JAB-norm",
               "type_offset": "type-off"}

print(f"{MODEL_ID}  ({BLOCK.n_layers} layers, hidden {BLOCK.hidden_size}, intermediate "
      f"{BLOCK.intermediate_size}, {BLOCK.num_heads} query heads / {BLOCK.num_kv_heads} KV heads)")
print("Quantized: all seven linear modules per block -- " + ", ".join(UNIT_KINDS) + ".")
print("fp32: RMSNorm gains, embed_tokens, lm_head.")
print(f"'bits/param' below is PARAMETER-WEIGHTED ({BLOCK.n_layers * len(UNIT_KINDS)} units "
      f"spanning 14x in size), never bits per matrix -- see section 14b.")
print(f"Perplexity is on a WikiText-2 test SUBSET: {N_EVAL_WINDOWS} windows x {EVAL_SEQ_LEN} "
      f"tokens.")
if "fp16" in results:
    print(f"fp32 control: {results['fp16']:.3f}\n")


def lookup(crit, avg_bits, ft):
    if crit == "uniform" and avg_bits == 4:
        return results.get("joint" if ft else "uniform4")
    key = f"{crit}_{avg_bits}_{'ft' if ft else 'gptq'}"
    return sweep_results.get(key)


col_keys = [(c, ft) for c in ("uniform", "jab", "jab_norm", "type_offset") for ft in (False, True)]
rows = sorted(set([2, 3] + list(SWEEP_BITS)))
header = f"{'bits/param':>11}" + "".join(f"{CRIT_LABELS[c] + ('/ft' if ft else '/gptq'):>15}"
                                         for c, ft in col_keys)
print(header)
for avg_bits in rows:
    row = f"{avg_bits:>11}"
    for c, ft in col_keys:
        v = lookup(c, avg_bits, ft)
        row += f"{v:>15.3f}" if v is not None else f"{'--':>15}"
    print(row)
print("\n'--' is a cell that was not run: uniform only exists at integer widths, and type_offset "
      "is\nrun GPTQ-only by default (it is specified as one evaluation per budget; "
      "--type-offset-ft adds\nthe fine-tuned column). There is no oracle column -- that criterion "
      "is infeasible at 7B and was\nremoved, not defaulted off.")

# --- the five metrics, per arm ---------------------------------------------------------------
print("\nWeight- and activation-error metrics (eps_W, eps_A, eps_M, rho_attn, rho_mlp):")
print("  eps_W  = ||W - W_hat||_F / ||W||_F, per unit, averaged per unit kind")
print("  eps_A  = relative Frobenius error of the PRE-o_proj attention output, teacher vs student,")
print("           both on the same input from the live partially-quantized model")
print("  eps_M  = the same for down_proj's input (the SwiGLU product)")
print("  rho_attn = mean eps_A / mean eps_W over ATTENTION units only")
print("  rho_mlp  = mean eps_M / mean eps_W over MLP units only")
print("  The two rhos are reported separately and never pooled: with the MLP also quantized, a "
      "block's\n  eps_A includes error propagated from quantized MLPs in lower layers, so these "
      "are NOT comparable\n  to a QKV-only measurement.")
_metric_rows = []
for _name in ARM_ORDER:
    _rec = RESULTS.get("arms", {}).get(_name) or {}
    if _rec.get("metrics"):
        _metric_rows.append((_name, _rec["metrics"]))
for _key in sorted(sweep_metrics):
    if sweep_metrics[_key]:
        _metric_rows.append((f"sweep: {_key}", sweep_metrics[_key]))

if _metric_rows:
    mh = (f"{'arm':<34}{'eps_W attn':>12}{'eps_W mlp':>12}{'eps_A':>12}{'eps_M':>12}"
          f"{'rho_attn':>10}{'rho_mlp':>10}")
    print()
    print(mh)
    print("-" * len(mh))

    def _f(x, w=12, p="e"):
        return f"{'--':>{w}}" if x is None else (f"{x:>{w}.4e}" if p == "e" else f"{x:>{w}.3f}")

    for _name, m in _metric_rows:
        print(f"{_name:<34}" + _f(m.get("eps_W_attn_mean")) + _f(m.get("eps_W_mlp_mean"))
              + _f(m.get("eps_A_mean")) + _f(m.get("eps_M_mean"))
              + _f(m.get("rho_attn"), 10, "f") + _f(m.get("rho_mlp"), 10, "f"))
    print("\nPer-unit-kind eps_W for each arm:")
    for _name, m in _metric_rows:
        bk = m.get("eps_W_by_kind") or {}
        print(f"  {_name:<34}" +
              ", ".join(f"{KIND_SHORT[k]}={bk.get(k):.3e}" if bk.get(k) is not None
                        else f"{KIND_SHORT[k]}=--" for k in UNIT_KINDS))
else:
    print("  (no arm with metrics has run in this session)")

print("\nPer-layer flip rate (fraction of grid positions moved by FT, pooled over all 7 units):")
for key, rates in sweep_flip_rates.items():
    if rates:
        mean_rate = sum(rates.values()) / len(rates)
        print(f"  {key:<24}: mean={mean_rate:.3f}  " +
              ", ".join(f"L{i}={r:.3f}" for i, r in sorted(rates.items())))

print("\nFlip rate by unit kind (mean over layers) -- a kind pinned at 0.000 is the "
      "fixed-absolute-rate\nfailure the per-unit learning rate exists to prevent, so read this "
      "column before anything else:")
for key, rates in sweep_unit_flip_rates.items():
    if rates:
        parts = []
        for kind in UNIT_KINDS:
            rs = [r for n, r in rates.items() if unit_kind_of(n) == kind]
            if rs:
                parts.append(f"{KIND_SHORT[kind]}={sum(rs) / len(rs):.3f}")
        print(f"  {key:<24}: " + ", ".join(parts))


# =============================================================================================
# 14b. Compression
#
# Every number here is PARAMETER-WEIGHTED. That is not a refinement of the GPT-2 scripts' table,
# it is a correction that those scripts did not need: their allocator units all held identical
# parameter counts, so "average bits per matrix" and "average bits per parameter" were the same
# number. Here they are not even close -- gate_proj is 14x k_proj -- and an unweighted average over
# the 224 units would understate the cost of exactly the units the allocator is most tempted to
# upgrade.
#
# This script also quantizes 100% of the non-embedding weight parameters, where the GPT-2 MLP
# script quantized 66.6% of them and the QKV notebook 25.0%. An arm labelled "4 bits/param" here is
# therefore a strictly more aggressive intervention than an identically labelled arm in either
# ancestor, and the row labels are NOT a shared x-axis across the three.
#
# The fp16 reference is a convention, not a claim about this run: everything here computes in fp32.
# =============================================================================================
section("## 14b. Compression")

REF_BITS = 16      # storage baseline for the compression ratio -- see the section header

NONEMB_PARAMS = BLOCK.block_numel * BLOCK.n_layers
ATTN_PARAMS = BLOCK.attn_numel * BLOCK.n_layers
MLP_PARAMS = BLOCK.mlp_numel * BLOCK.n_layers

print(f"{MODEL_ID}: {NONEMB_PARAMS:,} non-embedding WEIGHT parameters "
      f"(RMSNorm gains and the embed_tokens / lm_head matrices excluded -- none of them is a "
      f"quantization target)")
print(f"  attention (q,k,v,o):     {ATTN_PARAMS:>14,}  "
      f"({100 * ATTN_PARAMS / NONEMB_PARAMS:.1f}%)")
print(f"  MLP (gate,up,down):      {MLP_PARAMS:>14,}  "
      f"({100 * MLP_PARAMS / NONEMB_PARAMS:.1f}%)")
print(f"  quantized by this script:{NONEMB_PARAMS:>14,}  (100.0% -- all seven units)")
print(f"  at fp16 that is {NONEMB_PARAMS * REF_BITS / 8 / 1e9:.2f} GB of weights\n")


def arm_compression(bits_map):
    """
    bits_map: {unit_name: bits} for every quantized unit, or a plain int for a uniform arm.

    Returns (quantized_params, PARAMETER-WEIGHTED average bits, compression ratio vs fp16,
             stored size in GB, {kind: mean bits}).
    """
    if isinstance(bits_map, int):
        bits_map = {unit_name(i, k): bits_map
                    for i in range(BLOCK.n_layers) for k in UNIT_KINDS}
    q_params = sum(PARAMS[n] for n in bits_map)
    q_bits = sum(cost(b, PARAMS[n]) for n, b in bits_map.items())
    avg_bits = q_bits / q_params
    total_bits = q_bits + (NONEMB_PARAMS - q_params) * REF_BITS
    ratio = (NONEMB_PARAMS * REF_BITS) / total_bits
    by_kind = {}
    for kind in UNIT_KINDS:
        vals = [b for n, b in bits_map.items() if unit_kind_of(n) == kind]
        by_kind[kind] = sum(vals) / len(vals) if vals else None
    return q_params, avg_bits, ratio, total_bits / 8 / 1e9, by_kind


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

hdr = (f"{'arm':<40}{'q-params':>15}{'bits/param':>12}{'compress':>10}{'size GB':>10}{'ppl':>9}")
print(hdr)
print("-" * len(hdr))
for label, bits_map, ppl in arms:
    q_params, avg_bits, ratio, size_gb, _ = arm_compression(bits_map)
    print(f"{label:<40}{q_params:>15,}{avg_bits:>12.3f}{ratio:>9.2f}x{size_gb:>10.2f}{ppl:>9.3f}")

print("\nMean assigned bits by unit kind (parameter-weighted budgets do NOT imply equal widths --")
print("the allocator can and does buy a wider k_proj for a fraction of what a wider gate_proj "
      "costs):")
kh = f"{'arm':<40}" + "".join(f"{KIND_SHORT[k]:>7}" for k in UNIT_KINDS)
print(kh)
print("-" * len(kh))
for label, bits_map, _ in arms:
    _, _, _, _, by_kind = arm_compression(bits_map)
    print(f"{label:<40}" + "".join(
        f"{by_kind[k]:>7.2f}" if by_kind[k] is not None else f"{'--':>7}" for k in UNIT_KINDS))

print("\nHow to read this against the GPT-2 scripts: run_mlp_quant.py quantized 66.6% of "
      "GPT-2's\nnon-embedding weight parameters and the QKV notebook 25.0%, both at a uniform "
      "unit size that let\nthem label rows 'bits per matrix'. This script quantizes 100% of "
      "Mistral's, at unit sizes\nspanning 14x, and labels rows 'bits per parameter'. The two "
      "labels are different quantities;\nrow-matching the tables would systematically flatter "
      "the ancestors.")


# =============================================================================================
# done
# =============================================================================================
RESULTS["meta"]["finished"] = datetime.now().isoformat(timespec="seconds")
RESULTS["meta"]["total_elapsed_s"] = round(time.time() - _T_START, 2)
flush_results()
section(f"## run complete -- {(time.time() - _T_START) / 60:.1f} min total")
print(f"results  {RESULTS_PATH}")
print(f"log      {_LOG_PATH}")
_b = RESULTS["meta"].get("fp64_benchmark", {})
_bg = "--" if _b.get("gpu_s") is None else "%.2fs" % _b["gpu_s"]
_bc = "--" if _b.get("cpu_s") is None else "%.2fs" % _b["cpu_s"]
print(f"fp64 ({_b.get('dim')}, {_b.get('dim')}) solve benchmark -- GPU: {_bg}, CPU: {_bc}, "
      f"solves ran on {_b.get('solve_device')}")
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
