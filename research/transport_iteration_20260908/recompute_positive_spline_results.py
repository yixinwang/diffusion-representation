"""Recheck frozen saved coefficients and published evaluation streams; NEVER fit."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import platform
import subprocess
import sys
import time

import numpy as np
import scipy
from scipy.interpolate import BSpline
from scipy.spatial.distance import pdist

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True,
                    help="new directory for reconstructed records; never overwritten")
args = parser.parse_args()
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RESULTS = HERE / "psc_positive_spline"
WORK = args.output.resolve()
WORK.mkdir(parents=True, exist_ok=False)
COMMIT = "dc0169f50eb0eddfbd5bff6ace6dc2fb384347a2"
summary = json.loads((RESULTS / "summary.json").read_text())
manifest = summary["manifest"]
assert manifest["source_commit"] == COMMIT
scratch = WORK / "recompute-positive-spline-source"
(scratch / "qalt").mkdir(parents=True, exist_ok=True)
(scratch / "qalt" / "__init__.py").write_text("")
for name in ("positive_density_spline.py", "positive_spline_flow.py"):
    relative = "qalt/src/qalt/" + name
    source = subprocess.check_output(["git", "show", COMMIT + ":" + relative], cwd=REPO)
    assert hashlib.sha256(source).hexdigest() == manifest["source_hashes"][relative]
    (scratch / "qalt" / name).write_bytes(source)
sys.path.insert(0, str(scratch))
from qalt.positive_density_spline import PositiveDensitySpline
from qalt.positive_spline_flow import PositiveSplineFlow


def forbidden_fit(*args, **kwargs):
    raise AssertionError("reproducibility audit must never fit")


PositiveSplineFlow.fit = forbidden_fit


def frozen_stream(record, purpose):
    assert purpose in (2, 3), "training stream is outside this audit"
    identity = [record["seed"], record["dimension"], record["train_array_count"],
                {"local": 0, "distant": 1}[record["world"]], purpose]
    return np.random.default_rng(np.random.SeedSequence(identity))


def observed_arrays(record):
    """Invert the analytic cosine-pair conditional with the frozen 54 steps."""
    u = frozen_stream(record, 2).uniform(size=(4096, record["dimension"]))
    result = u.copy()
    children = np.arange(1, record["dimension"], 2) if record["world"] == "local" else np.array([record["dimension"] - 1])
    parents = children - 1 if record["world"] == "local" else np.array([0])
    amplitude = .65 * np.cos(2 * np.pi * u[:, parents])
    left, right = np.zeros_like(u[:, children]), np.ones_like(u[:, children])
    for _ in range(54):
        middle = left + (right - left) / 2
        integral = middle + amplitude * np.sin(2 * np.pi * middle) / (2 * np.pi)
        below = integral < u[:, children]
        left, right = np.where(below, middle, left), np.where(below, right, middle)
    result[:, children] = left + (right - left) / 2
    assert np.all((result > 0) & (result < 1))
    return result


def target_log_density(x, world):
    """Joint product of independent pair densities, or just the endpoint pair."""
    pairs = list(zip(range(0, x.shape[1], 2), range(1, x.shape[1], 2))) if world == "local" else [(0, x.shape[1]-1)]
    total = np.zeros(len(x))
    for parent, child in pairs:
        total += np.log1p(.65 * np.cos(2*np.pi*x[:, parent]) * np.cos(2*np.pi*x[:, child]))
    return total


def independent_spline_log_density(x, model_record):
    """Direct dense context basis and linear hat formula, separate from wrapper."""
    total = np.zeros(len(x))
    for j, (parents, group) in enumerate(zip(model_record["parents"], model_record["groups"])):
        head = model_record["heads"][group]
        b = head["bins"]
        coefficients = np.array(head["coefficients"])
        if parents:
            assert len(parents) == 1
            knots = np.r_[np.zeros(3), np.arange(1, b)/b, np.ones(3)]
            basis = BSpline(knots, np.eye(b+2), 2)(x[:, parents[0]])
            conditional_coefficients = basis @ coefficients
        else:
            conditional_coefficients = np.broadcast_to(coefficients, (len(x), b+1))
        position = b*x[:, j]
        cell = np.minimum(position.astype(int), b-1)
        fraction = position-cell
        density = conditional_coefficients[np.arange(len(x)), cell]*(1-fraction) + conditional_coefficients[np.arange(len(x)), cell+1]*fraction
        total += np.log(density)
    return total


def score_summary(values):
    return {"mean": float(np.mean(values)), "mcse": float(np.std(values, ddof=1)/np.sqrt(len(values))),
            "independent_array_count": len(values)}


def moments(x):
    y = np.cos(2*np.pi*x)
    return {"marginal_cosine": [score_summary(y[:, j]) for j in range(x.shape[1])],
            "adjacent_pair_cosine_product": [score_summary(y[:, j]*y[:, j+1]) for j in range(0,x.shape[1],2)],
            "first_last_cosine_product": score_summary(y[:, 0]*y[:, -1])}


def energy(observed, generated):
    # Independent implementation: sum cross distances in small blocks, and
    # use only upper-triangular generated pairs (each unordered pair once).
    cross_sum = 0.
    for start in range(0, len(generated), 16):
        differences = generated[start:start+16, None, :] - observed[None, :, :]
        cross_sum += np.sqrt(np.sum(differences*differences, axis=-1)).sum()
    cross = cross_sum/(len(generated)*len(observed))
    self_half = pdist(generated).sum()/(len(generated)*(len(generated)-1))
    return float((cross-self_half)/np.sqrt(observed.shape[1]))


def array_hash(x):
    x = np.ascontiguousarray(x)
    digest = hashlib.sha256(str((x.shape, x.dtype.str)).encode())
    digest.update(x.view(np.uint8))
    return digest.hexdigest()


def max_tree_difference(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        return max(max_tree_difference(first[k], second[k]) for k in first)
    if isinstance(first, list):
        assert len(first) == len(second)
        return max(max_tree_difference(a,b) for a,b in zip(first,second))
    return abs(float(first)-float(second))


start = time.perf_counter()
audits = []
for record in summary["results"]:
    raw = record["model"]
    heads = tuple(PositiveDensitySpline(np.array(h["coefficients"]), h["bins"], h["context_dimension"], h["lower"], h["upper"]) for h in raw["heads"])
    model = PositiveSplineFlow(tuple(tuple(p) for p in raw["parents"]), tuple(raw["groups"]), heads)
    observed = observed_arrays(record)
    gaussian = frozen_stream(record, 3).normal(size=(256, record["dimension"]))
    generated, inverse_ld = model.decode(gaussian)
    recovered, forward_ld = model.encode(generated)
    independent_log_q = independent_spline_log_density(observed, raw)
    wrapper_log_q = model.log_prob(observed)
    differences = target_log_density(observed, record["world"]) - independent_log_q
    calculated = {"joint_kl": score_summary(differences), "per_coordinate_kl": score_summary(differences/record["dimension"]),
                  "energy_score": energy(observed, generated),
                  "observed_moments": moments(observed), "generated_moments": moments(generated)}
    metric_differences = {key:max_tree_difference(value,record["evaluation"][key]) for key,value in calculated.items()}
    normal_lp = -.5*(record["dimension"]*np.log(2*np.pi)+(gaussian**2).sum(axis=1))
    numerical = {"source_roundtrip_max":float(np.max(abs(recovered-gaussian))),
                 "logdet_cancellation_max":float(np.max(abs(inverse_ld+forward_ld))),
                 "normalized_density_identity_max":float(np.max(abs(model.log_prob(generated)+inverse_ld-normal_lp)))}
    copied = PositiveSplineFlow(model.parents, model.groups, model.models)
    copied_samples, copied_ld = copied.decode(gaussian)
    exact_copy = bool(np.array_equal(copied_samples,generated) and np.array_equal(copied_ld,inverse_ld)
                      and np.array_equal(copied.log_prob(observed),wrapper_log_q))
    new_hashes = {"development_arrays":array_hash(observed),"gaussian_source":array_hash(gaussian),"generated_arrays":array_hash(generated)}
    audits.append({"cell":[record["world"],record["dimension"],record["train_array_count"],record["seed"]],
                   "recomputed":calculated,"metric_max_absolute_discrepancies":metric_differences,
                   "wrapper_vs_independent_log_density_max":float(np.max(abs(independent_log_q-wrapper_log_q))),
                   "numerical_checks_recomputed":numerical,
                   "numerical_check_absolute_discrepancies":{k:abs(v-record["numerical_checks"][k]) for k,v in numerical.items()},
                   "copy_exact":exact_copy,"hashes_recomputed":new_hashes,
                   "hash_matches":{k:v==record["hashes"][k] for k,v in new_hashes.items()}})
    assert exact_copy and max(numerical.values())<=1e-8
    assert max(metric_differences.values())<=1e-9
    assert audits[-1]["wrapper_vs_independent_log_density_max"]<=1e-10

report = {"source_commit":COMMIT,"cells_recomputed":len(audits),"fit_called":False,
          "training_arrays_regenerated":False,"new_seed_identities_used":False,"real_data_accessed":False,
          "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"platform":platform.platform(),
          "runtime_seconds":time.perf_counter()-start,"cells":audits}
(WORK/"psc-positive-spline-recomputed.json").write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+"\n")
maximum_metrics = {k:max(a["metric_max_absolute_discrepancies"][k] for a in audits) for k in audits[0]["metric_max_absolute_discrepancies"]}
maximum_checks = {k:max(a["numerical_checks_recomputed"][k] for a in audits) for k in audits[0]["numerical_checks_recomputed"]}
maximum_check_discrepancies = {k:max(a["numerical_check_absolute_discrepancies"][k] for a in audits) for k in audits[0]["numerical_check_absolute_discrepancies"]}
hash_counts = {k:sum(a["hash_matches"][k] for a in audits) for k in audits[0]["hash_matches"]}
text = "\n## Reproducibility addendum: frozen arrays and saved coefficients\n\n"
text += f"All {len(audits)} saved coefficient models were reconstructed locally from the frozen source primitives at `{COMMIT}`. Only the already registered development and Gaussian-source streams were regenerated. No fitting, training-array regeneration, new seed/cell, PSC job, real-data access or adaptive model choice occurred.\n\n"
text += f"Audit environment: Python {report['python']}, NumPy {report['numpy']}, SciPy {report['scipy']}, {report['platform']}. The original PSC environment used NumPy {manifest['numpy']} and SciPy {manifest['scipy']}.\n\n"
text += "The analytic cosine-pair joint density was evaluated independently. Fitted density was separately computed from dense quadratic context B-spline bases and linear response hats, then cross-checked against the frozen wrapper. Energy used blockwise raw Euclidean differences and upper-triangular self pairs. Whole-array KL/MCSE, all saved moment means/MCSEs, numerical checks and exact decoder copies were recomputed. Maximum wrapper-versus-independent joint log-density discrepancy: `"+format(max(a['wrapper_vs_independent_log_density_max'] for a in audits),'.6g')+"`.\n\n"
text += "Maximum absolute discrepancies from original recorded metrics:\n\n"+"\n".join(f"- {k}: `{v:.6g}`." for k,v in maximum_metrics.items())+"\n\n"
text += "Recomputed numerical-check maxima (and maximum differences from recorded check values):\n\n"+"\n".join(f"- {k}: `{v:.6g}`; check-value discrepancy `{maximum_check_discrepancies[k]:.6g}`." for k,v in maximum_checks.items())+"\n\n"
text += "All 36 copied decoders remained bitwise identical within this reconstruction, and all numerical maxima remain below the frozen1e-8 tolerance. Exact byte-hash matches against PSC: "+", ".join(f"{k} {v}/36" for k,v in hash_counts.items())+". Byte-hash differences, if any, are preserved in the audit JSON and are not relabeled exact reproduction; the small metric discrepancies above quantify cross-platform numerical reproducibility.\n\n"
text += "The earlier audit's inability to recompute these metrics from raw observations is resolved by this deterministic-stream reconstruction, within the documented numerical tolerances. Training traces/gaps were not recomputed because doing so would require a new fit. These checks do not cure the original22 optimization failures, remove the distant-dependence failure, or establish real-data quality.\n\n"
text += "Audit script: `research/transport_iteration_20260908/recompute_positive_spline_results.py`. Full scalar comparisons, regenerated-array hashes and metrics are written to the explicitly supplied output directory.\n"
text=text.replace('frozen1e-8','frozen 1e-8').replace('original22','original 22')
with (WORK/"psc-positive-spline-review.md").open('a') as handle:
    handle.write(text)
print(json.dumps({"cells":len(audits),"metric_max_discrepancies":maximum_metrics,"hash_matches":hash_counts,
                  "numerical_maxima":maximum_checks,"runtime_seconds":report['runtime_seconds']},indent=2))
