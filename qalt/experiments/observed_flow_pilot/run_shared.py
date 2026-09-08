"""Shared-analysis native-image operational pilot utilities.

No loader or automatic study execution: the reviewed caller supplies fit tensors
and eligible repair arrays from the frozen observed_flow_data loader. All timings
are synchronized device wall time. No FM likelihood approximation is reported.
"""
from __future__ import annotations
import hashlib
import importlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from qalt.flow_matching import heun_integrate
from qalt.learned_latent_flow_matching import GlobalConditionalVelocity

SEED = 77101
DEFAULTS = dict(analysis_seconds=90., coarse_seconds=90., decoder_seconds=180.,
                batch_size=32, learning_rate=1e-3, generation_batch=64,
                coarse_nfe=32, residual_nfes=[4, 8, 16, 32, 64], pairs_per_image=1)


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


def prepare_output(path, source_paths):
    """No overwrite; snapshot exact sources before the caller loads observations."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=False)
    hashes = {}
    snapshot = path / 'sources'
    snapshot.mkdir()
    for index, original in enumerate(source_paths):
        original = Path(original).resolve()
        payload = original.read_bytes()
        hashes[str(original)] = hashlib.sha256(payload).hexdigest()
        (snapshot / f'{index:02d}_{original.name}').write_bytes(payload)
    atomic_json(path / 'source_hashes.json', hashes)
    atomic_json(path / 'status.json', {'status': 'started', 'defaults': DEFAULTS, 'seed': SEED})
    return path


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def parameter_count(model):
    return sum(p.numel() for p in model.parameters())


def make_decoders(*, residual_channels=45, context_channels=3, size=8,
                  constructor='qalt.global_conditional_spline:GlobalConditionalSplineDecoder'):
    """Shape-only match; construction does not inspect observations or scores."""
    module, name = constructor.split(':')
    candidate = getattr(importlib.import_module(module), name)(
        residual_channels, context_channels, size, layers=4, width=32,
        bins=8, attention_heads=4)
    target = parameter_count(candidate)
    for width in range(32, 4097, 4):
        reference = GlobalConditionalVelocity(residual_channels, context_channels, size, width, 4)
        if parameter_count(reference) >= target:
            return candidate, reference, dict(candidate_parameters=target,
                fm_parameters=parameter_count(reference), fm_width=width)
    raise ValueError('shape-only matching range exhausted')


def logit_inputs(unit_values):
    """Apply shared logit in float64 before float32; return outer Jacobian."""
    x = np.asarray(unit_values)
    if x.dtype != np.float64 or not np.isfinite(x).all() or not ((x > 0) & (x < 1)).all():
        raise ValueError('expected finite strict-open float64 loader output')
    logits = np.log(x) - np.log1p(-x)
    jacobian = (-np.log(x)-np.log1p(-x)).reshape(len(x), -1).sum(1)
    return torch.from_numpy(logits.astype(np.float32)), jacobian


def train_stage(parameters, loss_function, *, sample_count, record_ids, device,
                seconds, batch_size=32, learning_rate=1e-3, batch_seed=SEED,
                path_seed=SEED+1, progress_path=None):
    """Callback loss(indices, generator); deterministic shared index prefixes.

    Wall cap is checked between updates: one update may overrun and is charged.
    Optimizer creation and device synchronization are included. Caller performs
    and separately charges source-cache/model construction and movement.
    """
    params = [p for p in parameters if p.requires_grad]
    if not params or seconds <= 0 or sample_count != len(record_ids):
        raise ValueError('invalid stage configuration')
    rng = np.random.Generator(np.random.PCG64(batch_seed))
    generator = torch.Generator(device=device).manual_seed(path_seed)
    synchronize(device)
    start = time.perf_counter()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    optimizer = torch.optim.Adam(params, lr=learning_rate)
    if torch.device(device).type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    digest = hashlib.sha256()
    report = dict(updates=0, examples=0, batch_seed=batch_seed, path_seed=path_seed,
                  requested_seconds=seconds, losses=[], status='running')
    try:
        while time.perf_counter()-start < seconds:
            index_np = rng.integers(0, sample_count, size=batch_size)
            digest.update(np.asarray(record_ids[index_np], dtype='<i8').tobytes())
            indices = torch.as_tensor(index_np, device=device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_function(indices, generator)
            if loss.ndim != 0 or not bool(torch.isfinite(loss)):
                raise FloatingPointError('nonfinite or nonscalar training loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in params):
                raise FloatingPointError('nonfinite training gradient')
            optimizer.step()
            if any(not bool(torch.isfinite(p).all()) for p in params):
                raise FloatingPointError('nonfinite updated parameter')
            synchronize(device)
            report['updates'] += 1
            report['examples'] += batch_size
            report['losses'].append(float(loss.detach()))
            if progress_path and report['updates'] % 10 == 0:
                atomic_json(progress_path, {**report, 'elapsed_seconds': time.perf_counter()-start})
        if not report['updates']:
            raise RuntimeError('training stage completed zero updates')
        report['status'] = 'completed_cap'
    except Exception as error:
        report['status'] = 'failed'
        report['error'] = repr(error)
        raise
    finally:
        synchronize(device)
        report['elapsed_seconds'] = time.perf_counter()-start
        report['record_order_sha256'] = digest.hexdigest()
        report['peak_allocated_bytes'] = (torch.cuda.max_memory_allocated(device)
            if torch.device(device).type == 'cuda' else None)
        report['peak_reserved_bytes'] = (torch.cuda.max_memory_reserved(device)
            if torch.device(device).type == 'cuda' else None)
        if progress_path:
            atomic_json(progress_path, report)
        optimizer.zero_grad(set_to_none=True)
    return report


def fm_loss(velocity, target, generator, context=None):
    noise = torch.randn(target.shape, device=target.device, dtype=target.dtype, generator=generator)
    t = torch.rand(target.shape[0], device=target.device, dtype=target.dtype, generator=generator)
    mix = t.reshape(-1, 1, 1, 1)
    prediction = velocity((1-mix)*noise+mix*target, t, context)
    return (prediction-(target-noise)).square().mean()


@torch.no_grad()
def cache_fit_codes(analysis, fit_logits, *, batch_size=64):
    """Only observed fitting codes; explicit cache creation cost shared by arms."""
    device = next(analysis.parameters()).device
    synchronize(device)
    start = time.perf_counter()
    values = []
    for chunk in fit_logits.split(batch_size):
        encoded, determinant = analysis.encode_analysis(chunk.to(device))
        if not bool(torch.isfinite(encoded).all()) or not bool(torch.isfinite(determinant).all()):
            raise FloatingPointError("nonfinite fitting analysis or Jacobian")
        values.append(encoded.cpu())
    code = torch.cat(values).to(device)
    coarse, residual = analysis._split_code(code)
    synchronize(device)
    return coarse, residual, dict(seconds=time.perf_counter()-start,
        code_bytes=code.numel()*code.element_size(),
        materialized_block_bytes=sum(v.numel()*v.element_size() for v in (coarse, residual)),
        unique_device_storage_bytes=sum({v.untyped_storage().data_ptr():
            v.untyped_storage().nbytes() for v in (code, coarse, residual)}.values()))


@torch.no_grad()
def generate(analysis, decoder, source, *, kind, residual_nfe=32, coarse_nfe=32):
    """Same complete Gaussian source; no real context accepted at generation."""
    if source.ndim != 2 or source.shape[1] != analysis.dimension:
        raise ValueError('full-dimensional source required')
    if kind == 'analysis_only':
        coarse = source[:, :analysis.latent_dimension].reshape(-1, analysis.channels, analysis.coarse_size, analysis.coarse_size)
        residual = source[:, analysis.latent_dimension:].reshape(-1, analysis.packed_residual_channels, analysis.coarse_size, analysis.coarse_size)
        logits, _ = analysis.decode_analysis(analysis._join_code(coarse, residual))
    else:
        if coarse_nfe <= 0 or coarse_nfe % 2 or residual_nfe <= 0 or residual_nfe % 2:
            raise ValueError('Heun requires positive even actual velocity-call counts')
        coarse = analysis.coarse_prior.sample_from_gaussian(
            source[:, :analysis.latent_dimension], steps=coarse_nfe//2)
        residual = source[:, analysis.latent_dimension:].reshape(-1,
            analysis.packed_residual_channels, analysis.coarse_size, analysis.coarse_size)
        if kind == 'coupling':
            residual = decoder.decode(residual, coarse)[0]
        elif kind == 'fm':
            residual = heun_integrate(decoder, residual, steps=residual_nfe//2, context=coarse)
        else:
            raise ValueError('unknown decoder kind')
        logits, _ = analysis.decode_analysis(analysis._join_code(coarse, residual))
    if not bool(torch.isfinite(logits).all()):
        raise FloatingPointError("nonfinite generated logits before sigmoid")
    output = torch.sigmoid(logits)
    if not bool(torch.isfinite(output).all()):
        raise FloatingPointError('nonfinite generated output')
    return output


def descriptors(images):
    """Fixed spatial dependency measurements; no fitted feature extractor."""
    x = np.asarray(images, dtype=np.float64)
    gray = x.mean(axis=1)
    a, b = gray[:, :16, :16].mean((1, 2)), gray[:, 16:, 16:].mean((1, 2))
    horizontal = np.square(np.diff(gray, axis=2)).mean((1, 2))
    vertical = np.square(np.diff(gray, axis=1)).mean((1, 2))
    return np.column_stack([a, b, a*b, horizontal, vertical, horizontal*vertical])


@torch.no_grad()
def evaluate_pairs(sampler, repair, *, dimension, device, seed=SEED+300,
                   batch_size=64, pairs_per_image=1, artifact_directory=None, record_ids=None):
    """Independent Gaussian pairs assigned per repair image; stream GPU batches.

    Returned per-image scores are the elementary sampling units. Paired method
    comparisons require the same seed, shape and pair count. Repair reuse makes
    resulting uncertainty descriptive; this function makes no coverage claim.
    """
    if pairs_per_image < 1 or len(repair) < 2:
        raise ValueError('insufficient evaluation units')
    ids = np.arange(len(repair), dtype="<i8") if record_ids is None else np.asarray(record_ids, dtype="<i8")
    if ids.shape != (len(repair),) or len(np.unique(ids)) != len(ids):
        raise ValueError("evaluation record IDs must be unique and match repair order")
    source_digest = hashlib.sha256()
    generator = torch.Generator(device='cpu').manual_seed(seed)
    count = len(repair)*pairs_per_image*2
    artifact_path = None
    if artifact_directory is not None:
        artifact_path = Path(artifact_directory)
        artifact_path.mkdir(parents=True, exist_ok=False)
    scores = np.empty(len(repair), dtype=np.float64)
    features = []
    boundaries = 0
    synchronize(device)
    start = time.perf_counter()
    # Batch complete images so each retains its own independent generated pairs.
    image_batch = max(1, batch_size//(2*pairs_per_image))
    for first in range(0, len(repair), image_batch):
        real = np.asarray(repair[first:first+image_batch], dtype=np.float64)
        n = len(real)
        cpu_source = torch.randn(n*pairs_per_image*2, dimension, generator=generator)
        source_digest.update(cpu_source.numpy().tobytes())
        z = cpu_source.to(device)
        generated = sampler(z).cpu().numpy().astype(np.float64)
        if artifact_path is not None:
            np.savez(artifact_path/f'chunk_{first:04d}.npz',
                     gaussian=z.cpu().numpy(), generated=generated,
                     repair_positions=np.arange(first, first+n))
        boundaries += int(((generated == 0) | (generated == 1)).sum())
        features.append(descriptors(generated))
        draws = generated.reshape(n, pairs_per_image, 2, dimension)
        target = real.reshape(n, 1, dimension)
        norm = lambda value: np.linalg.norm(value, axis=-1)/math.sqrt(dimension)
        values = .5*(norm(draws[:, :, 0]-target)+norm(draws[:, :, 1]-target)
                     -norm(draws[:, :, 0]-draws[:, :, 1]))
        scores[first:first+n] = values.mean(1)
    synchronize(device)
    features = np.concatenate(features)
    if artifact_path is not None:
        np.save(artifact_path/'per_image_energy.npy', scores)
        files = sorted(artifact_path.iterdir())
        atomic_json(artifact_path/'hashes.json', {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    real_features = descriptors(repair)
    return dict(energy_mean=float(scores.mean()),
        descriptive_energy_se=float(scores.std(ddof=1)/math.sqrt(len(scores))),
        per_image_energy=scores.tolist(), generated_count=count,
        generated_descriptor_mean=features.mean(0).tolist(),
        repair_descriptor_mean=real_features.mean(0).tolist(),
        descriptor_names=['upper_left_mean','lower_right_mean','opposite_product',
                          'horizontal_gradient_energy','vertical_gradient_energy','gradient_product'],
        rounded_boundary_values=boundaries, seconds=time.perf_counter()-start,
        source_seed=seed, pairs_per_image=pairs_per_image, dimension=dimension,
        source_stream_sha256=source_digest.hexdigest(),
        repair_ids_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        repair_values_sha256=hashlib.sha256(np.ascontiguousarray(repair, dtype=np.float64).tobytes()).hexdigest(),
        record_ids_supplied=record_ids is not None,
        uncertainty='descriptive only: repair images were reused in development')


def paired_energy(first, second):
    identity = ('source_seed', 'pairs_per_image', 'dimension', 'source_stream_sha256',
                'repair_ids_sha256', 'repair_values_sha256', 'generated_count')
    if any(first[k] != second[k] for k in identity):
        raise ValueError('paired evaluation requires identical actual sources and repair records')
    left, right = np.asarray(first['per_image_energy']), np.asarray(second['per_image_energy'])
    if left.ndim != 1 or left.shape != right.shape or len(left) < 2 or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('paired scores must have identical finite one-dimensional shapes')
    difference = left-right
    return dict(first_minus_second=float(difference.mean()),
        descriptive_paired_se=float(difference.std(ddof=1)/math.sqrt(len(difference))),
        images=len(difference), uncertainty='exploratory reused repair; no confirmation claim')


def fit_shared_and_decoders(analysis, candidate, reference, fit_logits, fit_ids,
                            output, *, device='cuda', settings=None):
    """Train only supplied fit arrays; preserve each stage and arm checkpoint.

    This scaffolding requires an already source-frozen, canonical-data-verified
    caller. It intentionally has no data-root or repair-fitting argument.
    """
    cfg = {**DEFAULTS, **(settings or {})}
    output = Path(output)
    reports = {}
    synchronize(device)
    shared_external_start = time.perf_counter()
    analysis.to(device).train()
    candidate.cpu()
    reference.cpu()
    logits = fit_logits.to(device)
    def stage(name, parameters, loss, seconds, path_seed):
        try:
            reports[name] = train_stage(parameters, loss, sample_count=len(logits),
                record_ids=fit_ids, device=device, seconds=seconds,
                batch_size=cfg['batch_size'], learning_rate=cfg['learning_rate'],
                batch_seed=SEED+10, path_seed=path_seed,
                progress_path=output/f'{name}_progress.json')
        finally:
            torch.save(analysis.state_dict(), output/'shared_latest.pt')
    stage('analysis', analysis._analysis_parameters(),
          lambda i, g: analysis.train_analysis_loss(logits[i]), cfg['analysis_seconds'], SEED+20)
    analysis.freeze_analysis()
    torch.save(analysis.state_dict(), output/'analysis_only.pt')
    coarse, residual, reports['cache'] = cache_fit_codes(analysis, logits)
    del logits
    # Closures below need only cached fitting codes; no observed repair array.
    logits = coarse
    stage('coarse', analysis.coarse_prior.parameters(),
          lambda i, g: fm_loss(analysis.coarse_prior.velocity, coarse[i], g),
          cfg['coarse_seconds'], SEED+30)
    analysis.eval()
    for parameter in analysis.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    synchronize(device)
    reports["shared_external_seconds"] = time.perf_counter()-shared_external_start
    reports["decoder_external_seconds"] = {}
    for name, model, loss in [
        ('coupling', candidate, lambda m, i, g: -m.log_prob(residual[i], coarse[i]).mean()/analysis.residual_dimension),
        ('residual_fm', reference, lambda m, i, g: fm_loss(m, residual[i], g, coarse[i]))]:
        arm_external_start = time.perf_counter()
        model.to(device).train()
        try:
            stage(name, model.parameters(), lambda i, g: loss(model, i, g),
                  cfg['decoder_seconds'], SEED+40)
        finally:
            torch.save(model.state_dict(), output/f'{name}_latest.pt')
            model.cpu().eval()
            if torch.device(device).type == 'cuda':
                torch.cuda.empty_cache()
            synchronize(device)
            reports["decoder_external_seconds"][name] = time.perf_counter()-arm_external_start
    reports["standalone_fitting_external_seconds"] = {
        k: reports["shared_external_seconds"]+reports["decoder_external_seconds"][k]
        for k in ("coupling", "residual_fm")}
    shared_seconds = sum(reports[k]['elapsed_seconds'] for k in ('analysis', 'coarse'))+reports['cache']['seconds']
    reports['standalone_training_seconds'] = {
        k: shared_seconds+reports[k]['elapsed_seconds'] for k in ('coupling', 'residual_fm')}
    reports['cost_exclusions'] = 'Caller must add loading, logit preprocessing, model construction/movement, source guard, checkpoint and evaluation wall times.'
    atomic_json(output/'training_report.json', reports)
    return reports


@torch.no_grad()
def latency(sampler, *, dimension, device, batch_sizes=(1, 64), repeats=3):
    """Measure whole supplied pipeline; caller offloads the other decoder first."""
    report = {}
    for batch in batch_sizes:
        generator = torch.Generator(device="cpu").manual_seed(SEED+400+batch)
        source = torch.randn(batch, dimension, generator=generator).to(device)
        sampler(source)  # charged warm-up reported separately by caller total wall time
        synchronize(device)
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            sampler(source)
            synchronize(device)
            times.append(time.perf_counter()-start)
        report[str(batch)] = dict(seconds=times, median_seconds=float(np.median(times)))
    return report
