"""Audit one saved CPU synthetic case without fitting or regenerating training."""
from pathlib import Path
import hashlib
import json
import math
import subprocess
import sys
import argparse

import numpy as np
import scipy
from scipy.spatial.distance import cdist
import torch

torch.set_num_threads(1)
REPO=Path(__file__).resolve().parents[2]
RESULTS=Path(__file__).resolve().parent/"psc_complete_cpu"
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--scratch", type=Path, required=True)
args=parser.parse_args()
if args.output.exists() or args.scratch.exists():
    raise FileExistsError("output and scratch must be new paths")
COMMIT="2b58285826a74e32882a6de0af2a1d951a85ba58"
summary=json.loads((RESULTS/"summary.json").read_text())
assert summary["commit"]==COMMIT and summary["config"]["device"]=="cpu"
assert summary==json.loads((RESULTS/"manifest.json").read_text())
assert summary["training"]==json.loads((RESULTS/"training.json").read_text())
assert summary["evaluation"]==json.loads((RESULTS/"evaluation.json").read_text())
assert len(summary["source_hashes"])==12
scratch=args.scratch
(scratch/"qalt").mkdir(parents=True,exist_ok=True)
for name,expected in summary["source_hashes"].items():
    source=subprocess.check_output(["git","show",COMMIT+":"+name],cwd=REPO)
    assert hashlib.sha256(source).hexdigest()==expected,name
    blob=hashlib.sha1(b"blob "+str(len(source)).encode()+b"\0"+source).hexdigest()
    assert blob==summary["source_git_blobs"][name]
    if name.startswith("qalt/src/qalt/"):
        (scratch/"qalt"/Path(name).name).write_bytes(source)
sys.path.insert(0,str(scratch))
from qalt.multiscale_flow import MultiscaleSplineFlow,CopiedStochasticLatentDecoder
from qalt.flow_matching import FullTensorFlowMatching,HierarchicalFlowMatching


def recreate_evaluation():
    """Independent implementation of the registered local world, seed103100."""
    rng=np.random.default_rng(3100+100000)
    coarse=torch.tensor(rng.normal(size=(256,1,2,2)).astype(np.float32))
    coarse=.7*torch.sinh(.7*coarse)
    for _ in range(2):
        noise=torch.tensor(rng.normal(size=(256,3,*coarse.shape[2:])).astype(np.float32))
        noise=.6*torch.sinh(.6*noise)
        context=coarse.repeat(1,3,1,1)
        detail=.3*torch.sin(1.5*context)+(.25+.2*torch.sigmoid(context))*noise
        horizontal,vertical,diagonal=detail.chunk(3,dim=1)
        output=torch.empty((256,1,coarse.shape[2]*2,coarse.shape[3]*2))
        output[...,::2,::2]=(coarse+horizontal+vertical+diagonal)/2
        output[...,::2,1::2]=(coarse+horizontal-vertical-diagonal)/2
        output[...,1::2,::2]=(coarse-horizontal+vertical-diagonal)/2
        output[...,1::2,1::2]=(coarse-horizontal-vertical+diagonal)/2
        coarse=output
    return coarse.numpy()


def independent_metrics(real,fake):
    # Scores use saved samples; no model or evaluator fitting occurs.
    x=fake.reshape(len(fake),-1).astype(np.float64)
    y=real.reshape(len(real),-1).astype(np.float64)
    xx=cdist(x,x)/np.sqrt(x.shape[1]);yy=cdist(y,y)/np.sqrt(x.shape[1]);xy=cdist(x,y)/np.sqrt(x.shape[1])
    n,m=len(x),len(y)
    result={"energy_score":float(xy.mean()-xx.sum()/(2*n*(n-1))),
        "mean_rms_gap":float(np.sqrt(np.mean((real.mean(axis=0)-fake.mean(axis=0))**2))),
        "second_moment_gap":float(np.mean(np.abs((real**2).mean(axis=0)-(fake**2).mean(axis=0))))}
    for scale in [.1,.3,1.]:
        kxx=np.exp(-xx**2/(2*scale**2));kyy=np.exp(-yy**2/(2*scale**2));kxy=np.exp(-xy**2/(2*scale**2))
        result[f"mmd2_unbiased_scale_{scale}"]=float((kxx.sum()-n)/(n*(n-1))+(kyy.sum()-m)/(m*(m-1))-2*kxy.mean())
    return result


assert summary["config"]=={"output":summary["config"]["output"],"seed":3100,"steps":100000,"seconds":90.,
    "size":8,"channels":1,"train_size":2048,"audit_size":256,"batch":64,"width":24,"world":"local","device":"cpu"}
expected_labels=["spline","full_fm_heun_8","full_fm_heun_32","hierarchical_fm_heun_8","hierarchical_fm_heun_32"]
with np.load(RESULTS/"generated_samples.npz",allow_pickle=False) as archive:
    assert set(archive.files)==set(expected_labels)
    saved={key:archive[key] for key in expected_labels}
with np.load(RESULTS/"copied_stochastic_latent_samples.npz",allow_pickle=False) as archive:
    copied_saved=archive["samples"]
assert np.array_equal(copied_saved,saved["spline"])
assert all(a.shape==(256,1,8,8) and a.dtype==np.float32 and np.isfinite(a).all() for a in saved.values())
evaluation=recreate_evaluation()
metric_check={}
for label,samples in saved.items():
    actual=independent_metrics(evaluation,samples)
    differences={k:abs(value-summary["evaluation"][label][k]) for k,value in actual.items()}
    assert max(differences.values())<1e-6
    metric_check[label]={"recomputed":actual,"absolute_discrepancies":differences}

constructors={"spline":lambda:MultiscaleSplineFlow(1,8,2,4,2,24,8),
              "full_fm":lambda:FullTensorFlowMatching(1,8,2,24),
              "hierarchical_fm":lambda:HierarchicalFlowMatching(1,8,2,24)}
models={}
for name,constructor in constructors.items():
    model=constructor()
    state=torch.load(RESULTS/(name+".pt"),map_location="cpu",weights_only=True)
    model.load_state_dict(state,strict=True);model.eval();models[name]=model
    assert sum(p.numel() for p in model.parameters())==summary["training"][name]["parameters"]
    assert all(torch.isfinite(value).all() for value in state.values())
    record=summary["training"][name]
    assert 0<record["steps"]<=100000
    assert 89.9<record["seconds"]+record["initialization_seconds"]<90.1

gaussian=torch.randn(256,64,generator=torch.Generator().manual_seed(3100+200000))
regeneration={}
with torch.no_grad():
    for label in expected_labels:
        name=label.split("_heun_")[0]
        if label=="spline":
            generated=models[name].sample_from_gaussian(gaussian)
            expected_calls=0
        else:
            steps=int(label.rsplit("_",1)[1]);generated=models[name].sample_from_gaussian(gaussian,steps=steps)
            expected_calls=2*steps*(1 if name=="full_fm" else 3)
        assert expected_calls==summary["evaluation"][label]["velocity_evaluations"]
        difference=float(np.max(np.abs(generated.numpy()-saved[label])))
        regeneration[label]={"max_saved_sample_absolute_difference":difference,
                             "bitwise_equal_to_saved":bool(np.array_equal(generated.numpy(),saved[label])),
                             "verified_velocity_evaluations":expected_calls}
    source_recovered,forward_ld=models["spline"].encode(torch.tensor(saved["spline"]))
    generated,inverse_ld=models["spline"].decode(gaussian)
    roundtrip=float((source_recovered-gaussian).abs().max())
    cancellation=float((forward_ld+inverse_ld).abs().max())
    assert roundtrip<=1e-3 and cancellation<=.01
    copied=CopiedStochasticLatentDecoder(models["spline"]).sample_from_gaussian(gaussian)
    assert torch.equal(copied,generated)
assert summary["source_roundtrip_max"]<=summary["source_roundtrip_tolerance"]==.001
assert summary["logdet_cancellation_max"]<=summary["logdet_cancellation_tolerance"]==.01
assert summary["completed_training_arms"]==list(constructors)
assert summary["all_fits_frozen"] and summary["all_numerical_checks_pass"]
assert not any(summary[k] for k in ["real_data_accessed","quality_advantage_established","cost_advantage_established","representation_advantage_established"])
payloads={p.name:{"sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"size_bytes":p.stat().st_size} for p in sorted(RESULTS.iterdir()) if p.is_file()}
audit={"source_commit":COMMIT,"source_hashes_and_git_blobs_verified":12,"downloaded_payloads":payloads,
       "original_payload_hash_manifest_present":False,"saved_copy_bitwise_equal":True,
       "independent_evaluation_recomputed":True,"training_data_regenerated":False,"fit_performed":False,
       "torch_reconstruction_version":torch.__version__,"numpy":np.__version__,"scipy":scipy.__version__,
       "metric_check":metric_check,"sample_regeneration":regeneration,
       "numerical_recomputed":{"source_roundtrip_max":roundtrip,"logdet_cancellation_max":cancellation},
       "all_registered_numerical_checks_pass":True,"copied_decoder_reconstructed_exact":True}
args.output.write_text(json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+"\n")
print(json.dumps({"max_metric_discrepancy":max(v for r in metric_check.values() for v in r["absolute_discrepancies"].values()),
                  "regeneration":regeneration,"numerical":audit["numerical_recomputed"],"payload_count":len(payloads)},indent=2))
