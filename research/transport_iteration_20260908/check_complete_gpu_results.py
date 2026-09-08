"""Portable metadata/sample audit of the four frozen GPU synthetic cases.

No training, checkpoint sampling, CUDA use, new evaluation identities or real
datasets. Recreates only the published synthetic evaluation stream on CPU.
Usage: python audit_complete_gpu.py --results RESULTS --repo REPO
       --output AUDIT_OUTPUT_DIRECTORY --scratch SCRATCH_DIRECTORY
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import tempfile

import numpy as np
import scipy
from scipy.spatial.distance import cdist
import torch

CASES={"local_image":(3100,1,"local"),"distant_image":(3101,1,"distant"),
       "local_multiframe":(3110,4,"local"),"distant_multiframe":(3111,4,"distant")}
LABELS=("spline","full_fm_heun_8","full_fm_heun_32","hierarchical_fm_heun_8","hierarchical_fm_heun_32")
METRICS=("energy_score","mean_rms_gap","second_moment_gap","mmd2_unbiased_scale_0.1",
         "mmd2_unbiased_scale_0.3","mmd2_unbiased_scale_1.0")
COMMIT="2b58285826a74e32882a6de0af2a1d951a85ba58"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate_world(seed,channels,kind):
    """Registered teacher law, independently implemented; evaluation seed only."""
    rng=np.random.default_rng(seed+100000)
    coarse=torch.from_numpy(rng.normal(size=(256,channels,2,2)).astype(np.float32))
    coarse=.7*torch.sinh(.7*coarse)
    for _ in range(2):
        noise=torch.from_numpy(rng.normal(size=(256,3*channels,*coarse.shape[2:])).astype(np.float32))
        noise=.6*torch.sinh(.6*noise)
        if kind=="distant":
            width=noise.shape[-1];half=width//2
            noise[...,half:]=noise[...,half:]+.8*(noise[...,:width-half].square()-.2)
        context=coarse.repeat(1,3,1,1)
        detail=.3*torch.sin(1.5*context)+(.25+.2*torch.sigmoid(context))*noise
        h,v,d=detail.chunk(3,dim=1)
        result=torch.empty((256,channels,coarse.shape[2]*2,coarse.shape[3]*2))
        result[...,::2,::2]=(coarse+h+v+d)/2
        result[...,::2,1::2]=(coarse+h-v-d)/2
        result[...,1::2,::2]=(coarse-h+v-d)/2
        result[...,1::2,1::2]=(coarse-h-v+d)/2
        coarse=result
    return coarse.numpy()


def scores(real,fake):
    x=fake.reshape(len(fake),-1).astype(np.float64)
    y=real.reshape(len(real),-1).astype(np.float64)
    xx=cdist(x,x)/math.sqrt(x.shape[1]);yy=cdist(y,y)/math.sqrt(x.shape[1]);xy=cdist(x,y)/math.sqrt(x.shape[1])
    n,m=len(x),len(y)
    result={"energy_score":float(xy.mean()-xx.sum()/(2*n*(n-1))),
            "mean_rms_gap":float(np.sqrt(np.mean((real.mean(0)-fake.mean(0))**2))),
            "second_moment_gap":float(np.abs((real**2).mean(0)-(fake**2).mean(0)).mean())}
    for scale in (.1,.3,1.):
        kxx=np.exp(-xx**2/(2*scale**2));kyy=np.exp(-yy**2/(2*scale**2));kxy=np.exp(-xy**2/(2*scale**2))
        result[f"mmd2_unbiased_scale_{scale}"]=float((kxx.sum()-n)/(n*(n-1))+(kyy.sum()-m)/(m*(m-1))-2*kxy.mean())
    return result


def audit_case(folder,case,repo):
    read=lambda name:json.loads((folder/name).read_text())
    summary=read("summary.json")
    assert summary==read("manifest.json")
    assert summary["commit"]==COMMIT
    seed,channels,kind=CASES[case]
    expected={"seed":seed,"channels":channels,"world":kind,"device":"cuda","steps":100000,
              "seconds":90.,"size":8,"train_size":2048,"audit_size":256,"batch":64,"width":24}
    assert {k:summary["config"][k] for k in expected}==expected
    assert len(summary["source_hashes"])==12
    for name,value in summary["source_hashes"].items():
        raw=subprocess.check_output(["git","show",COMMIT+":"+name],cwd=repo)
        assert hashlib.sha256(raw).hexdigest()==value
        assert hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()==summary["source_git_blobs"][name]
    assert summary["training"]==read("training.json")
    assert summary["evaluation"]==read("evaluation.json")
    numerical=read("numerical_validation.json")
    for key,value in numerical.items():
        assert summary[key]==value
    assert numerical["exact_stochastic_latent_copy"]
    assert numerical["source_roundtrip_max"]<=numerical["source_roundtrip_tolerance"]==.001
    assert numerical["logdet_cancellation_max"]<=numerical["logdet_cancellation_tolerance"]==.01
    assert summary["all_fits_frozen"] and summary["all_numerical_checks_pass"]
    assert summary["completed_training_arms"]==["spline","full_fm","hierarchical_fm"]
    assert all(not summary[k] for k in ["real_data_accessed","quality_advantage_established",
                                        "cost_advantage_established","representation_advantage_established"])
    with np.load(folder/"generated_samples.npz",allow_pickle=False) as archive:
        assert set(archive.files)==set(LABELS)
        samples={key:archive[key] for key in LABELS}
    with np.load(folder/"copied_stochastic_latent_samples.npz",allow_pickle=False) as archive:
        copy=archive["samples"]
    assert np.array_equal(copy,samples["spline"])
    assert all(a.shape==(256,channels,8,8) and a.dtype==np.float32 and np.isfinite(a).all() for a in samples.values())
    evaluation=evaluate_world(seed,channels,kind)
    verified={}
    for label,values in samples.items():
        actual=scores(evaluation,values)
        discrepancies={key:abs(value-summary["evaluation"][label][key]) for key,value in actual.items()}
        assert max(discrepancies.values())<1e-6
        calls=0 if label=="spline" else 2*int(label.rsplit("_",1)[1])*(1 if label.startswith("full") else 3)
        assert calls==summary["evaluation"][label]["velocity_evaluations"]
        verified[label]={"recomputed_scores":actual,"absolute_discrepancies":discrepancies,
                         "saved_array_sha256":hashlib.sha256(values.tobytes()).hexdigest(),
                         "verified_velocity_evaluations":calls}
    training={}
    for arm,record in summary["training"].items():
        state=torch.load(folder/(arm+".pt"),map_location="cpu",weights_only=True)
        assert all(torch.isfinite(t).all() for t in state.values())
        count=sum(t.numel() for t in state.values())
        assert count==record["parameters"]
        assert 0<record["steps"]<=100000
        elapsed=record["seconds"]+record["initialization_seconds"]
        assert elapsed>=89.9
        training[arm]={**record,"training_plus_initialization_seconds":elapsed,
                       "recorded_cap_overrun_seconds":max(0.,elapsed-90.),"checkpoint_scalars":count}
    filenames={"summary.json","manifest.json","training.json","evaluation.json","numerical_validation.json",
               "spline.pt","full_fm.pt","hierarchical_fm.pt","generated_samples.npz","copied_stochastic_latent_samples.npz"}
    assert {p.name for p in folder.iterdir() if p.is_file()}==filenames
    return {"case":case,"configuration":summary["config"],"source_hashes_verified":12,
            "source_commit":COMMIT,"payload_hash_manifest_originally_present":False,
            "downloaded_payloads":{p.name:{"sha256":digest(p),"size_bytes":p.stat().st_size} for p in sorted(folder.iterdir())},
            "saved_copy_bitwise_equal":True,"gaussian_coordinates":channels*64,
            "evaluation_reconstruction_seed":seed+100000,
            "evaluation_reconstructed_hash":hashlib.sha256(evaluation.tobytes()).hexdigest(),
            "metric_checks":verified,"training":training,"reported_evaluation":summary["evaluation"],
            "reported_numerical_validation":numerical,"reported_case_runtime_seconds":summary["runtime_seconds"],
            "checkpoint_sampling_rerun":False,"fit_performed":False,"training_arrays_regenerated":False}


def write_report(audit):
    cases=audit["cases"]
    text=["# Independent four-case GPU synthetic-generator audit","",
          f"Frozen source `{COMMIT}`; reported PSC job45558567 on V100/v024. All four registered cases and all three fitted arms per case are present. This audit did not contact PSC or independently query scheduler/GPU metadata.","",
          "All12 distinct source hashes/Git blobs verify in each case. All40 expected saved artifacts are present; JSON snapshots agree, checkpoints contain finite values and the reported parameter counts, and all20 generated arrays have the prescribed shape/dtype with finite values. Every saved copied-stochastic-latent sample array is bitwise equal to its saved spline sample array.","",
          "The older runner has no original COMPLETE/payload-hash manifest. A fresh local hash/size ledger for all downloaded artifacts is included in the machine-check output. This is preservation plus cross-file/recomputation evidence, not verification against a nonexistent remote payload ledger.","",
          "## Independent evaluation recomputation","",
          f"Only the four frozen evaluation seeds were recreated using an independent implementation of the recorded local/distant teacher. Independent NumPy/SciPy score formulas recomputed all120 metrics from saved generated samples. Maximum absolute discrepancy: `{audit['max_metric_absolute_discrepancy']:.9g}`. No fitting, training-stream regeneration, new seeds/cells, real datasets, or checkpoint sampling occurred. Local torch{audit['torch']} differs from PSC2.10.0+cu128; a bitwise evaluation-array match cannot be checked because original evaluation arrays/hashes were not retained.","",
          "The four-channel cases are synthetic multichannel arrays, not real video, and their8x8 spatial field is small enough for the networks to have broad receptive fields. The distant case is a stress test, not a demonstrated limitation of each neural architecture. The hierarchical FM retains fixed Haar analysis; it is not the newer learned-analysis/global-decoder baseline.","",
          "## Numerical gates (saved records)","",
          "| Case | Gaussian-source RT max | Logdet cancellation max | Saved copied samples |",
          "|---|---:|---:|---|"]
    for case in cases:
        n=case["reported_numerical_validation"]
        text.append(f"| {case['case']} | {n['source_roundtrip_max']:.8g} | {n['logdet_cancellation_max']:.8g} | bitwise equal |")
    text += ["", "All recorded numerical gates pass fixed limits.001 for source roundtrip and.01 for logdet cancellation. These values were integrity/threshold checked, not independently recomputed by sampling checkpoints. Same prior coordinate accounting is64 for one-channel and256 for four-channel arrays; the copied decoder has no extra uncounted noise.","",
             "## Complete descriptive score and cost table","",
             "All fixed scales and Heun grids are retained. Lower energy/moment/MMD is preferable, but there is one training seed per case and only256 evaluation tensors; no significance or uniform quality ordering is established. Generation timings are one batch, not repeated latency measurements.","",
             "| Case | Arm/grid | Energy | Mean RMS | Second moment | MMD .1 | MMD .3 | MMD1.0 | Seconds | Calls |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for case in cases:
        for label,r in case["reported_evaluation"].items():
            text.append(f"| {case['case']} | {label} | {r['energy_score']:.8f} | {r['mean_rms_gap']:.8f} | {r['second_moment_gap']:.8f} | {r['mmd2_unbiased_scale_0.1']:.8f} | {r['mmd2_unbiased_scale_0.3']:.8f} | {r['mmd2_unbiased_scale_1.0']:.8f} | {r['generation_seconds']:.6f} | {r['velocity_evaluations']} |")
    text += ["", "Adverse/mixed findings must remain explicit:","",
             "- In both image cases, hierarchical FM has lower energy and MMD(.3) than spline. In both multichannel cases, spline has lower energy/MMD(.3), but this is not real-video evidence. Other metrics need not follow those rankings.",
             "- Spline generation takes approximately31–41ms. Full FM32 and hierarchical FM8 each take about16ms in these recorded batches; the CPU spline timing advantage does not transfer to this GPU result.",
             "- Spline also uses more measured peak allocated GPU training memory than both FM controls in all four cases:6.23MiB versus2.52/1.46MiB for image, and19.36MiB versus4.18/3.36MiB for multichannel. It has substantially more parameters; this experiment does not support a memory-efficiency advantage.",
             "- The local-image full-FM8 batch takes about697ms, while32 steps take about16ms. This inversion is a warmup/cache/timing anomaly warning; it cannot support a clean solver-cost frontier. Preserve both observations without selecting the favorable one.",
             "- Heun8/32 are step counts: full FM uses16/64 velocity calls, hierarchical FM48/192 across three spatial resolutions. Call counts at different resolutions are not equivalent FLOPs or latency.","",
             "## Training/resource accounting","",
             "| Case | Fitted arm | Parameters | Updates | Training+init seconds | Cap overrun seconds | Peak GPU allocated MiB |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for case in cases:
        for arm,r in case["training"].items():
            memory="unavailable" if r["peak_training_allocated_bytes"] is None else f"{r['peak_training_allocated_bytes']/2**20:.2f}"
            text.append(f"| {case['case']} | {arm} | {r['parameters']:,} | {r['steps']:,} | {r['training_plus_initialization_seconds']:.6f} | {r['recorded_cap_overrun_seconds']:.6f} | {memory} |")
    text += ["", "All update counts remain below100,000. The90-second budget includes initialization and allows a final in-flight update overrun; actual overruns are retained above. These were different-size networks with unequal update counts. Measured PyTorch allocated GPU memory is not total driver/process/device memory. Summed case runtime is %.3fs; reported scheduler elapsed18:52 includes launch/testing/other overhead and is not independently queried here."%sum(c["reported_case_runtime_seconds"] for c in cases),"",
             "No theoretical bound is demonstrated by these finite-cap neural fits; all advantages in the original summary remain false. The results establish complete execution, passed recorded numerical gates, reproducible exploratory metrics, and mixed quality/cost outcomes. They do not establish a generation/representation/efficiency win against optimized latent flow/diffusion or the stronger new learned-analysis baseline.","",
             "Audit source is portable: `audit_complete_gpu.py --results RESULTS --repo REPO --output OUTPUT --scratch SCRATCH`. The machine-check JSON binds local payload hashes and records all recomputed scores/discrepancies. No repository mutation, commit, checkpoint sampling, real-data access or PSC action occurred."]
    result="\n".join(text)+"\n"
    for a,b in [("job455","job 455"),("All12","All 12"),("All40","All 40"),("all20","all 20"),("all120","all 120"),("torch2","torch 2"),("PSC2","PSC 2"),("their8x8","their 8x8"),("limits.001","limits .001"),("and.01","and .01"),("accounting is64","accounting is 64"),("and256","and 256"),("only256","only 256"),("approximately31","approximately 31"),("FM32","FM 32"),("FM8","FM 8"),("about16ms","about 16 ms"),("about697ms","about 697 ms"),("while32","while 32"),("Heun8/32","Heun 8/32"),("uses16/64","uses 16/64"),("FM48/192","FM 48/192"),("below100,000","below 100,000"),("The90-second","The 90-second"),("elapsed18:52","elapsed 18:52")]: result=result.replace(a,b)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--repo",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--scratch",type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True);args.scratch.mkdir(parents=True,exist_ok=True)
    destinations=[args.output/"psc-complete-gpu-review.md",args.output/"psc-complete-gpu-machinecheck.json"]
    if any(p.exists() for p in destinations):raise FileExistsError("refusing to overwrite an earlier audit")
    torch.set_num_threads(1)
    cases=[audit_case(args.results/name,name,args.repo) for name in CASES]
    maxdiff=max(v for c in cases for m in c["metric_checks"].values() for v in m["absolute_discrepancies"].values())
    audit={"source_commit":COMMIT,"cases":cases,"cases_audited":4,"scalar_metrics_recomputed":120,
           "max_metric_absolute_discrepancy":maxdiff,"torch":torch.__version__,"numpy":np.__version__,
           "scipy":scipy.__version__,"platform":platform.platform(),"fit_performed":False,
           "checkpoint_sampling_performed":False,"real_data_accessed":False,"psc_actions":False}
    with tempfile.TemporaryDirectory(prefix="gpu-audit-",dir=args.scratch) as temporary:
        for filename,value in [("psc-complete-gpu-review.md",write_report(audit)),
                               ("psc-complete-gpu-machinecheck.json",json.dumps(audit,indent=2,sort_keys=True,allow_nan=False)+"\n")]:
            staged=Path(temporary)/filename;staged.write_text(value)
            os.replace(staged,args.output/filename)
    print(json.dumps({"cases":4,"metrics":120,"max_metric_discrepancy":maxdiff,
                      "copy_equal_all":all(c["saved_copy_bitwise_equal"] for c in cases)},indent=2))


if __name__=="__main__":main()
