"""Audit saved readiness metadata and published repair IDs; never load pixels."""
from pathlib import Path
import hashlib
import json
import subprocess
import argparse

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "psc_observed_inputs"
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
COMMIT = "2b58285826a74e32882a6de0af2a1d951a85ba58"


def read(name):
    return json.loads((OUT/name).read_text())


def stable_hash(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()


def id_hash(values):
    return hashlib.sha256(np.asarray(values,dtype="<i8").tobytes()).hexdigest()


completion, provenance, ledger, ids, summary = [read(n) for n in
    ["COMPLETE.json","provenance.json","data_ledger.json","record_ids.json","summary.json"]]
assert completion["source_commit"]==provenance["source_commit"]==summary["source_commit"]==COMMIT
assert completion["status"]==summary["status"]=="input_readiness_checks_passed"
assert len(completion["payload_hashes"])==6
assert set(completion["payload_hashes"])=={p.name for p in OUT.glob("*.json") if p.name!="COMPLETE.json"}
for name,digest in completion["payload_hashes"].items():
    assert hashlib.sha256((OUT/name).read_bytes()).hexdigest()==digest,name
for name,digest in provenance["source_hashes"].items():
    raw=subprocess.check_output(["git","show",COMMIT+":"+name],cwd=REPO)
    assert hashlib.sha256(raw).hexdigest()==digest,name
assert len(provenance["source_hashes"])==6
assert stable_hash({k:v for k,v in ledger.items() if k!="ledger_sha256"})==ledger["ledger_sha256"]
frozen_manifest=json.loads(subprocess.check_output(["git","show",COMMIT+":qalt/data/observed_manifest_v1.json"],cwd=REPO))
file_hashes=frozen_manifest["cifar10"]["allowed_development_batches"]
allowlist=[f"data_batch_{i}" for i in range(1,6)]
assert ledger["file_allowlist"]==allowlist
assert [x["name"] for x in ledger["files"]]==allowlist
assert ledger["opened_files"]==[x["path"] for x in ledger["files"]]
for file in ledger["files"]:
    assert file["sha256"]==file_hashes[file["name"]]
assert ledger["canonical_dataset_verified"] and not ledger["allow_noncanonical_fixture"]
assert ledger["full_split_counts"]=={"fit":40000,"repair_holdout":5000,"excluded_discovery":5000}
assert ledger["full_split_sha256"]=="814c2280cea33403d9370b02ea12c83df3a5f89bbbf593567fed6e960ef321e7"
assert ledger["full_partition_id_sha256"]["fit"]=="4f002c7cfe2e3d1ca54a7c8847982941d655060e6989a66f87094e4d632be457"
assert ledger["full_partition_id_sha256"]["repair_holdout"]=="ced1fb315bba0eeb81c03b6f2fb296afd3208cb02a7cff6dc513327708ea8939"
for key,count,hash_key in [("fit",4000,"fit_ids_sha256"),("seen_repair",1000,"repair_ids_sha256")]:
    selected=ids[key]
    assert len(selected)==count and selected==sorted(set(selected))
    assert all(isinstance(i,int) and 0<=i<50000 for i in selected)
    assert id_hash(selected)==ledger[hash_key]
assert not set(ids["fit"])&set(ids["seen_repair"])
assert ledger["fit_per_class"]==400 and ledger["repair_per_class"]==100
assert ledger["fit_repair_intersection"]==ledger["selected_discovery_intersection"]==0

# Existing published development artifact only. NPZ lazily extracts just these
# two named metadata entries: none of its score arrays are materialized.
parent = REPO/"qalt/results/observed_b4_radial_child_20260830/seed_2100/scores.npz"
with np.load(parent,allow_pickle=False) as archive:
    parent_ids=archive["record_ids"]
    parent_labels=archive["labels"]
assert id_hash(parent_ids)==ledger["full_partition_id_sha256"]["repair_holdout"]
assert len(parent_ids)==len(parent_labels)==5000
assert np.array_equal(np.bincount(parent_labels,minlength=10),np.full(10,500))
assert set(ids["seen_repair"])<=set(parent_ids)
assert not set(ids["fit"])&set(parent_ids)
labels_by_id={int(i):int(label) for i,label in zip(parent_ids,parent_labels)}
assert np.array_equal(np.bincount([labels_by_id[i] for i in ids["seen_repair"]],minlength=10),np.full(10,100))
selected_again=[]
for label in range(10):
    candidates=[int(i) for i in parent_ids if labels_by_id[int(i)]==label]
    rank=lambda i:(hashlib.sha256(f"observed-flow-cifar-selection-v1|repair|{i:05d}".encode()).digest(),i)
    selected_again.extend(sorted(candidates,key=rank)[:100])
assert sorted(selected_again)==ids["seen_repair"]

checks={}
for role,count in [("fit",4000),("seen_repair",1000)]:
    numeric=read(role+"_numerical_checks.json")
    assert numeric==summary["numerical_checks"][role]
    assert numeric["record_count"]==count and numeric["coordinates_per_record"]==3072
    assert numeric["float64_roundtrip_limit"]==1e-12 and numeric["float32_logit_roundtrip_limit"]==1e-6
    assert numeric["float64_logit_sigmoid_roundtrip_max"]<=1e-12
    assert numeric["float32_logit_sigmoid_roundtrip_max"]<=1e-6
    assert numeric["all_checked_tensors_finite"] and numeric["unit_cube_input_strictly_interior"] and numeric["passed"]
    checks[role]=numeric
for obj in [summary,provenance]:
    assert obj["training_performed"]==obj["generation_performed"]==obj["quality_statistics_computed"]==False
assert not provenance["test_data_accessed"] and not ledger["test_data_accessed"]
assert not provenance["pro5_twenty_thousand_split_executed"]
assert summary["wall_seconds"]<270 and summary["peak_rss_bytes"]<2000*1024*1024

audit={"source_commit":COMMIT,"payload_hashes_verified":6,"source_hashes_verified":6,
       "ledger_hash_verified":True,"selected_id_hashes_verified":True,
       "selected_counts_sorted_unique_range_disjoint_verified":True,
       "repair_membership_class_quota_hash_ranking_independently_verified":True,
       "selected_fit_excludes_all_parent_repair_ids":True,
       "full_split_hashes_match_frozen_identity":True,
       "fit_class_membership_and_discovery_exclusion":
           "source/ledger verified, not independently reconstructed: readiness outputs omit full fit/discovery membership and fitting labels",
       "pixel_reads_performed":False,"score_array_reads_performed":False,
       "numerical_checks":checks,"no_training_generation_or_quality_claim":True}
args.output.write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
print(json.dumps({k:v for k,v in audit.items() if k!="numerical_checks"},indent=2))
