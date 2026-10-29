"""Bounded v2 control campaign with three genuinely fresh final matches.

Competition labels are never opened. Existing release dead-ball weights stay
frozen; only team control and its transition cost are candidates.
"""
from __future__ import annotations

import argparse
import copy
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from possession.config import load_config
from possession.control import feature_matrix as control_features, fit_model as fit_control, FEATURE_NAMES
from possession.decoder import decode
from possession.features import extract
from possession.io import sha256, write_json
from possession.learned import (feature_matrix as dead_features, load_model, save_model,
                                learned_probabilities, resolve_config)
from possession.tags import build
from tools.external_benchmark import SOURCE, LICENSE, digest, read_match
from tools.train_stoppages import CATALOG as OLD_CATALOG, SEEDS as OLD_SEEDS, corrupt_tracking, measure, paired_spell_bootstrap
from tools.control_diagnostics import diagnostic, training_spell_weights

CATALOG = dict(OLD_CATALOG, **{
    "J03WPY": {"metadata": (51643487,11922,"65a4568e4f374af94d3c1015cbc7c4f5"),
                "tracking": (51643526,371688499,"e56ff1336a334f8da51110fbe7cab764")},
    "J03WQQ": {"metadata": (51643484,11932,"5979209af316a914ce57347d0633df6c"),
                "tracking": (51643529,356219423,"9105bad987191d3c3c362c1fce9a4fcd")},
    "J03WR9": {"metadata": (51643490,11823,"5e2d580d3d8ed97e32959de90d04992b"),
                "tracking": (51643532,372934140,"e060bb7d7a5a3bfbd15bbf9bc631024d")},
})
SPLITS = {"train": ["J03WOH","J03WOY"], "validation": ["J03WN1","J03WMX"],
          "final": ["J03WPY","J03WQQ","J03WR9"]}
SEEDS = dict(OLD_SEEDS, J03WPY=71201, J03WQQ=71213, J03WR9=71237)
REGIMES = ("clean","degraded")
L2_VALUES = (0.001,0.01)
SWITCH_COSTS = (4.0,2.0,1.0)
WEIGHTINGS = ("uniform","spell-balanced")
BASELINE = "release-1.1"
MIN_GAIN = 0.02
MAX_CELL_DROP = 0.02


def candidate_ids():
    return {BASELINE, "decoder-cost-2", "decoder-cost-1"} | {
        f"control-{weighting}-l2-{l2:g}-cost-{cost:g}"
        for weighting in WEIGHTINGS for l2 in L2_VALUES for cost in SWITCH_COSTS}


def scientific_hashes():
    files = sorted((ROOT/"possession").glob("*.py")) + [
        ROOT/name for name in ("tools/train_control.py","tools/control_diagnostics.py",
            "tools/train_stoppages.py","tools/external_benchmark.py","evaluation/possession_metrics.py")]
    return {p.relative_to(ROOT).as_posix():sha256(p) for p in files}


def read_external(data_dir, match):
    paths=[]
    for kind in ("metadata","tracking"):
        _,size,expected=CATALOG[match][kind]
        p=data_dir/f"{match}_{kind}.xml"
        if not p.is_file() or p.stat().st_size!=size or digest(p)!=expected:
            raise ValueError(f"Missing or mismatched input: {p.name}")
        paths.append(p)
    data,reference,blocks,pitch,alignment=read_match(*paths,match)
    provenance={p.name:{"sha256":sha256(p),"bytes":p.stat().st_size} for p in paths}
    return data,reference,blocks,pitch,alignment,provenance


def download_inputs(data_dir):
    data_dir.mkdir(parents=True,exist_ok=True)
    for match in CATALOG:
        for kind in ("metadata","tracking"):
            ident,size,expected=CATALOG[match][kind]
            p=data_dir/f"{match}_{kind}.xml"
            if not p.exists():
                tmp=p.with_suffix(".download")
                print(f"Downloading {p.name} ({size:,} bytes)",flush=True)
                with urllib.request.urlopen(f"https://ndownloader.figshare.com/files/{ident}",timeout=60) as r,tmp.open("wb") as f:
                    while chunk:=r.read(1048576): f.write(chunk)
                if tmp.stat().st_size!=size or digest(tmp)!=expected:
                    raise ValueError(f"Download verification failed: {p.name}")
                tmp.replace(p)
            if p.stat().st_size!=size or digest(p)!=expected:
                raise ValueError(f"Input verification failed: {p.name}")


def compare_cells(candidate,baseline):
    by_cell=lambda rows:{(r["match"],r["regime"]):r for r in rows}
    c,b=by_cell(candidate),by_cell(baseline)
    if len(c)!=len(candidate) or len(b)!=len(baseline) or not c or c.keys()!=b.keys():
        raise ValueError("Candidate and baseline need identical unique match/regime cells")
    for rows in (candidate,baseline):
        if any(r["regime"] not in REGIMES or not all(np.isfinite(r[k]) and 0<=r[k]<=1 for k in ("spellF1","macroF1")) for r in rows):
            raise ValueError("Invalid regime or score")
    if any({regime for m,regime in c if m==match} != set(REGIMES) for match,_ in c):
        raise ValueError("Every match needs both regimes")
    gains={f"{m}/{r}":float(c[m,r]["spellF1"]-b[m,r]["spellF1"]) for m,r in sorted(c)}
    mean=float(np.mean(list(gains.values())))
    return {"passed":bool(mean>=MIN_GAIN and min(gains.values())>=-MAX_CELL_DROP),
            "mean_spellF1_gain":mean,"cell_gains":gains,
            "minimum_mean_gain":MIN_GAIN,"maximum_cell_drop":MAX_CELL_DROP}


def select_candidate(records):
    grouped={}
    for row in records: grouped.setdefault(row["candidate"],[]).append(row)
    if BASELINE not in grouped: raise ValueError("Missing frozen-release baseline")
    if set(grouped) != candidate_ids():
        raise ValueError("Candidate evidence differs from the predeclared candidate set")
    required={(match,regime) for match in SPLITS["validation"] for regime in REGIMES}
    if any({(r["match"],r["regime"]) for r in rows} != required for rows in grouped.values()):
        raise ValueError("Validation evidence needs every declared match/regime cell")
    gates={name:compare_cells(rows,grouped[BASELINE]) for name,rows in grouped.items()}
    rank=sorted(grouped,key=lambda name:(-np.mean([r["spellF1"] for r in grouped[name]]),
                                         -np.mean([r["macroF1"] for r in grouped[name]]),name))
    eligible=[name for name in rank if name==BASELINE or gates[name]["passed"]]
    return {"selected":eligible[0],"ranking":rank,"eligible_ranking":eligible,"gates":gates}


def prepared(data,blocks,pitch,regime,match,config):
    observed=data if regime=="clean" else corrupt_tracking(data,blocks,pitch,25,SEEDS[match])
    features=extract(observed,pitch,blocks,25,config)
    tags=build([],features,observed.team_names,len(observed.ball),25,blocks,config)
    return features,tags,dead_features(observed,features,blocks,pitch,25),control_features(observed,features,blocks,pitch,25)


def score_job(destination,match,candidate,regime,reference,probabilities,blocks,config,usable,bootstrap):
    """Decode the same joint probabilities used by production, then score externally."""
    labels=np.full(len(reference),"X",dtype="<U1")
    for block,has_evidence in zip(blocks,usable):
        _,a,b=block
        if has_evidence: labels[a:b+1]=np.array(list("HAD"))[decode(probabilities[a:b+1],25,config)]
    row=measure(destination,match,candidate,regime,reference,labels,bootstrap)
    row["control_diagnostics"]=diagnostic(reference,labels,25)
    write_json(destination/"control-diagnostics.json",row["control_diagnostics"])
    return row,labels


def evaluate_match(pool,out,match,data_dir,models,config,stage,bootstrap=0):
    print(f"Reading {stage.upper()} {match}",flush=True)
    data,reference,blocks,pitch,alignment,provenance=read_external(data_dir,match)
    write_json(out/f"{match}-inputs.json",provenance)
    write_json(out/f"{match}-alignment.json",alignment)
    records=[]; pairs={}
    for regime in REGIMES:
        features,tags,dead_x,control_x=prepared(data,blocks,pitch,regime,match,config)
        usable=[bool(features.reliability[a:b+1].sum()+tags.live[a:b+1].sum()+tags.restart_live[a:b+1].sum()) for _,a,b in blocks]
        jobs={}
        for candidate,model in models.items():
            probabilities=learned_probabilities(features,tags,dead_x,model,control_x)
            effective=resolve_config(config,model)
            destination=out/stage/match/regime/candidate
            job=pool.submit(score_job,destination,match,candidate,regime,reference,probabilities,blocks,effective,usable,bootstrap)
            jobs[job]=candidate
        labels_by_candidate={}
        for job in as_completed(jobs):
            row,labels=job.result()
            records.append(row)
            if stage=="final": labels_by_candidate[row["candidate"]]=labels
        if stage=="final" and len(models)==2:
            chosen=next(c for c in models if c!=BASELINE)
            pairs[f"{match}/{regime}"]=paired_spell_bootstrap(reference,labels_by_candidate[BASELINE],labels_by_candidate[chosen])
        del features,tags,dead_x,control_x,labels_by_candidate
    return sorted(records,key=lambda r:(r["match"],r["regime"],r["candidate"])),pairs


def run(args):
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("Use a fresh campaign output directory; do not overwrite evidence")
    args.out.mkdir(parents=True,exist_ok=True)
    baseline=load_model(args.baseline)
    if "control_model" in baseline or baseline.get("decoder_mode")!="joint":
        raise ValueError("Baseline must be the frozen v1.1 joint live/dead model")
    if sha256(args.baseline)!="3759734a29a07eafc11723029800d30863d00c0a9e478bea3fcafaf6805c2867":
        raise ValueError("Frozen v1.1 baseline bytes differ")
    config=load_config(args.config)
    if sha256(args.config)!="8d190bec7b92ae61157f6b41fbc839bc6d400e6f626908d4a6d8bdacc7dd76cf":
        raise ValueError("Frozen v1.1 baseline configuration bytes differ")
    frozen=scientific_hashes()
    protocol={"schema_version":2,"dataset":SOURCE,"license":LICENSE,"splits":SPLITS,
        "exposed_historical_test_now_validation":["J03WMX"],"catalog":CATALOG,"seeds":SEEDS,
        "regimes":list(REGIMES),"train_stride":5,"control_features":list(FEATURE_NAMES),
        "l2_values":list(L2_VALUES),"team_switch_costs":list(SWITCH_COSTS),"weightings":list(WEIGHTINGS),
        "weighting_definition":"Training-only inverse-sqrt H/A spell duration in seconds; final weights capped [0.5,3], live mean1, D/X0; no bridging D/X/half boundaries",
        "baseline":BASELINE,"baseline_sha256":sha256(args.baseline),"config":config.as_dict(),
        "config_sha256":sha256(args.config),"candidate_ids":sorted(candidate_ids()),
        "source_hashes":frozen,"candidate_count":15,"dead_head":"Frozen release1.1 weights; no refit",
        "tags":"External automatic tags unavailable. Existing production tag fusion retained, not validated by this benchmark.",
        "selection":"Gate each candidate against v1.1 on all four validation match/regime cells; rank eligible by equal-cell mean spellF1, then macroF1, then name; baseline fallback.",
        "gate":{"minimum_mean_spellF1_gain":MIN_GAIN,"maximum_any_match_regime_drop":MAX_CELL_DROP},
        "final_policy":"Freeze one winner and all bytes before any of three fresh final references are read. Compare only winner vs release1.1 across all six cells. Failure retains v1.1; no runner-up or final tuning.",
        "competition_labels_used":False,"pretrained_external_checkpoints_used":False}
    write_json(args.out/"protocol.json",protocol)
    protocol_hash=sha256(args.out/"protocol.json")
    train_x=[];train_y=[];train_w=[];train_meta={}
    for match in SPLITS["train"]:
        print(f"Reading TRAIN {match}",flush=True)
        data,reference,blocks,pitch,alignment,provenance=read_external(args.data_dir,match)
        train_meta[match]={"inputs":provenance,"reference_sha256":hashlib.sha256(reference.tobytes()).hexdigest()}
        weights=training_spell_weights(reference,25)
        indices=np.arange(0,len(reference),5)
        for regime in REGIMES:
            _,_,_,matrix=prepared(data,blocks,pitch,regime,match,config)
            train_x.append(matrix[indices]);train_y.append(reference[indices]);train_w.append(weights[indices])
        del data,reference,matrix
    X,y,w=np.concatenate(train_x),np.concatenate(train_y),np.concatenate(train_w)
    del train_x,train_y,train_w
    models={BASELINE:baseline}
    for cost in SWITCH_COSTS[1:]:
        model=copy.deepcopy(baseline);model["decoder_settings"]={"team_switch_cost":cost}
        models[f"decoder-cost-{cost:g}"]=model
    for weighting in WEIGHTINGS:
        for l2 in L2_VALUES:
            print(f"Fitting control {weighting} l2={l2}",flush=True)
            control=fit_control(X,y,l2=l2,sample_weight=None if weighting=="uniform" else w,
                metadata={"dataset":SOURCE,"license":LICENSE,"training_matches":train_meta,
                          "weighting":weighting,"protocol_sha256":protocol_hash})
            for cost in SWITCH_COSTS:
                cid=f"control-{weighting}-l2-{l2:g}-cost-{cost:g}"
                model=copy.deepcopy(baseline)
                model.update(control_model=control,decoder_settings={"team_switch_cost":cost},candidate_id=cid)
                model["metadata"]=dict(model["metadata"],control_training_matches=train_meta,
                    v2_validation_matches=SPLITS["validation"],v2_final_test_matches=SPLITS["final"],
                    v2_protocol_sha256=protocol_hash)
                models[cid]=model
    del X,y,w
    for cid,model in models.items(): save_model(args.out/"models"/f"{cid}.json",model)
    validation=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for match in SPLITS["validation"]:
            rows,_=evaluate_match(pool,args.out,match,args.data_dir,models,config,"validation")
            validation.extend(rows)
        if scientific_hashes()!=frozen: raise ValueError("Scientific code changed during validation")
        selection=select_candidate(validation)
        selected=selection["selected"]
        selection.update(validation_records=validation,selected_model_sha256=sha256(args.out/"models"/f"{selected}.json"),
                         protocol_sha256=protocol_hash)
        write_json(args.out/"selection-frozen-before-final.json",selection)
        print(f"Frozen validation winner: {selected}",flush=True)
        final=[];uncertainty={}
        if selected!=BASELINE:
            final_models={BASELINE:baseline,selected:models[selected]}
            for match in SPLITS["final"]:
                rows,pairs=evaluate_match(pool,args.out,match,args.data_dir,final_models,config,"final",1000)
                final.extend(rows);uncertainty.update(pairs)
    gate=compare_cells([r for r in final if r["candidate"]==selected],
                        [r for r in final if r["candidate"]==BASELINE]) if final else None
    if scientific_hashes()!=frozen: raise ValueError("Scientific code changed during final evaluation")
    promoted=selected!=BASELINE and gate is not None and gate["passed"]
    release=models[selected] if promoted else baseline
    save_model(args.out/"release-model.json",release)
    write_json(args.out/"release-config.json",resolve_config(config,release).as_dict())
    summary={"selected_on_validation":selected,"promoted":promoted,"final_gate":gate,
        "validation_records":validation,"final_records":final,"paired_final_uncertainty":uncertainty,
        "protocol":protocol,"protocol_sha256":protocol_hash,"source_hashes":frozen,
        "winner_frozen_before_final_sha256":sha256(args.out/"selection-frozen-before-final.json"),
        "release_model_sha256":sha256(args.out/"release-model.json"),
        "release_candidate":selected if promoted else BASELINE,
        "competition_public_labels_used":False,"final_set_retuning":False}
    write_json(args.out/"summary.json",summary)
    print(json.dumps({"selected":selected,"promoted":promoted,"final_gate":gate}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir",type=Path,default=ROOT/"data/external/idsse")
    parser.add_argument("--out",type=Path,default=ROOT/"runs/control-campaign-v2")
    parser.add_argument("--baseline",type=Path,default=ROOT/"models/baselines/v1.1.json")
    parser.add_argument("--config",type=Path,default=ROOT/"configs/baselines/v1.1.json")
    parser.add_argument("--workers",type=int,choices=range(1,5),default=2)
    parser.add_argument("--download",action="store_true")
    args=parser.parse_args()
    if args.download: download_inputs(args.data_dir)
    run(args)


if __name__=="__main__":
    main()
