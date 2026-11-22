import hashlib
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.ms_swift_aqa.papers28 import finish_queue, report


def test_replay_requires_identical_counts_and_allows_reduction_roundoff(tmp_path):
    matrix = [[160,5,3,2],[145,209,11,77],[72,12,107,54],[150,14,12,208]]
    recalls = [row[i]/sum(row) for i,row in enumerate(matrix)]
    ser = dict(uar=0.5981071274663712, accuracy=684/1241, utterances=1241, confusion_matrix=matrix)
    assert math.isclose(ser['uar'],sum(recalls)/4,abs_tol=1e-15)
    recorded = {'bora':dict(checkpoint={'adapter_sha256':'verified'},
                            ser=ser,mmau={'official_text_accuracy':.718})}
    independent = dict(ser,uar=math.nextafter(ser['uar'],0.0))
    paired = {'frozen-base':{},'bora':dict(ser=independent,selected_weights_sha256='verified',
                                            mmau={'official_accuracy':.718})}
    path = tmp_path/'paired-target-scores.json'
    put(path, {'scores':paired})
    assert report.current_replay(tmp_path,recorded)
    independent['uar'] -= 0.000001
    put(path, {'scores':paired})
    assert not report.current_replay(tmp_path,recorded)
    independent.update(uar=ser['uar'],confusion_matrix=[r.copy() for r in matrix])
    independent['confusion_matrix'][0][1] -= 1
    independent['confusion_matrix'][0][2] += 1
    put(path, {'scores':paired})
    assert not report.current_replay(tmp_path,recorded)


def put(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def arm(out, method):
    train, ser = out/"train"/method, out/"ser"/method
    folder = train/"checkpoint-32"
    folder.mkdir(parents=True)
    (folder/"adapter_model.safetensors").write_bytes(b"weights")
    checksum = hashlib.sha256(b"weights").hexdigest()
    cp = {"method":method, "step":32, "adapter_sha256":checksum, "config_sha256":"b"*64,
          "checkpoint":str(folder), "files":{"adapter_model.safetensors":checksum}}
    scores = {"method":method,"checkpoint":cp,"archived_training":False, "train_dir":str(train), "ser_dir":str(ser),
              "ser":{"utterances":1241,"uar":.5,"accuracy":.5}, "mmau":{"by_domain":{"all":{"total":1000}},
                     "official_text_accuracy":.4,"best_step":32,"trainable":42,"updates":32}}
    put(out/"protocol.json",{"model_revision":"rev"})
    put(out/"configs"/f"{method}-selected.json",cp)
    put(train/"result.json",{})
    for dest in (train/"reload-test",ser):
        put(dest/"metrics.json",{"adapter_sha256":checksum,"predictions_sha256":hashlib.sha256(b"rows").hexdigest()})
        (dest/"predictions.jsonl").write_bytes(b"rows")
    result = out/"results"/f"{method}.json"
    put(result,scores)
    put(out/f"status-{method}.json",{"method":method,"stage":"verified"})
    return result


@pytest.mark.parametrize("problem",["failed","missing","corrupt","wrong_method","newer_result","wrong_selection","changed_weights","changed_predictions"])
def test_stale_result_is_never_displayed_or_skipped(tmp_path, problem):
    method = "stable-lora"
    result = arm(tmp_path,method)
    status = tmp_path/f"status-{method}.json"
    if problem == "failed":
        put(status,{"method":method,"stage":"failed","error":"new checkpoint hash mismatch"})
    elif problem == "missing":
        status.unlink()
    elif problem == "corrupt":
        status.write_text("{")
    elif problem == "wrong_method":
        put(status,{"method":"lora","stage":"verified"})
    elif problem == "newer_result":
        ns = status.stat().st_mtime_ns + 1_000_000
        os.utime(result,ns=(ns,ns))
    elif problem == "wrong_selection":
        put(tmp_path/"configs"/f"{method}-selected.json",{"method":method})
    elif problem == "changed_weights":
        (tmp_path/"train"/method/"checkpoint-32"/"adapter_model.safetensors").write_bytes(b"altered")
    elif problem == "changed_predictions":
        (tmp_path/"ser"/method/"predictions.jsonl").write_bytes(b"other choices")
    report.publish(tmp_path)
    assert method not in report.load(tmp_path/"scores.json")["scores"]
    line = next(s for s in (tmp_path/"REPORT.md").read_text().splitlines() if s.startswith("| 27 |"))
    assert "50.00%" not in line
    assert not finish_queue.completed(tmp_path,method)
    assert result.is_file()


def test_terminal_disposition_keeps_paper_methods_theory_teacher_and_references_separate(tmp_path):
    arm(tmp_path,"stable-lora")
    arm(tmp_path,"lora")
    put(tmp_path/"status-lora.json",{"method":"lora","stage":"failed","error":"latest validation failed"})
    assert finish_queue.completed(tmp_path,"stable-lora")
    result = finish_queue.write_disposition(tmp_path)
    assert result == json.loads((tmp_path/"disposition.json").read_text())
    assert result["status"] == "failed"
    papers = {r["method"]:r for r in result["paper_methods"]}
    assert len(papers) == 27 and papers["stable-lora"]["status"] == "verified"
    assert "lora" not in papers and "known-target-qo" not in papers
    assert result["theory"]["method"] == "expressive-power" and result["theory"]["status"] == "missing"
    assert result["teacher_diagnostic"]["method"] == "known-target-qo"
    refs = {r["method"]:r for r in result["references"]}
    assert refs["lora"]["status"] == "failed" and "latest validation failed" in refs["lora"]["reason"]
    assert refs["frozen-base"]["status"] == "missing"


def test_disposition_complete_requires_all_four_separate_groups_and_current_replay(tmp_path,monkeypatch):
    names = [r["method"] for r in report.load(report.HERE/"catalog.json") if r["id"] > 1]
    for name in [*names,"known-target-qo","lora","lora-recompute"]:
        arm(tmp_path,name)
    monkeypatch.setattr(finish_queue,"theory_evidence",lambda out:{"method":"expressive-power","status":"verified"})
    monkeypatch.setattr(finish_queue,"frozen_evidence",lambda out:({"method":"frozen-base","status":"verified"},{}))
    monkeypatch.setattr(finish_queue,"current_replay",lambda out, records:len(records)==30)
    accepted = finish_queue.write_disposition(tmp_path)
    assert accepted["status"] == "complete"
    assert accepted["counts"]["paper_methods"] == {"expected":27,"verified":27}
    assert finish_queue.write_disposition(tmp_path,["independent replay failed"])["status"] == "failed"


def test_supervisor_retries_a_failed_earlier_result_and_exits_with_terminal_record(tmp_path,monkeypatch):
    arm(tmp_path,"stable-lora")
    put(tmp_path/"status-stable-lora.json",{"method":"stable-lora","stage":"failed"})
    calls = []
    def failed(command,check):
        calls.append(command)
        return SimpleNamespace(returncode=1)
    monkeypatch.setattr(finish_queue.subprocess,"run",failed)
    monkeypatch.setattr(finish_queue.sys,"argv",["finish_queue","--out",str(tmp_path)])
    assert finish_queue.main() == 1
    assert any(command[-1] == "stable-lora" for command in calls)
    assert report.load(tmp_path/"disposition.json")["status"] == "failed"
