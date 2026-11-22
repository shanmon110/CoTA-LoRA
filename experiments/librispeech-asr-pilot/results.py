"""Render verified pilot metrics recorded by the runner."""
import json
from pathlib import Path
import sys


def render(run):
    status = json.loads((run / "status.json").read_text())
    lines = ["# LibriSpeech dev-clean 单种子结果", "",
             f"运行状态 `{status['state']}`。仅已完成整份开发集评估的方法进入结果表。", "",
             "| 方法 | 可训练参数 | WER | 错词 / 参考词 | 无 EOS 撞限 / 录音数 | 训练峰值 allocated / reserved (GiB) |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, arm in status["results"].items():
        if "evaluation" not in arm:
            continue
        score = arm["evaluation"]
        peak = arm["train_peaks"]["train"]
        lines.append(f"| {name} | {arm['train']['trainable_parameters']:,} | {score['wer']:.2%} | "
                     f"{score['word_errors']:,} / {score['reference_words']:,} | "
                     f"{score['max_new_tokens_without_eos_count']} / {score['utterances']:,} | "
                     f"{peak['peak_allocated_bytes']/2**30:.2f} / {peak['peak_reserved_bytes']/2**30:.2f} |")
    lines.extend(["", f"[原始结果与分阶段耗时]({run}/status.json)"])
    if status["state"] == "failed":
        lines.extend(["", f"中止于 {status['method']} 的 {status['phase']} 阶段。详见该方法的 stderr。"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    print(render(Path(sys.argv[1]).resolve()), end="")
