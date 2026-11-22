"""Render the measured stages of the same-backbone teacher, expert and federated protocols."""
import json
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def stages(out):
    lines = ['# 特殊训练协议与额外开销','',
             '固定 Qwen2.5-Omni-3B 和同一批 AVQA 源数据。教师只用源训练和开发集；四个专家从源训练集取样；联邦客户端按源录音分片。全部方法沿用同一套 IEMOCAP 与 MMAU 测试流程。','',
             '| 方法 / 阶段 | 更新数与单位 | 实测秒数 | GPU 峰值 allocated / reserved（GiB） |',
             '|---|---|---:|---:|']

    def add(label, updates, sec=None, allocated=None, reserved=None):
        memory = '—' if allocated is None or reserved is None else f'{allocated/2**30:.2f} / {reserved/2**30:.2f}'
        seconds = '待实测' if sec is None else f'{sec:.2f}'
        lines.append(f'| {label} | {updates} | {seconds} | {memory} |')

    construction = out/'theory-01/omni-control/construction.json'
    if construction.is_file():
        c = read(construction)
        teacher = read(Path(c['teacher_selection_path']).parent/'result.json')
        add('已知目标 Q/O · 同源全参教师',f"{teacher['global_step']} 次，开发集选第 {c['teacher_step']} 步",
            teacher['training_seconds'],teacher['peak_allocated_bytes'],teacher['peak_reserved_bytes'])
        add('已知目标 Q/O · 教师选模','AVQA 开发验证',teacher['dev_seconds'])
        add('已知目标 Q/O · CPU 构造',f"{c['optimizer_steps']} 次优化，{c['num_projections']} 个投影",c['construction_seconds'])
    else:
        add('已知目标 Q/O · 同源教师及构造','等待构造记录')

    mixer = out/'train/lora-mixer'
    init = read(mixer/'initialization.json') if (mixer/'initialization.json').is_file() else {}
    if init:
        add('LoRA-Mixer / LoRAGen · 共用专家库',f"{init['warmup_steps']} 次专家更新，四域各 {init['source_examples_per_expert']} 条",
            init['stage_seconds'],init.get('stage_peak_allocated'),init.get('stage_peak_reserved'))
    else:
        add('LoRA-Mixer / LoRAGen · 共用专家库','等待专家训练记录')
    if (mixer/'result.json').is_file():
        m = read(mixer/'result.json')
        add('LoRA-Mixer · 联合训练',f"{m['global_step']} 次，开发集选第 {m['best_step']} 步",
            m['training_seconds'],m['peak_allocated_bytes'],m['peak_reserved_bytes'])
        add('LoRA-Mixer · 选模','AVQA 开发验证',m['dev_seconds'])
    else:
        add('LoRA-Mixer · 联合训练','等待训练记录')

    generator = out/'train/loragen/generation.json'
    if generator.is_file():
        g = read(generator)
        for key,label in [('lae','adapter 自编码器'),('diffusion','扩散去噪器')]:
            add('LoRAGen · '+label,f"{g[key+'_updates']} 次生成器更新",g[key+'_seconds'],
                g.get(key+'_peak_allocated_bytes'),g.get(key+'_peak_reserved_bytes'))
        add('LoRAGen · 已有专家库后的总开销','含上两行、描述嵌入和生成',g['generator_seconds_including_embedding'])
    else:
        add('LoRAGen · 自编码器、扩散及生成','等待生成器实测')

    federation = out/'train/rolora-federated'
    if (federation/'result.json').is_file():
        result = read(federation/'result.json')
        partition = read(federation/'client-partitions.json')
        updates = result['global_step']
        per_round = len(partition['clients']) * partition['local_steps']
        rounds, partial = divmod(updates, per_round)
        if partial:
            raise ValueError('Final federated model has an incomplete aggregation round')
        add('RoLoRA · 含本地更新与聚合',f'{updates} 次本地更新，{rounds} 轮聚合',result['training_seconds'],
            result['peak_allocated_bytes'],result['peak_reserved_bytes'])
    else:
        add('RoLoRA · 单卡串行模拟','4 个客户端，每个客户端 4 次本地更新合为一轮')
    lines += ['', '训练秒数与开发集评估分别计时。复用已训练教师或专家库时，可分别读取先决阶段和本次新增阶段的开销。计算整个实验的总资源时，共用专家训练只计一次。LoRAGen 的生成器更新与 3B 主模型更新分别计数。RoLoRA 的计时包含进程内聚合，网络通信开销未测。','',
              '第 1 篇的逐层构造和 LoRAGen 在主表中的参数量为导出的 adapter 大小。教师、专家和生成器各自的参数量见原始记录。','',
              '[主表与测试成绩](REPORT.md) · [已知教师构造](theory-01/omni-control/construction.json) · [专家准备](train/lora-mixer/initialization.json) · [阶段日志](logs)']
    return '\n'.join(lines)+'\n'
