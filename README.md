# SQA3D → Qwen 数据转换脚本（Qwen2.5 / Qwen3）

本仓库提供 **SQA3D（ScanNet-QA 3D）数据集** 到 **Qwen2.5 / Qwen3 多模态训练格式** 的通用转换脚本，用于将官方的 `balanced` JSON 标注文件转换为可直接用于 Qwen 系列模型训练的数据格式。

---

## 1. 项目背景与动机

SQA3D 是一个基于 **ScanNet 场景的 3D 问答数据集**，其原始标注包含：

- 自然语言问题（question）
- 多选或分类式答案（answer id → 文本）
- Agent 在 3D 场景中的 **绝对位姿**
  - `agent_position: [x, y, z]`
  - `agent_rotation: [qw, qx, qy, qz]`
- 3D 目标指代、空间关系等信息

而 **Qwen 系列模型** 对输入格式的要求不同：

| 模型 | 是否原生支持 3D 空间 | 坐标形式 |
|----|----|----|
| Qwen2.5 | ❌ 不支持 3D | ❌ 不使用坐标 |
| Qwen3 | ✅ 支持 3D / BEV | ✅ 相对坐标 |

因此，**需要针对不同模型做不同的数据转换策略**。

---

## 2. 数据目录结构要求

请确保你本地的 `balanced/` 目录结构如下（官方 SQA3D 提供）：

balanced/
├── v1_balanced_questions_train_scannetv2.json
├── v1_balanced_questions_val_scannetv2.json
├── v1_balanced_questions_test_scannetv2.json
│
├── v1_balanced_sqa_annotations_train_scannetv2.json
├── v1_balanced_sqa_annotations_val_scannetv2.json
├── v1_balanced_sqa_annotations_test_scannetv2.json
│
└── answer_dict.json


### 关键说明

- `questions_*.json`
  - 包含 `question_id` 与问题文本
- `annotations_*.json`
  - 通过 `question_id` 关联答案 ID、3D 位姿、空间信息
- `answer_dict.json`
  - 答案编号 → 答案文本映射（**必须存在，但无需单独转换**）

---

## 3. Qwen2.5 转换说明（不包含 3D）

### 3.1 设计原则

- Qwen2.5 **不支持显式 3D / BEV 建模**
- 转换时：
  - ❌ 忽略 agent 位姿
  - ❌ 忽略 3D 坐标、空间点、bbox
  - ✅ 仅保留语言层面的 Situated QA

### 3.2 输出适用场景

- 纯语言 / 视觉 + 语言 QA
- 可结合视频帧、多视角图像
- ❌ 不包含真实 3D 空间监督

---

## 4. Qwen2.5 转换逻辑（无 3D）

### 4.1 转换策略

- `questions_*.json` 与 `annotations_*.json` 通过 `question_id` 一一匹配
- 每条样本最终仅保留：
  - 问题文本
  - 答案文本（由 `answer_dict.json` 映射）

示意逻辑：

```python
answer_text = answer_dict[annotation["answers"][0]["answer"]]
```
## 4.2 输出格式示例（Qwen2.5）

Qwen2.5 的输出不包含任何 3D 或空间坐标信息，仅保留标准对话式监督格式：

```json
{
  "messages": [
    {
      "role": "user",
      "content": "What is in front of the chair?"
    },
    {
      "role": "assistant",
      "content": "table"
    }
  ]
}
```
## 5. Qwen2.5 转换脚本使用方法
使用以下命令将 SQA3D 数据转换为适用于 Qwen2.5 的格式：
```bash
python convert_sqa3d_to_qwen25.py \
  --base_dir /path/to/SQA3D/sqa3d/sqa_task/balanced \
  --split train
```
支持的 split
```
train
```
```
val
```
```
test
```
输出文件说明

根据不同的 split，生成以下文件之一：
```
sqa3d_qwen25_train.json
```
```
sqa3d_qwen25_val.json
```
```
sqa3d_qwen25_test.json
```
## 6. Qwen3 转换说明（支持 3D / 相对坐标）
### 6.1 设计原则
- Qwen3 原生支持 3D / BEV 空间感知
- 转换过程中：
  -  保留 agent 位姿信息
  -  引入显式空间结构
  -  使用 相对坐标（agent-centric） 表达空间关系
- 目标是最大化保留 SQA3D 的空间推理信息，同时避免全局坐标带来的尺度与泛化问题
### 6.2 坐标处理方式
以 agent 所在位置作为坐标原点：
```text
relative_position = object_position - agent_position
```
转换后：
- agent 的位置固定为 [0, 0, 0]
- 所有目标物体、空间点均使用相对 3D 坐标
- 不直接使用 ScanNet 的全局世界坐标系
## 7. Qwen3 输出格式示例
Qwen3 的输入在 user 侧包含结构化的 3D 信息：
```json
{
  "messages": [
    {
      "role": "user",
      "content": {
        "text": "What is to the left of the sofa?",
        "agent_position": [0.0, 0.0, 0.0],
        "relative_objects": [
          {
            "name": "table",
            "position": [1.2, -0.3, 0.0]
          }
        ]
      }
    },
    {
      "role": "assistant",
      "content": "table"
    }
  ]
}
```
该结构可直接扩展为：
- 多目标
- BEV token
- 3D bbox
- 多帧 / 多视角空间输入
## 8. Qwen3 转换脚本使用方法
使用以下命令进行 Qwen3 格式转换：
```bash
python convert_sqa3d_to_qwen3.py \
  --base_dir /path/to/SQA3D/sqa3d/sqa_task/balanced \
  --split train
```
输出文件说明
- ```sqa3d_qwen3_train.json```

- ```sqa3d_qwen3_val.json```

- ```sqa3d_qwen3_test.json```
## 9. Qwen2.5 与 Qwen3 转换对比总结
| 项目           | Qwen2.5    | Qwen3      |
| ------------ | ---------- | ---------- |
| 是否使用 3D 信息   | ❌          | ✅          |
| 是否使用坐标       | ❌          | ✅（相对坐标）    |
| 空间推理能力       | 无          | 强          |
| 适合任务         | 语言 / 视觉 QA | 真正 3D 空间问答 |
| 是否推荐用于 SQA3D | ⚠️ 仅作基线    | ✅ 强烈推荐     |
## 10. 备注
本仓库不会修改原始 SQA3D 数据

所有转换结果均为 可逆的中间表示

可直接接入：
- LLaMA-Factory
- Qwen 官方训练脚本
- 如需进一步扩展：
  - BEV token
  - 3D bounding box
- 多帧时序感知
可在wen3 转换脚本基础上继续演进
