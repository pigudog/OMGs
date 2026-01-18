# OMGs - MDT Multi-Expert AI System

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**OMGs** (Oncology Multi-Expert Guidance System) 是一个基于多智能体协作的医疗决策支持系统，专门用于肿瘤科MDT（多学科团队）讨论场景。系统通过模拟多个医疗专家的角色（主席、肿瘤内科、影像科、病理科、核医学科），进行多轮讨论并生成综合性的临床决策建议。

## ✨ 主要特性

- **🤖 多专家智能体系统**：模拟5个医疗专家角色（Chair、Oncologist、Radiologist、Pathologist、Nuclear Medicine）
- **📊 智能报告筛选**：基于角色权限和临床相关性自动筛选实验室、影像、病理报告
- **🔍 RAG检索增强**：集成ChromaDB向量数据库，检索相关临床指南和文献
- **💬 多轮讨论引擎**：支持专家之间的多轮交互式讨论，解决冲突、补充信息
- **🧪 临床试验匹配**：自动匹配相关的临床试验方案
- **📝 完整可观测性**：生成详细的讨论日志、HTML报告和交互矩阵
- **🔐 角色权限控制**：每个专家只能访问其专业相关的报告类型

## 🏗️ 系统架构

```
输入病例数据
    ↓
[1] 加载患者报告 (实验室/影像/病理/突变)
    ↓
[2] 按角色筛选相关报告 (权限控制)
    ↓
[3] RAG检索全局指南 (ChromaDB + Embeddings)
    ↓
[4] 初始化专家智能体 (5个角色)
    ↓
[5] MDT多轮讨论引擎 (2轮 × 2回合)
    ↓
[6] 临床试验匹配 (可选)
    ↓
[7] 生成最终MDT决策输出
    ↓
保存结果 (JSON + TXT + HTML报告)
```

### 专家角色与权限

| 角色 | 实验室报告 | 影像报告 | 病理报告 | 指南类型 |
|------|-----------|---------|---------|---------|
| Chair | ✅ | ✅ | ❌ | chair |
| Oncologist | ✅ | ❌ | ❌ | oncologist |
| Radiologist | ❌ | ✅ | ❌ | radiologist |
| Pathologist | ❌ | ❌ | ✅ | pathologist |
| Nuclear Medicine | ❌ | ✅ | ❌ | nuclear |

## 📋 系统要求

- **Python**: 3.8+
- **操作系统**: Linux, macOS, Windows
- **Azure OpenAI**: 需要有效的Azure OpenAI服务账号
- **GPU** (可选): 用于本地embedding模型加速

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆项目（如果是git仓库）
git clone <repository-url>
cd OMGs

# 安装Python依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

设置Azure OpenAI的访问凭证：

```bash
export AZURE_OPENAI_ENDPOINT="https://your-resource.openai.azure.com/"
export AZURE_OPENAI_API_KEY="your-api-key-here"
```

或在Windows上：

```cmd
set AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
set AZURE_OPENAI_API_KEY=your-api-key-here
```

### 3. 准备数据文件

确保以下数据文件存在：

- **输入病例文件** (`input_ehr/*.jsonl`): JSONL格式的病例数据
- **实验室报告** (`files/lab_reports_summary.jsonl`): 实验室检查报告
- **影像报告** (`files/imaging_reports.jsonl`): 影像检查报告
- **突变报告** (`files/mutation_reports.jsonl`): 基因突变检测报告
- **病理报告** (可选): `files/pathology_reports.jsonl`
- **RAG索引** (`rag_store/chair/index/chroma/`): 临床指南向量索引
- **临床试验库** (可选): `all_trials_filtered.json`

### 4. 运行系统

```bash
python main.py \
    --input_path input_ehr/test_guo.jsonl \
    --model gpt-5.1 \
    --agent omgs \
    --num_samples 10
```

## 📖 使用方法

### 命令行参数

```bash
python main.py [OPTIONS]
```

**必需参数：**
- `--input_path`: 输入JSONL文件路径（病例数据）

**可选参数：**
- `--model`: Azure部署模型名称（默认: `gpt-5.1`）
- `--agent`: 使用的agent类型（默认: `omgs`，当前仅支持omgs）
- `--num_samples`: 处理的样本数量（默认: 999999，即处理所有样本）

### 输入数据格式

输入JSONL文件每行应包含以下字段：

```json
{
  "meta_info": "患者标识符（用于匹配报告）",
  "Time": "2024-01-15",
  "question": {
    "CASE_CORE": {
      "DIAGNOSIS": "诊断信息",
      "LINE_OF_THERAPY": "治疗线数",
      "BIOMARKERS": {},
      "CURRENT_STATUS": "当前状态"
    },
    "TIMELINE": {},
    "MED_ONC": {},
    "RADIOLOGY": {},
    "PATHOLOGY": {},
    "LAB_TRENDS": {}
  },
  "question_raw": "原始问题文本",
  "scene": "场景标识",
  "gold_plan": "标准答案（可选）"
}
```

### 输出结果

系统会在 `output_answer/{agent}_{timestamp}/` 目录下生成：

1. **results.json**: 结构化的结果数据（包含问题、回答、元信息）
2. **results.txt**: 人类可读的文本格式结果

同时在 `mdt_logs/` 目录下生成：

1. **mdt_history_{timestamp}.jsonl**: 完整的MDT讨论历史（JSONL格式）
2. **mdt_history_{timestamp}.md**: Markdown格式的讨论记录
3. **mdt_report_{timestamp}.html**: 交互式HTML报告（包含讨论矩阵、报告选择表格等）

## 📁 项目结构

```
OMGs/
├── main.py                 # 主入口脚本
├── agent_published.py      # MDT处理流程核心逻辑
├── requirements.txt        # Python依赖列表
├── README.md              # 本文件
│
├── aoai/                  # Azure OpenAI包装器
│   ├── wrapper.py
│   └── logger.py
│
├── utils/                 # 工具函数模块
│   ├── core.py           # Agent类和基础函数
│   ├── role_utils.py     # 角色管理和专家初始化
│   ├── rag_utils.py      # RAG检索相关
│   ├── select_utils.py   # 报告筛选和加载
│   ├── omgs_reports.py   # 报告生成（HTML/Markdown）
│   ├── console_utils.py  # 控制台输出和格式化
│   ├── time_utils.py     # 时间处理和timeline构建
│   └── trace_utils.py    # 可观测性和追踪
│
├── config/               # 配置文件
│   └── prompts.json      # Prompt模板
│
├── files/                # 数据文件目录
│   ├── lab_reports_summary.jsonl
│   ├── imaging_reports.jsonl
│   └── mutation_reports.jsonl
│
├── input_ehr/           # 输入病例数据
│   ├── test_guo.jsonl
│   └── ...
│
├── output_answer/       # 输出结果目录
│   └── omgs_YYYY-MM-DD_HH-MM-SS/
│       ├── results.json
│       └── results.txt
│
├── mdt_logs/            # MDT讨论日志
│   ├── mdt_history_*.jsonl
│   ├── mdt_history_*.md
│   └── mdt_report_*.html
│
└── rag_store/           # RAG向量索引
    └── chair/
        └── index/
            └── chroma/
```

## 🔧 核心依赖

主要依赖包（详见 `requirements.txt`）：

- **openai** (≥1.0.0): Azure OpenAI客户端
- **chromadb** (≥0.4.0): 向量数据库
- **langchain-huggingface** (≥0.0.1): Embedding模型集成
- **torch** (≥2.0.0): 深度学习框架
- **tiktoken** (≥0.5.0): Token计数
- **tqdm** (≥4.65.0): 进度条
- **prettytable** (≥3.8.0): 表格格式化

## 💡 使用示例

### 基础使用

```bash
# 处理单个测试文件
python main.py --input_path input_ehr/test_guo.jsonl --num_samples 5
```

### 使用不同的模型

```bash
python main.py \
    --input_path input_ehr/test_guo.jsonl \
    --model gpt-4 \
    --num_samples 10
```

### 批量处理

```bash
# 处理所有输入文件中的所有样本
for file in input_ehr/*.jsonl; do
    python main.py --input_path "$file"
done
```

## 🔍 工作原理详解

### 1. 报告加载与筛选

系统根据患者标识符（`meta_info`）从JSONL文件中加载：
- 实验室报告（CBC、LFT、肾功能、肿瘤标志物等）
- 影像报告（CT、MRI、PET等）
- 病理报告（组织学、IHC、分子检测等）
- 基因突变报告

然后基于时间戳过滤（只保留就诊日期之前的报告），并针对每个专家角色筛选最相关的报告。

### 2. 角色特定视图构建

每个专家接收：
- **角色特定的病例视图**：只包含该专业相关的字段
- **筛选后的报告**：基于角色权限和临床相关性
- **全局指南摘要**：从RAG检索得到的相关指南

### 3. MDT讨论流程

1. **初始意见**：每个专家基于其信息给出初始意见
2. **多轮讨论**：
   - Round 1, Turn 1: 专家可以相互提问/回应
   - Round 1, Turn 2: 继续讨论
   - Round 2: 在更新的上下文中进行第二轮讨论
3. **最终计划**：每轮结束后，每个专家提供最终的细化计划

### 4. 决策输出

Chair专家综合所有讨论内容，生成最终的MDT决策输出，包括：
- 最终评估
- 核心治疗策略
- 变更触发条件
- 临床试验建议（如适用）

## 📊 输出示例

### JSON输出结构

```json
{
  "scene": "场景标识",
  "question": "处理后的问题",
  "response": "最终MDT决策输出",
  "gold_plan": "标准答案（如果有）",
  "question_raw": "原始问题",
  "Time": "2024-01-15",
  "meta_info": "患者ID"
}
```

### HTML报告特性

生成的HTML报告包含：
- 交互式讨论矩阵（显示专家之间的交互次数）
- 报告选择表格（显示每个专家选择的报告）
- RAG检索结果表格
- 完整的讨论历史时间线
- 最终输出和临床试验建议

## ⚙️ 高级配置

### 自定义报告路径

在 `agent_published.py` 的 `process_omgs_multi_expert_query` 函数中可以自定义：

```python
process_omgs_multi_expert_query(
    question=question,
    question_raw=question_raw,
    model=model,
    args=args,
    labs_json="custom/labs.jsonl",      # 自定义实验室报告路径
    imaging_json="custom/imaging.jsonl", # 自定义影像报告路径
    pathology_json="custom/pathology.jsonl", # 自定义病理报告路径
    mutation_json="custom/mutations.jsonl",  # 自定义突变报告路径
    trials_json_path="custom/trials.json"    # 自定义临床试验库路径
)
```

### RAG配置

- **索引路径**: `rag_store/chair/index/chroma/`
- **Embedding模型**: `BAAI/bge-m3` (在 `rag_utils.py` 中配置)
- **检索数量**: 默认 `topk=5` (可调整)

## 🐛 故障排除

### 常见问题

1. **环境变量未设置**
   ```
   RuntimeError: Missing AZURE_OPENAI_ENDPOINT or AZURE_OPENAI_API_KEY
   ```
   **解决**: 确保设置了 `AZURE_OPENAI_ENDPOINT` 和 `AZURE_OPENAI_API_KEY` 环境变量

2. **文件未找到**
   ```
   FileNotFoundError: files/lab_reports_summary.jsonl
   ```
   **解决**: 检查数据文件路径是否正确，确保文件存在

3. **ChromaDB索引不存在**
   ```
   无法加载RAG索引
   ```
   **解决**: 确保 `rag_store/chair/index/chroma/` 目录存在且包含有效的索引文件

4. **模型部署名称错误**
   ```
   Azure API错误
   ```
   **解决**: 检查 `--model` 参数是否与Azure中的部署名称一致

## 📝 开发说明

### 扩展新的专家角色

1. 在 `utils/role_utils.py` 中添加新角色到 `ROLES` 列表
2. 在 `ROLE_PERMISSIONS` 中定义权限
3. 在 `ROLE_PROMPTS` 中添加角色提示词
4. 在 `build_role_specific_case_view` 中添加角色视图构建逻辑

### 添加新的报告类型

1. 在 `utils/select_utils.py` 中添加加载函数
2. 更新 `ROLE_PERMISSIONS` 以包含新报告类型
3. 在 `expert_select_reports` 中添加选择规则

## 📄 许可证

本项目采用 MIT 许可证。详见 LICENSE 文件。

## 🙏 致谢

感谢所有为OMGs系统开发做出贡献的团队成员。

## 📧 联系方式

如有问题或建议，请通过以下方式联系：
- 提交 Issue
- 发送邮件至项目维护者

---

**⚠️ 医疗免责声明**: 本系统仅用于研究和教育目的，生成的建议不应替代专业医疗诊断和治疗。任何医疗决策都应在有资质的医疗专业人员指导下进行。
