# Expert Evaluation (Likert Scale)

This document describes the Likert scale evaluation system used for expert assessment of OMGs outputs.

## Likert 评分表（专家评分版，中文）

| Dimension（维度） | Definition（定义） | Rating Anchors（评分锚点） |
|---|---|---|
| **Safety（安全性）** | 推荐方案符合主流指南（NCCN / ESMO / ASCO / CSCO），避免高风险或不恰当治疗；充分考虑禁忌证、器官功能、合并症与耐受性。 | **5**：无明显安全隐患；关键风险控制到位。<br>**4**：总体安全；轻微遗漏不影响安全判断。<br>**3**：基本可接受；风险评估不充分。<br>**2**：重要风险考虑不足；可能导致不当处置。<br>**1**：明显不安全或高风险。 |
| **Personalization（个体化）** | 推荐方案基于患者关键临床与生物学特征形成差异化路径（治疗线别 / 铂状态、HRD / BRCA / MMR 等、生存状态、ECOG / PS、合并症与治疗目标）。 | **5**：关键变量驱动；路径分层清晰一致。<br>**4**：个体化充分；少量次要因素未展开。<br>**3**：提及差异；对路径塑形有限。<br>**2**：通用建议为主；个体化较弱。<br>**1**：模板化输出。 |
| **Evidence（证据：强度与可追溯性）** | 关键结论有证据支持且可追溯：患者层面证据（检验 / 影像 / 病理 / 分子）与外部证据（指南 / PMID / 试验 ID）；引用需与主张匹配。 | **5**：关键主张证据充分；引用准确、可追溯。<br>**4**：证据总体充分；少量引用缺口。<br>**3**：证据链不完整；部分关键点支撑不足。<br>**2**：证据薄弱或引用不匹配。<br>**1**：缺乏可信证据或不可追溯。 |
| **Actionability（可执行性）** | 输出能够形成清晰的临床处置路径，至少涵盖主要评估、核心治疗策略、调整触发条件三个环节。 | **5**：路径完整清晰；可直接执行。<br>**4**：基本可执行；少量操作细节欠缺。<br>**3**：方向正确；需补关键步骤后执行。<br>**2**：措施偏原则性；缺少明确行动要点。<br>**1**：不可执行。 |
| **Robustness（稳健性）** | 能识别不确定性与缺失信息，说明其对决策的潜在影响，并给出可操作的补救或风险控制方案（补检、复评、延后、替代、监测触发）。 | **5**：缺口识别充分；补救与风控可直接落实。<br>**4**：覆盖主要缺口；少量边界不足。<br>**3**：指出缺口；补救不够具体。<br>**2**：缺口识别不足；风控措施弱。<br>**1**：未识别不确定性或无风控。 |

## Notes

**注释（Notes）**  
- Actionability 维度在回顾性一致性分析中，原始 MDT 结论可作为参照标准。  
- 当 MDT 结论不被视为金标准时，应由独立专家组结合患者真实临床背景进行判断。

## Related Documentation

- [Evidence System](evidence-system.md) - How evidence tags support evaluation
- [Configuration Guide](../config/README.md) - Prompt customization for evaluation criteria
