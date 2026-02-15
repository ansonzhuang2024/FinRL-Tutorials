# CLAUDE.md

## Repository Overview

This is the **FinRL-Tutorials** repository by AI4Finance Foundation — a collection of Jupyter notebook tutorials for financial reinforcement learning. It covers stock trading, portfolio allocation, crypto trading, paper trading, and hyperparameter optimization using DRL (Deep Reinforcement Learning).

**License:** MIT

## Repository Structure

```
FinRL-Tutorials/
├── 1-Introduction/       # Beginner tutorials (Stock trading, Portfolio allocation, Fundamentals)
├── 2-Advance/            # Intermediate (Explainable DRL, Library comparison, Ensemble strategies)
├── 3-Practical/          # Paper trading, Multi-crypto, China A-Share market
├── 4-Optimization/       # Hyperparameter tuning (Optuna, Ray Tune, W&B)
├── 5-Others/             # Docker demos, environment variants
├── DQN-DDPG_Stock_Trading/  # Legacy DQN/DDPG examples
├── Contributing.md       # Contribution guidelines
├── LICENSE               # MIT License
└── README.md
```

## Development Conventions

- **Pre-commit hooks**: `pip install pre-commit && pre-commit install`
- **Testing**: `python3 -m unittest discover`
- **Code style**: PEP format with inline documentation
- **PR process**: Tag maintainers, reference issues, include tests and documentation

---

## 项目任务指令 (Project Task Instructions)

### 1. 角色设定 (Role)

你现在是一名资深的 Hedge Fund Quant Developer，擅长使用 Python 进行跨市场相关性分析和高频数据处理。你的目标是编写一个生产级别的监控脚本，用于追踪中海油（00883.HK）与其核心定价因子之间的偏离。

### 2. 核心逻辑 (Core Logic)

中海油（00883.HK）的股价受以下变量实时驱动，请建立监控逻辑：

- **锚定资产：** 布伦特原油期货 (Yahoo Finance Ticker: `BZ=F`)
- **汇率对：** 离岸人民币 (Yahoo Finance Ticker: `CNH=X`)
- **计算公式：** 实时计算 $\Delta \text{Stock} / \Delta \text{Oil}$ 的 20 日动态 Beta，并监控当前价格是否偏离预测区间 $2\sigma$。

### 3. 技术栈要求 (Tech Stack)

- **库：** `yfinance` (获取数据), `pandas` (处理时序), `statsmodels` (回归分析), `plotly` (交互式可视化)
- **数据频率：** `1m` (分钟级) 实时流式模拟
- **输出控制：** 仅显示结构化 DataFrame 和信号触发日志

### 4. 具体的函数需求 (Function Requirements)

- **`fetch_realtime_data()`**: 同时抓取 `00883.HK`, `BZ=F`, `CNH=X` 的最近 5 天分钟线。
- **`calculate_correlation()`**:
  - 对齐三个时间序列（处理港股与国际原油交易时段的差异）。
  - 计算 Pearson 相关系数。
- **`signal_generator()`**:
  - 当油价上涨 > 1.5% 且 00883.HK 涨幅滞后 < 0.5% 时，标记为 `"Long Signal (Lagging)"`。
  - 当 CNH 剧烈波动时，给出汇率风险预警。

### 5. 严格约束 (Constraints)

- **处理时差：** 必须考虑布伦特原油（24小时交易）与港股（09:30-16:00 HKT）的重合时段。
- **异常处理：** 若 Yahoo Finance 返回空值，必须具备 Retry 逻辑，不得中断主程序。
- **无废话：** 直接给出可执行代码，不要解释基础语法。
