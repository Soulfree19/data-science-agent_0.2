# Data Science AI Agent

A powerful Chinese-language AI agent for data analysis, machine learning, and scientific visualization. Built with DeepSeek + LangChain + LangGraph.

> 一个强大的中文 AI 数据分析 Agent，支持从数据探索到深度学习、从时间序列预测到科研级可视化的全流程自动化。

## Features | 功能

### Core | 核心能力
- **Python REPL**: Persistent workspace with 55+ pre-imported libraries (pandas, numpy, sklearn, torch, xgboost, statsmodels...)
- **File Analysis**: Read CSV / Excel / JSON / Parquet files, auto-sampling for large files (>100MB)
- **Conversation Memory**: Multi-turn dialog with automatic context retention via LangGraph InMemorySaver
- **Streaming Output**: Real-time token-by-token response display with Rich UI

### Data Science | 数据分析
- **Time Series**: ADF/KPSS stationarity tests, ACF/PACF analysis, ARIMA/SARIMAX modeling, residual diagnostics, forecasting with confidence intervals
- **Machine Learning**: Linear/Ridge/Lasso, RandomForest, GradientBoosting, XGBoost, LightGBM with GridSearchCV and cross-validation
- **Deep Learning**: PyTorch-based neural networks (nn.Module, DataLoader, training loops)
- **Model Explainability**: SHAP feature importance analysis
- **Scientific Visualization**: Publication-ready red-blue theme, Chinese font support, auto image display in terminal

### Research | 学术研究
- **Paper Search**: arXiv + Semantic Scholar dual-source academic paper lookup with real citations
- **Web Search**: DuckDuckGo integration for real-time information

## Quick Start | 快速开始

### 1. Install | 安装

```bash
git clone https://github.com/Soulfree19/data-science-agent_0.2.git
cd data-science-agent
pip install -r requirements.txt
```

### 2. Configure | 配置

```bash
cp .env.example .env
# Edit .env and add your DeepSeek API key
# 编辑 .env，填入你的 DeepSeek API Key
```

Get a free API key at [platform.deepseek.com](https://platform.deepseek.com).

### 3. Run | 运行

```bash
python main.py
```

### 4. Usage | 使用

```
📊 请输入分析需求: 读取 examples/sales.csv，分析销售趋势

📊 请输入分析需求: 做时间序列预测，ADF检验，拟合ARIMA模型

📊 请输入分析需求: 用RandomForest和XGBoost预测，SHAP分析特征重要性

📊 请输入分析需求: 搜索transformer时间序列预测的最新论文
```

- Type `q` to quit | 输入 `q` 退出
- Type `/clear` to reset conversation memory | 输入 `/clear` 清空对话记忆

## Architecture | 架构

```
data-science-agent/
├── main.py              # Entry point (入口)
├── requirements.txt     # Dependencies (依赖)
├── .env.example         # Config template (配置模板)
├── examples/            # Sample data (示例数据)
│   ├── sales.csv        #   Sales dataset
│   └── timeseries.csv   #   Time series dataset
└── src/
    ├── llm.py           # LLM connection factory (DeepSeek via OpenAI-compatible API)
    ├── tools.py         # 7 tools: REPL, file reader, search, export, image, papers
    ├── agent.py         # Agent builder with memory + system prompt
    └── cli.py           # Rich streaming CLI with image display
```

## Tools | 工具

| Tool | Description |
|------|-------------|
| `python_repl` | Persistent Python REPL with 55+ pre-imported libraries |
| `read_file` | Read CSV/Excel/JSON/Parquet, auto-sample large files |
| `list_files` | Browse directories |
| `show_image` | Display generated plots in terminal (iTerm2 protocol) |
| `export_data` | Export DataFrames to CSV/Excel/JSON/Parquet |
| `search_papers` | Search academic papers on arXiv + Semantic Scholar |
| `search` | DuckDuckGo web search |

## Requirements | 环境要求

- Python >= 3.10
- DeepSeek API key ([free signup](https://platform.deepseek.com))
- macOS / Linux / Windows

## Example Demos | 示例演示

### Time Series Forecasting
```
📊 读取 examples/timeseries.csv，做完整的ARIMA分析，预测未来4期
```
The agent will: load data → plot time series → ADF test → ACF/PACF → fit ARIMA → residual diagnostics → forecast → export results.

### ML Model Comparison + SHAP
```
📊 读取 examples/sales.csv，用RF和XGBoost预测sales，SHAP分析，画科研配色对比图
```
The agent will: encode features → train both models → compare MSE/R² → SHAP analysis → scientific red-blue comparison plots.

## License

MIT License - see [LICENSE](LICENSE)
