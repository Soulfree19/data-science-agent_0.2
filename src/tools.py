import sys
import os
from io import StringIO
from langchain.tools import tool

# Persistent workspace for Python REPL — variables survive across calls
_workspace: dict = {}

_WARMUP_CODE = """
import pandas as pd
import numpy as np
import json

# ---- Scientific Plotting Engine ----
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# Scientific red-blue theme, publication-ready
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['PingFang SC', 'Heiti SC', 'Arial Unicode MS', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.linewidth': 1.0,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'axes.grid.axis': 'y',
    'grid.alpha': 0.3,
    'grid.linestyle': '--',
    'figure.dpi': 150,
    'figure.figsize': (8, 5),
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'legend.frameon': False,
    'lines.linewidth': 1.8,
    'lines.markersize': 6,
})
# Scientific red-blue color palette
SC_RED = '#C0392B'
SC_BLUE = '#2980B9'
SC_DARK_RED = '#7B241C'
SC_DARK_BLUE = '#1A5276'
SC_PALETTE = [SC_BLUE, SC_RED, '#27AE60', '#E67E22', '#8E44AD', '#16A085']
sns.set_palette(SC_PALETTE)
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=SC_PALETTE)

# ---- Statistical / TS ----
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import stats
from scipy.cluster import hierarchy
from scipy.interpolate import interp1d

# ---- Classic ML ----
from sklearn.model_selection import (
    train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import (
    RandomForestRegressor, RandomForestClassifier,
    GradientBoostingRegressor, GradientBoostingClassifier,
)
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Optional: XGBoost, LightGBM (may fail if system libs missing)
try:
    import xgboost as xgb
except Exception:
    xgb = None
try:
    import lightgbm as lgb
except Exception:
    lgb = None

# ---- Deep Learning ----
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except Exception:
    torch = nn = optim = DataLoader = TensorDataset = None

# ---- Model Explainability ----
try:
    import shap
except Exception:
    shap = None

# ---- Helper functions ----
def _sci_plot_setup(title=None, xlabel=None, ylabel=None):
    \"\"\"Quick setup for a clean scientific plot. Returns fig, ax.\"\"\"
    fig, ax = plt.subplots()
    if title: ax.set_title(title, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel)
    if ylabel: ax.set_ylabel(ylabel)
    return fig, ax

def _sci_save(fig, filename):
    \"\"\"Save figure with scientific defaults.\"\"\"
    fig.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'saved: {filename}')
"""
exec(_WARMUP_CODE, _workspace)


# ============================================================
# Tools
# ============================================================

@tool
def python_repl(code: str) -> str:
    """Execute Python code in a persistent workspace. Variables persist across calls.

    Use for ALL data analysis: EDA, stats, ML, DL, time series, visualization.

    The workspace is pre-configured with a SCIENTIFIC RED-BLUE plotting theme.
    Colors: SC_RED='#C0392B', SC_BLUE='#2980B9', SC_PALETTE (6 colors).

    Pre-imported (no need to import again):
    Data: pd, np, json
    Plot: plt, sns, _sci_plot_setup(), _sci_save(), SC_RED, SC_BLUE, SC_PALETTE
    Stats: sm, adfuller, kpss, seasonal_decompose, ARIMA, SARIMAX, plot_acf, plot_pacf, stats
    Classic ML: train_test_split, TimeSeriesSplit, GridSearchCV, cross_val_score,
      StandardScaler, MinMaxScaler, LabelEncoder, PCA, SelectKBest,
      LinearRegression, LogisticRegression, Ridge, Lasso,
      RandomForestRegressor, RandomForestClassifier,
      GradientBoostingRegressor, GradientBoostingClassifier,
      xgb (xgboost), lgb (lightgbm),
      mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
    DL: torch, nn, optim, DataLoader, TensorDataset
    Explain: shap

    Plotting workflow (scientific standard):
    1. fig, ax = _sci_plot_setup('Title', 'X Label', 'Y Label')
    2. Plot data with color=SC_RED or color=SC_BLUE
    3. _sci_save(fig, 'output.png')
    4. Call show_image('output.png') to display

    For multi-panel: use plt.subplots(nrows, ncols) and iterate over axes.
    """
    global _workspace

    old_stdout = sys.stdout
    sys.stdout = StringIO()

    try:
        exec(code, _workspace)
        output = sys.stdout.getvalue()
        return output or "(执行成功，无输出)"
    except Exception as e:
        return f"执行错误: {type(e).__name__}: {e}"
    finally:
        sys.stdout = old_stdout


@tool
def search_papers(query: str, max_results: int = 5) -> str:
    """Search for real academic papers on arXiv + Semantic Scholar. Returns title, authors, year, abstract, and URL.

    Use this when the user asks for references, related work, or wants to find papers
    on a specific topic (deep learning, time series forecasting, etc.). Always cite real papers.

    Args:
        query: Search query, e.g. 'transformer time series forecasting'
        max_results: Number of papers to return (1-10, default 5)
    """
    import urllib.request
    import urllib.parse
    import xml.etree.ElementTree as ET
    import json

    n = min(max(1, int(max_results)), 10)
    papers = []

    # --- arXiv API (primary) ---
    try:
        arxiv_url = (
            "http://export.arxiv.org/api/query?"
            + urllib.parse.urlencode({
                "search_query": f"all:{query.strip()}",
                "start": 0,
                "max_results": n,
                "sortBy": "relevance",
                "sortOrder": "descending",
            })
        )
        req = urllib.request.Request(arxiv_url, headers={"User-Agent": "DataScienceAgent/1.0"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            xml_data = resp.read().decode("utf-8")

        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }
        root = ET.fromstring(xml_data)
        for entry in root.findall("atom:entry", ns):
            title_el = entry.find("atom:title", ns)
            title = title_el.text.strip().replace("\n", " ") if title_el is not None else "N/A"

            authors = []
            for author in entry.findall("atom:author", ns):
                name_el = author.find("atom:name", ns)
                if name_el is not None:
                    authors.append(name_el.text.strip())

            summary_el = entry.find("atom:summary", ns)
            summary = summary_el.text.strip().replace("\n", " ")[:400] if summary_el is not None else ""

            published_el = entry.find("atom:published", ns)
            year = published_el.text[:4] if published_el is not None else "?"

            url_el = entry.find("atom:id", ns)
            url = url_el.text.strip() if url_el is not None else ""

            papers.append({
                "title": title,
                "authors": ", ".join(authors[:5]) + (" et al." if len(authors) > 5 else ""),
                "year": year,
                "abstract": summary,
                "url": url,
                "source": "arXiv",
            })
    except Exception as e:
        papers.append({"error": f"arXiv: {e}", "source": "arXiv"})

    # --- Semantic Scholar fallback ---
    if not papers or "error" in papers[0]:
        import time
        time.sleep(0.5)  # Be nice to the API
        try:
            ss_url = (
                "https://api.semanticscholar.org/graph/v1/paper/search?"
                + urllib.parse.urlencode({
                    "query": query.strip(),
                    "limit": n,
                    "fields": "title,authors,year,abstract,url",
                })
            )
            req = urllib.request.Request(ss_url, headers={"User-Agent": "DataScienceAgent/1.0"})
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            for p in data.get("data", []):
                authors = ", ".join(
                    a.get("name", "") for a in (p.get("authors") or [])[:5]
                )
                if len(p.get("authors") or []) > 5:
                    authors += " et al."

                papers.append({
                    "title": p.get("title", "N/A"),
                    "authors": authors,
                    "year": str(p.get("year", "?")),
                    "abstract": (p.get("abstract") or "No abstract.")[:400],
                    "url": p.get("url", ""),
                    "source": "Semantic Scholar",
                })
        except Exception as e:
            papers.append({"error": f"SemanticScholar: {e}", "source": "S2"})

    # --- Format output ---
    real_papers = [p for p in papers if "error" not in p]
    if not real_papers:
        errors = "; ".join(p.get("error", "Unknown") for p in papers)
        return f"论文检索失败: {errors}"

    lines = [f"=== 学术论文检索: '{query}' ({len(real_papers)} 篇) ===\n"]
    for i, p in enumerate(real_papers, 1):
        source_tag = f"[{p['source']}]"
        lines.append(f"【{i}】{source_tag} {p['title']}")
        lines.append(f"  作者: {p['authors']}")
        lines.append(f"  发表: {p['year']}")
        lines.append(f"  摘要: {p['abstract']}...")
        lines.append(f"  URL: {p['url']}")
        lines.append("")
    return "\n".join(lines)


@tool
def show_image(file_path: str) -> str:
    """Display a saved image file (PNG, JPG) to the user. Use after _sci_save() or plt.savefig()."""
    path = file_path.strip()
    if not os.path.isfile(path):
        return f"文件不存在: {path}"
    size = os.path.getsize(path)
    if size == 0:
        return f"文件为空: {path}"
    return f"@@IMAGE:{os.path.abspath(path)}@@ (图片 {path}, {size/1024:.1f}KB)"


@tool
def export_data(file_path: str, var_name: str = "df") -> str:
    """Export a DataFrame from the workspace to a file.

    Args:
        file_path: e.g. 'results.csv', 'predictions.xlsx'
        var_name: Variable name, default 'df'. Use for 'forecast_df', 'results', etc.
    """
    global _workspace

    path = file_path.strip()
    name = var_name.strip()

    if name not in _workspace:
        available = [k for k in _workspace.keys() if not k.startswith("_")][:15]
        return f"错误: 工作空间中没有 '{name}'。可用: {available}"

    var = _workspace[name]
    import pandas as pd

    if not isinstance(var, pd.DataFrame):
        return f"错误: '{name}' 不是 DataFrame (类型: {type(var).__name__})"

    fmt = os.path.splitext(path)[1].lstrip(".").lower()

    try:
        if fmt in ("csv",):
            var.to_csv(path, index=False)
        elif fmt in ("xlsx", "xls", "excel"):
            var.to_excel(path, index=False)
        elif fmt in ("json",):
            var.to_json(path, orient="records", force_ascii=False, indent=2)
        elif fmt in ("parquet", "pq"):
            var.to_parquet(path, index=False)
        else:
            return f"不支持的格式: {fmt}。支持: csv, excel, json, parquet"

        size = os.path.getsize(path)
        return f"已导出: {path} ({var.shape[0]}行 x {var.shape[1]}列, {size/1024:.1f}KB)"
    except Exception as e:
        return f"导出失败: {type(e).__name__}: {e}"


@tool
def read_file(file_path: str, nrows: int = 0) -> str:
    """Read a data file and return structure info. Handles large files gracefully.

    For files > 100MB, automatically samples to show structure without loading all data.
    Use nrows to limit rows (e.g. nrows=1000 for a quick preview of large CSV).

    Supports: CSV, Excel (.xlsx/.xls), JSON, Parquet (.parquet)
    """
    import pandas as pd

    path = file_path.strip()
    file_size = os.path.getsize(path) if os.path.isfile(path) else 0
    size_mb = file_size / (1024 * 1024)
    is_large = size_mb > 100

    try:
        if path.endswith(".csv"):
            # For large CSV, read only first N rows for schema
            if is_large and nrows == 0:
                nrows = 5000
            df = pd.read_csv(path, nrows=nrows if nrows else None)
            file_type = "CSV"
        elif path.endswith((".xlsx", ".xls")):
            df = pd.read_excel(path, nrows=nrows if nrows else None)
            file_type = "Excel"
        elif path.endswith(".json"):
            df = pd.read_json(path, nrows=nrows if nrows else None)
            file_type = "JSON"
        elif path.endswith(".parquet"):
            df = pd.read_parquet(path)
            file_type = "Parquet"
        else:
            return f"不支持的文件格式: {path}。支持: CSV, Excel, JSON, Parquet"
    except FileNotFoundError:
        return f"文件不存在: {path}"
    except Exception as e:
        return f"读取失败: {type(e).__name__}: {e}"

    lines = [f"=== {file_type} 文件: {path} ==="]

    if is_large:
        lines.append(f"文件大小: {size_mb:.1f}MB (大文件，预览前 {len(df)} 行)")

    lines.append(f"形状: {df.shape[0]} 行 x {df.shape[1]} 列")
    lines.append(f"列名: {list(df.columns)}")

    # Memory usage for larger datasets
    mem = df.memory_usage(deep=True).sum() / (1024 * 1024)
    if mem > 1:
        lines.append(f"内存占用: {mem:.1f}MB")

    lines.append(f"\n数据类型:\n{df.dtypes.to_string()}")

    # Missing values summary
    missing = df.isnull().sum()
    if missing.sum() > 0:
        missing_pct = (missing / len(df) * 100).round(2)
        missing_info = pd.DataFrame({"缺失数": missing[missing > 0], "占比%": missing_pct[missing > 0]})
        lines.append(f"\n缺失值:\n{missing_info.to_string()}")

    lines.append(f"\n--- 前 5 行 ---\n{df.head().to_string()}")

    # Truncate describe for wide tables
    desc = df.describe(include="all").to_string()
    if len(desc) > 800:
        desc = desc[:800] + f"\n... [截断，共 {len(desc)} 字符]"
    lines.append(f"\n--- 描述性统计 ---\n{desc}")

    return "\n".join(lines)


@tool
def list_files(directory: str = ".") -> str:
    """List files in a directory. Useful for discovering available data files."""
    try:
        items = os.listdir(directory)
        files = []
        dirs = []
        for item in sorted(items):
            full = os.path.join(directory, item)
            if os.path.isdir(full):
                dirs.append(f"📁 {item}/")
            else:
                size = os.path.getsize(full)
                if size < 1024:
                    size_str = f"{size}B"
                elif size < 1024 * 1024:
                    size_str = f"{size / 1024:.1f}KB"
                else:
                    size_str = f"{size / (1024 * 1024):.1f}MB"
                files.append(f"📄 {item} ({size_str})")

        output = [f"目录: {os.path.abspath(directory)}"]
        if dirs:
            output.append("\n".join(dirs))
        if files:
            output.append("\n".join(files))
        if not dirs and not files:
            output.append("(空目录)")
        return "\n".join(output)
    except Exception as e:
        return f"列出文件失败: {e}"
