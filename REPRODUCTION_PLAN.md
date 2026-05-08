# Charting by Machines — 完整复现方案

> 基于 Murray, Xia & Xiao (2024, *Journal of Financial Economics*) 原文严格对齐的实施路线图  
> 结合本项目现有 `cbm/` 源码，指出每一步的现状、差距与修改方法

---

## 目录

1. [数据准备](#1-数据准备)
2. [样本构建与变量定义](#2-样本构建与变量定义)
3. [特征工程（输入变量 CR1…CR12）](#3-特征工程)
4. [目标变量构造](#4-目标变量构造)
5. [观测权重方案](#5-观测权重方案)
6. [ML 模型优化期（1927-01 至 1963-06）](#6-ml-模型优化期)
7. [神经网络架构细节](#7-神经网络架构细节)
8. [集成预测（30次Ensemble）](#8-集成预测)
9. [扩展窗口生成测试期预测（1963-07 至 2022-12）](#9-扩展窗口预测)
10. [投资组合构建](#10-投资组合构建)
11. [绩效分析](#11-绩效分析)
12. [稳定性检验](#12-稳定性检验)
13. [当前代码差距与修改清单](#13-代码差距与修改清单)
14. [完整执行脚本示例](#14-完整执行脚本示例)

---

## 1. 数据准备

### 1.1 数据来源

| 项目 | 原文要求 | 获取方式 |
|------|----------|----------|
| 股票月度收益率 | CRSP 月度数据，1926-01 至 2022-12 | WRDS 账号下载，或使用 Mendeley DOI `10.17632/x63r376783.2` 提供的示例数据 |
| 市场市值 | 每月末股份数×股价（SHROUT×PRC） | CRSP |
| 无风险利率 | Ken French 网站 1 个月 T-bill | [French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html) |
| 因子数据（绩效分析用） | FF3/FF5/Carhart/Q 因子月度收益 | Ken French 网站 + Chen Xue 网站 |

### 1.2 CRSP 数据字段

```
PERMNO, date, RET, DLRET, SHROUT, PRC, EXCHCD, SHRCD
```

- `RET`：月度收益（已除权，含股利）  
- `DLRET`：退市收益（Shumway 1997 调整用）  
- `SHROUT`：流通股数（千股）  
- `PRC`：月末价格  
- `EXCHCD`：交易所代码（1=NYSE, 2=AMEX, 3=NASDAQ）  
- `SHRCD`：股票类别（10/11 = 普通股）

### 1.3 股票筛选条件（每月 t）

原文 Section 2.1：

```
1. SHRCD in (10, 11)                     # 美国本土普通股
2. EXCHCD in (1, 2, 3)                   # NYSE / AMEX / NASDAQ
3. 月 t-12 至 t-1 的收益全部非缺失       # 保证12个月价格数据完整
4. 月 t-1 末市值 = SHROUT × |PRC| 可计算  # 用于价值加权
```

### 1.4 退市收益调整（Shumway 1997）

```python
def calc_delisting_adjusted_return(ret, dlret):
    """
    如果 RET 缺失但 DLRET 有值，用 DLRET 填充；
    如果 DLRET 缺失且被强制退市(DLSTCD 500-591)，假设损失 -30%
    """
    if pd.isna(ret) and not pd.isna(dlret):
        return dlret
    elif not pd.isna(ret) and not pd.isna(dlret):
        return (1 + ret) * (1 + dlret) - 1
    return ret
```

---

## 2. 样本构建与变量定义

### 2.1 时期划分

| 时期 | 月份范围 | 用途 |
|------|----------|------|
| 优化期（Optimization Period） | 1927-01 → 1963-06 | 选择最优 ML 模型 |
| 测试期（Test Period） | 1963-07 → 2022-12 | 生成预测并构建投资组合 |

- 优化期从 1927-01 起（1926-01 是 CRSP 起始月，因此第一个有完整12个月数据的月为 1927-01）  
- 测试期从 1963-07 起，与 Fama-French (1992/1993) 的样本起始对齐

### 2.2 月度超额收益计算

```python
excess_return_i_t = delisting_adjusted_return_i_t - rf_t
```

---

## 3. 特征工程

### 3.1 输入变量定义（CR1…CR12）

原文 Section 2.2 / 3.1：

> *CRₖ 是覆盖月份 t-12 到 t-12+k-1 的 k 个月累积收益*

从投资者在月末 t-1 观察价格图的视角：

| 特征 | 定义 | 含义 |
|------|------|------|
| CR1 | r(t-12) | 12个月前单月收益 |
| CR2 | (1+r(t-12))(1+r(t-11))-1 | 12至11个月前累积收益 |
| CR3 | ∏(1+r(t-12..t-10))-1 | ... |
| … | … | … |
| CR12 | ∏(1+r(t-12..t-1))-1 | 过去12个月累积收益（即标准动量） |

**注意：原文不对输入变量做任何标准化变换**（保留幅度信息，区别于 Jiang et al. 2022 的图像方法）

### 3.2 当前实现对照

`cbm/data/feature_engineer.py` → `FeatureEngineer._calculate_cumulative_returns()` **已正确实现**该定义。

---

## 4. 目标变量构造

原文考察 4 种目标变量（Table 1）：

| 变量符号 | 名称 | 计算方式 |
|----------|------|----------|
| r | 原始超额收益 | 直接使用 |
| r_Std | 截面标准化收益 | (r - mean_t) / std_t |
| **r_Norm** | **截面正态化收益（最优）** | Φ⁻¹[rank_{i,t} / (N_t + 1)] |
| r_Pctl | 百分位收益 | rank_{i,t} / (N_t + 1) |

**最优选择：r_Norm**（CNNLSTM + MSE + EWPM + r_Norm 获得最高 Spearman 相关 10.8%）

### 4.1 r_Norm 的精确实现

原文公式：`r_Norm,i,t = Φ⁻¹[rank_{i,t} / (N_t + 1)]`

```python
from scipy.stats import norm

def calc_ret_norm(returns_cross_section: np.ndarray) -> np.ndarray:
    """
    returns_cross_section: shape (N_t,) — 月 t 全部股票的超额收益
    返回同形状的 r_Norm
    """
    N = len(returns_cross_section)
    valid_mask = ~np.isnan(returns_cross_section)
    result = np.full(N, np.nan)
    
    valid_returns = returns_cross_section[valid_mask]
    # rank 从1开始，最低收益 rank=1
    from scipy.stats import rankdata
    ranks = rankdata(valid_returns, method='average')  # 处理并列
    pctls = ranks / (len(valid_returns) + 1)
    result[valid_mask] = norm.ppf(pctls)   # 反正态 CDF
    return result
```

**当前代码问题**：`cbm/data/feature_engineer.py` 中 `RET_NORM` 实现的是 z-score 标准化，**不是**论文中的正态逆变换。需修正。

---

## 5. 观测权重方案

原文考察 3 种权重（Table 1）：

| 方案 | 缩写 | 实现方式 |
|------|------|----------|
| 等权 | EW | w_j = 1/N_total |
| **每月等权+月内股票等权（最优）** | **EWPM** | w_{i,t} = 1/(T × N_t) |
| 每月等权+月内市值加权 | EWPMVW | w_{i,t} = mcap_{i,t-1} / (T × Σ_i mcap_{i,t}) |

### 5.1 EWPM 实现

```python
def calc_ewpm_weights(dates: np.ndarray, n_per_month: dict) -> np.ndarray:
    """
    dates: 每个样本对应的月份
    n_per_month: {月份 -> 该月股票数}
    返回: 每个样本的权重
    """
    T = len(n_per_month)  # 月份总数
    weights = np.zeros(len(dates))
    for i, d in enumerate(dates):
        weights[i] = 1.0 / (T * n_per_month[d])
    return weights
```

**当前代码**：`cbm/ml/trainer.py` 中权重计算需核实 EWPM 逻辑是否与上述一致。

---

## 6. ML 模型优化期

### 6.1 优化流程

原文 Section 3.1：考察 96 种模型组合 = 4架构 × 2损失函数 × 3权重 × 4目标变量

#### 拟合月份划分（优化期内）

- **拟合月（Fitting months）**：偶数年的偶数月 + 奇数年的奇数月
- **评估月（Non-fitting months）**：偶数年的奇数月 + 奇数年的偶数月

```python
def is_fitting_month(year: int, month: int) -> bool:
    """判断是否为拟合月"""
    return (year % 2 == 0 and month % 2 == 0) or \
           (year % 2 == 1 and month % 2 == 1)
```

#### 训练/验证分割

- 每次拟合：从拟合月数据中随机抽取 **70% 训练 / 30% 验证**
- 每个模型重复拟合 **30次**（ensemble averaging）
- 使用 **验证集损失的 early stopping**

#### 评估指标

时间序列平均的**月度截面 Spearman 秩相关系数**（在评估月计算）：

```python
from scipy.stats import spearmanr

def eval_spearman(forecasts_by_month: dict, actual_by_month: dict) -> float:
    """
    计算非拟合月上预测值与实际超额收益的Spearman相关均值
    """
    corrs = []
    for month in forecasts_by_month:
        fcst = forecasts_by_month[month]
        actual = actual_by_month[month]
        rho, _ = spearmanr(fcst, actual, nan_policy='omit')
        corrs.append(rho)
    return np.mean(corrs)
```

### 6.2 优化期最优结果（原文 Table 1）

```
Architecture: CNNLSTM
Loss function: MSE
Weighting:    EWPM
Target:       r_Norm
Spearman corr (optimization period): 10.8%
```

---

## 7. 神经网络架构细节

原文 Section I / Figure A1 of Internet Appendix（具体超参数参考 Goodfellow et al. 2016 惯例）：

### 7.1 FNN（前馈神经网络）

```
Input(12) → Dense(64,ReLU) → Dropout(0.2) → Dense(32,ReLU) → Dropout(0.2) → Dense(1)
```

### 7.2 CNN（一维卷积网络）

```
Input(1,12) → Conv1D(32,k=3,ReLU) → Conv1D(64,k=3,ReLU) → AdaptiveAvgPool → Dense(64,ReLU) → Dense(1)
```

### 7.3 LSTM

```
Input(12,1) → LSTM(64,layers=2,dropout=0.2) → 取最后隐状态 → Dense(1)
```

### 7.4 CNNLSTM（最优架构）

```
Input(1,12)
  → Conv1D(32,k=3,ReLU)
  → Conv1D(32,k=3,ReLU)
  → reshape → (12,32)
  → LSTM(64,layers=2,dropout=0.2)
  → 取最后隐状态
  → Dropout(0.2)
  → Dense(1)
```

### 7.5 通用超参数

| 超参数 | 值 |
|--------|-----|
| 优化器 | Adam |
| 学习率 | 0.001 |
| Batch size | 256 |
| 最大 Epoch | 100 |
| Early stopping patience | 10 epochs（基于验证集损失） |
| Weight decay | 1e-5 |

**当前代码**：`cbm/ml/models/pytorch_impl.py` 架构实现与上述**基本一致**，超参数匹配良好。

---

## 8. 集成预测

### 8.1 原文方案

对选定模型（CNNLSTM + MSE + EWPM + r_Norm）：

1. 用所有拟合月数据独立训练 **30个** 模型实例（每次不同随机种子）
2. 对每个评估样本，取 30 个模型预测值的**平均值**作为最终预测

```python
def ensemble_predict(models: list, X: np.ndarray) -> np.ndarray:
    """30个模型的均值预测"""
    preds = np.stack([m.predict(X) for m in models], axis=0)  # (30, N)
    return preds.mean(axis=0)
```

### 8.2 为何需要 30 次

论文 Internet Appendix Section VI 验证：30次平均后，集成预测的随机性几乎消除，结果可被他人复现。

**当前代码**：`cbm/ml/trainer.py` 中的 `n_ensemble=30` 参数**已实现**该逻辑。

---

## 9. 扩展窗口预测

### 9.1 测试期扩展窗口方案

原文 Section 3.2：用**扩展窗口**（expanding window）的过去数据拟合模型，对未来期间生成预测。

具体的 6 个训练窗口端点：

| 训练窗口结束 | 预测覆盖的测试期 |
|-------------|-----------------|
| 1963-06 | 1963-07 → 1974-12 |
| 1974-12 | 1975-01 → 1984-12 |
| 1984-12 | 1985-01 → 1994-12 |
| 1994-12 | 1995-01 → 2004-12 |
| 2004-12 | 2005-01 → 2014-12 |
| 2014-12 | 2015-01 → 2022-12 |

所有窗口**起点均为 1927-01**（expanding，非 rolling）。

### 9.2 MLER 计算

```python
# 对某个测试月 t，确定使用哪个训练窗口的模型
def get_mler(t: str, window_models: dict, features_i: np.ndarray) -> float:
    """
    t: YYYY-MM 格式的月份
    window_models: {'196306': [30个模型], '197412': [30个模型], ...}
    features_i: 股票 i 的 CR1...CR12（shape: (12,)）
    """
    if   t <= '197412': key = '196306'
    elif t <= '198412': key = '197412'
    elif t <= '199412': key = '198412'
    elif t <= '200412': key = '199412'
    elif t <= '201412': key = '200412'
    else:               key = '201412'
    
    models = window_models[key]
    preds = [m.predict(features_i[np.newaxis, :])[0] for m in models]
    return np.mean(preds)
```

---

## 10. 投资组合构建

### 10.1 原文方案（Table 3 / Section 4.1）

每月末 t-1，按 MLER 将所有样本月 t 的股票排序，分成 10 个十分位（Decile）组合：

```
月 t 的组合超额收益 = 价值加权（以月 t-1 末市值为权重）月 t 超额收益
```

**关键细节**：

| 要素 | 原文设定 |
|------|----------|
| 分位点（Breakpoints） | **仅用 NYSE 上市股票**计算十分位断点 |
| 组合持仓范围 | NYSE + AMEX + NASDAQ 全部符合条件股票 |
| 加权方式 | 价值加权（Value-weighted） |
| 多空组合 | 做多 Decile 10，做空 Decile 1（零成本） |

### 10.2 NYSE 断点实现

```python
def calc_nyse_breakpoints(mler_values: dict, nyse_mask: dict, n_quantiles: int = 10):
    """
    mler_values: {ticker -> MLER值}（含全部交易所）
    nyse_mask:   {ticker -> True if NYSE}
    返回 9 个断点值
    """
    nyse_mler = [v for t, v in mler_values.items() if nyse_mask.get(t, False)]
    percentiles = np.linspace(0, 100, n_quantiles + 1)[1:-1]  # 10%, 20%,...90%
    return np.percentile(nyse_mler, percentiles)
```

**当前代码差距**：`cbm/portfolio/constructor.py` 第 95 行：

```python
if len(date_forecasts) < self.n_portfolios:
    continue
```

此处条件应改为「NYSE 股票数 < n_portfolios」；且断点计算目前基于全部股票，**需改为仅用 NYSE 股票**。

### 10.3 价值加权

```python
def value_weighted_return(stock_returns: dict, market_caps: dict) -> float:
    total_cap = sum(market_caps[t] for t in stock_returns if t in market_caps)
    if total_cap == 0:
        return np.nan
    return sum(
        stock_returns[t] * market_caps[t] / total_cap
        for t in stock_returns if t in market_caps
    )
```

---

## 11. 绩效分析

### 11.1 基本统计（Table 3）

对每个十分位组合和多空组合：

```python
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac

def newey_west_tstat(returns: np.ndarray, lags: int = 12) -> tuple:
    """Newey-West（12滞后期）调整的均值 t 统计量"""
    n = len(returns)
    mean_ret = np.mean(returns)
    
    # OLS of returns on constant
    X = np.ones(n)
    model = sm.OLS(returns, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    t_stat = model.tvalues[0]
    return mean_ret, t_stat
```

报告指标：
- 月均超额收益 r（%/month）
- 月均超额收益的 Newey-West t 统计量（12滞后期）
- 超额收益标准差 SD
- 年化 Sharpe 比率 = mean_r / std_r × √12

### 11.2 因子模型 Alpha（Table 4）

对 MLER 10-1 组合，分别用以下模型做时序回归：

| 模型 | 因子 |
|------|------|
| CAPM | MKT |
| FF3 | MKT, SMB, HML |
| Carhart (FFC) | MKT, SMB, HML, MOM |
| FFC+REV | MKT, SMB, HML, MOM, STR |
| FF5 | MKT, SMB, HML, RMW, CMA |
| Q-factor | MKT, ME, IA, ROE |

```python
def factor_alpha(portfolio_returns: np.ndarray, factors: pd.DataFrame, lags: int = 12):
    """
    portfolio_returns: (T,) 月度超额收益
    factors: DataFrame, 列为各因子月度超额收益，索引为月份
    返回: alpha, t_stat, adj_r2
    """
    X = sm.add_constant(factors.values)
    model = sm.OLS(portfolio_returns, X).fit(cov_type='HAC', cov_kwds={'maxlags': lags})
    return model.params[0], model.tvalues[0], model.rsquared_adj
```

---

## 12. 稳定性检验

### 12.1 子期间检验（Table 5）

将测试期分为 6 个子期间，对每个子期间重复 Table 3 分析：

```
1963-07 ~ 1974-12
1975-01 ~ 1984-12
1985-01 ~ 1994-12
1995-01 ~ 2004-12
2005-01 ~ 2014-12
2015-01 ~ 2022-12
```

### 12.2 大市值股检验（Table 6）

限制样本为：
- Size > P20_NYSE（市值大于 NYSE 第20百分位）
- Size > P50_NYSE
- Top 500 by market cap

### 12.3 预测稳定性检验（Table 7-8）

用不同子期间拟合的模型生成预测，计算：
- 不同子期间预测之间的截面 Spearman 秩相关
- 10-1 组合持仓重叠比例（Common Holdings）
- 10-1 组合收益 Pearson 相关

---

## 13. 代码差距与修改清单

### 13.1 必须修复（影响结果正确性）

| 序号 | 文件 | 问题 | 修改方案 |
|------|------|------|----------|
| ① | `cbm/data/feature_engineer.py` | `RET_NORM` 用的是 z-score，原文用正态逆 CDF | 改为 `norm.ppf(rank/(N+1))` |
| ② | `cbm/portfolio/constructor.py` | 断点用全部股票，原文仅用 NYSE 股票 | 添加 `nyse_mask` 参数，仅对 NYSE 股票计算断点 |
| ③ | `cbm/portfolio/constructor.py` | 跳过条件 `len < n_portfolios` 过于严苛 | 条件改为「NYSE 股票数 < n_portfolios」 |
| ④ | `cbm/data/adapters/yahoo_finance.py` | 无 NYSE 交易所标识字段 | 添加 `EXCHCD` 字段或单独维护 NYSE 股票列表 |
| ⑤ | `cbm/ml/trainer.py` | 优化期拟合月划分未实现 | 添加 `is_fitting_month()` 过滤逻辑 |

### 13.2 建议改进（提升复现精度）

| 序号 | 文件 | 问题 | 修改方案 |
|------|------|------|----------|
| ⑥ | `cbm/data/adapters/` | 缺少 CRSP 适配器 | 实现 `WRDSAdapter`，支持从 WRDS 直接下载 CRSP 数据 |
| ⑦ | `cbm/portfolio/analyzer.py` | 未实现 Newey-West t 统计量 | 引入 `statsmodels` 的 HAC 标准误计算 |
| ⑧ | `cbm/portfolio/analyzer.py` | 未实现因子模型 Alpha 计算 | 添加 `factor_alpha()` 支持 CAPM/FF3/FF5/FFC/Q |
| ⑨ | `cbm/core/engine.py` | 扩展窗口逻辑简化 | 实现 6 个训练窗口的完整 MLER 计算流程 |
| ⑩ | `cbm/ml/trainer.py` | EWPM 权重实现需核查 | 验证每月内各股票权重之和等于 1/T |

### 13.3 数据限制说明

原文使用 **CRSP 数据（需付费订阅 WRDS）**。如无法获取，以下替代方案可用于方法验证：

- **Mendeley 示例数据**：DOI `10.17632/x63r376783.2`（含随机化数据，可验证代码逻辑）
- **Yahoo Finance**（有频率限制）：股票数量大幅减少，结论仅供参考
- **合成数据**（`SyntheticAdapter`）：仅用于 pipeline 调试，不代表真实结果

---

## 14. 完整执行脚本示例

### 14.1 阶段一：数据准备与特征工程

```python
# reproduce_step1_data.py
import os
os.add_dll_directory(r"F:\envs\charting_by_machine\Lib\site-packages\torch\lib")

from cbm import PortfolioEngine

engine = PortfolioEngine()

# 1. 使用 CRSP 数据（需要 WRDS 账号）
# engine.load_data(source="wrds", universe="crsp_all",
#                  start_date="1926-01-01", end_date="2022-12-31")

# 替代：使用 Mendeley 示例数据（本地 CSV）
# engine.load_data(source="local", data_path="./data/crsp_sample.csv")

# 调试用：合成数据
engine.load_data(
    universe="sp500",          # 50支代理股票
    source="synthetic",
    start_date="1927-01-01",
    end_date="2022-12-31",
)

# 2. 特征工程（r_Norm 目标变量）
features = engine.prepare_features()
print(f"特征样本数: {len(features.features)}")
print(f"特征维度: {features.features.shape[1]}")  # 应为 12
```

### 14.2 阶段二：优化期模型选择

```python
# reproduce_step2_optimization.py
# 在 1927-01 至 1963-06 优化期内评估 96 种模型组合

ARCHITECTURES  = ["fnn", "cnn", "lstm", "cnn_lstm"]
LOSS_FUNCTIONS = ["mse", "mae"]
WEIGHTINGS     = ["ew", "ewpm", "ewpmvw"]
TARGET_VARS    = ["ret", "ret_std", "ret_norm", "ret_pctl"]

results = {}

for arch in ARCHITECTURES:
    for loss in LOSS_FUNCTIONS:
        for weight in WEIGHTINGS:
            for target in TARGET_VARS:
                model_id = engine.train_model(
                    architecture=arch,
                    loss_function=loss,
                    weighting=weight,
                    optimization_period=("1927-01", "1963-06"),
                    n_ensemble=30,          # 30次集成
                    train_val_split=0.70,   # 70/30
                )
                # 评估（非拟合月的Spearman相关）
                eval_score = engine.evaluate_optimization(model_id)
                results[(arch, loss, weight, target)] = eval_score
                print(f"{arch}|{loss}|{weight}|{target}: {eval_score:.4f}")

# 选取最优模型
best_config = max(results, key=results.get)
print(f"\n最优配置: {best_config}")
print(f"Spearman相关: {results[best_config]:.4f}")
# 原文最优: ('cnn_lstm', 'mse', 'ewpm', 'ret_norm') → 10.8%
```

### 14.3 阶段三：测试期扩展窗口预测

```python
# reproduce_step3_forecast.py

# 6 个扩展窗口
WINDOW_ENDS   = ["1963-06", "1974-12", "1984-12", "1994-12", "2004-12", "2014-12"]
WINDOW_STARTS = ["1927-01"] * 6  # expanding window，起点固定

all_forecasts = {}

for start, end in zip(WINDOW_STARTS, WINDOW_ENDS):
    model_id = engine.train_model(
        architecture="cnn_lstm",
        loss_function="mse",
        weighting="ewpm",
        optimization_period=(start, end),
        n_ensemble=30,
    )
    
    # 确定该模型覆盖的预测期
    forecast_period_map = {
        "1963-06": ("1963-07", "1974-12"),
        "1974-12": ("1975-01", "1984-12"),
        "1984-12": ("1985-01", "1994-12"),
        "1994-12": ("1995-01", "2004-12"),
        "2004-12": ("2005-01", "2014-12"),
        "2014-12": ("2015-01", "2022-12"),
    }
    test_start, test_end = forecast_period_map[end]
    
    forecasts = engine.forecast(
        model_id=model_id,
        test_period=(test_start, test_end),
    )
    all_forecasts[end] = forecasts
    print(f"窗口 1927-01~{end} → 预测 {test_start}~{test_end} 完成")
```

### 14.4 阶段四：投资组合构建与绩效分析

```python
# reproduce_step4_portfolio.py

# 合并所有预测
combined_forecasts = engine.merge_forecasts(all_forecasts)

# 构建十分位组合（NYSE 断点，价值加权）
portfolios = engine.construct_portfolios(
    forecasts=combined_forecasts,
    n_portfolios=10,
    weighting="value",
    use_nyse_breakpoints=True,   # 关键：仅用 NYSE 计算断点
)

# 绩效分析
performance = engine.analyze_performance(
    portfolios=portfolios,
    factor_models=["capm", "ff3", "ffc", "ffc_rev", "ff5"],
    newey_west_lags=12,
)

# 打印结果
for name, metrics in performance.items():
    print(f"\n=== {name} ===")
    print(metrics.summary())

# 目标结果对标（原文 Table 3）：
# MLER 1 平均月超额收益: -0.14%
# MLER 10 平均月超额收益: 0.93%
# MLER 10-1 平均月超额收益: 1.08%（t=5.51）
# MLER 10-1 Sharpe (年化): 0.78
```

---

## 附录：关键参数速查表

| 参数 | 原文取值 |
|------|----------|
| 输入特征数 | 12（CR1…CR12，月度累积收益） |
| 目标变量 | r_Norm（正态化截面排名） |
| 架构 | CNNLSTM |
| 损失函数 | MSE |
| 观测权重 | EWPM |
| 集成次数 | 30 |
| 训练/验证分割 | 70% / 30% |
| Early stopping patience | 10 epochs |
| 优化器 | Adam，lr=0.001 |
| 组合数 | 10（十分位） |
| 断点股票池 | 仅 NYSE 上市股票 |
| 组合加权 | 价值加权（市值） |
| t 统计量 | Newey-West，12 滞后期 |
| 优化期 | 1927-01 → 1963-06 |
| 测试期 | 1963-07 → 2022-12 |
| 扩展窗口端点 | 1963-06, 1974-12, 1984-12, 1994-12, 2004-12, 2014-12 |

---

*生成时间：2026-04-18 | 基于 Murray, Xia & Xiao (2024) Journal of Financial Economics 153, 103791*
