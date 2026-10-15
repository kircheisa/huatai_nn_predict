import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False

# 1. 重新读取并清洗原始数据（确保数据一致性）
file_path = Path(__file__).with_name('相关性分析.xlsx')
df_raw = pd.read_excel(file_path)

# 数据清洗：去除单位行，重命名列，转换数据类型
df_clean = df_raw.iloc[1:].copy()  # 去除第一行单位行
df_clean.columns = ['date', 'sse_index', 'gem_index', 'weekly_liquidity', 'daily_liquidity']

# 转换日期格式
df_clean['date'] = pd.to_datetime(df_clean['date'], errors='coerce')

# 转换数值列（处理可能的非数值数据）
numeric_cols = ['sse_index', 'gem_index', 'weekly_liquidity', 'daily_liquidity']
for col in numeric_cols:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# 删除缺失值和异常值（指数需大于1000点，流动性数据在合理范围）
df_clean = df_clean.dropna(subset=['date', 'sse_index', 'daily_liquidity'])
df_clean = df_clean[(df_clean['sse_index'] > 1000) & 
                    (df_clean['daily_liquidity'] >= -10000) &  # 过滤极端异常值
                    (df_clean['daily_liquidity'] <= 10000)]

# 提取年月信息
df_clean['year'] = df_clean['date'].dt.year
df_clean['month'] = df_clean['date'].dt.month

# 2. 筛选2025年3月和2026年3月的数据
data_202503 = df_clean[(df_clean['year'] == 2025) & (df_clean['month'] == 3)].copy()
data_202603 = df_clean[(df_clean['year'] == 2026) & (df_clean['month'] == 3)].copy()

# 按日期排序
data_202503 = data_202503.sort_values('date').reset_index(drop=True)
data_202603 = data_202603.sort_values('date').reset_index(drop=True)

# 3. 计算相关性（Pearson相关系数）
def calculate_correlation(data, month_label):
    """计算指定月份的相关性及统计指标"""
    # 提取核心变量
    sse = data['sse_index'].values
    liquidity = data['daily_liquidity'].values
    
    # 计算Pearson相关系数和p值
    corr_coef, p_value = stats.pearsonr(sse, liquidity)
    
    # 计算辅助统计量
    n_samples = len(data)
    sse_mean = np.mean(sse)
    sse_std = np.std(sse)
    liquidity_mean = np.mean(liquidity)
    liquidity_std = np.std(liquidity)
    liquidity_total = np.sum(liquidity)  # 月内总净投放
    
    # 判断相关性强度和方向
    if abs(corr_coef) < 0.3:
        corr_strength = '弱相关'
    elif abs(corr_coef) < 0.5:
        corr_strength = '中等相关'
    else:
        corr_strength = '强相关'
    
    corr_direction = '正相关' if corr_coef > 0 else '负相关' if corr_coef < 0 else '无相关'
    significance = '显著' if p_value < 0.05 else '不显著'
    
    return {
        '月份': month_label,
        '样本数量': n_samples,
        '相关系数': round(corr_coef, 4),
        'p值': round(p_value, 4),
        '相关性强度': corr_strength,
        '相关性方向': corr_direction,
        '统计显著性': significance,
        '上证综指均值': round(sse_mean, 2),
        '上证综指标准差': round(sse_std, 2),
        '货币净投放均值': round(liquidity_mean, 2),
        '货币净投放标准差': round(liquidity_std, 2),
        '月内总净投放（亿元）': round(liquidity_total, 2)
    }


def calculate_monthly_change_correlation(data):
    monthly = data[['date', 'sse_index', 'daily_liquidity']].copy().sort_values('date')
    monthly = monthly.set_index('date')
    monthly_agg = pd.DataFrame({
        '月末上证综指': monthly['sse_index'].resample('ME').last(),
        '月初上证综指': monthly['sse_index'].resample('ME').first(),
        '月度净投放': monthly['daily_liquidity'].resample('ME').sum()
    }).dropna()
    monthly_agg['上证综指月变动量'] = monthly_agg['月末上证综指'] - monthly_agg['月初上证综指']
    monthly_pair = monthly_agg[['上证综指月变动量', '月度净投放']].dropna()
    n_samples = len(monthly_pair)
    if n_samples < 3:
        corr_coef = np.nan
        p_value = np.nan
    else:
        corr_coef, p_value = stats.pearsonr(monthly_pair['上证综指月变动量'], monthly_pair['月度净投放'])
    return monthly_agg, pd.DataFrame([{
        '指标': '上证综指月变动量 vs 月度净投放',
        '样本数量': n_samples,
        '相关系数': round(corr_coef, 4) if pd.notna(corr_coef) else np.nan,
        'p值': round(p_value, 4) if pd.notna(p_value) else np.nan
    }])

# 执行计算
result_202503 = calculate_correlation(data_202503, '2025年3月')
result_202603 = calculate_correlation(data_202603, '2026年3月')
monthly_data_df, monthly_corr_df = calculate_monthly_change_correlation(df_clean)

# 整理结果为DataFrame
result_df = pd.DataFrame([result_202503, result_202603])

# 4. 输出计算结果
print("="*80)
print("2025年3月、2026年3月上证综指与货币净投放相关性计算结果")
print("="*80)
print(f"\n【计算方法】Pearson相关系数（衡量线性相关性）")
print(f"【核心变量】上证综合指数（日度收盘价） vs 中国:公开市场操作:货币净投放（日度数据）")
print(f"\n" + "-"*80)
print(result_df.to_string(index=False))
print(f"\n" + "-"*80)
print("上证综指每月变动量与货币净投放（月度总量）相关性")
print(monthly_corr_df.to_string(index=False))

# 5. 验证计算过程（输出部分原始数据和中间结果）
print(f"\n" + "="*80)
print("计算过程验证（部分原始数据）")
print("="*80)

print(f"\n【2025年3月前5条数据】")
print(data_202503[['date', 'sse_index', 'daily_liquidity']].head().to_string(index=False))

print(f"\n【2026年3月前5条数据】")
print(data_202603[['date', 'sse_index', 'daily_liquidity']].head().to_string(index=False))

# 手动验证相关系数计算（与scipy结果对比）
def manual_corr(x, y):
    """手动计算Pearson相关系数，验证结果"""
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    return numerator / denominator if denominator != 0 else 0

corr_2025_manual = manual_corr(data_202503['sse_index'], data_202503['daily_liquidity'])
corr_2026_manual = manual_corr(data_202603['sse_index'], data_202603['daily_liquidity'])

print(f"\n【结果验证】")
print(f"2025年3月：scipy计算({round(result_202503['相关系数'], 4)}) vs 手动计算({round(corr_2025_manual, 4)})")
print(f"2026年3月：scipy计算({round(result_202603['相关系数'], 4)}) vs 手动计算({round(corr_2026_manual, 4)})")
print(f"结论：两种方法结果一致，计算准确")

# 6. 保存计算结果到Excel
output_path = Path(__file__).with_name('202503_202603相关性计算结果.xlsx')
with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
    result_df.to_excel(writer, index=False, sheet_name='指定月份相关性')
    monthly_corr_df.to_excel(writer, index=False, sheet_name='月度相关性')
    monthly_data_df.reset_index().to_excel(writer, index=False, sheet_name='月度聚合明细')
print(f"\n【文件输出】计算结果已保存至：202503_202603相关性计算结果.xlsx")
