导师皱眉：“你拿 XGBoost 直接怼时序预测？！”我：“不行吗？”导师：“先做时间序列分解！再让模型补残差，效果才稳！”
哈喽，我是小白~

今儿和大家聊一个极为重要的一个思路，也是一个案例：基于XGBoost和时间序列分解的电力负荷预测模型。

今天这篇文章，我们会把 时间序列分解+XGBoost 的核心逻辑讲透，再给一个完整的落地案例，带大家一步一步把模型做起来~

核心逻辑：先分解，再建模，最后合成
首先，电力负荷是一个“带有规律的复杂时间序列”。

直观地讲，它同时包含老三种东西：

趋势（Trend）：比如随着经济增长，负荷逐渐增加；
季节性（Seasonality）：每天白天负荷高、晚上低；夏季制冷高、冬季采暖高；
剩余项/扰动（Remainder/Residual）：各种突发事件、温度突然变化、节假日等造成的临时偏差。
也就是：

其中， 是时刻  的负荷， 是趋势， 是季节性， 是剩余项。

这是“加性分解”的范式。也可以用乘性分解，但负荷预测里加性分解通常够用且解释直观。

为什么先分解？

因为直接对原始负荷建模，模型会混杂地同时学习趋势、季节性、波动，复杂又容易过拟合。

拆开后，每部分的规律性不同，就能用合适的模型分别搞定：

趋势和季节性：结构稳定，统计分解方法（比如STL分解）就很香；
剩余项：更随机、更依赖外部因素（温度、节假日、时段等），适合强非线性的机器学习模型，比如XGBoost。
然后再把预测结果按公式合起来：

其中， 和  来自时间序列分解， 来自XGBoost。这样分工明确，模型更稳、更好解释、更容易调试。

XGBoost在这里的角色是：

不直接去拟合全部负荷，而是去拟合“剩余项”；
从大量特征（时段、温度、历史滞后值、滚动统计等）中抽取非线性关系；
通过加法模型累积弱学习器（树）来逼近复杂函数。
XGBoost的目标函数一般是：

其中  是损失（如平方误差）， 是复杂度正则， 是第  棵树。逐步加树的过程（梯度提升）可写成：

其中  是学习率。

直白说，就是把问题拆解（分解成趋势、季节性、剩余项），把最难那块（剩余项）交给XGBoost，最后把预测拼回去。

好处：

减少模型“分心”，专注难点；
解释友好，可视化直观；
鲁棒性强，工程效果好；
方便叠加其他模块。
完整案例
我们整个流程，包含数据模拟、分解、特征工程、建模、评估、可视化。

考虑到现实中电力负荷很多时候受温度、节假日、时段影响，我们会把这些信息也模拟进来。

数据集
我们用小时级数据，180天，包含以下因素：

每日负荷曲线（高峰在白天低在夜间），用正弦波表达；
每周周期（工作日高、周末略低）；
长期趋势（缓慢上升）；
温度对负荷的U型影响（热/冷都提升负荷，比如空调和采暖），可简化为与“离舒适温度”的绝对偏差相关；
节假日效果（负荷下降或变化）；
噪声与少量异常点。
温度影响可简单写成：

其中  是舒适温度，常取20℃左右。

最终负荷：

趋
势
日
季
节
性
周
效
应
温
度
效
应
节
假
日
效
应
模型流程概览
STL分解得到 ；

用XGBoost拟合 ，输入包括时段特征、历史滞后、温度、节假日等；

用PyTorch构建一个简单的MLP作为“残差校准器”，对XGBoost的预测进行小幅校正（stacking思想：把XGBoost的输出当特征之一）；

汇总得到最终预测 ；

评估指标：RMSE/MAE/MAPE；

可视化。

评估指标定义：

完整代码
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")
np.random.seed(42)

# 时间与数据模拟
period_hours = 24 * 180  # 180天小时数据
date_index = pd.date_range(start='2022-01-01', periods=period_hours, freq='H')
df = pd.DataFrame({'timestamp': date_index})
df['hour'] = df['timestamp'].dt.hour
df['dow'] = df['timestamp'].dt.dayofweek  # 0=Monday, 6=Sunday
df['day'] = df['timestamp'].dt.day
df['month'] = df['timestamp'].dt.month
df['dayofyear'] = df['timestamp'].dt.dayofyear
df['weekofyear'] = df['timestamp'].dt.isocalendar().week.astype(int)
df['is_weekend'] = df['dow'].isin([5,6]).astype(int)

# 模拟温度：季节波动+昼夜波动+噪声
# 年内温度: 冬季低、夏季高，用年内正弦
year_temp = 15 + 10 * np.sin(2 * np.pi * (df['dayofyear'] / 365.0))
# 昼夜波动：夜间略低
day_temp = -2 * np.cos(2 * np.pi * (df['hour'] / 24.0))
# 随机扰动
noise_temp = np.random.normal(0, 1.5, size=period_hours)
df['temp'] = year_temp + day_temp + noise_temp

# 模拟节假日（随机挑选若干天）
holiday_days = np.random.choice(df['timestamp'].dt.date.unique(), size=10, replace=False)
df['is_holiday'] = df['timestamp'].dt.date.isin(holiday_days).astype(int)

# 模拟负荷的组成：趋势+日季节性+周效应+温度U型+节假日影响+噪声
alpha0 = 500  # 基础负荷
alpha1 = 0.02  # 线性趋势斜率（每小时增长0.02）
A, B = 40, 25  # 日周期幅度
comfort_T = 20
gamma = 6.0  # 温度影响强度
holiday_effect = -50  # 节假日负荷降低
weekly_effect = {0: 20, 1: 15, 2: 10, 3: 5, 4: 0, 5: -10, 6: -15}  # 周一到周日

# 日季节性（双频叠加）
daily_season = (A * np.sin(2 * np.pi * df['hour'] / 24.0)
                + B * np.cos(2 * np.pi * df['hour'] / 24.0))

# 周效应映射
df['weekly_effect'] = df['dow'].map(weekly_effect)

# 温度U型效应
temp_effect = gamma * np.abs(df['temp'] - comfort_T)

# 随机异常点（突发波动）
anomaly = np.zeros(period_hours)
anomaly_idx = np.random.choice(period_hours, size=30, replace=False)
anomaly[anomaly_idx] = np.random.choice([80, -60, 100, -90], size=30)

# 噪声
epsilon = np.random.normal(0, 12, size=period_hours)

# 最终负荷
df['trend'] = alpha0 + alpha1 * np.arange(period_hours)
df['season_daily'] = daily_season
df['load'] = df['trend'] + df['season_daily'] + df['weekly_effect'] + temp_effect \
             + holiday_effect * df['is_holiday'] + anomaly + epsilon

# 可视化1：原始时间序列
plt.figure(figsize=(14, 4))
plt.plot(df['timestamp'], df['load'], color='#FF00FF', linewidth=1.2, label='Load (Magenta)')
plt.title('图1：模拟电力负荷时间序列（180天，小时级）', fontsize=14)
plt.xlabel('时间')
plt.ylabel('负荷')
plt.grid(alpha=0.3)
# 标注测试集区域（最后30天）
test_start = df['timestamp'].iloc[-24*30]
plt.axvspan(test_start, df['timestamp'].iloc[-1], color='#00FFFF', alpha=0.15, label='Test Period')
plt.legend()
plt.tight_layout()
plt.show()

# STL分解
from statsmodels.tsa.seasonal import STL

# 使用24小时作为周期
stl = STL(df['load'], period=24, robust=True)
res_stl = stl.fit()
df['stl_trend'] = res_stl.trend
df['stl_season'] = res_stl.seasonal
df['stl_resid'] = res_stl.resid

# 可视化2：STL分解结果
fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
axes[0].plot(df['timestamp'], df['stl_trend'], color='#00FF00', label='Trend (Lime)', linewidth=1.5)
axes[0].set_title('趋势（STL分解）'); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].plot(df['timestamp'], df['stl_season'], color='#FF8C00', label='Seasonality (Orange)', linewidth=1.2)
axes[1].set_title('季节性（STL分解）'); axes[1].legend(); axes[1].grid(alpha=0.3)
axes[2].plot(df['timestamp'], df['stl_resid'], color='#00BFFF', label='Remainder (DeepSkyBlue)', linewidth=1.0)
axes[2].set_title('剩余项（STL分解）'); axes[2].legend(); axes[2].grid(alpha=0.3)
plt.xlabel('时间')
plt.suptitle('图2：STL分解三要素', y=0.98, fontsize=16)
plt.tight_layout()
plt.show()

# 特征工程：时段/温度/滞后/滚动
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)

# 滞后与滚动特征（针对剩余项）
for lag in [1, 2, 24, 48, 168]:
    df[f'resid_lag_{lag}'] = df['stl_resid'].shift(lag)

df['temp_ma_24'] = df['temp'].rolling(24).mean()
df['temp_ma_168'] = df['temp'].rolling(168).mean()
df['resid_ma_24'] = df['stl_resid'].rolling(24).mean()
df['resid_std_24'] = df['stl_resid'].rolling(24).std()

# 滤除前期空值
df_model = df.dropna().copy()

# 划分训练/验证/测试（时间顺序）
n = len(df_model)
test_hours = 24 * 30
val_hours = 24 * 20
train_end = n - (test_hours + val_hours)
val_end = n - test_hours

train_df = df_model.iloc[:train_end]
val_df = df_model.iloc[train_end:val_end]
test_df = df_model.iloc[val_end:]

feature_cols = [
    'hour', 'dow', 'is_weekend', 'is_holiday',
    'hour_sin', 'hour_cos',
    'temp', 'temp_ma_24', 'temp_ma_168',
    'resid_lag_1', 'resid_lag_2', 'resid_lag_24', 'resid_lag_48', 'resid_lag_168',
    'resid_ma_24', 'resid_std_24'
]

X_train = train_df[feature_cols]
y_train = train_df['stl_resid']
X_val = val_df[feature_cols]
y_val = val_df['stl_resid']
X_test = test_df[feature_cols]
y_test_resid = test_df['stl_resid']  # 仅用于残差的评估

# XGBoost训练
from xgboost import XGBRegressor
xgb = XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    random_state=42,
    tree_method='hist'
)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# XGBoost残差预测
pred_resid_train_xgb = xgb.predict(X_train)
pred_resid_val_xgb = xgb.predict(X_val)
pred_resid_test_xgb = xgb.predict(X_test)

# 可视化3：XGBoost特征重要性
importances = xgb.feature_importances_
imp_df = pd.DataFrame({'feature': feature_cols, 'importance': importances}).sort_values(by='importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(data=imp_df, x='importance', y='feature', palette='bright')
plt.title('图3：XGBoost特征重要性', fontsize=14)
plt.xlabel('重要性'); plt.ylabel('特征')
plt.tight_layout()
plt.show()

# PyTorch：残差校准器（MLP）
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

# 将XGBoost预测作为额外输入特征（stacking）
train_df_stack = train_df.copy()
val_df_stack = val_df.copy()
test_df_stack = test_df.copy()

train_df_stack['xgb_pred'] = pred_resid_train_xgb
val_df_stack['xgb_pred'] = pred_resid_val_xgb
test_df_stack['xgb_pred'] = pred_resid_test_xgb

stack_cols = feature_cols + ['xgb_pred']

X_train_stack = torch.tensor(train_df_stack[stack_cols].values, dtype=torch.float32)
y_train_stack = torch.tensor(train_df_stack['stl_resid'].values, dtype=torch.float32).view(-1, 1)
X_val_stack = torch.tensor(val_df_stack[stack_cols].values, dtype=torch.float32)
y_val_stack = torch.tensor(val_df_stack['stl_resid'].values, dtype=torch.float32).view(-1, 1)
X_test_stack = torch.tensor(test_df_stack[stack_cols].values, dtype=torch.float32)
y_test_stack = torch.tensor(test_df_stack['stl_resid'].values, dtype=torch.float32).view(-1, 1)

train_ds = TensorDataset(X_train_stack, y_train_stack)
val_ds = TensorDataset(X_val_stack, y_val_stack)

train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=256, shuffle=False)

# 定义MLP
class ResidualCalibrator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)

in_dim = X_train_stack.shape[1]
model = ResidualCalibrator(in_dim)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
criterion = nn.MSELoss()

# 训练（简易早停）
best_val = np.inf
best_state = None
epochs = 50

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * xb.size(0)
    train_loss /= len(train_loader.dataset)
    # 验证
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            pred = model(xb)
            loss = criterion(pred, yb)
            val_loss += loss.item() * xb.size(0)
    val_loss /= len(val_loader.dataset)
    if val_loss < best_val:
        best_val = val_loss
        best_state = model.state_dict()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# 加载最优参数
model.load_state_dict(best_state)

model.eval()
with torch.no_grad():
    pred_resid_train_final = model(X_train_stack).numpy().flatten()
    pred_resid_val_final = model(X_val_stack).numpy().flatten()
    pred_resid_test_final = model(X_test_stack).numpy().flatten()

# 合成最终预测：y_hat = trend + season + resid_pred_final
# 注意：STL的trend和season是在df（含前段）上计算的，这里在各分段取对应列
def reconstruct_y(df_part, resid_pred):
    return df_part['stl_trend'].values + df_part['stl_season'].values + resid_pred

y_hat_train = reconstruct_y(train_df_stack, pred_resid_train_final)
y_hat_val = reconstruct_y(val_df_stack, pred_resid_val_final)
y_hat_test = reconstruct_y(test_df_stack, pred_resid_test_final)

# 评估指标
from sklearn.metrics import mean_squared_error, mean_absolute_error

def mape(y, yhat):
    return (np.mean(np.abs((y - yhat) / y)) * 100)

rmse_test = np.sqrt(mean_squared_error(test_df_stack['load'].values, y_hat_test))
mae_test = mean_absolute_error(test_df_stack['load'].values, y_hat_test)
mape_test = mape(test_df_stack['load'].values, y_hat_test)

print(f"Test RMSE: {rmse_test:.3f}, MAE: {mae_test:.3f}, MAPE: {mape_test:.2f}%")

# 可视化4：测试集真实 vs 预测
plt.figure(figsize=(14, 4))
plt.plot(test_df_stack['timestamp'], test_df_stack['load'], color='#FF1493', label='真实负荷 (DeepPink)', linewidth=1.5)
plt.plot(test_df_stack['timestamp'], y_hat_test, color='#00FF7F', label='预测负荷 (SpringGreen)', linewidth=1.2)
plt.title('图4：测试集真实 vs 预测（叠加展示）', fontsize=14)
plt.xlabel('时间'); plt.ylabel('负荷')
plt.legend(); plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# 可视化5：剩余项的诊断
plt.figure(figsize=(8, 5))
sns.histplot(y_test_resid - pred_resid_test_final, bins=50, kde=True, color='#FF4500')
plt.title('图5：测试集剩余项预测误差分布', fontsize=14)
plt.xlabel('误差'); plt.ylabel('频数')
plt.tight_layout()
plt.show()

# 可视化6：小时x周几的平均负荷热力图
pivot = df.pivot_table(index='hour', columns='dow', values='load', aggfunc='mean')
plt.figure(figsize=(10, 6))
sns.heatmap(pivot, cmap='rainbow')
plt.title('图6：小时-周几平均负荷热力图', fontsize=14)
plt.xlabel('周几（0=周一, 6=周日）'); plt.ylabel('小时')
plt.tight_layout()
plt.show()

# 可视化7：特征相关性热力图
corr_cols = ['load', 'temp', 'is_holiday', 'is_weekend', 'hour_sin', 'hour_cos', 'stl_resid',
             'resid_lag_1', 'resid_lag_24', 'resid_ma_24', 'resid_std_24']
corr = df_model[corr_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=False, cmap='Spectral')
plt.title('图7：关键特征相关性光谱图', fontsize=14)
plt.tight_layout()
plt.show()
核心步骤
数据模拟：

我们用日周期的正弦和余弦来刻画每日负荷规律，周效应用映射字典简单表达，温度用U型函数反映对负荷的提升（凉/热都提升电力需求），再叠加趋势和噪声，形成更贴近现实的序列。这样训练出来的模型能处理“有结构+有随机”的情形。

STL分解：

把序列拆分成趋势（长期变化）、季节性（稳定周期）、剩余项（杂波）。此时剩余项是我们要用XGBoost重点学习的对象，它浓缩了“难以用简单周期表达的外生影响”。

特征工程：

时段特征（hour、hour_sin/cos）帮助模型识别日内周期；

周几、节假日、是否周末代表制度性行为模式；

温度、温度滚动均值代表外部气象影响；

残差的滞后与滚动统计捕捉短时序相关性（例如1小时、24小时、168小时的滞后）。

XGBoost拟合剩余项：

避免直接建模整个负荷，减少难度和过拟合风险；

使用多树组合的非线性能力从复杂特征集中提取残差模式；

模型的特征重要性可用于解释哪些因素驱动了剩余波动。

PyTorch残差校准器（MLP）：

将XGBoost的预测作为输入之一，再用一个轻量神经网络对其进行“二次校正”；

相当于“stacking”或“后验校准”，可以减小系统性偏差；

PyTorch使我们可以灵活定义结构、正则、早停等训练流程。

合成最终预测：

使用  的思想，把趋势和季节性（来自STL）与校准后的残差预测合起来；

这样最终的预测既具备分解的稳健性，又吸收了机器学习模型的强非线性能力。

评估与可视化：

综合使用RMSE、MAE和MAPE体现尺度误差与相对误差；

使用多图分别展示原始序列、分解结果、特征重要性、预测效果对比、误差分布、热力图与相关性光谱等。

可视化分析
图1：模拟电力负荷时间序列

图片
展示负荷的整体走势，阴影标注测试集。

它帮助我们直观感受趋势上升、周期波动与噪声程度。

图2：STL分解三要素

三条彩色曲线分别显示趋势、季节性、剩余项。

图片
这图告诉我们：趋势非常平滑，季节性具有明显规律，剩余项是相对随机的部分。

图3：XGBoost特征重要性

条形图展示各特征对残差预测的贡献强弱。

图片
它是模型可解释性的关键图：让我们知道温度、滞后项、时段特征等谁更重要。

图4：测试集真实 vs 预测

图片
呈现预测与真实在测试集的贴合程度。可以直观检查是否存在系统性偏差、滞后问题以及峰谷跟随能力。

图5：测试集剩余项预测误差分布

图片
直方图+核密度展示误差的分布形态。若误差集中在零附近且长尾不明显，说明模型校准较好；若偏态明显，说明存在系统性偏差。

图6：小时-周几平均负荷热力图

图片
用矩阵热力图展示不同小时与不同周几负荷的平均水平，它帮助我们识别周内和日内模式的交互结构。

图7：关键特征相关性光谱图

图片
显示负荷、温度、节假日、残差滞后等特征之间的相关性网络。用于检查多重共线性与潜在的信息冗余，为后续特征简化提供依据。

一些细节注意点
分解方法选择：

本文使用STL，对季节性稳定且含噪声的序列表现优异。若季节性随时间变化较强，可以尝试变周期或多尺度分解。

XGBoost参数调优：

可以使用交叉验证、网格搜索或贝叶斯优化调参，如max_depth、learning_rate、subsample和colsample_bytree等，对偏差/方差折中非常关键。

残差校准器的作用：

不是必须，但在复杂场景下能进一步提升拟合质量。也可以替换为更强的序列模型（如TCN、LSTM）进行校准，但要注意训练稳定性与部署成本。

特征工程的深挖：

更多气象指标（湿度、风速）、假期类型、工作时段标签、突发事件标记、节能策略信息等，都可能显著提升效果。

评价与上线：

评估除了RMSE/MAE/MAPE，还可关注峰值误差、错峰惩罚、时段加权指标（比如白天权重大于夜间）。上线要考虑模型鲁棒性、解释性、更新频率与监控体系。

总结
这套“时间序列分解+XGBoost”的方法，本质上是“复杂问题分块化”的思路，用分解分清“结构化”和“非结构化”的部分；

把最难处理的剩余交给强非线性模型（XGBoost），再用PyTorch做轻量校准，把误差进一步压小；

它具有很强的工程可落地性、解释友好且性能稳健。在实际业务中，常能在不投入巨大算力与极深模型的情况下，拿到足够好的预测精度与稳定性，尤其对需要快速部署与可解释的场景非常适合~

