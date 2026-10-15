"""
华泰证券债券预测数据集预处理脚本

功能概述：
1. 读取多频率数据（季频、月频、周频、日频、特殊频率、分类变量）
2. 对各频率数据进行滚动鲁棒标准化（基于中位数和四分位距）
3. 按发布日期进行多频率数据合并
4. 处理分类变量（布尔化编码）
5. 计算利率数据的动量特征

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体，解决matplotlib中文显示问题
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


if __name__ == '__main__':
    # ==================== 第一部分：读取多频率数据 ====================
    file_name = "data/43_origin_data.xlsx"

    # 读取各频率数据（Excel不同sheet）
    # sheet0: 季频数据（如GDP同比）
    # sheet1: 季频数据发布日期
    # sheet2: 月频数据
    # sheet3: 月频数据发布日期
    # sheet4: 周频数据
    # sheet5: 日频数据
    quar_data = pd.read_excel(file_name, sheet_name=0)
    quar_date = pd.read_excel(file_name, sheet_name=1)
    month_data = pd.read_excel(file_name, sheet_name=2)
    month_date = pd.read_excel(file_name, sheet_name=3)
    week_data = pd.read_excel(file_name, sheet_name=4)
    daily_data = pd.read_excel(file_name, sheet_name=5)
    daily_data = daily_data.bfill()  # 日频数据缺失值后向填充

    # ==================== 第二部分：季频数据标准化处理 ====================
    # 将日期列转换为datetime格式
    quar_data['Unnamed: 0'] = pd.to_datetime(quar_data['Unnamed: 0'])
    print("quar_data['Unnamed: 0'] ", quar_data['Unnamed: 0'])

    # 按日期升序排序
    quar_data = quar_data.sort_values(by='Unnamed: 0', ascending=True)
    quar_data = quar_data.set_index('Unnamed: 0')
    print("quar_data.head() ", quar_data.head())

    # 创建标准化结果DataFrame，    # 创建一个空的DataFrame用于存放标准化后的季频数据

    df_standardized = pd.DataFrame(index=quar_data.index, columns=quar_data.columns)
    print("df_standardized.head() ", df_standardized.head())

    # 对GDP同比数据进行累积标准化（非滚动窗口）
    # 使用累积数据的中位数和四分位距进行标准化
    for i in range(1, 59):
        window_data = quar_data['中国:GDP:不变价:当季同比'].iloc[0:i + 1]
        median = window_data.median()
        iqr = window_data.quantile(0.75) - window_data.quantile(0.25)

        # 当四分位距为0时，标准化结果设为NaN
        if iqr == 0:
            df_standardized.loc[quar_data.index[i], '中国:GDP:不变价:当季同比'] = np.nan
        else:
            df_standardized.loc[quar_data.index[i], '中国:GDP:不变价:当季同比'] = (
                quar_data['中国:GDP:不变价:当季同比'].iloc[i] - median
            ) / iqr

    quar_data_std = df_standardized

    # 合并季频标准化数据与发布日期
    quar_date['Unnamed: 0'] = pd.to_datetime(quar_date['Unnamed: 0'])
    quar_data_std = quar_data_std.reset_index()
    quar_merge = pd.merge(quar_data_std, quar_date, on='Unnamed: 0', how='left')
    quar_merge['GDP同比'] = pd.to_datetime(quar_merge['GDP同比'])  # 转换为datetime类型
    quar_merge = quar_merge.set_index('GDP同比')  # 以发布日期作为索引

    # ==================== 第三部分：定义滚动鲁棒标准化函数 ====================
    def rolling_robust_standardize(arr, window):
        """
        滚动窗口鲁棒标准化函数

        使用中位数和四分位距(IQR)进行标准化，相比均值和标准差，
        对异常值更加鲁棒。

        参数：
            arr: 输入数组（一维）
            window: 滚动窗口大小

        返回：
            标准化后的数组，前window-1个元素为NaN
        """
        arr = arr.astype(np.float64)

        # 创建滑动窗口视图，避免循环提高效率
        window_view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)

        # 计算滚动中位数和四分位数
        rolling_median = np.median(window_view, axis=1)
        rolling_q1 = np.percentile(window_view, 25, axis=1)
        rolling_q3 = np.percentile(window_view, 75, axis=1)

        # 计算四分位距(IQR)
        rolling_iqr = rolling_q3 - rolling_q1
        rolling_iqr[rolling_iqr == 0] = np.nan  # 避免除零错误

        # 标准化：(值 - 中位数) / 四分位距
        standardized_values = (arr[window - 1:] - rolling_median) / rolling_iqr

        # 构建结果数组，前window-1个位置填充NaN
        result = np.full_like(arr, np.nan)
        result[window - 1:] = standardized_values

        return result

    def apply_rolling_robust_std_to_df(df_origin, window):
        """
        对DataFrame的所有列应用滚动鲁棒标准化

        参数：
            df_origin: 原始DataFrame，第一列为日期
            window: 滚动窗口大小

        返回：
            标准化后的DataFrame
        """
        df_origin['Unnamed: 0'] = pd.to_datetime(df_origin['Unnamed: 0'])
        df = df_origin.sort_values(by='Unnamed: 0', ascending=True)
        df = df.set_index('Unnamed: 0')

        result_df = pd.DataFrame(index=df.index)
        for column in df.columns:
            result_df[column] = rolling_robust_standardize(df[column].values, window)

        return result_df

    # ==================== 第四部分：月频数据标准化和合并 ====================
    # 月频数据使用12期（12个月）滚动窗口
    month_data_std = apply_rolling_robust_std_to_df(month_data, 12)
    month_date['Unnamed: 0'] = pd.to_datetime(month_date['Unnamed: 0'])
    month_data_std = month_data_std.reset_index()
    month_merge = pd.merge(month_data_std, month_date, on='Unnamed: 0', how='left')
    month_merge = month_merge.set_index('Unnamed: 0')

    # 预先将所有发布日期列转换为datetime类型（避免merge_asof类型不兼容）
    for col in month_merge.columns[11:]:
        month_merge[col] = pd.to_datetime(month_merge[col], errors='coerce')

    # ==================== 第五部分：周频和日频数据标准化 ====================
    # 周频数据使用52期（52周≈1年）滚动窗口
    week_data_std = apply_rolling_robust_std_to_df(week_data, 52)
    # 日频数据使用252期（252个交易日≈1年）滚动窗口
    daily_data_std = apply_rolling_robust_std_to_df(daily_data, 252)

    print("daily_data_std")
    print(daily_data_std.head())

    # ==================== 第六部分：多频率数据合并 ====================
    # 使用pd.merge_asof进行时间对齐合并
    # direction='forward'表示使用最近的未来值进行匹配

    # 第一步：合并日频和周频数据
    merge1 = pd.merge_asof(
        daily_data_std,
        week_data_std,
        left_index=True,
        right_index=True,
        direction='forward'
    )
    print("merge1")
    print(merge1.head())
    print("quar_merge['中国:GDP:不变价:当季同比']",quar_merge['中国:GDP:不变价:当季同比'])

    # 第二步：合并季频数据（GDP同比）
    merge2 = pd.merge_asof(
        merge1,
        quar_merge['中国:GDP:不变价:当季同比'],
        left_index=True,
        right_index=True,
        direction='forward'
    )
    print("merge2 ",merge2.head())
    # 第三步：合并月频数据（逐列合并）
    merge3 = merge2
    month_merge.drop(month_merge.index[-1], inplace=True)  # 删除最后一行

    # 月频数据有11个指标列和对应的11个发布日期列
    for i in range(0, 11):
        print(i)
        right_key = month_merge.columns[i + 11]  # 发布日期列
        month_merge.iloc[:, i + 11] = pd.to_datetime(month_merge.iloc[:, i + 11])
        col_to_merge = month_merge.columns[i]  # 数据列

        if i == 3:
            print(1)
            pass

        # 按发布日期进行时间对齐合并
        merge3 = pd.merge_asof(
            merge3,
            month_merge[[right_key, col_to_merge]],
            left_index=True,
            right_on=right_key,
            direction='forward'
        ).drop(right_key, axis=1)

    # ==================== 第七部分：特殊频率数据处理 ====================
    # 读取特殊频率数据（如逆回购利率、存款准备金率）
    spec_data = pd.read_excel(file_name, sheet_name=6)

    # 标记利率变动：当值与下一期相同时设为0，否则保留原值
    # 这样可以识别出利率发生变化的时点
    spec_data['A_modified'] = spec_data['逆回购利率:7天'].where(
        spec_data['逆回购利率:7天'] != spec_data['逆回购利率:7天'].shift(-1), 0
    )
    spec_data['B_modified'] = spec_data['存款准备金率:大型金融机构(变动公告日期)'].where(
        spec_data['存款准备金率:大型金融机构(变动公告日期)'] != spec_data['存款准备金率:大型金融机构(变动公告日期)'].shift(-1), 0
    )

    spec_data = spec_data.sort_values(by='Unnamed: 0')

    # 合并特殊频率数据
    merge4 = pd.merge_asof(
        merge3,
        spec_data[['Unnamed: 0', 'A_modified', 'B_modified']],
        left_index=True,
        right_on='Unnamed: 0',
        direction='forward'
    ).drop('Unnamed: 0', axis=1)

    # ==================== 第八部分：分类变量布尔化处理 ====================
    # 读取分类变量数据
    cate_data = pd.read_excel(file_name, sheet_name=7)

    # 对分类变量进行独热编码（One-Hot Encoding）
    # 第2列有3个类别，第3列有5个类别，第4列有4个类别
    for i in range(1, 3):
        col_name = cate_data.columns[1]
        cate_data[f'{col_name}-{i}'] = cate_data[col_name] == i
        cate_data[f'{col_name}-{i}'] = cate_data[f'{col_name}-{i}'] * 1  # 布尔值转0/1

    for i in range(1, 6):
        col_name = cate_data.columns[2]
        cate_data[f'{col_name}-{i}'] = cate_data[col_name] == i
        cate_data[f'{col_name}-{i}'] = cate_data[f'{col_name}-{i}'] * 1

    for i in range(1, 5):
        col_name = cate_data.columns[3]
        cate_data[f'{col_name}-{i}'] = cate_data[col_name] == i
        cate_data[f'{col_name}-{i}'] = cate_data[f'{col_name}-{i}'] * 1

    # 删除原始分类列，保留独热编码列
    cate_data = cate_data.drop(cate_data.columns[[1, 2, 3]], axis=1)
    cate_data = cate_data.sort_values(by='Unnamed: 0')

    # 合并分类变量数据
    merge5 = pd.merge_asof(
        merge4,
        cate_data,
        left_index=True,
        right_on='Unnamed: 0',
        direction='forward'
    ).drop('Unnamed: 0', axis=1)

    # ==================== 第九部分：缺失值处理和输出 ====================
    # 先用前向填充处理缺失值
    print("merge5",merge5.head())
    # merge5_filled = merge5.fillna(method='ffill')
    # print("merge5_filled",merge5_filled.head())
    merge5_filled = merge5.ffill()
    print("merge5_filled",merge5_filled.head())

    # 剩余缺失值用99.00填充（作为特殊标记）
    merge5_filled_total = merge5_filled.fillna(99.00)

    # 将日期索引转换为[年, 月, 日]列表格式
    merge5_filled_total['date_list'] = merge5_filled_total.index.to_series().apply(
        lambda x: [x.year, x.month, x.day]
    )

    # 输出处理后的数据集
    merge5_filled_total.to_csv('data/43_data.csv')

    # ==================== 第十部分：利率动量特征计算 ====================
    # 读取原始利率数据
    df = pd.read_csv('data/43_origin_rate.csv')
    df.columns = ["time", "rate"]

    # 定义短期和长期动量窗口
    short_windows = [1, 7]      # 短期：1天、7天
    long_windows = [30, 60, 180]  # 长期：30天、60天、180天

    # 定义滚动标准化的窗口参数
    short_window_size = 252     # 短期动量标准化窗口：252个交易日（约1年）
    short_min_periods = 66      # 最小观测期：66个交易日（约3个月）

    long_window_size = 756      # 长期动量标准化窗口：756个交易日（约3年）
    long_min_periods = 188      # 最小观测期：188个交易日（约9个月）

    def calculate_momentum(df, window, rolling_window_size, min_periods):
        """
        计算动量并进行鲁棒标准化

        动量 = 当前值 - window天前的值
        标准化使用滚动中位数和四分位距

        参数：
            df: 包含'rate'列的DataFrame
            window: 动量计算窗口（天数）
            rolling_window_size: 标准化滚动窗口大小
            min_periods: 最小观测期

        返回：
            标准化后的动量序列
        """
        # 计算动量（差分）
        momentum = df['rate'].diff(window)

        # 计算滚动中位数
        rolling_median = momentum.rolling(
            window=rolling_window_size, min_periods=min_periods
        ).median()

        # 计算滚动四分位距
        rolling_iqr = (
            momentum.rolling(window=rolling_window_size, min_periods=min_periods).quantile(0.75) -
            momentum.rolling(window=rolling_window_size, min_periods=min_periods).quantile(0.25)
        )

        # 鲁棒标准化
        robust_scaled_momentum = (momentum - rolling_median) / rolling_iqr
        return robust_scaled_momentum

    # 计算短期动量（1天、7天）
    for window in short_windows:
        df[f'momentum_{window}_robust_scaled'] = calculate_momentum(
            df, window, short_window_size, short_min_periods
        )

    # 计算长期动量（30天、60天、180天）
    for window in long_windows:
        df[f'momentum_{window}_robust_scaled'] = calculate_momentum(
            df, window, long_window_size, long_min_periods
        )

    # 输出动量特征数据
    df.to_csv('data/43_rate.csv')

    print(1)
