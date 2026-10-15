import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题

if __name__ == '__main__':
    file_name = "data/43_origin_data.xlsx"

    quar_data = pd.read_excel(file_name, sheet_name=0)
    quar_date = pd.read_excel(file_name, sheet_name=1)
    month_data = pd.read_excel(file_name, sheet_name=2)
    month_date = pd.read_excel(file_name, sheet_name=3)
    week_data = pd.read_excel(file_name, sheet_name=4)
    daily_data = pd.read_excel(file_name, sheet_name=5)
    daily_data = daily_data.fillna(method='bfill')  ####读取各频率数据以及月频季频发布日期

    quar_data['Unnamed: 0'] = pd.to_datetime(quar_data['Unnamed: 0'])
    quar_data = quar_data.sort_values(by='Unnamed: 0', ascending=True)
    quar_data = quar_data.set_index('Unnamed: 0')

    df_standardized = pd.DataFrame(index=quar_data.index, columns=quar_data.columns)
    for i in range(1, 59):
        window_data = quar_data['中国:GDP:不变价:当季同比'].iloc[0:i + 1]
        median = window_data.median()
        iqr = window_data.quantile(0.75) - window_data.quantile(0.25)
        if iqr == 0:
            df_standardized.loc[quar_data.index[i], '中国:GDP:不变价:当季同比'] = np.nan
        else:
            df_standardized.loc[quar_data.index[i], '中国:GDP:不变价:当季同比'] = (quar_data['中国:GDP:不变价:当季同比'].iloc[
                                                                              i] - median) / iqr
    quar_data_std = df_standardized
    quar_date['Unnamed: 0'] = pd.to_datetime(quar_date['Unnamed: 0'])
    quar_data_std = quar_data_std.reset_index()
    quar_merge = pd.merge(quar_data_std, quar_date, on='Unnamed: 0', how='left')
    quar_merge = quar_merge.set_index('GDP同比')  #####季频数据标准化及发布日期拼接



    def rolling_robust_standardize(arr, window):
        arr = arr.astype(np.float64)

        window_view = np.lib.stride_tricks.sliding_window_view(arr, window_shape=window)

        rolling_median = np.median(window_view, axis=1)
        rolling_q1 = np.percentile(window_view, 25, axis=1)
        rolling_q3 = np.percentile(window_view, 75, axis=1)

        rolling_iqr = rolling_q3 - rolling_q1
        rolling_iqr[rolling_iqr == 0] = np.nan
        standardized_values = (arr[window - 1:] - rolling_median) / rolling_iqr
        result = np.full_like(arr, np.nan)
        result[window - 1:] = standardized_values

        return result

    def apply_rolling_robust_std_to_df(df_origin, window):
        df_origin['Unnamed: 0'] = pd.to_datetime(df_origin['Unnamed: 0'])
        df = df_origin.sort_values(by='Unnamed: 0', ascending=True)
        df = df.set_index('Unnamed: 0')
        result_df = pd.DataFrame(index=df.index)
        for column in df.columns:
            result_df[column] = rolling_robust_standardize(df[column].values, window)  #####标准化
        return result_df

    month_data_std = apply_rolling_robust_std_to_df(month_data, 12)
    month_date['Unnamed: 0'] = pd.to_datetime(month_date['Unnamed: 0'])
    month_data_std = month_data_std.reset_index()
    month_merge = pd.merge(month_data_std, month_date, on='Unnamed: 0', how='left')  #####月频数据标准化及发布日期拼接
    month_merge = month_merge.set_index('Unnamed: 0')

    week_data_std = apply_rolling_robust_std_to_df(week_data, 52)
    daily_data_std = apply_rolling_robust_std_to_df(daily_data, 252)  ###周频日频标准化b

    merge1 = pd.merge_asof(daily_data_std, week_data_std, left_index=True, right_index=True,
                           direction='forward')  ###拼接日频和周频
    merge2 = pd.merge_asof(merge1, quar_merge['中国:GDP:不变价:当季同比'], left_index=True, right_index=True,
                           direction='forward')  ###拼接季频数据

    merge3 = merge2
    month_merge.drop(month_merge.index[-1], inplace=True)

    for i in range(0, 11):
        print(i)
        right_key = month_merge.columns[i + 11]
        month_merge.iloc[:,i + 11] = pd.to_datetime(month_merge.iloc[:,i + 11])
        col_to_merge = month_merge.columns[i]
        if i ==3:
            print(1)
            pass
        merge3 = pd.merge_asof(
            merge3,
            month_merge[[right_key, col_to_merge]],
            left_index=True,
            right_on=right_key,
            direction='forward'
        ).drop(right_key, axis=1)  ####拼接月频数据
    spec_data = pd.read_excel(file_name, sheet_name=6)  ####特殊频率数据
    spec_data['A_modified'] = spec_data['逆回购利率:7天'].where(spec_data['逆回购利率:7天'] != spec_data['逆回购利率:7天'].shift(-1), 0)
    spec_data['B_modified'] = spec_data['存款准备金率:大型金融机构(变动公告日期)'].where(
        spec_data['存款准备金率:大型金融机构(变动公告日期)'] != spec_data['存款准备金率:大型金融机构(变动公告日期)'].shift(-1), 0)

    spec_data = spec_data.sort_values(by='Unnamed: 0')

    merge4 = pd.merge_asof(merge3, spec_data[['Unnamed: 0', 'A_modified', 'B_modified']], left_index=True,
                           right_on='Unnamed: 0', direction='forward').drop('Unnamed: 0', axis=1)  ##拼接特殊频率
    cate_data = pd.read_excel(file_name, sheet_name=7)  ###分类时代变量

    for i in range(1, 3):
        col_name = cate_data.columns[1]
        cate_data[f'{col_name}-{i}'] = cate_data[col_name] == i
        cate_data[f'{col_name}-{i}'] = cate_data[f'{col_name}-{i}'] * 1
    for i in range(1, 6):
        col_name = cate_data.columns[2]
        cate_data[f'{col_name}-{i}'] = cate_data[col_name] == i
        cate_data[f'{col_name}-{i}'] = cate_data[f'{col_name}-{i}'] * 1
    for i in range(1, 5):
        col_name = cate_data.columns[3]
        cate_data[f'{col_name}-{i}'] = cate_data[col_name] == i
        cate_data[f'{col_name}-{i}'] = cate_data[f'{col_name}-{i}'] * 1  #####处理分类变量布尔化
    cate_data = cate_data.drop(cate_data.columns[[1, 2, 3]], axis=1)
    cate_data = cate_data.sort_values(by='Unnamed: 0')

    merge5 = pd.merge_asof(merge4, cate_data, left_index=True, right_on='Unnamed: 0', direction='forward').drop(
        'Unnamed: 0', axis=1)  ####拼接布尔数据
    merge5_filled = merge5.fillna(method='ffill')
    merge5_filled_total = merge5_filled.fillna(99.00)  ###缺失值处理
    merge5_filled_total['date_list'] = merge5_filled_total.index.to_series().apply(lambda x: [x.year, x.month, x.day])

    merge5_filled_total.to_csv('data/43_data.csv')



    df = pd.read_csv('data/43_origin_rate.csv')  ####计算动量
    df.columns = ["time","rate"]
    short_windows = [1, 7]
    long_windows = [30, 60, 180]

    short_window_size = 252
    short_min_periods = 66

    long_window_size = 756
    long_min_periods = 188


    # 计算动量并进行鲁棒标准化
    def calculate_momentum(df, window, rolling_window_size, min_periods):
        momentum = df['rate'].diff(window)

        rolling_median = momentum.rolling(window=rolling_window_size, min_periods=min_periods).median()
        rolling_iqr = (
                momentum.rolling(window=rolling_window_size, min_periods=min_periods).quantile(0.75) -
                momentum.rolling(window=rolling_window_size, min_periods=min_periods).quantile(0.25)
        )

        robust_scaled_momentum = (momentum - rolling_median) / rolling_iqr
        return robust_scaled_momentum


    # 计算1天和7天的动量
    for window in short_windows:
        df[f'momentum_{window}_robust_scaled'] = calculate_momentum(df, window, short_window_size, short_min_periods)

    # 计算30天、60天和180天的动量
    for window in long_windows:
        df[f'momentum_{window}_robust_scaled'] = calculate_momentum(df, window, long_window_size, long_min_periods)

    df.to_csv('data/43_rate.csv')

    print(1)
