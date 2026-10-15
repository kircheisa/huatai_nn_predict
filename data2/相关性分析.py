from pathlib import Path
import numpy as np
import pandas as pd


def _normalize_column_name(name):
    return str(name).strip().replace(" ", "")


def _find_date_column(columns):
    normalized = {_normalize_column_name(col): col for col in columns}
    preferred = ["Unnamed:0", "日期", "date", "Date", "time", "Time"]
    for key in preferred:
        key_norm = _normalize_column_name(key)
        if key_norm in normalized:
            return normalized[key_norm]
    for col in columns:
        if "日期" in str(col) or "date" in str(col).lower() or "time" in str(col).lower():
            return col
    return None


def _score_column(col_name, keywords):
    text = _normalize_column_name(col_name)
    return sum(1 for kw in keywords if _normalize_column_name(kw) in text)


def _find_best_column(columns, keywords):
    scores = [(col, _score_column(col, keywords)) for col in columns]
    scores = [item for item in scores if item[1] > 0]
    if not scores:
        return None
    scores.sort(key=lambda x: x[1], reverse=True)
    return scores[0][0]


def _load_target_columns(excel_path):
    xls = pd.ExcelFile(excel_path, engine="openpyxl")
    index_keywords = ["上证综合指数", "上证综指", "上证指数", "000001.SH"]
    net_keywords = ["中国:公开市场操作:货币净投放", "货币净投放", "净投放"]
    best = None
    best_score = -1
    available_columns = {}

    for sheet in xls.sheet_names:
        df = pd.read_excel(excel_path, sheet_name=sheet, engine="openpyxl")
        available_columns[sheet] = df.columns.tolist()
        date_col = _find_date_column(df.columns)
        index_col = _find_best_column(df.columns, index_keywords)
        net_col = _find_best_column(df.columns, net_keywords)
        score = int(date_col is not None) + int(index_col is not None) + int(net_col is not None)
        if score > best_score:
            best_score = score
            best = (sheet, df, date_col, index_col, net_col)
        if score == 3:
            break

    if best is None:
        raise ValueError("Excel中未读取到任何可用数据表")

    sheet, df, date_col, index_col, net_col = best
    if date_col is None or index_col is None or net_col is None:
        raise ValueError(
            f"无法定位目标字段。当前最优sheet={sheet}，date={date_col}，index={index_col}，net={net_col}，"
            f"可用列={available_columns.get(sheet)}"
        )

    work = df[[date_col, index_col, net_col]].copy()
    work.columns = ["date", "sh_index", "net_injection"]
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["sh_index"] = pd.to_numeric(work["sh_index"], errors="coerce")
    work["net_injection"] = pd.to_numeric(work["net_injection"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values("date")
    work = work.drop_duplicates(subset=["date"], keep="last").set_index("date")
    return work


def _calc_correlation(x, y):
    pair = pd.concat([x.rename("x"), y.rename("y")], axis=1).dropna()
    if len(pair) < 3:
        return {"样本数": int(len(pair)), "Pearson": np.nan, "Spearman": np.nan}
    return {
        "样本数": int(len(pair)),
        "Pearson": float(pair["x"].corr(pair["y"], method="pearson")),
        "Spearman": float(pair["x"].corr(pair["y"], method="spearman")),
    }


def analyze_relationship(excel_path):
    data = _load_target_columns(excel_path)

    daily_index_change = data["sh_index"].pct_change()
    daily_net_injection = data["net_injection"]
    daily_result = _calc_correlation(daily_index_change, daily_net_injection)

    weekly_index = data["sh_index"].resample("W-FRI").last()
    weekly_index_change = weekly_index.pct_change()
    weekly_net_injection = data["net_injection"].resample("W-FRI").sum(min_count=1)
    weekly_result = _calc_correlation(weekly_index_change, weekly_net_injection)

    result_df = pd.DataFrame(
        [
            {"频率": "日度", **daily_result},
            {"频率": "周度", **weekly_result},
        ]
    )
    return data, result_df


def main():
    excel_path = Path(__file__).with_suffix(".xlsx")
    print("1")
    if not excel_path.exists():
        raise FileNotFoundError(f"未找到数据文件: {excel_path}")
    data, result_df = analyze_relationship(excel_path)
    print("匹配到的数据起止日期:", data.index.min(), "->", data.index.max())
    print("有效数据行数:", len(data))
    print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
