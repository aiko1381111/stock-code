import requests
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

# ========== 你的配置 ==========
INFOWAY_API_KEY = "b89c9c13bda04c9686a1086ba20fe3ab-infoway"


# =============================

class InfowayMetalFetcher:
    """通过Infoway API获取贵金属历史数据"""

    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://data.infoway.io"
        self.headers = {
            "accept": "application/json",
            "apiKey": self.api_key
        }

    def fetch_historical_kline(self, symbol_code, kline_type=8, kline_num=500):
        """获取指定数量的历史K线数据"""
        url = f"{self.base_url}/common/batch_kline/{kline_type}/{kline_num}/{symbol_code}"

        try:
            print(f"正在请求 {symbol_code} 历史数据 (数量: {kline_num})...")
            response = requests.get(url, headers=self.headers, timeout=30)
            response.raise_for_status()
            data = response.json()

            if data.get("ret") == 200:
                for item in data.get("data", []):
                    if item.get("s") == symbol_code:
                        klines = item.get("respList", [])
                        if klines:
                            print(f"  成功获取 {len(klines)} 条K线数据")
                            return klines
                print(f"错误: 在返回数据中未找到代码为 {symbol_code} 的条目。")
            else:
                print(f"错误: API返回状态码非200。消息: {data.get('msg')}")

            return []

        except Exception as e:
            print(f"获取历史数据错误 ({symbol_code}): {e}")
            return []

    def parse_kline_data(self, klines, symbol_code):
        """解析K线数据为结构化格式"""
        parsed_data = []

        for kline in klines:
            try:
                # 解析时间戳（假设是秒级时间戳）
                timestamp = int(kline.get("t", 0))
                if timestamp == 0:
                    continue

                date_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")

                # 解析价格
                parsed_data.append({
                    "date": date_str,
                    "timestamp": timestamp,
                    "open": float(kline.get("o", 0)),
                    "high": float(kline.get("h", 0)),
                    "low": float(kline.get("l", 0)),
                    "close": float(kline.get("c", 0)),
                    "volume": float(kline.get("v", 0)),
                    "symbol": symbol_code
                })
            except (ValueError, TypeError) as e:
                print(f"解析K线数据出错: {e}, 数据: {kline}")
                continue

        return parsed_data


def calculate_daily_gold_silver_ratio(gold_data, silver_data):
    """计算每日金银比"""
    # 转换为DataFrame
    gold_df = pd.DataFrame(gold_data)
    silver_df = pd.DataFrame(silver_data)

    if gold_df.empty or silver_df.empty:
        print("错误: 黄金或白银数据为空")
        return pd.DataFrame()

    # 按日期合并数据
    merged_df = pd.merge(
        gold_df[["date", "open", "high", "low", "close"]].rename(
            columns={"open": "gold_open", "high": "gold_high",
                     "low": "gold_low", "close": "gold_close"}),
        silver_df[["date", "open", "high", "low", "close"]].rename(
            columns={"open": "silver_open", "high": "silver_high",
                     "low": "silver_low", "close": "silver_close"}),
        on="date",
        how="inner"
    )

    if merged_df.empty:
        print("错误: 黄金和白银数据没有匹配的日期")
        return pd.DataFrame()

    # 计算金银比
    merged_df["ratio_open"] = merged_df["gold_open"] / merged_df["silver_open"]
    merged_df["ratio_high"] = merged_df["gold_high"] / merged_df["silver_high"]
    merged_df["ratio_low"] = merged_df["gold_low"] / merged_df["silver_low"]
    merged_df["ratio_close"] = merged_df["gold_close"] / merged_df["silver_close"]
    merged_df["ratio_avg"] = (merged_df["ratio_high"] + merged_df["ratio_low"]) / 2

    # 按日期排序
    merged_df = merged_df.sort_values("date")

    # 计算每日波动
    merged_df["ratio_range"] = merged_df["ratio_high"] - merged_df["ratio_low"]
    merged_df["daily_change"] = merged_df["ratio_close"] - merged_df["ratio_open"]
    merged_df["change_pct"] = (merged_df["daily_change"] / merged_df["ratio_open"]) * 100

    return merged_df


def generate_statistics_report(ratio_df):
    """生成统计报告"""
    if ratio_df.empty:
        return {}

    # 整体统计
    overall_stats = {
        "period_start": ratio_df["date"].iloc[0],
        "period_end": ratio_df["date"].iloc[-1],
        "total_days": len(ratio_df),
        "avg_ratio": ratio_df["ratio_close"].mean(),
        "median_ratio": ratio_df["ratio_close"].median(),
        "max_ratio": ratio_df["ratio_close"].max(),
        "min_ratio": ratio_df["ratio_close"].min(),
        "std_ratio": ratio_df["ratio_close"].std(),
        "avg_daily_range": ratio_df["ratio_range"].mean(),
        "avg_daily_change_pct": ratio_df["change_pct"].mean()
    }

    # 按年统计
    ratio_df["year"] = pd.to_datetime(ratio_df["date"]).dt.year
    yearly_stats = {}

    for year, group in ratio_df.groupby("year"):
        yearly_stats[year] = {
            "avg_ratio": group["ratio_close"].mean(),
            "max_ratio": group["ratio_close"].max(),
            "min_ratio": group["ratio_close"].min(),
            "volatility": group["ratio_close"].std()
        }

    return {
        "overall": overall_stats,
        "yearly": yearly_stats,
        "detailed_data": ratio_df
    }


def save_results(ratio_df, stats, output_format="excel"):
    """保存结果到文件"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 保存详细数据
    detailed_df = ratio_df[[
        "date", "ratio_open", "ratio_high", "ratio_low",
        "ratio_close", "ratio_avg", "ratio_range",
        "daily_change", "change_pct"
    ]].copy()

    if output_format == "excel":
        # 保存到Excel文件（包含多个sheet）
        with pd.ExcelWriter(f'gold_silver_ratio_3year_{timestamp}.xlsx', engine='openpyxl') as writer:
            # Sheet 1: 详细数据
            detailed_df.to_excel(writer, sheet_name='每日金银比', index=False)

            # Sheet 2: 月度统计
            monthly_stats = detailed_df.copy()
            monthly_stats["month"] = pd.to_datetime(monthly_stats["date"]).dt.to_period("M")
            monthly_agg = monthly_stats.groupby("month").agg({
                "ratio_open": "first",
                "ratio_high": "max",
                "ratio_low": "min",
                "ratio_close": "last",
                "ratio_avg": "mean",
                "ratio_range": "mean"
            }).reset_index()
            monthly_agg["month"] = monthly_agg["month"].astype(str)
            monthly_agg.to_excel(writer, sheet_name='月度统计', index=False)

            # Sheet 3: 年度统计
            yearly_data = []
            for year, year_stats in stats["yearly"].items():
                yearly_data.append({
                    "年份": year,
                    "平均比例": year_stats["avg_ratio"],
                    "最高比例": year_stats["max_ratio"],
                    "最低比例": year_stats["min_ratio"],
                    "波动率": year_stats["volatility"]
                })
            pd.DataFrame(yearly_data).to_excel(writer, sheet_name='年度统计', index=False)

        print(f"💾 数据已保存至: gold_silver_ratio_3year_{timestamp}.xlsx (Excel格式)")

    else:
        # 保存到CSV
        detailed_df.to_csv(f'gold_silver_ratio_3year_{timestamp}.csv', index=False, encoding='utf-8-sig')

        # 保存统计摘要
        with open(f'gold_silver_stats_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write("金银比例三年统计分析报告\n")
            f.write("=" * 60 + "\n")
            f.write(f"分析期间: {stats['overall']['period_start']} 至 {stats['overall']['period_end']}\n")
            f.write(f"总天数: {stats['overall']['total_days']}\n")
            f.write(f"平均比例: {stats['overall']['avg_ratio']:.2f}\n")
            f.write(f"比例范围: {stats['overall']['min_ratio']:.2f} - {stats['overall']['max_ratio']:.2f}\n")
            f.write(f"标准差: {stats['overall']['std_ratio']:.2f}\n")
            f.write(f"平均每日波动范围: {stats['overall']['avg_daily_range']:.2f}\n\n")

            f.write("年度统计:\n")
            for year, year_stats in stats["yearly"].items():
                f.write(f"  {year}年: 平均={year_stats['avg_ratio']:.2f}, "
                        f"范围={year_stats['min_ratio']:.2f}-{year_stats['max_ratio']:.2f}, "
                        f"波动率={year_stats['volatility']:.2f}\n")

        print(f"💾 数据已保存至:")
        print(f"  详细数据: gold_silver_ratio_3year_{timestamp}.csv")
        print(f"  统计报告: gold_silver_stats_{timestamp}.txt")


def print_summary_report(stats):
    """打印摘要报告"""
    print("\n" + "=" * 80)
    print("金银比例三年波动分析报告")
    print("=" * 80)

    overall = stats["overall"]
    print(f"📅 分析期间: {overall['period_start']} 至 {overall['period_end']}")
    print(f"📊 总交易日数: {overall['total_days']}")
    print(f"📈 平均金银比: {overall['avg_ratio']:.2f}")
    print(f"📉 比例范围: {overall['min_ratio']:.2f} - {overall['max_ratio']:.2f}")
    print(f"📊 标准差: {overall['std_ratio']:.2f}")
    print(f"📈 平均每日波动幅度: {overall['avg_daily_range']:.2f}")
    print(f"📉 平均日涨跌幅: {overall['avg_daily_change_pct']:.2f}%")

    print("\n📅 年度统计:")
    for year, year_stats in sorted(stats["yearly"].items()):
        print(f"  {year}年: 平均={year_stats['avg_ratio']:.2f}, "
              f"范围={year_stats['min_ratio']:.2f}-{year_stats['max_ratio']:.2f}, "
              f"波动={year_stats['volatility']:.2f}")

    print("\n💡 分析解读:")
    if overall['avg_ratio'] > 80:
        print("  * 三年平均比例 > 80: 期间白银总体相对便宜")
    elif overall['avg_ratio'] > 60:
        print("  * 三年平均比例 60-80: 期间金银比处于中性区间")
    else:
        print("  * 三年平均比例 < 60: 期间黄金总体相对便宜")

    print("=" * 80)


# ========== 主程序 ==========
def main():
    print("开始计算最近3年每日金银比例波动...")
    print("=" * 60)

    # 1. 初始化数据获取器
    fetcher = InfowayMetalFetcher(INFOWAY_API_KEY)

    # 2. 计算需要获取的数据量（3年 ≈ 750个交易日，分批获取）
    # 注意：Infoway API可能对单次请求的kline_num有限制，这里分批获取
    total_days_needed = 750  # 3年约750个交易日
    batch_size = 500  # 单次最大获取数量（根据API限制调整）
    num_batches = (total_days_needed + batch_size - 1) // batch_size

    print(f"计划获取最近3年数据，约{total_days_needed}个交易日")
    print(f"分{num_batches}批获取，每批最多{batch_size}条数据")

    # 3. 获取黄金历史数据
    print("\n获取黄金历史数据...")
    all_gold_klines = []
    for batch in range(num_batches):
        gold_klines = fetcher.fetch_historical_kline("XAUUSD", kline_num=batch_size)
        all_gold_klines.extend(gold_klines)
        if len(gold_klines) < batch_size:  # 如果返回数据少于请求，说明已获取完所有数据
            break
        time.sleep(1)  # 避免请求过于频繁

    gold_data = fetcher.parse_kline_data(all_gold_klines, "XAUUSD")
    print(f"黄金数据解析完成，共 {len(gold_data)} 条记录")

    # 4. 获取白银历史数据
    print("\n获取白银历史数据...")
    all_silver_klines = []
    for batch in range(num_batches):
        silver_klines = fetcher.fetch_historical_kline("XAGUSD", kline_num=batch_size)
        all_silver_klines.extend(silver_klines)
        if len(silver_klines) < batch_size:
            break
        time.sleep(1)

    silver_data = fetcher.parse_kline_data(all_silver_klines, "XAGUSD")
    print(f"白银数据解析完成，共 {len(silver_data)} 条记录")

    # 5. 计算每日金银比
    print("\n计算每日金银比例...")
    ratio_df = calculate_daily_gold_silver_ratio(gold_data, silver_data)

    if ratio_df.empty:
        print("错误: 无法计算金银比例")
        return

    print(f"金银比计算完成，共 {len(ratio_df)} 个交易日数据")

    # 6. 生成统计报告
    stats = generate_statistics_report(ratio_df)

    # 7. 打印摘要报告
    print_summary_report(stats)

    # 8. 保存结果
    save_results(ratio_df, stats, output_format="excel")

    # 9. 显示数据预览
    print("\n📋 数据预览（最近10个交易日）:")
    preview_df = ratio_df[["date", "ratio_open", "ratio_high", "ratio_low",
                           "ratio_close", "ratio_avg", "ratio_range", "change_pct"]].tail(10)
    print(preview_df.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

    print(f"\n✅ 分析完成！详细数据已保存到Excel文件。")


if __name__ == "__main__":
    main()