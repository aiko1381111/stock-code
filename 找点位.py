import baostock as bs
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta


def calculate_bollinger_bands(data, window=20, num_std=2):
    """
    计算布林带指标
    """
    df = data.copy()

    # 计算中轨（移动平均线）
    df['MA'] = df['close'].rolling(window=window).mean()

    # 计算标准差
    df['STD'] = df['close'].rolling(window=window).std()

    # 计算上轨和下轨
    df['BOLL_Upper'] = df['MA'] + num_std * df['STD']
    df['BOLL_Lower'] = df['MA'] - num_std * df['STD']

    # 计算当日涨跌幅
    if 'pctChg' not in df.columns:
        df['pctChg'] = df['close'].pct_change() * 100

    return df


def find_buy_signals(data):
    """
    找出符合买入条件的点位

    修正后的买入条件：
    1. 收盘价 <= 布林下轨 (100%)
    2. 单日跌幅 >= 1%
    """
    buy_signals = []

    for i in range(1, len(data)):
        current_row = data.iloc[i]

        # 检查是否有有效的布林带数据
        if pd.isna(current_row['BOLL_Lower']):
            continue

        # 检查买入条件
        condition1 = current_row['close'] <= current_row['BOLL_Lower']  # 收盘价低于布林下轨
        condition2 = current_row['pctChg'] <= -1  # 单日跌幅 >= 1%

        if condition1 and condition2:
            buy_signal = {
                'date': current_row['date'],
                'stock_code': current_row['code'],
                'buy_price': round(current_row['BOLL_Lower'], 2),  # 固定买入价：布林下轨的100%
                'close_price': round(current_row['close'], 2),  # 当日收盘价
                'boll_lower': round(current_row['BOLL_Lower'], 2),
                'pct_change': round(current_row['pctChg'], 2),
                'volume': current_row['volume'],
                'ma_price': round(current_row['MA'], 2)
            }
            buy_signals.append(buy_signal)

    return buy_signals


def calculate_holding_period(buy_signals, stock_data):
    """
    计算每个买点的持仓天数和卖出信息

    修正后的卖出逻辑：
    以未来某日的布林中轨的102%作为卖出价
    """
    signals_with_holding = []

    for buy_signal in buy_signals:
        buy_date = buy_signal['date']
        buy_index = stock_data[stock_data['date'] == buy_date].index

        if len(buy_index) == 0:
            continue

        buy_idx = buy_index[0]

        # 从买入点后开始寻找卖出点
        holding_days = 0
        sell_date = None
        sell_price = None
        sell_trigger_ma = None
        sell_reason = "未达到卖出条件"
        profit_pct = 0

        # 只查找买入点之后的数据
        for i in range(buy_idx + 1, len(stock_data)):
            current_row = stock_data.iloc[i]
            holding_days += 1

            # 修正卖出条件：最高价 >= 未来某日的布林中轨的102%
            future_sell_price = round(current_row['MA'] * 1.02, 2)
            if current_row['high'] >= future_sell_price:
                sell_date = current_row['date']
                sell_price = future_sell_price  # 卖出价：未来某日布林中轨的102%
                sell_trigger_ma = round(current_row['MA'], 2)
                sell_reason = "最高价触及未来布林中轨102%"
                profit_pct = round((sell_price - buy_signal['buy_price']) / buy_signal['buy_price'] * 100, 2)
                break

            # 可选：添加最大持仓天数限制
            if holding_days >= 250:  # 最多持有250个交易日（约1年）
                sell_date = current_row['date']
                sell_price = round(current_row['close'], 2)  # 最大持仓时用收盘价卖出
                sell_trigger_ma = round(current_row['MA'], 2)
                sell_reason = "最大持仓天数"
                profit_pct = round((sell_price - buy_signal['buy_price']) / buy_signal['buy_price'] * 100, 2)
                break

        # 如果没有卖出，使用最后一天的数据
        if sell_date is None and holding_days > 0:
            last_row = stock_data.iloc[-1]
            sell_date = last_row['date']
            sell_price = round(last_row['close'], 2)  # 持有至今用收盘价
            sell_trigger_ma = round(last_row['MA'], 2)
            sell_reason = "持有至今"
            profit_pct = round((sell_price - buy_signal['buy_price']) / buy_signal['buy_price'] * 100, 2)
            holding_days = len(stock_data) - buy_idx - 1

        # 更新买入信号信息
        buy_signal_with_holding = buy_signal.copy()
        buy_signal_with_holding.update({
            'holding_days': holding_days,
            'sell_date': sell_date,
            'sell_price': sell_price,
            'sell_trigger_ma': sell_trigger_ma,
            'sell_reason': sell_reason,
            'profit_pct': profit_pct,
            'status': '已卖出' if sell_reason != "持有至今" else '持有中',
            'max_price_reached': round(current_row['high'], 2) if sell_date else None
        })

        signals_with_holding.append(buy_signal_with_holding)

    return signals_with_holding


def download_stock_data(stock_code, years=5):
    """
    下载指定股票代码的日线数据
    """
    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    # 登陆Baostock系统
    lg = bs.login()
    print(f'登录响应: {lg.error_code} - {lg.error_msg}')

    try:
        # 查询历史K线数据
        rs = bs.query_history_k_data_plus(stock_code,
                                          "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg",
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag="2")

        print(f'数据查询响应: {rs.error_code} - {rs.error_msg}')

        if rs.error_code != '0':
            print("数据查询失败，请检查股票代码和网络连接")
            return None

        # 构建DataFrame
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())

        if not data_list:
            print("未获取到数据，请检查日期范围或股票代码")
            return None

        result = pd.DataFrame(data_list, columns=rs.fields)

        # 数据类型转换
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')

        # 日期格式转换
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
            result = result.sort_values('date')

        print(f"成功获取 {len(result)} 条日线数据")
        return result

    except Exception as e:
        print(f"数据获取过程中出现错误: {str(e)}")
        return None
    finally:
        bs.logout()


def analyze_stock_buy_signals(stock_code, stock_name, years=3):
    """
    分析单个股票的买入信号和持仓情况
    """
    print(f"\n🔍 正在分析 {stock_name} ({stock_code})...")

    # 下载数据
    stock_data = download_stock_data(stock_code, years=years)

    if stock_data is None or stock_data.empty:
        print(f"❌ 无法获取 {stock_name} 的数据")
        return None

    # 计算布林带指标
    stock_data_with_boll = calculate_bollinger_bands(stock_data)

    # 找出买入信号
    buy_signals = find_buy_signals(stock_data_with_boll)

    # 计算持仓信息
    if buy_signals:
        buy_signals_with_holding = calculate_holding_period(buy_signals, stock_data_with_boll)
    else:
        buy_signals_with_holding = []

    # 打印结果
    if buy_signals_with_holding:
        print(f"✅ 找到 {len(buy_signals_with_holding)} 个买入信号:")
        print("-" * 160)
        print(
            f"{'日期':<12} {'买入价':<8} {'收盘价':<8} {'布林下轨':<8} {'跌幅%':<6} {'持仓天数':<8} {'卖出价':<8} {'触发MA':<8} {'收益%':<8} {'状态':<8} {'卖出原因':<15}")
        print("-" * 160)

        for signal in buy_signals_with_holding:
            status_color = "✅" if signal['profit_pct'] > 0 else "❌"
            print(f"{signal['date'].strftime('%Y-%m-%d')} "
                  f"{signal['buy_price']:<8} "
                  f"{signal['close_price']:<8} "
                  f"{signal['boll_lower']:<8} "
                  f"{signal['pct_change']:<6} "
                  f"{signal['holding_days']:<8} "
                  f"{signal.get('sell_price', 'N/A'):<8} "
                  f"{signal.get('sell_trigger_ma', 'N/A'):<8} "
                  f"{status_color}{signal['profit_pct']:<7} "
                  f"{signal['status']:<8} "
                  f"{signal['sell_reason']:<15}")

            # 显示买卖点位关系
            print(
                f"          买入逻辑: 收盘价{signal['close_price']} ≤ 布林下轨{signal['boll_lower']} (买入价{signal['buy_price']})")
            if signal.get('sell_price'):
                print(
                    f"          卖出逻辑: 最高价触及未来布林中轨{signal['sell_trigger_ma']}的102% → 卖出价{signal['sell_price']}")
    else:
        print("❌ 未找到符合条件的买入信号")

    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'data': stock_data_with_boll,
        'buy_signals': buy_signals_with_holding
    }


def save_analysis_results(results, filename="/Users/mac/Desktop/股票买入信号分析.xlsx"):
    """
    保存分析结果到Excel文件
    """
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            for result in results:
                if result and result['data'] is not None:
                    stock_name = result['stock_name']
                    data = result['data'].copy()

                    # 添加中文表头
                    column_mapping = {
                        'date': '交易日期', 'code': '股票代码', 'open': '开盘价',
                        'high': '最高价', 'low': '最低价', 'close': '收盘价',
                        'volume': '成交量(股)', 'amount': '成交额(元)',
                        'adjustflag': '前复权', 'turn': '换手率(%)', 'pctChg': '涨跌幅(%)',
                        'MA': '布林中轨', 'BOLL_Upper': '布林上轨', 'BOLL_Lower': '布林下轨'
                    }
                    data = data.rename(columns=column_mapping)

                    # 标记买入信号
                    if result['buy_signals']:
                        buy_dates = [signal['date'] for signal in result['buy_signals']]
                        data['买入信号'] = data['交易日期'].isin(buy_dates)

                    # Sheet名称
                    sheet_name = f"{stock_name}_{result['stock_code']}"[:31]
                    data.to_excel(writer, sheet_name=sheet_name, index=False)

            # 保存交易信号汇总表
            all_signals = []
            for result in results:
                if result and result['buy_signals']:
                    for signal in result['buy_signals']:
                        signal_summary = {
                            '股票代码': result['stock_code'],
                            '股票名称': result['stock_name'],
                            '买入日期': signal['date'],
                            '买入价格': signal['buy_price'],
                            '当日收盘价': signal['close_price'],
                            '布林下轨': signal['boll_lower'],
                            '买入跌幅%': signal['pct_change'],
                            '持仓天数': signal['holding_days'],
                            '卖出日期': signal.get('sell_date'),
                            '卖出价格': signal.get('sell_price'),
                            '触发卖出MA': signal.get('sell_trigger_ma'),
                            '当日最高价': signal.get('max_price_reached'),
                            '收益率%': signal['profit_pct'],
                            '状态': signal['status'],
                            '卖出原因': signal['sell_reason']
                        }
                        all_signals.append(signal_summary)

            if all_signals:
                signals_df = pd.DataFrame(all_signals)
                signals_df.to_excel(writer, sheet_name='交易信号汇总', index=False)

        print(f"\n💾 分析结果已保存到: {filename}")
        return True
    except Exception as e:
        print(f"保存分析结果时出错: {str(e)}")
        return False


# 主程序
if __name__ == "__main__":
    # 定义要分析的股票列表
    stock_configs = [
        {"code": "sh.518880", "name": "黄金ETF"},
        {"code": "sz.000001", "name": "平安银行"},
        {"code": "sh.600519", "name": "贵州茅台"},
        {"code": "sz.300750", "name": "宁德时代"},
        {"code": "sh.600036", "name": "招商银行"},
        {"code": "sz.000858", "name": "五粮液"},
    ]

    print(f"{'🎯 开始分析股票买入信号 ':~^80}")
    print("买入条件：")
    print("1. 收盘价 ≤ 布林下轨 (100%)")
    print("2. 单日跌幅 ≥ 1%")
    print("卖出条件：")
    print("最高价 ≥ 未来某日布林中轨 × 1.02")
    print(f"{'~' * 80}")

    all_results = []

    for config in stock_configs:
        result = analyze_stock_buy_signals(
            stock_code=config["code"],
            stock_name=config["name"],
            years=3
        )
        all_results.append(result)

    # 保存分析结果
    save_analysis_results([r for r in all_results if r is not None])

    # 汇总统计
    valid_results = [r for r in all_results if r and r['buy_signals']]
    total_signals = sum(len(r['buy_signals']) for r in valid_results)

    if valid_results:
        # 计算收益统计
        all_profits = []
        for result in valid_results:
            for signal in result['buy_signals']:
                all_profits.append(signal['profit_pct'])

        profitable_trades = sum(1 for p in all_profits if p > 0)
        avg_profit = sum(all_profits) / len(all_profits) if all_profits else 0
        max_profit = max(all_profits) if all_profits else 0
        min_profit = min(all_profits) if all_profits else 0

        print(f"\n📊 分析完成！总共找到 {total_signals} 个买入信号")
        print(f"📈 盈利交易: {profitable_trades} 个 ({profitable_trades / total_signals * 100:.1f}%)")
        print(f"💰 平均收益率: {avg_profit:.2f}%")
        print(f"🎯 最高收益率: {max_profit:.2f}%")
        print(f"📉 最低收益率: {min_profit:.2f}%")

        # 显示每个股票的信号数量
        print(f"\n{'📈 各股票信号统计 ':~^50}")
        for result in valid_results:
            signal_count = len(result['buy_signals'])
            profits = [s['profit_pct'] for s in result['buy_signals']]
            avg_stock_profit = sum(profits) / len(profits) if profits else 0
            print(
                f"{result['stock_name']}({result['stock_code']}): {signal_count} 个信号, 平均收益: {avg_stock_profit:.2f}%")
    else:
        print(f"\n📊 分析完成！总共找到 {total_signals} 个买入信号")