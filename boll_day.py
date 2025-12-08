import baostock as bs
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
import chinese_calendar as calendar



def is_trading_day(date):
    """判断是否为交易日"""
    # 周末不是交易日
    if date.weekday() >= 5:
        return False
    # 法定节假日不是交易日
    if not calendar.is_workday(date):
        return False
    return True

def get_calendar_days(start_date, end_date):
    """计算两个日期之间的自然日天数"""
    if start_date and end_date:
        return (end_date - start_date).days
    return 0

def calculate_annualized_return(profit_pct, calendar_days):
    """计算年化收益率"""
    if calendar_days <= 0:
        return 0
    # 年化收益率 = (1 + 总收益率) ^ (365 / 持仓自然日) - 1
    annualized_return = ((1 + profit_pct / 100) ** (365 / calendar_days) - 1) * 100
    return round(annualized_return, 2)

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

    买入条件：
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
            # 确保所有字段都有默认值
            buy_signal = {
                'date': current_row['date'],
                'stock_code': current_row.get('code', '未知代码'),
                'buy_price': round(current_row['BOLL_Lower'], 2) if not pd.isna(current_row['BOLL_Lower']) else 0,
                'close_price': round(current_row['close'], 2) if not pd.isna(current_row['close']) else 0,
                'boll_lower': round(current_row['BOLL_Lower'], 2) if not pd.isna(current_row['BOLL_Lower']) else 0,
                'pct_change': round(current_row['pctChg'], 2) if not pd.isna(current_row['pctChg']) else 0,
                'volume': current_row.get('volume', 0),
                'ma_price': round(current_row['MA'], 2) if not pd.isna(current_row['MA']) else 0
            }
            buy_signals.append(buy_signal)

    return buy_signals


def calculate_holding_period(buy_signals, stock_data):
    """
    计算每个买点的持仓天数和卖出信息
    """
    signals_with_holding = []

    for buy_signal in buy_signals:
        buy_date = buy_signal['date']
        buy_index = stock_data[stock_data['date'] == buy_date].index

        if len(buy_index) == 0:
            continue

        buy_idx = buy_index[0]
        cost_price = buy_signal['buy_price']

        # 初始化变量
        holding_days = 0
        sell_date = None
        sell_price = None
        sell_trigger_ma = None
        sell_reason = "未达到卖出条件"
        profit_pct = 0
        status = '持有中'

        # 只查找买入点之后的数据
        for i in range(buy_idx + 1, len(stock_data)):
            current_row = stock_data.iloc[i]
            holding_days += 1

            # 计算未来卖出价
            future_sell_price = round(current_row['MA'] * 1.02, 2)

            condition1 = current_row['high'] >= future_sell_price
            condition2 = future_sell_price > cost_price

            if condition1 and condition2:
                sell_date = current_row['date']  # 确保这是datetime对象
                sell_price = future_sell_price
                sell_trigger_ma = round(current_row['MA'], 2)
                sell_reason = "盈利卖出(>成本价)"
                profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)
                status = '已卖出'
                break

            elif condition1 and not condition2:
                if sell_date is None:
                    sell_reason = "触及卖出点但亏本，继续持有"

            # 最大持仓天数限制
            if holding_days >= 60:
                sell_date = current_row['date']  # 确保这是datetime对象
                if current_row['close'] > cost_price:
                    sell_price = round(current_row['close'], 2)
                    sell_reason = "最大持仓(盈利)"
                else:
                    sell_price = round(current_row['close'], 2)
                    sell_reason = "最大持仓(止损)"
                sell_trigger_ma = round(current_row['MA'], 2)
                profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)
                status = '已卖出'
                break

        # 如果没有卖出，使用最后一天的数据
        if sell_date is None and holding_days > 0:
            last_row = stock_data.iloc[-1]
            sell_date = last_row['date']  # 确保这是datetime对象
            if last_row['close'] > cost_price:
                sell_price = round(last_row['close'], 2)
                sell_reason = "最终盈利卖出"
                profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)
                status = '已卖出'
            else:
                sell_price = None
                sell_reason = "持有中(低于成本价)"
                profit_pct = round((last_row['close'] - cost_price) / cost_price * 100, 2)
                status = '持有中'
            sell_trigger_ma = round(last_row['MA'], 2)
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
            'status': status,
            'max_price_reached': round(current_row['high'], 2) if sell_date else None,
            'cost_price': cost_price
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


def analyze_stock_buy_signals(stock_code, stock_name, years=5):
    """
    分析单个股票的买入信号和持仓情况
    """
    print(f"\n🔍 正在分析 {stock_name} ({stock_code})...")

    # 下载数据
    stock_data = download_stock_data(stock_code, years=5)

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

    # 打印结果 - 使用最安全的字符串拼接方式
    if buy_signals_with_holding:
        print(f"✅ 找到 {len(buy_signals_with_holding)} 个买入信号:")
        print("-" * 160)
        print(
            f"{'日期':<12} {'成本价':<8} {'收盘价':<8} {'布林下轨':<8} {'跌幅%':<6} {'持仓天数':<8} {'卖出价':<8} {'触发MA':<8} {'收益%':<8} {'状态':<10} {'卖出原因':<20}")
        print("-" * 160)

        for signal in buy_signals_with_holding:
            # 不使用f-string，改用传统字符串拼接
            date_str = str(signal.get('date', '未知日期'))[:10]  # 只取前10个字符
            cost_price = str(signal.get('cost_price', '0'))
            close_price = str(signal.get('close_price', '0'))
            boll_lower = str(signal.get('boll_lower', '0'))
            pct_change = str(signal.get('pct_change', '0'))
            holding_days = str(signal.get('holding_days', '0'))
            sell_price_display = str(signal.get('sell_price', '持有中'))
            sell_trigger_ma = str(signal.get('sell_trigger_ma', 'N/A'))
            profit_pct = str(signal.get('profit_pct', '0'))
            status = str(signal.get('status', '未知'))
            sell_reason = str(signal.get('sell_reason', '未知原因'))

            status_color = "✅" if float(profit_pct or 0) > 0 else "❌"

            # 使用传统字符串格式化，避免f-string问题
            line = (
                    date_str.ljust(12) + " " +
                    cost_price.ljust(8) + " " +
                    close_price.ljust(8) + " " +
                    boll_lower.ljust(8) + " " +
                    pct_change.ljust(6) + " " +
                    holding_days.ljust(8) + " " +
                    sell_price_display.ljust(8) + " " +
                    sell_trigger_ma.ljust(8) + " " +
                    status_color + profit_pct.ljust(7) + " " +
                    status.ljust(10) + " " +
                    sell_reason.ljust(20)
            )
            print(line)

    else:
        print("❌ 未找到符合条件的买入信号")

    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'data': stock_data_with_boll,
        'buy_signals': buy_signals_with_holding
    }


def save_results_to_excel(all_results, filename=None):
    """
    将分析结果保存为Excel文件
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"股票买入信号分析_{timestamp}.xlsx"

    # 创建DataFrame列表
    all_signals = []

    for result in all_results:
        if result is not None and result['buy_signals']:
            for signal in result['buy_signals']:
                # 添加股票代码和名称到每个信号中
                signal_record = signal.copy()
                signal_record['stock_code'] = result['stock_code']
                signal_record['stock_name'] = result['stock_name']
                all_signals.append(signal_record)

    if not all_signals:
        print("❌ 没有找到任何买入信号，无法生成Excel文件")
        return

    # 转换为DataFrame
    df = pd.DataFrame(all_signals)

    # 重新排列列的顺序，让股票代码和名称在前面
    columns_order = ['stock_code', 'stock_name'] + [col for col in df.columns if
                                                    col not in ['stock_code', 'stock_name']]
    df = df[columns_order]

    # 保存为Excel文件
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 保存详细信号数据
            df.to_excel(writer, sheet_name='买入信号详情', index=False)

            # 创建汇总统计表
            summary_data = []
            for result in all_results:
                if result is not None:
                    total_signals = len(result['buy_signals'])
                    profitable_signals = len([s for s in result['buy_signals'] if s.get('profit_pct', 0) > 0])
                    success_rate = (profitable_signals / total_signals * 100) if total_signals > 0 else 0

                    summary_data.append({
                        '股票代码': result['stock_code'],
                        '股票名称': result['stock_name'],
                        '买入信号数量': total_signals,
                        '盈利信号数量': profitable_signals,
                        '成功率%': round(success_rate, 2)
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

        print(f"✅ 分析结果已保存到: {filename}")
        print(f"📊 共保存了 {len(all_signals)} 个买入信号")

    except Exception as e:
        print(f"❌ 保存Excel文件时出错: {str(e)}")

# 主程序和其他函数保持不变...
if __name__ == "__main__":
    # 定义要分析的股票列表
    stock_configs = [
        {"code": "sh.600406", "name": "国电南瑞"},
        {"code": "sh.603288", "name": "海天味业"},
        {"code": "sz.000333", "name": "美的"}
        #{"code": "sz.300015", "name": "爱尔眼科"},
        #{"code": "sz.300760", "name": "迈瑞医疗"},
    ]

    print(f"{'🎯 开始分析股票买入信号 ':~^80}")
    print("买入条件：")
    print("1. 收盘价 ≤ 布林下轨 (100%)")
    print("2. 单日跌幅 ≥ 1%")
    print("卖出条件：")
    print("1. 最高价 ≥ 未来某日布林中轨 × 1.02")
    print("2. 卖出价必须 > 成本价 (保本原则)")
    print(f"{'~' * 80}")

    all_results = []

    for config in stock_configs:
        result = analyze_stock_buy_signals(
            stock_code=config["code"],
            stock_name=config["name"],
            years=3
        )
        all_results.append(result)

    # 保存分析结果和汇总统计代码保持不变...
    # 保存结果到Excel
    save_results_to_excel(all_results, filename="/Users/mac/Desktop/分析.xlsx")

