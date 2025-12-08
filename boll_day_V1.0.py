import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
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



def calculate_bollinger_bands_with_volume(data, window=20, num_std=2, volume_multiplier=1.3):
    """
    计算布林带指标并分析成交量
    """
    df = data.copy()

    # 计算中轨（20日移动平均线）
    df['MA'] = df['close'].rolling(window=window).mean()

    # 计算标准差和布林带
    df['STD'] = df['close'].rolling(window=window).std()
    df['BOLL_Upper'] = df['MA'] + num_std * df['STD']
    df['BOLL_Lower'] = df['MA'] - num_std * df['STD']

    # 计算当日涨跌幅
    if 'pctChg' not in df.columns:
        df['pctChg'] = df['close'].pct_change() * 100

    # 计算成交量相关指标
    df['volume_ma'] = df['volume'].rolling(window=5).mean()  # 5日平均成交量
    df['volume_ratio'] = df['volume'] / df['volume_ma']  # 成交量比率

    # 判断中轨方向（今日MA > 昨日MA 表示向上）
    df['ma_direction'] = df['MA'] > df['MA'].shift(1)

    return df


def find_buy_signals_optimized(data, volume_multiplier=1.2):
    """
    找出符合优化买入条件的点位

    买入条件：
    1. 收盘价 ≤ 布林下轨
    2. 单日跌幅 ≥ 1%
    3. 布林中轨方向向上（今日MA > 昨日MA）
    4. 成交量显著放大（当日成交量 > 5日均量 × volume_multiplier）
    """
    buy_signals = []

    for i in range(1, len(data)):
        current_row = data.iloc[i]
        prev_row = data.iloc[i - 1] if i > 0 else None

        # 检查是否有有效的布林带数据
        if pd.isna(current_row['BOLL_Lower']) or pd.isna(current_row['MA']):
            continue

        # 检查买入条件
        condition1 = current_row['close'] <= current_row['BOLL_Lower']  # 收盘价低于布林下轨
        condition2 = current_row['pctChg'] <= -1  # 单日跌幅 ≥ 1%
        condition3 = current_row['ma_direction']  # 布林中轨方向向上

        # 成交量显著放大（当日成交量大于5日均量的volume_multiplier倍）
        condition4 = current_row['volume_ratio'] >= volume_multiplier if not pd.isna(
            current_row['volume_ratio']) else False

        if condition1 and condition2 and condition4:
            # 记录买入时的中轨价格，用于后续止盈计算
            ma_at_buy = current_row['MA']

            buy_signal = {
                'date': current_row['date'],
                'stock_code': current_row.get('code', '未知代码'),
                'buy_price': round(current_row['close'], 2),
                'close_price': round(current_row['close'], 2),
                'boll_lower': round(current_row['BOLL_Lower'], 2),
                'ma_at_buy': round(ma_at_buy, 2),  # 记录买入时的中轨价格
                'pct_change': round(current_row['pctChg'], 2),
                'volume': current_row.get('volume', 0),
                'volume_ratio': round(current_row['volume_ratio'], 2) if not pd.isna(
                    current_row['volume_ratio']) else 0,
                'current_ma': round(current_row['MA'], 2),
                'ma_direction': '向上' if current_row['ma_direction'] else '向下'
            }
            buy_signals.append(buy_signal)

    return buy_signals


def calculate_holding_period_optimized(buy_signals, stock_data, take_profit_ratio=1.10, stop_loss_ratio=0.95):
    """
    计算每个买点的持仓天数和卖出信息（优化版）

    卖出条件：
    1. 止盈：实时价 ≥ 买入时布林中轨价格 × take_profit_ratio (默认110%)
    2. 止损：实时价 ≤ 买入价 × stop_loss_ratio (默认95%)
    """
    signals_with_holding = []

    for buy_signal in buy_signals:
        buy_date = buy_signal['date']
        buy_index = stock_data[stock_data['date'] == buy_date].index

        if len(buy_index) == 0:
            continue

        buy_idx = buy_index[0]
        cost_price = buy_signal['buy_price']
        ma_at_buy = buy_signal['ma_at_buy']

        # 计算止盈价和止损价
        take_profit_price = round(ma_at_buy * take_profit_ratio, 2)
        stop_loss_price = round(cost_price * stop_loss_ratio, 2)

        # 初始化变量
        holding_days = 0
        sell_date = None
        sell_price = None
        sell_reason = "未达到卖出条件"
        profit_pct = 0
        status = '持有中'

        # 查找买入点之后的数据
        for i in range(buy_idx + 1, len(stock_data)):
            current_row = stock_data.iloc[i]
            current_price = current_row['close']
            holding_days += 1

            # 检查止盈条件
            if current_price >= take_profit_price:
                sell_date = current_row['date']
                sell_price = take_profit_price
                sell_reason = f"止盈触发(≥买入时中轨×{take_profit_ratio})"
                profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)
                status = '已卖出'
                break

            # 检查止损条件
            elif current_price <= stop_loss_price:
                sell_date = current_row['date']
                sell_price = current_price  # 按实际价格卖出
                sell_reason = f"止损触发(≤买入价×{stop_loss_ratio})"
                profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)
                status = '已卖出'
                break

            # 最大持仓天数限制（90天）
            if holding_days >= 90:
                sell_date = current_row['date']
                sell_price = current_price
                sell_reason = "最大持仓天数(90天)"
                profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)
                status = '已卖出'
                break

        # 如果没有卖出，使用最后一天的数据
        if sell_date is None and holding_days > 0:
            last_row = stock_data.iloc[-1]
            sell_date = last_row['date']
            sell_price = last_row['close']
            profit_pct = round((sell_price - cost_price) / cost_price * 100, 2)

            if sell_price >= take_profit_price:
                sell_reason = "最终止盈"
            elif sell_price <= stop_loss_price:
                sell_reason = "最终止损"
            else:
                sell_reason = "最终平仓"
            status = '已卖出'

        # 更新买入信号信息
        buy_signal_with_holding = buy_signal.copy()
        buy_signal_with_holding.update({
            'holding_days': holding_days,
            'sell_date': sell_date,
            'sell_price': sell_price,
            'take_profit_price': take_profit_price,
            'stop_loss_price': stop_loss_price,
            'sell_reason': sell_reason,
            'profit_pct': profit_pct,
            'status': status,
            'max_price_reached': round(current_row['high'], 2) if sell_date else None,
            'min_price_reached': round(current_row['low'], 2) if sell_date else None,
            'cost_price': cost_price
        })

        signals_with_holding.append(buy_signal_with_holding)

    return signals_with_holding

def save_results_to_excel(all_results, filename=None):
    """
    将分析结果保存为Excel文件
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"股票周线买入信号分析_{timestamp}.xlsx"

    # 创建DataFrame列表
    all_signals = []

    for result in all_results:
        if result is not None and result['buy_signals']:
            for signal in result['buy_signals']:
                # 添加股票代码和名称到每个信号中
                signal_record = signal.copy()
                signal_record['stock_code'] = result['stock_code']
                signal_record['stock_name'] = result['stock_name']

                # 确保卖出日期是字符串格式，避免Excel保存问题
                if 'sell_date' in signal_record and signal_record['sell_date'] is not None:
                    if isinstance(signal_record['sell_date'], (pd.Timestamp, datetime)):
                        signal_record['sell_date'] = signal_record['sell_date'].strftime('%Y-%m-%d')
                else:
                    signal_record['sell_date'] = "尚未卖出"

                all_signals.append(signal_record)

    if not all_signals:
        print("❌ 没有找到任何买入信号，无法生成Excel文件")
        return

    # 转换为DataFrame
    df = pd.DataFrame(all_signals)

    # 重新排列列的顺序，让关键信息在前面
    preferred_columns = ['stock_code', 'stock_name', 'date', 'sell_date', 'cost_price', 'sell_price',
                         'profit_pct', 'holding_weeks', 'status', 'sell_reason']

    # 构建最终的列顺序
    final_columns = []
    for col in preferred_columns:
        if col in df.columns:
            final_columns.append(col)

    # 添加其他列
    for col in df.columns:
        if col not in final_columns:
            final_columns.append(col)

    df = df[final_columns]

    # 重命名列名为中文，便于阅读
    column_mapping = {
        'stock_code': '股票代码',
        'stock_name': '股票名称',
        'date': '买入日期',
        'sell_date': '卖出日期',
        'cost_price': '成本价',
        'close_price': '买入收盘价',
        'boll_lower': '布林下轨',
        'pct_change': '买入跌幅%',
        'holding_weeks': '持仓周数',
        'sell_price': '卖出价',
        'sell_trigger_ma': '触发MA',
        'profit_pct': '收益率%',
        'status': '状态',
        'sell_reason': '卖出原因',
        'ma_price': 'MA价格',
        'volume': '成交量',
        'data_type': '数据类型'
    }

    df = df.rename(columns=column_mapping)

    # 保存为Excel文件
    try:
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 保存详细信号数据
            df.to_excel(writer, sheet_name='周线买入信号详情', index=False)

            # 创建汇总统计表
            summary_data = []
            for result in all_results:
                if result is not None and result['buy_signals']:
                    total_signals = len(result['buy_signals'])
                    profitable_signals = len([s for s in result['buy_signals'] if s.get('profit_pct', 0) > 0])
                    success_rate = (profitable_signals / total_signals * 100) if total_signals > 0 else 0

                    # 计算平均持仓周数
                    avg_holding_weeks = np.mean([s.get('holding_weeks', 0) for s in result['buy_signals']])

                    # 计算平均收益率
                    avg_profit_pct = np.mean([s.get('profit_pct', 0) for s in result['buy_signals']])

                    summary_data.append({
                        '股票代码': result['stock_code'],
                        '股票名称': result['stock_name'],
                        '买入信号数量': total_signals,
                        '盈利信号数量': profitable_signals,
                        '成功率%': round(success_rate, 2),
                        '平均持仓周数': round(avg_holding_weeks, 1),
                        '平均收益率%': round(avg_profit_pct, 2)
                    })

            if summary_data:
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='汇总统计', index=False)

        print(f"✅ 分析结果已保存到: {filename}")
        print(f"📊 共保存了 {len(all_signals)} 个周线买入信号")

    except Exception as e:
        print(f"❌ 保存Excel文件时出错: {str(e)}")


def analyze_stock_buy_signals_optimized(stock_code, stock_name, years=3, volume_multiplier=1.3):
    """
    分析单个股票的买入信号和持仓情况（优化版）
    """
    print(f"\n🔍 正在分析 {stock_name} ({stock_code})...")

    # 下载数据（使用原有函数）
    stock_data = download_stock_data(stock_code, years=years)

    if stock_data is None or stock_data.empty:
        print(f"❌ 无法获取 {stock_name} 的数据")
        return None

    # 计算布林带指标（优化版）
    stock_data_with_boll = calculate_bollinger_bands_with_volume(stock_data, volume_multiplier=volume_multiplier)

    # 找出买入信号（优化版）
    buy_signals = find_buy_signals_optimized(stock_data_with_boll, volume_multiplier=volume_multiplier)

    # 计算持仓信息（优化版）
    if buy_signals:
        buy_signals_with_holding = calculate_holding_period_optimized(buy_signals, stock_data_with_boll)
    else:
        buy_signals_with_holding = []

    # 打印结果
    if buy_signals_with_holding:
        print(f"✅ 找到 {len(buy_signals_with_holding)} 个买入信号:")
        print("-" * 180)
        print(
            f"{'日期':<12} {'成本价':<8} {'买入MA':<8} {'止盈价':<8} {'止损价':<8} {'成交量比':<8} {'MA方向':<8} {'持仓天数':<8} {'卖出价':<8} {'收益%':<10} {'状态':<10} {'卖出原因':<20}")
        print("-" * 180)

        for signal in buy_signals_with_holding:
            date_str = str(signal.get('date', '未知日期'))[:10]
            cost_price = str(signal.get('cost_price', '0'))
            ma_at_buy = str(signal.get('ma_at_buy', '0'))
            take_profit = str(signal.get('take_profit_price', '0'))
            stop_loss = str(signal.get('stop_loss_price', '0'))
            volume_ratio = str(signal.get('volume_ratio', '0'))
            ma_direction = str(signal.get('ma_direction', '未知'))
            holding_days = str(signal.get('holding_days', '0'))
            sell_price_display = str(signal.get('sell_price', '持有中'))
            profit_pct = signal.get('profit_pct', 0)
            status = str(signal.get('status', '未知'))
            sell_reason = str(signal.get('sell_reason', '未知原因'))

            # 收益着色
            if profit_pct > 0:
                profit_str = f"✅+{profit_pct:.2f}%"
            elif profit_pct < 0:
                profit_str = f"❌{profit_pct:.2f}%"
            else:
                profit_str = f"{profit_pct:.2f}%"

            line = (f"{date_str:<12} {cost_price:<8} {ma_at_buy:<8} {take_profit:<8} {stop_loss:<8} "
                    f"{volume_ratio:<8} {ma_direction:<8} {holding_days:<8} {sell_price_display:<8} "
                    f"{profit_str:<10} {status:<10} {sell_reason:<20}")
            print(line)

        # 计算统计信息
        total_signals = len(buy_signals_with_holding)
        profitable_signals = len([s for s in buy_signals_with_holding if s.get('profit_pct', 0) > 0])
        stop_loss_signals = len([s for s in buy_signals_with_holding if "止损" in s.get('sell_reason', '')])
        take_profit_signals = len([s for s in buy_signals_with_holding if "止盈" in s.get('sell_reason', '')])
        max_holding_signals = len([s for s in buy_signals_with_holding if "最大持仓" in s.get('sell_reason', '')])

        success_rate = (profitable_signals / total_signals * 100) if total_signals > 0 else 0
        avg_profit = np.mean([s.get('profit_pct', 0) for s in buy_signals_with_holding]) if total_signals > 0 else 0
        avg_holding_days = np.mean(
            [s.get('holding_days', 0) for s in buy_signals_with_holding]) if total_signals > 0 else 0

        print(f"\n📊 统计摘要:")
        print(f"   总信号数: {total_signals}")
        print(f"   盈利信号数: {profitable_signals} (成功率: {success_rate:.2f}%)")
        print(f"   平均收益率: {avg_profit:.2f}%")
        print(f"   平均持仓天数: {avg_holding_days:.1f}天")
        print(
            f"   止盈触发: {take_profit_signals}次, 止损触发: {stop_loss_signals}次, 最大持仓触发: {max_holding_signals}次")
    else:
        print("❌ 未找到符合条件的买入信号")

    return {
        'stock_code': stock_code,
        'stock_name': stock_name,
        'data': stock_data_with_boll,
        'buy_signals': buy_signals_with_holding
    }


# 主程序
if __name__ == "__main__":
    # 定义要分析的股票列表
    stock_configs = [
       ## {"code": "sh.600406", "name": "国电南瑞"},
       ## {"code": "sh.600900", "name": "长江电力"},
       ## {"code": "sh.600900", "name": "三一重工"},

        {"code": "sh.600941", "name": "中国移动"},
        {"code": "sz.000333", "name": "美的集团"}
    ]

    print(f"{'🎯 开始分析股票买入信号 (优化策略) ':~^100}")
    print("买入条件：")
    print("1. 收盘价 ≤ 布林下轨")
    print("2. 单日跌幅 ≥ 1%")
    print("3. 布林中轨方向向上 (今日MA > 昨日MA)")
    print("4. 成交量显著放大 (当日成交量 > 5日均量 × 1.2)")
    print("\n卖出条件：")
    print("1. 止盈：实时价 ≥ 买入时布林中轨价格 × 110%")
    print("2. 止损：实时价 ≤ 买入价 × 95%")
    print("3. 最大持仓：90天强制平仓")
    print(f"{'~' * 100}")

    all_results = []

    for config in stock_configs:
        result = analyze_stock_buy_signals_optimized(
            stock_code=config["code"],
            stock_name=config["name"],
            years=3,
            volume_multiplier=1.3  # 成交量放大倍数
        )
        all_results.append(result)

    # 保存结果到Excel
    save_results_to_excel(all_results, filename="/Users/mac/Desktop/股票买入信号分析_优化策略.xlsx")