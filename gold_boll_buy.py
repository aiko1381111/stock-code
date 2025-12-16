import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import matplotlib.pyplot as plt
import os
import json
import hashlib
from typing import Optional, Dict, List, Tuple
import config

import warnings

warnings.filterwarnings('ignore')


# ==================== 自定义JSON编码器 ====================
class NumpyEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理NumPy数据类型"""

    def default(self, obj):  # 移除类型注解
        if isinstance(obj, (np.integer, int)):  # 添加int类型
            return int(obj)
        elif isinstance(obj, (np.floating, float)):  # 添加float类型
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, datetime):
            return obj.strftime('%Y-%m-%d %H:%M:%S')
        elif isinstance(obj, timedelta):
            return str(obj)
        elif pd.isna(obj):
            return None
        else:
            return super().default(obj)
# ==================== 缓存管理器类 ====================
class DataCache:
    """数据缓存管理器"""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def _get_cache_key(self, symbol: str, start_date: str, end_date: str) -> str:
        cache_str = f"{symbol}_{start_date}_{end_date}"
        return hashlib.md5(cache_str.encode()).hexdigest()

    def _get_cache_file(self, cache_key: str) -> str:
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def get(self, symbol: str, start_date: str, end_date: str,
            max_age_hours: int = 24) -> Optional[pd.DataFrame]:
        cache_key = self._get_cache_key(symbol, start_date, end_date)
        cache_file = self._get_cache_file(cache_key)

        if not os.path.exists(cache_file):
            return None

        try:
            file_mtime = os.path.getmtime(cache_file)
            file_age = (time.time() - file_mtime) / 3600

            if file_age > max_age_hours:
                print(f"缓存已过期 ({file_age:.1f} 小时 > {max_age_hours} 小时)")
                return None

            with open(cache_file, 'r', encoding='utf-8') as f:
                cache_data = json.load(f)

            df = pd.DataFrame(cache_data['data'])
            df['date'] = pd.to_datetime(df['date'])

            print(f"从缓存加载数据 ({file_age:.1f} 小时前)")
            return df

        except Exception as e:
            print(f"读取缓存失败: {e}")
            return None

    def save(self, symbol: str, start_date: str, end_date: str,
             df: pd.DataFrame) -> bool:
        try:
            cache_key = self._get_cache_key(symbol, start_date, end_date)
            cache_file = self._get_cache_file(cache_key)

            cache_data = {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'cached_at': datetime.now().isoformat(),
                'data': json.loads(df.to_json(orient='records', date_format='iso'))
            }

            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(cache_data, f, ensure_ascii=False, indent=2, cls=Numpy)

            print(f"数据已缓存到: {cache_file}")
            return True

        except Exception as e:
            print(f"保存缓存失败: {e}")
            return False

# ==================== 辅助函数 ====================
def convert_to_serializable(obj):
    """将对象转换为JSON可序列化的类型"""
    if isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64,
                       np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    elif isinstance(obj, pd.Series):
        return obj.tolist()
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient='records')
    elif isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, timedelta):
        return str(obj)
    elif pd.isna(obj):
        return None
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj

# ==================== 增强版回测类 ====================
class GoldTradingBacktestEnhanced:
    """增强版黄金交易回测系统（带缓存、止损、最大持有天数）"""

    def __init__(self, api_key: str, cache_enabled: bool = True):
        self.api_key = api_key
        self.base_url = "https://api.twelvedata.com"
        self.session = requests.Session()
        self.cache_enabled = cache_enabled

        if cache_enabled:
            self.cache = DataCache()
        else:
            self.cache = None

        # 策略参数
        self.strategy_params = {
            'bb_period': 20,
            'bb_std': 2,
            'buy_bb_lower_multiplier': 0.995,
            'buy_ma_period': 120,
            'sell_bb_upper_multiplier': 1.005,
            'stop_loss_percent': 0.92,
            'max_hold_days': 180
        }

    def fetch_historical_data(self, symbol: str = "XAU/USD",
                              years: int = 5) -> pd.DataFrame:
        print(f"正在获取 {years} 年历史数据...")

        end_date = datetime.now()
        start_date = end_date - timedelta(days=years * 365 + 100)

        start_str = start_date.strftime("%Y-%m-%d")
        end_str = end_date.strftime("%Y-%m-%d")

        # 尝试从缓存获取
        if self.cache_enabled and self.cache:
            cached_df = self.cache.get(symbol, start_str, end_str)
            if cached_df is not None:
                cached_start = cached_df['date'].min()
                cached_end = cached_df['date'].max()
                request_start = pd.to_datetime(start_str)
                request_end = pd.to_datetime(end_str)

                if cached_start <= request_start and cached_end >= request_end:
                    filtered_df = cached_df[
                        (cached_df['date'] >= request_start) &
                        (cached_df['date'] <= request_end)
                        ].copy()

                    if len(filtered_df) > 0:
                        print(f"从缓存获取 {len(filtered_df)} 条数据")
                        return filtered_df
                else:
                    print("缓存数据时间范围不足，从API获取")

        # 从API获取数据
        params = {
            "symbol": symbol,
            "interval": "1day",
            "outputsize": 5000,
            "start_date": start_str,
            "end_date": end_str,
            "apikey": self.api_key,
            "format": "JSON"
        }

        try:
            print(f"从API获取数据: {start_str} 到 {end_str}")
            response = self.session.get(f"{self.base_url}/time_series",
                                        params=params, timeout=30)
            response.raise_for_status()
            data = response.json()

            if "values" not in data:
                print(f"获取数据失败: {data.get('message', '未知错误')}")
                if self.api_key != "demo":
                    print("尝试使用演示密钥...")
                    self.api_key = "demo"
                    return self.fetch_historical_data(symbol, years)
                return pd.DataFrame()

            # 转换为DataFrame
            df = pd.DataFrame(data["values"])

            # 重命名列
            column_mapping = {
                'datetime': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            }
            df = df.rename(columns=column_mapping)

            # 转换数据类型
            df['date'] = pd.to_datetime(df['date'])
            numeric_cols = ['open', 'high', 'low', 'close']
            for col in numeric_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            # 按日期排序
            df = df.sort_values('date').reset_index(drop=True)

            print(f"成功获取 {len(df)} 条历史数据")
            print(f"数据时间范围: {df['date'].min()} 到 {df['date'].max()}")

            # 保存到缓存
            if self.cache_enabled and self.cache:
                self.cache.save(symbol, start_str, end_str, df)

            return df

        except Exception as e:
            print(f"获取历史数据失败: {e}")
            return pd.DataFrame()

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        params = self.strategy_params

        # 计算120日简单移动平均线
        df['ma_120'] = df['close'].rolling(window=params['buy_ma_period'],
                                           min_periods=1).mean()

        # 计算布林带
        window = params['bb_period']
        df['bb_middle'] = df['close'].rolling(window=window, min_periods=1).mean()
        df['bb_std'] = df['close'].rolling(window=window, min_periods=1).std()

        # 布林带上轨
        df['bb_upper'] = df['bb_middle'] + (df['bb_std'] * params['bb_std'])

        # 布林带下轨
        df['bb_lower'] = df['bb_middle'] - (df['bb_std'] * params['bb_std'])

        # 计算价格与布林带上下轨的关系
        df['below_bb_lower'] = df['low'] <= (df['bb_lower'] * params['buy_bb_lower_multiplier'])
        df['above_bb_upper'] = df['high'] >= (df['bb_upper'] * params['sell_bb_upper_multiplier'])

        # 计算价格与120日均线的关系
        df['below_ma_120'] = df['low'] < df['ma_120']

        # 向前填充NaN值
        df = df.fillna(method='ffill')

        return df

    def generate_signals_with_stop_loss(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df

        df = df.copy()
        params = self.strategy_params

        # 初始化信号列
        df['signal'] = 0
        df['position'] = 0
        df['buy_price'] = np.nan
        df['sell_price'] = np.nan
        df['stop_loss_price'] = np.nan
        df['buy_date'] = pd.NaT
        df['sell_reason'] = ''

        position = 0
        buy_price = 0
        buy_date = None
        stop_loss_price = 0
        entry_index = 0

        for i in range(params['buy_ma_period'], len(df)):
            current = df.iloc[i]
            current_date = current['date']

            # 买入条件
            if position == 0:
                condition1 = current['below_bb_lower']
                #condition2 = current['below_ma_120']

                if condition1 :
                    position = 1
                    buy_price = current[('low'
                                         '')]
                    buy_date = current_date
                    entry_index = i

                    stop_loss_price = buy_price * params['stop_loss_percent']

                    df.at[i, 'signal'] = 1
                    df.at[i, 'position'] = 1
                    df.at[i, 'buy_price'] = buy_price
                    df.at[i, 'buy_date'] = buy_date
                    df.at[i, 'stop_loss_price'] = stop_loss_price

                    print(f"买入信号 [{current_date.strftime('%Y-%m-%d')}]: "
                          f"价格=${buy_price:.2f}, 止损价=${stop_loss_price:.2f}")

            # 卖出条件
            elif position == 1:
                current_low = current['low']
                current_high = current['high']
                current_close = current['close']

                hold_days = (current_date - buy_date).days

                sell_reason = ''
                should_sell = False

                # 止损条件
                if current_low <= stop_loss_price:
                    sell_reason = '止损'
                    should_sell = True
                    sell_price = min(stop_loss_price, current_close)

                # 最大持有天数
                elif hold_days >= params['max_hold_days']:
                    sell_reason = '超时'
                    should_sell = True
                    sell_price = current_close

                # 正常卖出
                elif current['above_bb_upper']:
                    sell_reason = '止盈'
                    should_sell = True
                    sell_price = current_close

                # 执行卖出
                if should_sell:
                    position = 0

                    df.at[i, 'signal'] = -1
                    df.at[i, 'position'] = 0
                    df.at[i, 'sell_price'] = sell_price
                    df.at[i, 'sell_reason'] = sell_reason

                    return_rate = (sell_price - buy_price) / buy_price * 100
                    print(f"卖出信号 [{current_date.strftime('%Y-%m-%d')}]: "
                          f"买入=${buy_price:.2f}, 卖出=${sell_price:.2f}, "
                          f"持有{hold_days}天, 收益率={return_rate:.2f}%, "
                          f"原因={sell_reason}")

                    buy_price = 0
                    buy_date = None
                    stop_loss_price = 0
                    entry_index = 0

        # 如果最后一天仍然持仓，强制平仓
        if position == 1:
            last_idx = len(df) - 1
            last_date = df.iloc[last_idx]['date']
            hold_days = (last_date - buy_date).days
            sell_price = df.iloc[last_idx]['close']

            df.at[last_idx, 'signal'] = -1
            df.at[last_idx, 'position'] = 0
            df.at[last_idx, 'sell_price'] = sell_price
            df.at[last_idx, 'sell_reason'] = '强制平仓'

            return_rate = (sell_price - buy_price) / buy_price * 100
            print(f"强制平仓 [{last_date.strftime('%Y-%m-%d')}]: "
                  f"持有{hold_days}天, 收益率={return_rate:.2f}%")

        return df

    def calculate_trades(self, df: pd.DataFrame) -> pd.DataFrame:
        trades = []
        buy_info = None

        for i in range(len(df)):
            signal = df.iloc[i]['signal']

            if signal == 1:
                buy_info = {
                    'buy_date': df.iloc[i]['date'],
                    'buy_price': df.iloc[i]['buy_price'],
                    'stop_loss_price': df.iloc[i]['stop_loss_price']
                }

            elif signal == -1 and buy_info is not None:
                sell_date = df.iloc[i]['date']
                sell_price = df.iloc[i]['sell_price']
                sell_reason = df.iloc[i]['sell_reason']

                hold_days = (sell_date - buy_info['buy_date']).days
                return_rate = (sell_price - buy_info['buy_price']) / buy_info['buy_price'] * 100

                if hold_days > 0:
                    annual_return = (return_rate / hold_days) * 365
                else:
                    annual_return = 0

                # 计算最大浮亏
                start_idx = df[df['date'] == buy_info['buy_date']].index[0]
                end_idx = i

                if start_idx < end_idx:
                    period_df = df.iloc[start_idx:end_idx + 1]
                    min_price = period_df['low'].min()
                    max_drawdown = (min_price - buy_info['buy_price']) / buy_info['buy_price'] * 100
                else:
                    max_drawdown = 0

                trades.append({
                    'trade_id': len(trades) + 1,
                    'buy_date': buy_info['buy_date'],
                    'buy_price': buy_info['buy_price'],
                    'sell_date': sell_date,
                    'sell_price': sell_price,
                    'sell_reason': sell_reason,
                    'hold_days': hold_days,
                    'return_rate': return_rate,
                    'annual_return': annual_return,
                    'max_drawdown': max_drawdown,
                    'stop_loss_price': buy_info['stop_loss_price'],
                    'stop_loss_triggered': 1 if sell_reason == '止损' else 0
                })

                buy_info = None

        if trades:
            return pd.DataFrame(trades)
        else:
            return pd.DataFrame()


    def calculate_statistics(self, trades_df, initial_capital=10000):
        if trades_df.empty:
            return {}

        stats = {}

        # 基础统计
        stats['total_trades'] = len(trades_df)
        stats['winning_trades'] = len(trades_df[trades_df['return_rate'] > 0])
        stats['losing_trades'] = len(trades_df[trades_df['return_rate'] <= 0])
        stats['win_rate'] = (stats['winning_trades'] / stats['total_trades'] * 100
                             if stats['total_trades'] > 0 else 0)

        # 卖出原因统计
        if 'sell_reason' in trades_df.columns:
            sell_reasons = trades_df['sell_reason'].value_counts()
            stats['sell_reasons'] = sell_reasons.to_dict()

            stats['stop_loss_trades'] = len(trades_df[trades_df['sell_reason'] == '止损'])
            stats['timeout_trades'] = len(trades_df[trades_df['sell_reason'] == '超时'])
            stats['profit_taking_trades'] = len(trades_df[trades_df['sell_reason'] == '止盈'])

        # 收益率统计
        stats['avg_return'] = trades_df['return_rate'].mean()
        stats['max_return'] = trades_df['return_rate'].max()
        stats['min_return'] = trades_df['return_rate'].min()
        stats['avg_annual_return'] = trades_df['annual_return'].mean()

        # 持有天数统计
        stats['avg_hold_days'] = trades_df['hold_days'].mean()
        stats['max_hold_days'] = trades_df['hold_days'].max()
        stats['min_hold_days'] = trades_df['hold_days'].min()

        # 最大回撤统计
        if 'max_drawdown' in trades_df.columns:
            stats['avg_max_drawdown'] = trades_df['max_drawdown'].mean()
            stats['max_max_drawdown'] = trades_df['max_drawdown'].max()

        # 累计收益计算
        capital = initial_capital
        capital_history = [capital]
        date_history = [trades_df['buy_date'].min() - timedelta(days=1)]

        for _, trade in trades_df.iterrows():
            return_rate = trade['return_rate'] / 100
            capital = capital * (1 + return_rate)
            capital_history.append(capital)
            date_history.append(trade['sell_date'])

        stats['final_capital'] = capital
        stats['total_return'] = ((capital - initial_capital) / initial_capital) * 100
        stats['annualized_return'] = stats['total_return'] / 5

        # 最大回撤计算
        peak = capital_history[0]
        max_drawdown_capital = 0
        max_drawdown_start = None
        max_drawdown_end = None

        for i, value in enumerate(capital_history):
            if value > peak:
                peak = value
                peak_date = date_history[i]

            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown_capital:
                max_drawdown_capital = drawdown
                max_drawdown_start = peak_date
                max_drawdown_end = date_history[i]

        stats['max_drawdown_capital'] = max_drawdown_capital
        stats['max_drawdown_period'] = f"{max_drawdown_start} 到 {max_drawdown_end}" if max_drawdown_start else "N/A"

        # 夏普比率
        returns = trades_df['return_rate'].values / 100
        if len(returns) > 1 and np.std(returns) > 0:
            stats['sharpe_ratio'] = (np.mean(returns) / np.std(returns)) * np.sqrt(252)
        else:
            stats['sharpe_ratio'] = 0

        # 胜率按卖出原因分类
        if 'sell_reason' in trades_df.columns:
            for reason in ['止盈', '止损', '超时', '强制平仓']:
                if reason in trades_df['sell_reason'].values:
                    reason_trades = trades_df[trades_df['sell_reason'] == reason]
                    reason_wins = len(reason_trades[reason_trades['return_rate'] > 0])
                    reason_total = len(reason_trades)
                    stats[f'win_rate_{reason}'] = (reason_wins / reason_total * 100
                                                   if reason_total > 0 else 0)

        return stats

    def plot_results(self, df: pd.DataFrame, trades_df: pd.DataFrame):
        if df.empty or trades_df.empty:
            print("无数据可绘图")
            return

        fig = plt.figure(figsize=(16, 14))

        # 子图1: 价格走势和交易点
        ax1 = plt.subplot(3, 1, 1)
        ax1.plot(df['date'], df['close'], label='收盘价', linewidth=1, alpha=0.7, color='black')
        ax1.plot(df['date'], df['ma_120'], label='120日均线', linewidth=1, alpha=0.7, color='blue')
        ax1.plot(df['date'], df['bb_upper'], label='布林上轨', linewidth=0.5, alpha=0.5,
                 linestyle='--', color='orange')
        ax1.plot(df['date'], df['bb_lower'], label='布林下轨', linewidth=0.5, alpha=0.5,
                 linestyle='--', color='orange')
        ax1.fill_between(df['date'], df['bb_lower'], df['bb_upper'],
                         alpha=0.1, color='orange', label='布林带')

        # 标记买入点和卖出点
        for _, trade in trades_df.iterrows():
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']
            sell_reason = trade['sell_reason']

            ax1.scatter(buy_date, buy_price, color='green', s=80,
                        marker='^', zorder=5, alpha=0.8)

            if sell_reason == '止盈':
                color = 'red'
                marker = 'v'
                size = 80
            elif sell_reason == '止损':
                color = 'purple'
                marker = 'x'
                size = 100
            elif sell_reason == '超时':
                color = 'orange'
                marker = 's'
                size = 80
            else:
                color = 'gray'
                marker = 'o'
                size = 80

            ax1.scatter(sell_date, sell_price, color=color, s=size,
                        marker=marker, zorder=5, alpha=0.8)

            ax1.plot([buy_date, sell_date], [buy_price, sell_price],
                     linewidth=0.5, alpha=0.3, color='gray')

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='green', marker='^', linestyle='None',
                   markersize=8, label='买入点'),
            Line2D([0], [0], color='red', marker='v', linestyle='None',
                   markersize=8, label='止盈卖出'),
            Line2D([0], [0], color='purple', marker='x', linestyle='None',
                   markersize=10, label='止损卖出'),
            Line2D([0], [0], color='orange', marker='s', linestyle='None',
                   markersize=8, label='超时卖出'),
            Line2D([0], [0], color='black', linewidth=1, label='收盘价'),
            Line2D([0], [0], color='blue', linewidth=1, label='120日均线'),
            Line2D([0], [0], color='orange', linewidth=0.5, linestyle='--',
                   label='布林带')
        ]

        ax1.legend(handles=legend_elements, loc='upper left')
        ax1.set_title('黄金价格走势与交易信号（带止损和最大持有天数）', fontsize=14)
        ax1.set_ylabel('价格 (USD)')
        ax1.grid(True, alpha=0.3)

        # 子图2: 持仓状态
        ax2 = plt.subplot(3, 1, 2)

        position_colors = []
        for pos in df['position']:
            if pos == 1:
                position_colors.append('lightgreen')
            else:
                position_colors.append('lightcoral')

        ax2.bar(df['date'], df['position'], width=1, color=position_colors,
                edgecolor='none', alpha=0.6)

        for _, trade in trades_df.iterrows():
            buy_date = trade['buy_date']
            sell_date = trade['sell_date']
            hold_days = trade['hold_days']

            mid_date = buy_date + (sell_date - buy_date) / 2
            ax2.text(mid_date, 0.5, f'{hold_days}天',
                     ha='center', va='center', fontsize=8, alpha=0.8)

        ax2.set_ylim(0, 1.1)
        ax2.set_title('持仓状态与持有天数', fontsize=14)
        ax2.set_ylabel('仓位')
        ax2.grid(True, alpha=0.3)

        # 子图3: 累计收益率对比
        ax3 = plt.subplot(3, 1, 3)

        initial_price = df['close'].iloc[0]
        buy_hold_return = (df['close'] - initial_price) / initial_price * 100
        ax3.plot(df['date'], buy_hold_return, label='买入持有策略',
                 linewidth=2, alpha=0.7, color='blue')

        if not trades_df.empty:
            strategy_values = [10000]
            strategy_dates = [df['date'].iloc[0]]

            current_capital = 10000
            in_position = False

            for i in range(1, len(df)):
                current_date = df['date'].iloc[i]

                if df['signal'].iloc[i] == 1 and not in_position:
                    in_position = True
                    position_capital = current_capital

                elif df['signal'].iloc[i] == -1 and in_position:
                    in_position = False
                    recent_trades = trades_df[trades_df['sell_date'] <= current_date]
                    if not recent_trades.empty:
                        last_trade = recent_trades.iloc[-1]
                        return_rate = last_trade['return_rate'] / 100
                        current_capital = position_capital * (1 + return_rate)

                strategy_values.append(current_capital)
                strategy_dates.append(current_date)

            strategy_return = [(v - 10000) / 10000 * 100 for v in strategy_values]
            ax3.plot(strategy_dates, strategy_return, label='布林带策略（带止损）',
                     linewidth=2, alpha=0.7, color='red')

        ax3.set_title('累计收益率对比 (%)', fontsize=14)
        ax3.set_ylabel('收益率 (%)')
        ax3.set_xlabel('日期')
        ax3.legend(loc='best')
        ax3.grid(True, alpha=0.3)

        # 添加回测统计信息文本框
        if not trades_df.empty:
            stats_text = f"总交易次数: {len(trades_df)}\n"
            stats_text += f"胜率: {len(trades_df[trades_df['return_rate'] > 0]) / len(trades_df) * 100:.1f}%\n"
            stats_text += f"平均持有天数: {trades_df['hold_days'].mean():.1f}\n"
            stats_text += f"平均收益率: {trades_df['return_rate'].mean():.2f}%\n"

            if 'sell_reason' in trades_df.columns:
                for reason, count in trades_df['sell_reason'].value_counts().items():
                    reason_return = trades_df[trades_df['sell_reason'] == reason]['return_rate'].mean()
                    stats_text += f"{reason}: {count}次 ({reason_return:.2f}%)\n"

            plt.figtext(0.02, 0.02, stats_text, fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

        plt.tight_layout()
        plt.savefig('gold_trading_backtest_enhanced.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_results(self, trades_df: pd.DataFrame, stats: Dict):
        print("\n" + "=" * 80)
        print("增强版黄金交易策略回测结果")
        print("策略规则:")
        print("  买入条件:")
        print("    1. 最低价 ≤ 布林日线下轨 × 0.995 (低于下轨0.5%)")
        print("    2. 最低价 < 120日均线")
        print("  卖出条件:")
        print("    1. 最高价 > 布林日线上轨 × 1.005 (高于上轨0.5%)")
        print("    2. 止损: 价格低于买入价92%")
        print("    3. 最大持有天数: 180天")
        print("=" * 80)

        if trades_df.empty:
            print("没有交易记录")
            return

        # 打印交易记录
        print("\n交易记录明细:")
        print("-" * 120)

        display_cols = ['trade_id', 'buy_date', 'buy_price', 'sell_date',
                        'sell_price', 'sell_reason', 'hold_days',
                        'return_rate', 'annual_return', 'max_drawdown']

        formatted_trades = trades_df.copy()
        formatted_trades['buy_date'] = formatted_trades['buy_date'].dt.strftime('%Y-%m-%d')
        formatted_trades['sell_date'] = formatted_trades['sell_date'].dt.strftime('%Y-%m-%d')
        formatted_trades['buy_price'] = formatted_trades['buy_price'].round(2)
        formatted_trades['sell_price'] = formatted_trades['sell_price'].round(2)
        formatted_trades['return_rate'] = formatted_trades['return_rate'].round(2)
        formatted_trades['annual_return'] = formatted_trades['annual_return'].round(2)
        formatted_trades['max_drawdown'] = formatted_trades['max_drawdown'].round(2)

        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_columns', None)

        print(formatted_trades[display_cols].to_string(index=False))

        # 打印统计信息
        print("\n" + "=" * 80)
        print("回测统计摘要")
        print("=" * 80)

        print(f"\n📊 基础统计:")
        print(f"  总交易次数: {stats.get('total_trades', 0)}")
        print(f"  盈利交易: {stats.get('winning_trades', 0)}")
        print(f"  亏损交易: {stats.get('losing_trades', 0)}")
        print(f"  胜率: {stats.get('win_rate', 0):.2f}%")

        print(f"\n📈 收益率统计:")
        print(f"  平均收益率: {stats.get('avg_return', 0):.2f}%")
        print(f"  最大收益率: {stats.get('max_return', 0):.2f}%")
        print(f"  最小收益率: {stats.get('min_return', 0):.2f}%")
        print(f"  平均年化收益率: {stats.get('avg_annual_return', 0):.2f}%")
        print(f"  总收益率: {stats.get('total_return', 0):.2f}%")
        print(f"  年化收益率: {stats.get('annualized_return', 0):.2f}%")

        print(f"\n⏰ 持有时间统计:")
        print(f"  平均持有天数: {stats.get('avg_hold_days', 0):.2f}")
        print(f"  最大持有天数: {stats.get('max_hold_days', 0)}")
        print(f"  最小持有天数: {stats.get('min_hold_days', 0)}")

        print(f"\n⚠️ 风险统计:")
        print(f"  最大回撤（资金）: {stats.get('max_drawdown_capital', 0):.2f}%")
        if 'max_drawdown_period' in stats:
            print(f"  最大回撤期间: {stats.get('max_drawdown_period', 'N/A')}")
        print(f"  平均最大浮亏: {stats.get('avg_max_drawdown', 0):.2f}%")
        print(f"  最大浮亏: {stats.get('max_max_drawdown', 0):.2f}%")
        print(f"  夏普比率: {stats.get('sharpe_ratio', 0):.2f}")

        print(f"\n🎯 卖出原因统计:")
        if 'sell_reasons' in stats:
            for reason, count in stats['sell_reasons'].items():
                win_rate_key = f'win_rate_{reason}'
                win_rate = stats.get(win_rate_key, 0)
                print(f"  {reason}: {count}次 ({win_rate:.1f}%胜率)")

        print(f"\n💰 资金统计:")
        print(f"  最终资金: ${stats.get('final_capital', 0):.2f}")
        print(f"  总收益: ${stats.get('final_capital', 0) - 10000:.2f}")

        buy_hold_return = stats.get('annualized_return', 0)
        strategy_return = stats.get('annualized_return', 0)

        if buy_hold_return > 0 and strategy_return > 0:
            outperformance = strategy_return - buy_hold_return
            print(f"\n📊 策略对比:")
            print(f"  策略年化收益: {strategy_return:.2f}%")
            print(f"  买入持有年化收益: {buy_hold_return:.2f}%")
            print(f"  超额收益: {outperformance:.2f}%")

        print("=" * 80)

    def run_backtest(self, symbol: str = "XAU/USD",
                     years: int = 5) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        print("=" * 80)
        print("增强版黄金交易策略回测系统")
        print("=" * 80)

        # 1. 获取历史数据（带缓存）
        start_time = time.time()
        df = self.fetch_historical_data(symbol, years)

        if df.empty:
            print("无法获取数据，请检查API密钥或网络连接")
            return pd.DataFrame(), pd.DataFrame(), {}

        # 2. 计算技术指标
        print("\n正在计算技术指标...")
        df = self.calculate_indicators(df)

        # 3. 生成交易信号（带止损和最大持有天数）
        print("正在生成交易信号...")
        df = self.generate_signals_with_stop_loss(df)

        # 4. 计算交易记录
        print("正在计算交易记录...")
        trades_df = self.calculate_trades(df)

        if trades_df.empty:
            print("没有生成任何交易记录")
            return df, trades_df, {}

        print(f"共生成 {len(trades_df)} 笔交易记录")

        # 5. 计算统计信息
        print("正在计算回测统计...")
        stats = self.calculate_statistics(trades_df)

        # 计算运行时间
        end_time = time.time()
        run_time = end_time - start_time
        print(f"回测完成，耗时 {run_time:.2f} 秒")

        return df, trades_df, stats


# ==================== 主函数 ====================
def main():
    """主程序"""
    print("增强版黄金交易策略回测系统")
    print("=" * 80)
    api_key = config.API_KEY
    years = config.YEARS
    cache_enabled = config.enable_cache
    stop_loss = config.stop_loss
    max_hold_days = config.max_hold_days

    # 配置参数
    #DEFAULT_API_KEY = "1711a6d605444df78cfd2371e51e986b"

    #print("配置回测参数:")

    # 获取API密钥

    #use_custom_key = input("是否使用自定义API密钥？(y/n, 默认n): ").strip().lower()

    # if use_custom_key == 'y':
    #     api_key = input("请输入您的Twelve Data API密钥: ").strip()
    #     if not api_key:
    #         print("未提供API密钥，使用演示密钥")
    #         api_key = config.API_KEY
    # else:
    #     api_key = config.API_KEY
    #     print(f"使用演示密钥: {api_key}")

    # 是否启用缓存
    # enable_cache = input("是否启用缓存？(y/n, 默认n): ").strip().lower()
    # cache_enabled = not (enable_cache == 'y')
    #
    # if cache_enabled:
    #     print("缓存已启用，数据将保存到./cache目录")
    #
    # # 回测年数
    # try:
    #     years = int(input("请输入回测年数 (默认2年): ").strip() or "2")
    # except:
    #     years = 2

    # 策略参数调整
    # print("\n策略参数调整 (按Enter使用默认值):")
    #
    # try:
    #     stop_loss = float(input(f"止损比例 (默认0.92): ").strip() or "0.92")
    #     if 0 < stop_loss < 1:
    #         print(f"止损比例设置为: {stop_loss}")
    #     else:
    #         print("无效的止损比例，使用默认值0.92")
    #         stop_loss = 0.92
    # except:
    #     stop_loss = 0.92

    # try:
    #     max_hold_days = int(input(f"最大持有天数 (默认180): ").strip() or "180")
    #     if max_hold_days > 0:
    #         print(f"最大持有天数设置为: {max_hold_days}")
    #     else:
    #         print("无效的天数，使用默认值180")
    #         max_hold_days = 180
    # except:
    #     max_hold_days = 180

    try:
        # 初始化回测系统
        print(f"\n开始回测: {years}年数据，止损{stop_loss * 100:.0f}%，最大持有{max_hold_days}天")
        backtester = GoldTradingBacktestEnhanced(api_key, cache_enabled)  # 这里使用正确的类名

        # 更新策略参数
        backtester.strategy_params['stop_loss_percent'] = stop_loss
        backtester.strategy_params['max_hold_days'] = max_hold_days

        # 运行回测
        df, trades_df, stats = backtester.run_backtest("XAU/USD", years)

        if not trades_df.empty:
            # 打印结果
            backtester.print_results(trades_df, stats)

            # 绘制图表
            # print("\n正在生成图表...")
            # backtester.plot_results(df, trades_df)

            # 保存结果
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            # 保存交易记录
            trades_file = f"gold_trades_{timestamp}.csv"
            trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')

            # 保存完整数据
            data_file = f"gold_data_{timestamp}.csv"
            df.to_csv(data_file, index=False, encoding='utf-8-sig')

            # 保存统计信息
            stats_file = f"gold_stats_{timestamp}.json"
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, ensure_ascii=False, indent=2)

            print(f"\n💾 结果已保存:")
            print(f"  交易记录: {trades_file}")
            print(f"  完整数据: {data_file}")
            print(f"  统计信息: {stats_file}")
            #print(f"  图表: gold_trading_backtest_enhanced.png")

            # 打印关键建议


    except KeyboardInterrupt:
        print("\n\n用户中断回测")
    except Exception as e:
        print(f"\n回测过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()