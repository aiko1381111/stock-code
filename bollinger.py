import pandas as pd
import numpy as np


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

    # 计算买入点位：布林下轨 * 0.98
    df['BUY_POINT'] = df['BOLL_Lower'] * 0.98

    return df


# 您还可以在这个文件中添加其他相关函数
def another_related_function():
    """其他相关技术指标函数"""
    pass