import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta


def process_multiple_stocks(stock_codes, base_path="/Users/mac/Desktop/Stock", years=3):
    """
    批量处理多个股票数据

    参数:
    stock_codes (list): 股票代码列表
    base_path (str): 文件保存的基础路径
    years (int): 数据年份
    """
    results = {}

    for stock_code in stock_codes:
        print(f"\n{'=' * 60}")
        print(f"正在处理: {stock_code}")
        print(f"{'=' * 60}")

        try:
            # 下载数据
            stock_data = download_stock_data(stock_code, years=years)

            if stock_data is not None:
                # 生成文件名
                filename = os.path.join(base_path, f"{stock_code}_日线数据.xlsx")

                # 保存数据
                if save_to_excel(stock_data, filename=filename):
                    results[stock_code] = "成功"
                    print(f"✅ {stock_code} 处理成功！")
                else:
                    results[stock_code] = "保存失败"
                    print(f"❌ {stock_code} 保存失败！")
            else:
                results[stock_code] = "下载失败"
                print(f"❌ {stock_code} 下载失败！")

        except Exception as e:
            results[stock_code] = f"异常: {str(e)}"
            print(f"❌ {stock_code} 处理异常: {str(e)}")

    # 打印汇总结果
    print(f"\n{'🎯 批量处理汇总 ':~^50}")
    success_count = sum(1 for status in results.values() if status == "成功")
    print(f"总计处理: {len(stock_codes)} 个股票")
    print(f"成功: {success_count} 个")
    print(f"失败: {len(stock_codes) - success_count} 个")

    for stock, status in results.items():
        print(f"  {stock}: {status}")

    return results

def download_stock_data(stock_code, years=5):
    """
    下载指定股票代码的日线数据

    参数:
    stock_code (str): 股票代码，格式如 'sh.600000' (上海) 或 'sz.000001' (深圳)
    years (int): 需要下载数据的年份数，默认5年

    返回:
    pandas.DataFrame: 包含日线数据的DataFrame
    """

    # 计算日期范围
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime('%Y-%m-%d')

    # 登陆Baostock系统
    lg = bs.login()
    print(f'登录响应: {lg.error_code} - {lg.error_msg}')

    try:
        # 查询历史K线数据
        # 字段说明: date-日期, open-开盘价, high-最高价, low-最低价, close-收盘价,
        # volume-成交量, amount-成交额, adjustflag-复权状态, turn-换手率, pctChg-涨跌幅[citation:9]
        rs = bs.query_history_k_data_plus(stock_code,
                                          "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg",
                                          start_date=start_date, end_date=end_date,
                                          frequency="d", adjustflag="2")  # d-日线, adjustflag-2:前复权[citation:6]

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

        # 数据类型转换[citation:8]
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
        for col in numeric_columns:
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')

        # 日期格式转换
        if 'date' in result.columns:
            result['date'] = pd.to_datetime(result['date'])
            result = result.sort_values('date')  # 按日期排序

        print(f"成功获取 {len(result)} 条日线数据")
        return result

    except Exception as e:
        print(f"数据获取过程中出现错误: {str(e)}")
        return None
    finally:
        # 登出系统
        bs.logout()
        print("已登出Baostock系统")


def save_to_excel(data, filename=None):
    """
    将数据保存为Excel文件[citation:10]

    参数:
    data (pandas.DataFrame): 要保存的数据
    filename (str): 保存的文件名，如果为None则自动生成
    """
    if data is None or data.empty:
        print("没有数据可保存")
        return False

    if filename is None:
        stock_code = data['code'].iloc[0] if 'code' in data.columns else 'stock'
        filename = f"{stock_code}_日线数据_{datetime.now().strftime('%Y%m%d')}.xlsx"

    try:
        if os.path.isdir(filename):
            # 如果用户提供的是目录路径，自动在目录下生成文件名
            stock_code = data['code'].iloc[0] if 'code' in data.columns else 'stock'
            auto_filename = f"{stock_code}_日线数据_{datetime.now().strftime('%Y%m%d')}.xlsx"
            filename = os.path.join(filename, auto_filename)
            print(f"检测到路径为目录，已自动生成文件名: {os.path.basename(filename)}")

            # 确保目标目录存在
        os.makedirs(os.path.dirname(filename), exist_ok=True)

        # ★★★ 创建数据副本并重命名列名为中文 ★★★
        data_chinese = data.copy()

        # 定义中文字段名映射
        column_mapping = {
            'date': '交易日期',
            'code': '股票代码',
            'open': '开盘价',
            'high': '最高价',
            'low': '最低价',
            'close': '收盘价',
            'volume': '成交量(股)',
            'amount': '成交额(元)',
            'adjustflag': '前复权',
            'turn': '换手率(%)',
            'pctChg': '涨跌幅(%)'
        }

        # 重命名列
        data_chinese = data_chinese.rename(columns=column_mapping)

        # 保存为Excel文件（使用中文列名）
        data_chinese.to_excel(filename, index=False, sheet_name='日线数据')
        print(f"数据已成功保存到: {filename}")
        print(f"【重要提示】数据文件已保存至：{os.path.abspath(filename)}")

        # 显示中文表头预览
        print("\nExcel文件表头预览:")
        print(list(data_chinese.columns))
        return True
    except Exception as e:
        print(f"保存文件时出现错误: {str(e)}")
        print("请确保没有重复打开同名Excel文件，并检查文件路径权限")
        return False


# 主程序
if __name__ == "__main__":
    # 设置股票代码 (示例: 平安银行)
    # 格式: 上海股票 - sh.600000, 深圳股票 - sz.000001[citation:6]
    '''
    stock_code = "sh.518880"  # 修改为您想要下载的股票代码
    print(f"开始下载 {stock_code} 的日线数据...")
    '''
    """
    stock_codes = [
        "sh.518880",  # 黄金ETF
        "sz.000001",  # 平安银行
        "sh.600519",  # 贵州茅台
        "sz.300750",  # 宁德时代
        "sh.600036",  # 招商银行
        "sz.000858",  # 五粮液
    ]
    process_multiple_stocks(stock_codes, years=3)
    """

    # 下载数据
    #stock_data = download_stock_data(stock_code, years=3)
stock_configs = [
    {"code": "sz.000333", "name": "美的", "years": 3},
    '''
    {"code": "sz.000001", "name": "平安银行", "years": 3},
    {"code": "sh.600519", "name": "贵州茅台", "years": 3},
    {"code": "sz.300750", "name": "宁德时代", "years": 3},
    {"code": "sh.600036", "name": "招商银行", "years": 3},
    {"code": "sz.000858", "name": "五粮液", "years": 3},
    '''
]

print(f"{'🚀 开始批量下载股票数据 ':~^60}")

for config in stock_configs:
    stock_code = config["code"]
    stock_name = config["name"]
    years = config["years"]

    print(f"\n📊 正在处理: {stock_name} ({stock_code})")
    print(f"⏰ 数据范围: 最近{years}年")

    # 下载数据
    stock_data = download_stock_data(stock_code, years=years)

    if stock_data is not None:
        # 使用股票名称作为文件名，更友好
        filename = f"/Users/mac/Desktop/Stock/{stock_name}_{stock_code}_数据.xlsx"

        # 保存数据
        if save_to_excel(stock_data, filename=filename):
            print(f"✅ {stock_name} 数据保存成功！")
        else:
            print(f"❌ {stock_name} 数据保存失败！")
    else:
        print(f"❌ {stock_name} 数据下载失败！")

print(f"\n{'🎉 所有股票数据处理完成 ':~^60}")