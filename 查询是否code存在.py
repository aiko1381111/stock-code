# (接登录代码)
# 查询黄金ETF的基本信息
import baostock as bs
import pandas as pd
import os
from datetime import datetime, timedelta

rs = bs.query_stock_basic(code="sh.518880")

data_list = []
while (rs.error_code == '0') & rs.next():
    data_list.append(rs.get_row_data())

if data_list:
    result = pd.DataFrame(data_list, columns=rs.fields)
    print("黄金ETF基本信息：")
    print(result)
else:
    print("未找到该ETF的基本信息。")

# (接登出代码)