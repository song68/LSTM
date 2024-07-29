import pandas as pd



# 设置pandas显示选项
pd.set_option('display.max_rows', None)                # 显示所有行
pd.set_option('display.max_columns', None)             # 显示所有列
pd.set_option('display.max_colwidth', None)            # 显示完整的列内容
pd.set_option('display.width', None)                   # 自动调整控制台宽度
pd.set_option('display.expand_frame_repr', False)      # 防止自动换行



df = pd.read_excel(r"data.xlsx")  # 读取数据e

# 把sensor1-sensor23和健康预测数据分离出来
sensors = df.drop(columns=["Unit", "Time(h)", "Health Indicator"]).to_numpy()
indicator = df["Health Indicator"].to_numpy()  # 健康预测数据

print(df)
# print(sensors)
