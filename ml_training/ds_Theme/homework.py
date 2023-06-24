import numpy as np
import pandas as pd
from pandas import DataFrame

# week3
# url_winequality_data = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
# def homework(url_winequality_data, n):
#     df = pd.read_csv(url_winequality_data, sep=';')
#     df['group total'] = pd.qcut(df["total sulfur dioxide"], n)
#     df1 = df.groupby('group total').mean()
#     ph_max = df1['pH'].max()
#     ph_min = df1['pH'].min()
#     my_result = ph_min + ph_max
#     return my_result
# homework(url_winequality_data, 4)



# week4
# def homework(target_online_retail_data_tb, n):
#     data = target_online_retail_data_tb.groupby('CustomerID')['TotalPrice'].sum().sort_values(ascending=False)
#     cut  = pd.qcut(data, n)
#     df = pd.concat([data, cut], axis=1)
#     df.columns = ['TotalPrice', 'Group']
#     group_total = df.groupby('Group')['TotalPrice'].sum().sort_index(ascending=False)
#     my_result = group_total / group_total.sum()
#     return my_result



# week5
# def homework(path_winequality_data, X_column, Y_column):
#     reg = linear_model.LinearRegression()
#     df = pd.read_csv(path_winequality_data, sep=';')
#     X = df.loc[:, [X_column]].values
#     Y = df.loc[:, [Y_column]].values
#     reg.fit(X, Y)
#     reg.coef_
#     reg.intercept_
#     my_result = reg.score(X, Y)
#     return my_result



# week6
# def homework(target_online_retail_data_tb):
#     trans_all = set(target_online_retail_data_tb.InvoiceNo)
#     trans_a = set(target_online_retail_data_tb[target_online_retail_data_tb['StockCode']=='20725'].InvoiceNo)
#     trans_b = set(target_online_retail_data_tb[target_online_retail_data_tb['StockCode']=='22383'].InvoiceNo)
#     trans_ab = trans_a & trans_b
#     support_b = len(trans_b) / len(trans_all)
#     confidence = len(trans_ab) / len(trans_a)
#     my_result = confidence / support_b
#     return my_result



# week7
# sql
# terminalでsqlite3を起動
# sqlite3 data.db

# テーブルの確認
# .table
# select * from data と　 select * from goods

# 結合してgoods_idとpriceとgoods_genre_idカラムを表示
# create view df as select data.goods_id, data.price, goods.goods_genre_id  >>> 結合結果をdfというviewテーブルに格納
# from data inner join goods on data.goods_id = goods.goods_id;
# headers on カラム名を表示
# select * from df;

# goods_genre_idごとにグループ化して平均値を求める
# create view db1 as >>> 結合結果をdb1というviewテーブルに格納
# select goods_genre_id, avg(price) as avg_price from df group by goods_genre_id;
# select * from db1

# dfにしてcsvで保存
# import pandas as pd
# import numpy as np
# import sqlite3
# db = sqlite3.connect('data.db')
# df = pd.read_sql_query('SELECT * FROM db1', db)
# df
# df.to_csv('avg_price.csv', index=False)