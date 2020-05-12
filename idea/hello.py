import pandas as pd
from sqlalchemy import create_engine

if __name__ == '__main__':
    print('hello world')
    conn_169 = create_engine('mysql+pymysql://root:123456@111.230.136.169:3308/test?charset=utf8')
    if conn_169:
        print("connect success")

    df_temp = pd.read_sql_table('user_info', conn_169)
    print(df_temp)
