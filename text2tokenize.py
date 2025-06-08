"""
对文本数据进行分词处理，这是一个独立的脚本，直接运行
"""

from data_preprocessing import tokenize_df
from data_load import (
    after_clear_train,
    after_clear_test,
    train_token_file,
    test_token_file,
)
import os
import polars as pl


if __name__ == "__main__":
    if os.path.exists(train_token_file):
        try:
            df_train_token = pl.read_ndjson(train_token_file)
            print(f"成功加载 {train_token_file}")

        except Exception as e:
            print(f"加载 {train_token_file} 时出错: {e}")
    else:
        df_train_token = tokenize_df(after_clear_train)
        df_train_token.write_ndjson(train_token_file)
        print(f"文件 {train_token_file} 不存在，跳过加载。")

    if os.path.exists(test_token_file):
        try:
            df_test_token = pl.read_ndjson(test_token_file)
            print(f"成功加载 {test_token_file}")
        except Exception as e:
            print(f"加载 {test_token_file} 时出错: {e}")
    else:
        df_test_token = tokenize_df(after_clear_test)
        df_test_token.write_ndjson(test_token_file)
        print(f"文件 {test_token_file} 不存在，跳过加载。")
