import polars as pl
from data_preprocessing import clear_df
import os

# 加载训练数据
df_train = pl.read_csv("virus_train.csv")
after_clear_train = clear_df(df_train)

# 加载测试数据
df_test = pl.read_csv("virus_test.csv")
after_clear_test = clear_df(df_test)

# 尝试加载分词后的 ndjson 文件
df_train_token = None
df_test_token = None

df_train_token_vector = None
df_test_token_vector = None

train_token_file = "virus_train_token.ndjson"
test_token_file = "virus_test_token.ndjson"

train_token_vector_file = "virus_train_token_vectors.ndjson"
test_token_vector_file = "virus_test_token_vectors.ndjson"

if os.path.exists(train_token_file):
    try:
        df_train_token = pl.read_ndjson(train_token_file)
        print(f"成功加载 {train_token_file}")
    except Exception as e:
        print(f"加载 {train_token_file} 时出错: {e}")
else:
    print(f"文件 {train_token_file} 不存在，跳过加载。")

if os.path.exists(test_token_file):
    try:
        df_test_token = pl.read_ndjson(test_token_file)
        print(f"成功加载 {test_token_file}")
    except Exception as e:
        print(f"加载 {test_token_file} 时出错: {e}")
else:
    print(f"文件 {test_token_file} 不存在，跳过加载。")

if os.path.exists(train_token_vector_file):
    try:
        df_train_token_vector = pl.read_ndjson(train_token_vector_file)
        print(f"成功加载 {train_token_vector_file}")
    except Exception as e:
        print(f"加载 {train_token_vector_file} 时出错: {e}")
else:
    print(f"文件 {train_token_vector_file} 不存在，跳过加载。")

if os.path.exists(test_token_vector_file):
    try:
        df_test_token_vector = pl.read_ndjson(test_token_vector_file)
        print(f"成功加载 {test_token_vector_file}")
    except Exception as e:
        print(f"加载 {test_token_vector_file} 时出错: {e}")
else:
    print(f"文件 {test_token_vector_file} 不存在，跳过加载。")

if __name__ == "__main__":
    # 打印原始数据
    print("--- 原始数据 ---")
    print("原始训练 DataFrame head:\n", df_train.head())
    print("\n原始测试 DataFrame head:\n", df_test.head())

    # 清理后
    print("\n\n--- 清理后数据 ---")
    print("训练数据清理后 (after_clear_train) head:\n", after_clear_train.head())
    print("\n测试数据清理后 (after_clear_test) head:\n", after_clear_test.head())

    # 分词后
    print("\n\n--- 分词后数据 (如果已加载) ---")
    if df_train_token is not None:
        print("训练数据分词后 (df_train_token) head:\n", df_train_token.head())
    else:
        print("df_train_token 未加载。")

    if df_test_token is not None:
        print("\n测试数据分词后 (df_test_token) head:\n", df_test_token.head())
    else:
        print("df_test_token 未加载。")

    # 向量化后
    print("\n\n--- 向量化后数据 (如果已加载) ---")
    if df_train_token_vector is not None:
        print(
            "训练数据向量化后 (df_train_token_vector) head:\n",
            df_train_token_vector.head(),
        )
    else:
        print("df_train_token_vector 未加载。")
    if df_test_token_vector is not None:
        print(
            "\n测试数据向量化后 (df_test_token_vector) head:\n",
            df_test_token_vector.head(),
        )
    else:
        print("df_test_token_vector 未加载。")
