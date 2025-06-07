import polars as pl
import data_preprocessing

# 加载训练数据
df_train = pl.read_csv("virus_train.csv")
after_clear_train = data_preprocessing.clear_df(df_train)

# 加载测试数据
df_test = pl.read_csv("virus_test.csv")
after_clear_test = data_preprocessing.clear_df(df_test)


if __name__ == "__main__":
    print("--- Training Data ---")
    print("Original training DataFrame head:\n", df_train.head())
    print("\nTraining DataFrame after preprocessing (after_clear_train) head:\n")
    if after_clear_train is not None and not after_clear_train.is_empty():
        print(f"Columns in after_clear_train: {after_clear_train.columns}")
        cols_to_show = ["id"]
        if "cleaned_content" in after_clear_train.columns:
            cols_to_show.append("cleaned_content")
        elif "content" in after_clear_train.columns:
            cols_to_show.append("content")

        existing_cols_to_show = [
            col for col in cols_to_show if col in after_clear_train.columns
        ]
        if existing_cols_to_show:
            print(after_clear_train.select(existing_cols_to_show).head())
        else:
            print(
                "Neither 'cleaned_content' nor 'content' found in after_clear_train DataFrame for head printing.",
                after_clear_train.head(),
            )
    else:
        print("after_clear_train DataFrame is None or empty.")

    print("\n\n--- Test Data ---")
    print("Original test DataFrame head:\n", df_test.head())
    print("\nTest DataFrame after preprocessing (after_clear_test) head:\n")
    if after_clear_test is not None and not after_clear_test.is_empty():
        print(f"Columns in after_clear_test: {after_clear_test.columns}")
        cols_to_show_test = ["id"]  # 测试集通常没有标签列，但可能有id和content
        if "cleaned_content" in after_clear_test.columns:
            cols_to_show_test.append("cleaned_content")
        elif "content" in after_clear_test.columns:
            cols_to_show_test.append("content")

        existing_cols_to_show_test = [
            col for col in cols_to_show_test if col in after_clear_test.columns
        ]
        if existing_cols_to_show_test:
            print(after_clear_test.select(existing_cols_to_show_test).head())
        else:
            print(
                "Neither 'cleaned_content' nor 'content' found in after_clear_test DataFrame for head printing.",
                after_clear_test.head(),
            )
    else:
        print("after_clear_test DataFrame is None or empty.")
