import polars as pl
from spacy.tokens import Doc
from data_preprocessing import nlp
import warnings
import os
from data_load import (
    train_token_file,
    test_token_file,
    train_token_vector_file,
    test_token_vector_file,
)


def token2vec(df: pl.DataFrame) -> pl.DataFrame:
    """
    将DataFrame中的'token'列（预分词的词列表）转换为句子向量。
    - 使用当前加载的spaCy模型（例如 'zh_core_web_sm'）计算句子向量。
    - 句子向量是其组成词向量的平均值（由spaCy的Doc.vector属性提供）。
    """
    if "token" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'token' column with tokenized text (list of strings)."
        )

    if nlp is None:
        raise RuntimeError(
            "spaCy model (nlp) not loaded. Cannot generate sentence vectors."
        )

    if nlp.vocab.vectors_length == 0:
        warnings.warn(
            f"The loaded spaCy model ('{nlp.meta['name']}') does not have word vectors. "
            "Sentence vectors will be empty. "
            "Consider using a model with vectors (e.g., 'zh_core_web_md' or 'zh_core_web_lg').",
            UserWarning,
        )

    def get_sentence_vector(tokens: list[str]) -> list[float]:
        """
        辅助函数：将词列表转换为句子向量。

        - 直接从词列表创建spaCy Doc对象
        - 这利用了spaCy的词汇表和向量计算能力。
        - 对于空的词列表，doc.vector将是正确维度的零向量。
        """

        doc = Doc(nlp.vocab, words=tokens)
        return doc.vector.tolist()

    # 将get_sentence_vector函数应用于'token'列中的每个元素（词列表）。
    vector_column = (
        df["token"]
        .map_elements(get_sentence_vector, return_dtype=pl.List(pl.Float32))
        .alias("sentence_vectors")
    )

    df_with_vectors = df.with_columns(vector_column)
    return df_with_vectors


if __name__ == "__main__":
    if os.path.exists(train_token_file):
        try:
            df_train_token = pl.read_ndjson(train_token_file)
            print(f"成功加载 {train_token_file}")
            df_train_vectors = token2vec(df_train_token)
            df_train_vectors.write_ndjson(train_token_vector_file)
            print(f"训练数据向量化结果已保存到 {train_token_vector_file}")

        except Exception as e:
            print(f"加载 {train_token_file} 时出错: {e}")
    else:
        print(f"文件 {train_token_file} 不存在，跳过加载。")

    if os.path.exists(test_token_file):
        try:
            df_test_token = pl.read_ndjson(test_token_file)
            print(f"成功加载 {test_token_file}")
            df_test_vectors = token2vec(df_test_token)
            df_test_vectors.write_ndjson(test_token_vector_file)
            print(f"测试数据向量化结果已保存到 {test_token_vector_file}")

        except Exception as e:
            print(f"加载 {test_token_file} 时出错: {e}")
    else:
        print(f"文件 {test_token_file} 不存在，跳过加载。")
