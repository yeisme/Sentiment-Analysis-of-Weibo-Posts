import re
import emoji
from zhconv import convert
import polars as pl
import spacy

# 加载spacy中文模型
try:
    nlp = spacy.load("zh_core_web_md")
except OSError:
    print(
        "未找到'zh_core_web_md'模型。请运行 'python -m spacy download zh_core_web_md' 来下载它。"
    )
    nlp = None


def clean_text(text: str) -> str:
    """
    文本清洗函数
    - 去除URL、@用户名、特殊符号
    - 处理表情符号和emoji
    - 统一繁简体转换
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    # 去除URL链接
    text = re.sub(
        r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )
    text = re.sub(
        r"www\.(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
        "",
        text,
    )

    # 去除@用户名以及类似 //@xxx: 的模式
    text = re.sub(r"//@.*?:", "", text)

    # 处理emoji - 将emoji转换为文字描述然后移除
    text = emoji.demojize(text, language="zh")
    text = re.sub(r":[^:]*:", "", text)

    # 去除表情符号和特殊字符，保留中文、英文、数字和基本标点
    text = re.sub(
        r'[^\u4e00-\u9fa5a-zA-Z0-9，。！？；：、""\'\"\[\]（）【】《》#]', "", text
    )

    # 繁简体转换 - 统一转换为简体中文
    text = convert(text, "zh-cn")

    # 去除多余的空白字符
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clear_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    清理DataFrame中的content列
    """
    if "content" not in df.columns:
        raise ValueError("DataFrame must contain a 'content' column.")

    df_cleaned = df.with_columns(
        pl.col("content")
        .map_elements(clean_text, return_dtype=pl.String)
        .alias("cleaned_content")
    )
    return df_cleaned


def tokenize_text(text: str) -> list[str]:
    """
    使用spacy对文本进行分词
    """
    if nlp is None or not isinstance(text, str):
        return []
    doc = nlp(text)
    return [token.text for token in doc]


def tokenize_df(df: pl.DataFrame) -> pl.DataFrame:
    """
    将DataFrame中的cleaned_content列转换为分词结果，并添加为新列'token'
    使用nlp.pipe()进行批量处理以提高效率。
    """
    if "cleaned_content" not in df.columns:
        raise ValueError(
            "DataFrame must contain a 'cleaned_content' column for tokenization."
        )

    if nlp is None:
        raise RuntimeError("Spacy 'zh_core_web_sm' model not loaded. Cannot tokenize.")

    texts_to_tokenize = df["cleaned_content"].to_list()

    tokenized_texts = [
        [token.text for token in doc] for doc in nlp.pipe(texts_to_tokenize)
    ]

    # 创建一个新的Polars Series包含分词结果
    token_series = pl.Series("token", tokenized_texts, dtype=pl.List(pl.String))

    df_tokenized = df.with_columns(token_series)

    return df_tokenized
