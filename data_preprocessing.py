import re
import emoji
from zhconv import convert
import polars as pl


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

    # 去除@用户名
    text = re.sub(r"@[^\s@]+", "", text)

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
