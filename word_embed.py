#!/usr/bin/env python3
import polars as pl
from gensim.models import Word2Vec
import logging
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import click

logging.basicConfig(
    format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO
)


def prepare_corpus_from_dataframe(
    df: pl.DataFrame, cleaned_text_column: str = "cleaned_content", min_len: int = 1
) -> list[list[str]]:
    """
    # 从已加载和预处理的 DataFrame 准备 Word2Vec 训练所需的语料库。

    Args:
        df (pl.DataFrame): 包含文本数据的 Polars DataFrame。
        cleaned_text_column (str, optional): 包含已清洗文本的列名。默认为 "cleaned_content"。
        min_len (int, optional): 句子分词后的最小长度。默认为 1。

    Returns:
        list[list[str]]: 语料库，每个元素是一个分词后的句子 (list of str)。
    """
    if df is None:
        print("Error: Input DataFrame is None. Cannot prepare corpus.")
        sample_sentences = [
            "示例 语料库 因为 数据 加载 失败",
            "请 检查 data_load 模块",
        ]
        return [s.split() for s in sample_sentences]

    if cleaned_text_column not in df.columns:
        print(
            f"Error: Column '{cleaned_text_column}' not found in DataFrame. Available columns: {df.columns}"
        )
        # 尝试使用 'content' 列并应用 clean_text，如果 'data_preprocessing' 可用
        if "content" in df.columns:
            print("Attempting to use 'content' column and clean it now.")
            try:
                from data_preprocessing import clean_text as fallback_clean_text

                corpus = []
                for text_series in df.select(pl.col("content")).iter_rows():
                    text = text_series[0]
                    if text and isinstance(text, str):
                        cleaned = fallback_clean_text(text)
                        tokens = cleaned.split()
                        if len(tokens) >= min_len:
                            corpus.append(tokens)
                if corpus:
                    print(
                        f"Corpus prepared from 'content' column with {len(corpus)} sentences."
                    )
                    return corpus
                else:
                    print("Failed to prepare corpus from 'content' column.")
                    return []
            except ImportError:
                print(
                    "Fallback clean_text not available. Cannot process 'content' column."
                )
                return []
        else:
            return []

    print(f"Tokenizing text from column: '{cleaned_text_column}'...")
    corpus = []
    # 确保迭代非空文本
    for text_val in df.get_column(cleaned_text_column):
        if text_val and isinstance(text_val, str):
            tokens = text_val.split()  # 假设 cleaned_text_column 中的文本已准备好分词
            if len(tokens) >= min_len:
                corpus.append(tokens)

    print(f"Corpus prepared with {len(corpus)} sentences.")
    if not corpus:
        print(
            "Warning: Corpus is empty after processing. Check your data and cleaning steps in data_load.py and data_preprocessing.py."
        )
    return corpus


def train_word2vec_model(
    corpus: list[list[str]],
    vector_size: int = 100,
    window: int = 5,
    min_count: int = 5,
    workers: int = 4,
    sg: int = 0,
    epochs: int = 10,
    model_save_path: str = "word2vec.model",
) -> Word2Vec | None:
    """
    # 训练 Word2Vec 模型并保存。

    Args:
        corpus (list[list[str]]): 预处理好的语料库。
        vector_size (int, optional): 词向量维度。默认为 100。
        window (int, optional): 上下文窗口大小。默认为 5。
        min_count (int, optional): 忽略总频率低于此值的词。默认为 5。
        workers (int, optional): 训练时使用的线程数。默认为 4。
        sg (int, optional): 训练算法 (0: CBOW, 1: Skip-gram)。默认为 0 (CBOW)。
        epochs (int, optional): 迭代次数。默认为 10。
        model_save_path (str, optional): 模型保存路径。默认为 "word2vec.model"。

    Returns:
        Word2Vec | None: 训练好的 Word2Vec 模型，如果语料库为空则返回 None。
    """
    if not corpus:
        print("Cannot train Word2Vec model: Corpus is empty.")
        return None

    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=corpus,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        sg=sg,
        epochs=epochs,
    )

    print(f"Word2Vec model trained. Vocabulary size: {len(model.wv.key_to_index)}")

    if model_save_path:
        model.save(model_save_path)
        print(f"Model saved to {model_save_path}")

    return model


def load_word2vec_model(model_path: str = "word2vec.model") -> Word2Vec | None:
    """
    # 加载预训练的 Word2Vec 模型。

    Args:
        model_path (str, optional): 模型文件路径。默认为 "word2vec.model"。

    Returns:
        Word2Vec | None: 加载的 Word2Vec 模型，如果加载失败则返回 None。
    """
    try:
        model = Word2Vec.load(model_path)
        print(f"Word2Vec model loaded from {model_path}")
        print(f"Vocabulary size: {len(model.wv.key_to_index)}")
        return model
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return None
    except Exception as e:
        print(f"Error loading Word2Vec model: {e}")
        return None


def debug_word2vec_model(model: Word2Vec | None, sample_word: str = "疫情"):
    """
    # 调试 Word2Vec 模型，打印示例词的词向量和相似词。

    Args:
        model (Word2Vec | None): 加载的 Word2Vec 模型。
        sample_word (str, optional): 用于调试的示例单词。默认为 "疫情"。
    """
    if not model:
        print("Model not loaded, cannot perform debug.")
        return

    if not model.wv.key_to_index:
        print("Model vocabulary is empty. Cannot perform debug.")
        return

    # 检查示例词是否在词汇表中
    if sample_word in model.wv:
        print(f"\n--- Debugging Model with word: '{sample_word}' ---")
        print(f"Vector for '{sample_word}':")
        print(model.wv[sample_word])

        try:
            similar_words = model.wv.most_similar(sample_word, topn=5)
            print(f"\nWords similar to '{sample_word}':")
            for word, score in similar_words:
                print(f"- {word}: {score:.4f}")
        except KeyError:
            print(
                f"\nCannot find similar words for '{sample_word}' as it might be too infrequent after pruning or not in vocab."
            )
    else:
        # 如果示例词不在词汇表中，尝试使用词汇表中的第一个词
        fallback_word = model.wv.index_to_key[0]
        print(
            f"\nWarning: Sample word '{sample_word}' not in vocabulary. "
            f"Using fallback word '{fallback_word}' for debugging."
        )
        if fallback_word:
            print(f"Vector for '{fallback_word}':")
            print(model.wv[fallback_word])
            try:
                similar_words = model.wv.most_similar(fallback_word, topn=5)
                print(f"\nWords similar to '{fallback_word}':")
                for word, score in similar_words:
                    print(f"- {word}: {score:.4f}")
            except KeyError:
                print(f"\nCannot find similar words for '{fallback_word}'.")
        else:
            print("Vocabulary is empty, cannot pick a fallback word.")
    print("--- End of Debug ---")


def visualize_word_embeddings(
    model: Word2Vec | None,
    num_words: int = 100,
    plot_title: str = "t-SNE visualization of Word Embeddings",
):
    """
    # Visualizes word embeddings using t-SNE.

    Args:
        model (Word2Vec | None): The Word2Vec model to visualize.
        num_words (int, optional): The number of words to visualize. Defaults to 100.
        plot_title (str, optional): The title of the plot. Defaults to "t-SNE visualization of Word Embeddings".
    """
    if not model:
        print("Model not loaded, cannot perform visualization.")
        return

    if not model.wv.key_to_index:
        print("Model vocabulary is empty. Cannot perform visualization.")
        return

    vocab = list(model.wv.index_to_key)
    if not vocab:
        print("Vocabulary is empty, cannot visualize.")
        return

    # 获取词向量
    words_to_visualize = vocab[: min(num_words, len(vocab))]
    word_vectors = np.array([model.wv[word] for word in words_to_visualize])

    if word_vectors.shape[0] < 2:
        print(
            f"Not enough word vectors ({word_vectors.shape[0]}) to visualize. Need at least 2."
        )
        return

    print(f"Performing t-SNE on {len(words_to_visualize)} word vectors...")
    # 使用 t-SNE 进行降维
    tsne = TSNE(
        n_components=2, random_state=42, perplexity=min(30, len(words_to_visualize) - 1)
    )  # Perplexity must be less than n_samples
    vectors_2d = tsne.fit_transform(word_vectors)

    try:
        font_path = fm.findfont(fm.FontProperties(family="SimHei"))
        if not font_path:
            font_path = fm.findfont(fm.FontProperties(family=None))
        custom_font = fm.FontProperties(fname=font_path)
        print(f"Using font: {custom_font.get_name()} from {font_path}")
    except Exception as e:
        print(
            f"Font setup failed: {e}. Chinese characters might not display correctly."
        )
        custom_font = fm.FontProperties()  # Fallback to default

    # 可视化
    plt.figure(figsize=(12, 12))
    for i, word in enumerate(words_to_visualize):
        plt.scatter(vectors_2d[i, 0], vectors_2d[i, 1])
        plt.annotate(
            word,
            xy=(vectors_2d[i, 0], vectors_2d[i, 1]),
            xytext=(5, 2),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontproperties=custom_font,
        )
    plt.title(plot_title, fontproperties=custom_font)
    plt.xlabel("t-SNE Dimension 1", fontproperties=custom_font)
    plt.ylabel("t-SNE Dimension 2", fontproperties=custom_font)
    plt.grid(True)
    print("Showing plot. Please close the plot window to continue.")
    plt.show()
    print("Plot closed.")


@click.command()
@click.option(
    "--act",
    default="prepare",
    type=click.Choice(["prepare", "train", "load", "visualize"], case_sensitive=False),
    help="Action (prepare/train/load/visualize) to perform.",
)
def word_option(act: str):
    from data_load import after_clear_train  # 修改导入

    if act == "prepare":
        prepare_corpus_from_dataframe(after_clear_train)  # 使用 after_clear_train
    elif act == "train":
        corpus = prepare_corpus_from_dataframe(
            after_clear_train
        )  # 使用 after_clear_train
        if corpus:
            train_word2vec_model(corpus)
        else:
            print("No corpus available for training.")
    elif act == "load":
        model = load_word2vec_model()
        if model:
            print(
                "Model loaded successfully. You can now use it for other tasks or debug/visualize."
            )
    elif act == "visualize":
        model = load_word2vec_model()
        if model:
            try:
                plt.rcParams["font.sans-serif"] = ["SimHei"]
                plt.rcParams["axes.unicode_minus"] = False
                print("Attempted to set Chinese font for matplotlib.")
            except Exception as e:
                print(
                    f"Could not set Chinese font for matplotlib: {e}. Characters might not display correctly."
                )
            visualize_word_embeddings(model)
        else:
            print("Failed to load model for visualization.")
    return


if __name__ == "__main__":
    word_option()
