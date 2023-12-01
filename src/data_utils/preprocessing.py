# include build-in libraries

import pandas as pd
from transformers import PreTrainedModel, PreTrainedTokenizer

# include self-defined libraries
from src.data_utils import constants


def encode_sapo(
    news_df: pd.DataFrame = None,
    tokenizer: PreTrainedTokenizer = None,
    MAX_SAPO_LENGTH: int = 64,
):
    """
    This function is to encode sapo col information to vector using pre-train language models
    to save memory while training for low-memory PC

    Args:
        news_path: path to the news.tsv file
        tokenizer: a tokenizer to encoder, such as roberta,...

    Returns:
        True, encoded_sapo: List if successful
        False, None if otherwise
    """

    # Check arguments is not empty
    if news_df is None or tokenizer is None:
        raise ValueError("Args must not be specified")
    try:
        # Get the information about sapo column
        sapo_list = list(news_df.get(constants.SAPO))

        # Define a list to store the sapo information sorted like in newss.tsv file
        sapo_encoding_list = []

        for sapo_element in sapo_list:
            sapo_encoding = tokenizer.encode(
                sapo_element,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_SAPO_LENGTH,
            )
            sapo_encoding_list.append(sapo_encoding)

        return True, sapo_encoding_list

    except:
        return False, None


def encode_titles(
    news_df: pd.DataFrame = None,
    tokenizer: PreTrainedTokenizer = None,
    MAX_TITLE_LENGTH: int = 32,
):
    """
    This function is to encode title col information to vector using pre-train language models
    to save memory while training for low-memory PC

    Args:
        news_path: path to the news.tsv file
        tokenizer: a tokenizer to encoder, such as roberta,...

    Returns:
        True, encoded_title: List if successful
        False, None if otherwise
    """

    # Check arguments is not empty
    if news_df is None or tokenizer is None:
        raise ValueError("Args must not be specified")
    try:
        # Get the information about title column
        title_list = list(news_df.get(constants.TITLE))

        # Define a list to store the sapo information sorted like in newss.tsv file
        title_encoding_list = []

        for title_element in title_list:
            title_encoding = tokenizer.encode(
                title_element,
                add_special_tokens=True,
                truncation=True,
                max_length=MAX_TITLE_LENGTH,
            )
            title_encoding_list.append(title_encoding)

        return True, title_encoding_list

    except:
        return False, None


def encode_news_data(
    news_path: str = None,
    tokenizer: PreTrainedTokenizer = None,
    model: PreTrainedModel = None,
    MAX_SAPO_LENGTH: int = 64,
    MAX_TITLE_LENGTH: int = 32,
):
    """
    This function is to encode text format in news.tsv file to vector
    and overwrite this file.
    Args:
        news_path: path to the news.tsv file
        tokenizer: a tokenizer to encoder, such as roberta,...
    Returns:
        True, news_path if successful,
        False, None if otherwise
    """
    if news_path is None or tokenizer is None:
        raise ValueError("Args must not be specified")
    try:
        # Read the news.tsv file
        news_df = pd.read_csv(news_path, sep="\t", header=None)
        news_df = news_df.dropna(how="any")
        sapo_check_flag, encoding_sapo_list = encode_sapo(
            news_df, tokenizer, MAX_SAPO_LENGTH
        )
        title_check_flag, encoding_title_list = encode_titles(
            news_df, tokenizer, MAX_TITLE_LENGTH
        )
        if sapo_check_flag:
            news_df[constants.SAPO] = encoding_sapo_list
        if title_check_flag:
            news_df[constants.TITLE] = encoding_title_list
        news_df.to_csv(news_path, sep="\t", index=False)
        return True, news_path
    except:
        return False, None


if __name__ == "__main__":
    # flag, dest_path = category2id()
    # print(dest_path)

    flag, dest_path = user2id("../data/mind-demo/train/behaviors.tsv")
    print(dest_path)

    # encode_news_data(../data/mind-demo/train/behaviors.tsv)
