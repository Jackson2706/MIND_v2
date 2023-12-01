# include built-in libraries
import os

from transformers import PreTrainedModel, PreTrainedTokenizer

# include self-defined libraries
from src.data_utils.download_utils import (
    download_path,
    maybe_download,
    unzip_file,
)
from src.data_utils.preprocessing import encode_news_data

URL_MIND_LARGE_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_train.zip"
)
URL_MIND_LARGE_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDlarge_dev.zip"
)
URL_MIND_SMALL_TRAIN = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_train.zip"
)
URL_MIND_SMALL_VALID = (
    "https://mind201910small.blob.core.windows.net/release/MINDsmall_dev.zip"
)
URL_MIND_DEMO_TRAIN = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_train.zip"
)
URL_MIND_DEMO_VALID = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_dev.zip"
)
URL_MIND_DEMO_UTILS = (
    "https://recodatasets.z20.web.core.windows.net/newsrec/MINDdemo_utils.zip"
)

URL_MIND = {
    "large": (URL_MIND_LARGE_TRAIN, URL_MIND_LARGE_VALID),
    "small": (URL_MIND_SMALL_TRAIN, URL_MIND_SMALL_VALID),
    "demo": (URL_MIND_DEMO_TRAIN, URL_MIND_DEMO_VALID),
}


def download_mind(size="small", dest_path=None):
    """Download MIND dataset

    Args:
        size (str): Dataset size. One of ["small", "large"]
        dest_path (str): Download path. If path is None, it will download the dataset on a temporal path

    Returns:
        str, str: Path to train and validation sets.
    """
    size_options = ["small", "large", "demo"]
    if size not in size_options:
        raise ValueError(
            f"Wrong size option, available options are {size_options}"
        )
    url_train, url_valid = URL_MIND[size]
    with download_path(dest_path) as path:
        train_path = maybe_download(url=url_train, work_directory=path)
        valid_path = maybe_download(url=url_valid, work_directory=path)
    return train_path, valid_path


def extract_mind(
    train_zip,
    valid_zip,
    train_folder="train",
    valid_folder="valid",
    clean_zip_file=True,
):
    """Extract MIND dataset

    Args:
        train_zip (str): Path to train zip file
        valid_zip (str): Path to valid zip file
        train_folder (str): Destination forder for train set
        valid_folder (str): Destination forder for validation set

    Returns:
        str, str: Train and validation folders
    """
    dir_folder = os.path.dirname(train_zip)
    train_path = os.path.join(dir_folder, train_folder)
    valid_path = os.path.join(dir_folder, valid_folder)
    unzip_file(train_zip, train_path, clean_zip_file=clean_zip_file)
    unzip_file(valid_zip, valid_path, clean_zip_file=clean_zip_file)
    return train_path, valid_path


def download_extract_small_mind(
    size: str = "small", dest_path: str = None, clean_zip_file: bool = True
):
    train_zip_path, valid_zip_path = download_mind(size, dest_path)
    train_path, valid_path = extract_mind(
        train_zip=train_zip_path,
        valid_zip=valid_zip_path,
        clean_zip_file=clean_zip_file,
    )
    return train_path, valid_path


def preprocess(
    train_path: str = None,
    valid_path: str = None,
    tokenizer: PreTrainedTokenizer = None,
    model: PreTrainedModel = None,
    MAX_SAPO_LENGTH: int = 50,
    MAX_TITLE_LENGTH: int = 50,
):
    """
    Processes the news.tsv file in train phase or validation phase
    """

    # Check if train_path or validation_path is empty
    if train_path is None and valid_path is None:
        raise ValueError("At least train_path or valid_path must be specified")

    # Check if tokenizer is empty
    if tokenizer is None:
        raise ValueError("tokenizer must be specified")
    try:
        if train_path:
            # Create path
            train_news_path = os.path.join(train_path, "news.tsv")
            # Encode train news data
            train_news_check_flag, train_news_path = encode_news_data(
                news_path=train_news_path,
                tokenizer=tokenizer,
                MAX_SAPO_LENGTH=MAX_SAPO_LENGTH,
                MAX_TITLE_LENGTH=MAX_TITLE_LENGTH,
            )
        if valid_path:
            valid_news_path = os.path.join(valid_path, "news.tsv")
            valid_news_check_flag, valid_news_path = encode_news_data(
                news_path=valid_news_path,
                tokenizer=tokenizer,
                MAX_SAPO_LENGTH=MAX_SAPO_LENGTH,
                MAX_TITLE_LENGTH=MAX_TITLE_LENGTH,
            )

        return True
    except:
        return False


if __name__ == "__main__":
    train_path, valid_path = download_extract_small_mind(
        size="small", dest_path="./data/mind-demo"
    )
