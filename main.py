import argparse
import os

from src.data_preprocessing.behavior_preprocess import behavior_preprocessing
from src.data_preprocessing.news_preprocess import news_preprocessing
from src.glove.generate_glove_dict import glovePKL
from src.mind import download_extract_small_mind


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="where is the dataset path")
    parser.add_argument("--data_size", help="the size of MIND data")
    parser.add_argument("--pkl_dir", help="Where is the pkl folder")
    parser.add_argument("--glove_url", help="the url of glove representation")
    args = parser.parse_args()
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        print("*" * 100)
        print("Do not find the path of dataset ---> Preparing the dataset:")
        print(f"\t Download the MIND dataset ({args.data_size})")
        train_path, validation_path = download_extract_small_mind(
            size=args.data_size, dest_path=data_dir, clean_zip_file=True
        )

        print(
            "Training path: {}\nValidation path: {}".format(
                train_path, validation_path
            )
        )
    else:
        print("Found the path of dataset ---> Skipping the dataset processing")

    # Glove dict
    print("*" * 100)
    pkl_dir = args.pkl_dir
    glove_url = args.glove_url
    if not os.path.exists(pkl_dir):
        os.mkdir(pkl_dir)
    if not os.path.exists("./pkl/glove_dict.pkl"):
        glove_dict_path = glovePKL(glove_url, pkl_dir)
        print(f"Glove dict path: {glove_dict_path}")
    print("*" * 100)
    behavior_preprocessing(data_path=data_dir, pkl_dir=pkl_dir)
    print("*" * 100)
    news_preprocessing(data_path=data_dir, pkl_path=pkl_dir)


if __name__ == "__main__":
    main()
