import argparse
import os
import pickle
from glob import glob

import numpy as np

from src.data_preprocessing.behavior_preprocess import behavior_preprocessing
from src.data_preprocessing.behavior_preprocess_evaluation import (
    generate_validation_data_pipeline,
)
from src.data_preprocessing.news_preprocess import news_preprocessing
from src.glove.generate_glove_dict import glovePKL
from src.mind import download_extract_small_mind
from utils import train_and_evaluate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", help="where is the dataset path")
    parser.add_argument("--data_size", help="the size of MIND data")
    parser.add_argument("--pkl_dir", help="Where is the pkl folder")
    parser.add_argument("--glove_url", help="the url of glove representation")
    parser.add_argument(
        "--training_data",
        type=str,
        default="./pkl/data_train.pkl",
        help="Input training data path",
    )
    parser.add_argument(
        "--validation_data",
        type=str,
        default="./pkl/validation_data.pkl",
        help="Input validation data path",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1, help="batch size of training"
    )
    parser.add_argument(
        "--news_final_embed_size",
        type=int,
        default=50,
        help="the size of a piece of embeded news",
    )
    parser.add_argument(
        "--history_num_head",
        type=int,
        default=10,
        help="num of attention heads for user histories",
    )
    parser.add_argument(
        "--history_attn_vector_size",
        type=int,
        default=32,
        help="dim of attn vector for encoding history",
    )
    parser.add_argument(
        "--recent_num_head",
        type=int,
        default=10,
        help="num of attention heads for user recent behavior",
    )
    parser.add_argument(
        "--recent_attn_vector_size",
        type=int,
        default=32,
        help="dim of attn vector for encoding recent reading behaviors",
    )
    parser.add_argument(
        "--category_id",
        type=str,
        default="./pkl/category_id.pkl",
        help="category_id dictionary",
    )

    parser.add_argument(
        "--TitleWordId_embeddings",
        type=str,
        default="./pkl/TitleWord_Id_embeddings.npy",
        help="TitleWordId_embeddings matrix",
    )

    parser.add_argument(
        "--AbstractWordId_embeddings",
        type=str,
        default="./pkl/AbstractWord_Id_embeddings.npy",
        help="AbstractWordId_embeddings matrix",
    )

    parser.add_argument(
        "--EntityId_embeddings",
        type=str,
        default="./pkl/EntityId_embeddings.npy",
        help="EntityId_embeddings matrix",
    )
    parser.add_argument(
        "--num_head_text",
        type=int,
        default=16,
        help="num of attention heads for text",
    )
    parser.add_argument(
        "--num_head_entity",
        type=int,
        default=8,
        help="num of attention heads for entity",
    )
    parser.add_argument(
        "--text_attn_vector_size",
        type=int,
        default=200,
        help="dim of attn vector for encoding title/abstract",
    )
    parser.add_argument(
        "--entity_attn_vector_size",
        type=int,
        default=64,
        help="dim of attn vector for encoding title entities/abstract entities",
    )
    parser.add_argument(
        "--news_final_attn_vector_size",
        type=int,
        default=48,
        help="dim of attn vector for encoding a news",
    )
    parser.add_argument(
        "--final_attn_vector_size",
        type=int,
        default=32,
        help="dim of attn vector for encoding a news",
    )
    parser.add_argument(
        "--newsID_categoryID",
        type=str,
        default="./pkl/newID_catgoryID.pkl",
        help="newsID_categoryID dictionary",
    )
    parser.add_argument(
        "--newsID_subcategoryID",
        type=str,
        default="./pkl/newID_subcategoryID.pkl",
        help="newsID_subcategoryID dictionary",
    )

    parser.add_argument(
        "--newsID_TitleWordID",
        type=str,
        default="./pkl/newsID_TitleWord_ID.pkl",
        help="newsID_TitleWordId dictionary",
    )

    parser.add_argument(
        "--newsID_AbstractWordID",
        type=str,
        default="./pkl/newsID_AbstractWord_ID.pkl",
        help="newsID_AbstractWordId dictionary",
    )

    parser.add_argument(
        "--newsID_titleEntityId_conf",
        type=str,
        default="./pkl/newsID_titleEntityId_conf.pkl",
        help="newsID_titleEntityId dictionary",
    )

    parser.add_argument(
        "--newsID_abstractEntityId_conf",
        type=str,
        default="./pkl/newsID_abstractEntityId_conf.pkl",
        help="newsID_abstractEntityId dictionary",
    )
    parser.add_argument(
        "--pack_loss",
        type=str,
        default="pack_loss.pkl",
        help="the name of the loss file",
    )
    parser.add_argument(
        "--ranking_name",
        type=str,
        default="prediction.txt",
        help="the name of the prediction file",
    )
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
    if not len(pkl_dir + "./*.pkl") + len(pkl_dir + "./*.npy"):
        glove_dict_path = glovePKL(glove_url, pkl_dir)
        print(f"Glove dict path: {glove_dict_path}")
        print("*" * 100)
        behavior_preprocessing(data_path=data_dir, pkl_dir=pkl_dir)
        print("*" * 100)
        news_preprocessing(data_path=data_dir, pkl_path=pkl_dir)
        print("*" * 100)
        generate_validation_data_pipeline(data_dir=data_dir, pkl_dir=pkl_dir)
        print("*" * 100)
    print("Loading the training data")
    f = open(args.training_data, "rb")
    training_data = pickle.load(f)["training_data"]
    f.close()

    print("*" * 100)
    print("Loading the validation data")
    f = open(args.validation_data, "rb")
    validation_data = pickle.load(f)["validation_data"]
    f.close()

    # trim very long reading sequences
    for i, j in enumerate(training_data):
        if type(j[1]) != float and len(j[1]) > 60:
            training_data[i][1] = j[1][-60:]
        if type(j[2]) != float and len(j[2]) > 10:
            training_data[i][2] = j[2][-10:]

    for i, j in enumerate(validation_data):
        if type(j[1]) != float and len(j[1]) > 60:
            validation_data[i][1] = j[1][-60:]
        if type(j[2]) != float and len(j[2]) > 10:
            validation_data[i][2] = j[2][-10:]

    print("*" * 100)
    print("loading all dictionaries")

    f = open(args.category_id, "rb")
    category_id = pickle.load(f)
    num_category = len(category_id["category_id"].keys())

    f = open(args.subcategory_id, "rb")
    subcategory_id = pickle.load(f)
    num_subcategory = len(subcategory_id["subcategory_id"].keys())

    print("*" * 100)
    print("Model training")
    batch_size = args.batch_size
    training_data = np.array(training_data)
    validation_data = np.array(validation_data)
    training_len = len(training_data)
    num_iterations = training_len // batch_size
    validation_len = len(validation_data)
    num_iterations_vali = validation_len // batch_size
    print("number of training samples", training_len)
    news_final_embed_size = args.news_final_embed_size
    history_num_head = args.history_num_head
    history_attn_vector_size = args.history_attn_vector_size
    recent_num_head = args.recent_num_head
    recent_attn_vector_size = args.recent_attn_vector_size
    title_embed_matrix = np.load(args.TitleWordId_embeddings)
    abstract_embed_matrix = np.load(args.AbstractWordId_embeddings)
    entity_embed_matrix = np.load(args.EntityId_embeddings)
    num_head_text = args.num_head_text
    num_head_entity = args.num_head_entity
    text_attn_vector_size = args.text_attn_vector_size
    entity_attn_vector_size = args.entity_attn_vector_size
    news_final_attn_vector_size = args.news_final_attn_vector_size
    final_attn_vector_size = args.final_attn_vector_size
    f = open(args.newsID_categoryID, "rb")
    newsID_categoryID = pickle.load(f)
    f.close()
    f = open(args.newsID_subcategoryID, "rb")
    newsID_subcategoryID = pickle.load(f)
    f.close()
    f = open(args.newsID_TitleWordID, "rb")
    newsID_TitleWordID = pickle.load(f)
    f.close()
    f = open(args.newsID_AbstractWordID, "rb")
    newsID_AbstractWordID = pickle.load(f)
    f.close()
    f = open(args.newsID_titleEntityId_conf, "rb")
    newsID_titleEntityId_conf = pickle.load(f)
    f.close()
    f = open(args.newsID_abstractEntityId_conf, "rb")
    newsID_abstractEntityId_conf = pickle.load(f)
    f.close()
    model_name = args.model_name
    pack_loss = args.pack_loss
    train_and_evaluate(
        batch_size=batch_size,
        training_data=training_data,
        validation_data=validation_data,
        news_final_embed_size=news_final_embed_size,
        history_num_head=history_num_head,
        history_attn_vector_size=history_attn_vector_size,
        recent_num_head=recent_num_head,
        recent_attn_vector_size=recent_attn_vector_size,
        num_category=num_category,
        num_subcategory=num_subcategory,
        title_embed_matrix=title_embed_matrix,
        abstract_embed_matrix=abstract_embed_matrix,
        entity_embed_matrix=entity_embed_matrix,
        num_head_text=num_head_text,
        num_head_entity=num_head_entity,
        text_attn_vector_size=text_attn_vector_size,
        entity_attn_vector_size=entity_attn_vector_size,
        news_final_attn_vector_size=news_final_attn_vector_size,
        final_attn_vector_size=final_attn_vector_size,
        newsID_categoryID=newsID_categoryID,
        newsID_subcategoryID=newsID_subcategoryID,
        newsID_TitleWordID=newsID_TitleWordID,
        newsID_AbstractWordID=newsID_AbstractWordID,
        newsID_titleEntityId_conf=newsID_titleEntityId_conf,
        newsID_abstractEntityId_conf=newsID_abstractEntityId_conf,
        num_iterations=num_iterations,
        num_iterations_vali=num_iterations_vali,
        pack_loss=pack_loss,
        model_name=model_name,
    )


if __name__ == "__main__":
    main()
