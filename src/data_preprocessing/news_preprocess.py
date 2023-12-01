import os
import pickle
from ast import literal_eval

import numpy as np
import pandas as pd


def save_category_subcategory(
    category: pd.Series, subcategory: pd.Series, newsid: pd.Series, pkl_dir: str
):
    """
    This function is used to create dictionary to convert category and subcategory to ID and newid to categoryID and subcategoryID
    finally, saving to .pkl file
    """
    # remove the same rows, keep only one record
    cate_all = category.drop_duplicates()
    subcate_all = subcategory.drop_duplicates()

    # Create a dictionary with key is id and value is category element
    id2category = dict(zip(range(len(cate_all)), cate_all))
    # Create a dictionary with ket is category elemnet and value is id
    category2id = dict(zip(cate_all, range(len(cate_all))))

    # Create a dictionary to save category2id and id2category
    category_dict = {"category2id": category2id, "id2category": id2category}
    # Save to .pkl file
    category_id_path = os.path.join(pkl_dir, "category_id.pkl")
    f = open(category_id_path, "wb")
    pickle.dump(category_dict, f)
    f.close()

    # Create a dictionary with key is id and value is subcategory element
    id2subcategory = dict(zip(range(len(subcate_all)), subcate_all))
    # Create a dictionary with key is subcategory element and value is id
    subcategory2id = dict(zip(subcate_all, range(len(subcate_all))))

    # Create a dictionary to save subcategor2id and id2subcategory
    subcategory_dict = {
        "subcategory2id": subcategory2id,
        "id2subcategory": id2subcategory,
    }

    # Save to .pkl file
    subcategory_id_path = os.path.join(pkl_dir, "subcategory_id.pkl")
    f = open(subcategory_id_path, "wb")
    pickle.dump(subcategory_dict, f)
    f.close()
    # Create a dicttionary with key is newid and value is category_id
    cateid = category.apply(lambda x: category2id[x])
    new2categoryid = dict(zip(newsid, cateid))

    # Save to .pkl file
    newID_catgoryID_path = os.path.join(pkl_dir, "newID_catgoryID.pkl")
    f = open(newID_catgoryID_path, "wb")
    pickle.dump(new2categoryid, f)
    f.close()
    # Create a dictionary with key is new id and value is subcatgory_id
    subcateid = subcategory.apply(lambda x: subcategory2id[x])
    new2subcategoryid = dict(zip(newsid, subcateid))

    # Save to .pkl file
    newID_subcategoryID_path = os.path.join(pkl_dir, "newID_subcategoryID.pkl")
    f = open(newID_subcategoryID_path, "wb")
    pickle.dump(new2subcategoryid, f)

    f.close()  # Close file


def save_word_embeddings(
    column_words: pd.Series, newsid: pd.Series, file_name: str, pkl_dir: str
):
    """
    This function is used to embed title or abstract to vector
        Param:
            @param column_words may be a title and an abstract
            @param newsid is matching to titile or abstrac
            @param file_name: the name of file which need saving
    """
    # convert columnwords from pd.Series to string datatype
    column_word = column_words.apply(lambda x: str(x))
    split = column_word.str.split(" ")

    # removes all trailing characters from a string that belong to the specified set, in this case, the characters ' and 's'
    split = split.apply(lambda x: [i.rstrip("'s") for i in x])

    # removes all trailing characters from a string that belong to the specified set, which includes common punctuation marks such as ',.!?\'\"'
    split = split.apply(lambda x: [i.rstrip(",.!?'\"") for i in x])
    split = split.apply(lambda x: [i.lstrip(",.!?'\"") for i in x])
    words = split.tolist()  # Convert from pd.Series to List
    # Create vocabulary
    word_vocab = list(set(np.concatenate(np.array(words, dtype=object))))
    # Load word embeding from glove
    glove_dict_path = os.path.join(pkl_dir, "glove_dict.pkl")
    f = open(glove_dict_path, "rb")
    glove = pickle.load(f)  # Datatype: Dictionary
    glove_keys = glove.keys()
    f.close()
    # Remove element which is not existed in embeding word
    word_vocab = [i for i in word_vocab if i in glove_keys]

    # Create word embedding vocab and save to .npy file which help load data faster
    word_embedding = np.array([glove[i] for i in word_vocab])
    word_embedding_path = os.path.join(
        pkl_dir, file_name + "_Id_embeddings.npy"
    )
    np.save(word_embedding_path, word_embedding)
    print("Generated word embedding matrix")

    # Create a dictionary for word vocab and saving to .pkl file
    id2word = dict(zip(range(len(word_vocab)), word_vocab))
    word2id = dict(zip(word_vocab, range(len(word_vocab))))
    word_dict = {"word2id": word2id, "id2word": id2word}
    word_dictionary_path = os.path.join(
        pkl_dir, file_name + "_word_dictionary.pkl"
    )
    f = open(word_dictionary_path, "wb")
    pickle.dump(word_dict, f)
    print("Generated word dictionary !")
    f.close()
    # convert each word in titile or abstract to id in word dictionary
    split = split.apply(lambda x: [word2id[i] for i in x if i in word_vocab])

    # Create a dictionary with key is newsid and value is tittle or abstract which is convert to id in dictionary (word vocab) and save to .pkl file
    newsid2wordsid = dict(zip(newsid, split))
    newid2wordid_path = os.path.join(pkl_dir, "newsID_" + file_name + "_ID.pkl")
    f = open(newid2wordid_path, "wb")
    pickle.dump(newsid2wordsid, f)
    f.close()
    print("newsID_wordsID generated")


def save_entity(newsdata, data_dir, pkl_dir):
    # Loading entity_embedding.vec and create dictionaries id to entity ID and entity ID to id. Saving to .pkl file
    entity1 = pd.read_csv(
        os.path.join(data_dir, "train/entity_embedding.vec"),
        header=None,
        sep="\t",
    )
    entity2 = pd.read_csv(
        os.path.join(data_dir, "valid/entity_embedding.vec"),
        header=None,
        sep="\t",
    )
    entity = pd.concat((entity1, entity2), axis=0)
    entity = entity.drop_duplicates()
    entity_IDs = entity[0].drop_duplicates()
    entityID2id = dict(zip(entity_IDs, range(len(entity_IDs))))
    id2entityID = dict(zip(range(len(entity_IDs)), entity_IDs))
    entity_dict = {"entityID2id": entityID2id, "id2entityID": id2entityID}
    entityID_id_path = os.path.join(pkl_dir, "entityID_id.pkl")
    f = open(entityID_id_path, "wb")
    pickle.dump(entity_dict, f)
    f.close()
    # Create a numpy array to save vector matching each enity_ID and save .npy file
    entity["vector"] = entity.iloc[:, 1:101].values.tolist()
    entity["id"] = entity[0].apply(lambda x: entityID2id[x])
    entity = entity.sort_values(by=["id"])
    entity_embeddings = np.array(list(entity["vector"]))
    EntityId_embeddings_path = os.path.join(pkl_dir, "EntityId_embeddings.npy")
    np.save(EntityId_embeddings_path, entity_embeddings)

    newsdata = newsdata.copy()
    # Check if value in this cell is null -> if yes, convert to []
    # -> no, convert value from str or sequence to turple, list, ...
    newsdata["TitleEntities"] = newsdata["TitleEntities"].apply(
        lambda x: literal_eval(x) if pd.notnull(x) else []
    )

    # filter to select all record which have TitleEntities is not null
    newsdata_tittle_entity = newsdata.loc[
        newsdata["TitleEntities"].str.len() != 0
    ]

    # remove value of this cell is not a list
    newsdata_tittle_entity = newsdata.loc[
        newsdata["TitleEntities"].apply(lambda x: isinstance(x, list))
    ]

    # create a new column contain entityID and confidence
    newsdata_tittle_entity = newsdata_tittle_entity.copy()
    newsdata_tittle_entity["TitleEntity_Confidence"] = newsdata_tittle_entity[
        "TitleEntities"
    ].apply(
        lambda x: [
            [entityID2id[i["WikidataId"]], float(i["Confidence"])]
            for i in list(x)
            if i["WikidataId"] in entityID2id.keys()
        ]
    )

    # filter to select all record which have TitleEntities is null
    newsdata_titleno = newsdata.loc[newsdata["TitleEntities"].str.len() == 0]

    newsdata_titleno = newsdata_titleno.copy()
    newsdata_titleno["TitleEntity_Confidence"] = newsdata_titleno[
        "TitleEntities"
    ]

    # Create a new dataframe contain newsID and title confiden(include entity ID with this confidence)
    newsdata_all = pd.concat(
        (
            newsdata_tittle_entity.loc[:, ["NewsID", "TitleEntity_Confidence"]],
            newsdata_titleno.loc[:, ["NewsID", "TitleEntity_Confidence"]],
        ),
        axis=0,
    )

    # convert this dataframe mentioned above to dictionary datatype and save to .pkl file
    news_title_entity = dict(
        zip(newsdata_all["NewsID"], newsdata_all["TitleEntity_Confidence"])
    )
    newsID_titleEntityId_conf_path = os.path.join(
        pkl_dir, "newsID_titleEntityId_conf.pkl"
    )
    f = open(newsID_titleEntityId_conf_path, "wb")
    pickle.dump(news_title_entity, f)
    f.close()
    # Same as the task above, but target is Astract Entities
    newsdata = newsdata.copy()
    newsdata["AbstractEntities"] = newsdata["AbstractEntities"].apply(
        lambda x: literal_eval(x) if pd.notnull(x) else []
    )
    newsdata_absentity = newsdata.loc[
        newsdata["AbstractEntities"].str.len() != 0
    ]

    # remove value of this cell is not a list
    newsdata_absentity = newsdata.loc[
        newsdata["AbstractEntities"].apply(lambda x: isinstance(x, list))
    ]

    newsdata_absentity = newsdata_absentity.copy()
    newsdata_absentity["AbstractEntity_Confidence"] = newsdata_absentity[
        "AbstractEntities"
    ].apply(
        lambda x: [
            [entityID2id[i["WikidataId"]], float(i["Confidence"])]
            for i in x
            if i["WikidataId"] in entityID2id.keys()
        ]
    )

    newsdata_absentityno = newsdata.loc[
        newsdata["AbstractEntities"].str.len() == 0
    ]
    newsdata_absentityno = newsdata_absentityno.copy()
    newsdata_absentityno["AbstractEntity_Confidence"] = newsdata_absentityno[
        "AbstractEntities"
    ]

    newsdata_all = pd.concat(
        (
            newsdata_absentity.loc[:, ["NewsID", "AbstractEntity_Confidence"]],
            newsdata_absentityno.loc[
                :, ["NewsID", "AbstractEntity_Confidence"]
            ],
        ),
        axis=0,
    )
    news_abstract_entity = dict(
        zip(newsdata_all["NewsID"], newsdata_all["AbstractEntity_Confidence"])
    )
    newsID_abstractEntityId_conf_path = os.path.join(
        pkl_dir, "newsID_abstractEntityId_conf.pkl"
    )
    f = open(newsID_abstractEntityId_conf_path, "wb")
    pickle.dump(news_abstract_entity, f)
    f.close()


def news_preprocessing(data_path, pkl_path):
    news_train_path = os.path.join(data_path, "train/news.tsv")
    news_train = pd.read_csv(
        news_train_path,
        sep="\t",
        header=None,
        names=[
            "NewsID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "TitleEntities",
            "AbstractEntities",
        ],
    )

    news_dev_path = os.path.join(data_path, "valid/news.tsv")
    news_dev = pd.read_csv(
        news_dev_path,
        sep="\t",
        header=None,
        names=[
            "NewsID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "TitleEntities",
            "AbstractEntities",
        ],
    )
    data = pd.concat((news_train, news_dev), axis=0)
    data = data.drop_duplicates()
    save_category_subcategory(
        data["Category"], data["SubCategory"], data["NewsID"], pkl_path
    )

    save_word_embeddings(data["Title"], data["NewsID"], "TitleWord", pkl_path)
    save_word_embeddings(
        data["Abstract"], data["NewsID"], "AbstractWord", pkl_path
    )
    save_entity(data, data_path, pkl_path)


if __name__ == "__main__":
    news_train = pd.read_csv(
        "data/train/news.tsv",
        sep="\t",
        header=None,
        names=[
            "NewsID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "TitleEntities",
            "AbstractEntities",
        ],
    )
    news_dev = pd.read_csv(
        "data/valid/news.tsv",
        sep="\t",
        header=None,
        names=[
            "NewsID",
            "Category",
            "SubCategory",
            "Title",
            "Abstract",
            "URL",
            "TitleEntities",
            "AbstractEntities",
        ],
    )
    data = pd.concat((news_train, news_dev), axis=0)
    data = data.drop_duplicates()
    save_category_subcategory(
        data["Category"], data["SubCategory"], data["NewsID"]
    )

    save_word_embeddings(data["Title"], data["NewsID"], "TitleWord")
    save_word_embeddings(data["Abstract"], data["NewsID"], "AbstractWord")
    save_entity(data)
