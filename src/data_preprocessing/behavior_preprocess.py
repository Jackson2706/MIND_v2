import os
import pickle
import random

import numpy as np
import pandas as pd


def preprocess(
    data: pd.DataFrame,
    nagetive_sampling_num: int,
    data_name: str,
    save_folder_dir: str,
    validation_users: list = None,
):
    """
    This function to preprocess data from behavior.tsv which contains 5 fields:
        - ImpressiveID
        - User ID
        - Time
        - History
        - Impression
    become a list whose element is like
        [[userID], [History (view or click) at the time before the latest day], [Clicked history at the timr before the latest day], [clicked new in the latest day, k non-clicked new in the latest day]]
    and save in a .pkl file
    """
    mat_data = []
    group_user = data.groupby("UserID")
    for key in group_user.groups.keys():
        # for each to colect unique user id
        if data_name == "validation_data":
            print("generating validation data")
            if key not in validation_users:
                print(key + " is not in validation users.")
                continue
        user_list = []
        user_list.append([key])
        user_data = group_user.get_group(
            key
        )  # Get all information of each user via unique user id
        user_long_his = (
            user_data["Histories"].str.split(" ").iloc[0]
        )  # Get unique newId from each user's history
        user_list.append(
            user_long_his
        )  # user_list = [[userID], [user_long_hist, for example, 'N61704', 'N49103', 'N63554', 'N44399', 'N32117', 'N2186', 'N37123']]

        user_data = user_data.sort_values(
            ["Time"], ascending=True
        )  # sort dataframe by timestamp with ascending ordor
        user_data["date"] = user_data["Time"].apply(
            lambda x: x.date()
        )  # create a new colum only saving date information instead of date_time
        max_date = user_data["date"].tolist()[
            -1
        ]  # get the latest day of each userID
        user_short_his = user_data.loc[
            user_data["date"] != max_date
        ]  # Get information of user in all day before latest day
        user_short_his = user_short_his["Impressions"].str.split(" ")
        short_his = []
        for i in user_short_his:
            for j in i:
                if j.endswith("-1"):
                    short_his.append(j.split("-")[0])
        user_list.append(
            short_his
        )  # user_list = [[userID], [history], [clicked history]]
        user_pre = user_data.loc[user_data["date"] == max_date]
        user_pre = user_pre["Impressions"].str.split(" ")

        for i in user_pre:
            pre_true = []
            pre_false = []
            flag = 0  # count the number of positive clicks
            for j in i:
                if j.endswith("-1"):
                    flag += 1
                    pre_true.append(j.split("-")[0])
                else:
                    pre_false.append(j.split("-")[0])

            if (
                flag != 0
            ):  # if there's a positive click in this impression, then sample k negative samples for each positive click
                if len(pre_false) >= nagetive_sampling_num:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = random.sample(
                            pre_false, nagetive_sampling_num
                        )
                        li.extend(pre_false_sample)
                        user_li = user_list.copy()
                        user_li.append(
                            li
                        )  # user_li = [[userID], [History], [clicked history], [clicked canidate news, k non-clicked candidate news]]
                        mat_data.append(user_li)
                else:
                    for i in pre_true:
                        li = [i]
                        pre_false_sample = pre_false + list(
                            np.random.choice(
                                pre_false,
                                nagetive_sampling_num - len(pre_false),
                            )
                        )
                        li.extend(pre_false_sample)
                        user_li = user_list.copy()
                        user_li.append(li)
                        mat_data.append(user_li)
    random.shuffle(mat_data)
    f = open(os.path.join(save_folder_dir, data_name + ".pkl"), "wb")
    data = {data_name: mat_data}
    pickle.dump(data, f)


def behavior_preprocessing(data_path, pkl_dir):
    behavior_train_path = os.path.join(data_path, "train/behaviors.tsv")
    behaviors_train = pd.read_csv(
        behavior_train_path,
        sep="\t",
        header=None,
        names=[
            "ImpressionID",
            "UserID",
            "Time",
            "Histories",
            "Impressions",
        ],
    )
    behaviors_train["Time"] = pd.to_datetime(
        behaviors_train["Time"], format="%m/%d/%Y %I:%M:%S %p"
    )

    preprocess(behaviors_train, 4, "data_train", pkl_dir)

    behavior_valid_path = os.path.join(data_path, "valid/behaviors.tsv")
    behaviors_valid = pd.read_csv(
        behavior_valid_path,
        sep="\t",
        header=None,
        names=[
            "ImpressionID",
            "UserID",
            "Time",
            "Histories",
            "Impressions",
        ],
    )
    behaviors_valid["Time"] = pd.to_datetime(
        behaviors_valid["Time"], format="%m/%d/%Y %I:%M:%S %p"
    )
    validation_users = behaviors_valid["UserID"].tolist()
    behaviors = pd.concat((behaviors_train, behaviors_valid), axis=0)
    preprocess(behaviors, 4, "validation_data", pkl_dir, validation_users)


if __name__ == "__main__":
    behaviors_train = pd.read_csv(
        "data/train/behaviors.tsv",
        sep="\t",
        header=None,
        names=[
            "ImpressionID",
            "UserID",
            "Time",
            "Histories",
            "Impressions",
        ],
    )
    behaviors_train["Time"] = pd.to_datetime(
        behaviors_train["Time"], format="%m/%d/%Y %I:%M:%S %p"
    )

    preprocess(behaviors_train, 4, "data_train")

    behaviors_valid = pd.read_csv(
        "data/valid/behaviors.tsv",
        sep="\t",
        header=None,
        names=[
            "ImpressionID",
            "UserID",
            "Time",
            "Histories",
            "Impressions",
        ],
    )
    behaviors_valid["Time"] = pd.to_datetime(
        behaviors_valid["Time"], format="%m/%d/%Y %I:%M:%S %p"
    )
    validation_users = behaviors_valid["UserID"].tolist()
    behaviors = pd.concat((behaviors_train, behaviors_valid), axis=0)
    print(behaviors.head())
    preprocess(behaviors, 4, "validation_data", validation_users)
