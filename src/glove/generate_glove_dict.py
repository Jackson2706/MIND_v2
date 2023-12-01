import os
import pickle
import zipfile
from glob import glob
from urllib import request

import numpy as np


def loadGloveModel(File):
    print("Loading Glove Model")
    f = open(File, "r", encoding="utf-8")
    gloveModel = {}
    for line in f:
        splitLines = line.split(" ")
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = wordEmbedding
    print(len(gloveModel), " words loaded!")
    return gloveModel


def glovePKL(glove_url: str, save_folder_dir):
    zip_name = glove_url.split("/")[-1]
    target_path = os.path.join(save_folder_dir, zip_name)
    if not os.path.exists(target_path):
        request.urlretrieve(glove_url, target_path)
        zip = zipfile.ZipFile(target_path)
        zip.extractall(save_folder_dir)
        zip.close

    for filename in os.listdir(save_folder_dir):
        if filename.startswith("glove") and filename.endswith(".txt"):
            glove_txt = os.path.join(save_folder_dir, filename)

            glove = loadGloveModel(glove_txt)
            glove_dict_path = os.path.join(save_folder_dir, "glove_dict.pkl")
            f = open(glove_dict_path, "wb")
            pickle.dump(glove, f)
            return glove_dict_path
    return None


if __name__ == "__main__":
    glove = loadGloveModel("glove.840B.300d.txt")
    f = open("glove_dict.pkl", "wb")
    pickle.dump(glove, f)
