import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.utils.data as Data

from NRMS import NewsEncoder, NRMS_new, TextEncoder

"the difference between evaluation and validation is batch_user_valid does not contain just five news (one pos + four neg)"
"batch_user_valid need to be padded to the length of the longest sequence"


def pad_validation_news(batch_user_valid):
    "pad validation news for all users in a batch, the length should be the length of the longest sequence in the batch"
    length = [len(i) for i in batch_user_valid]
    max_len = max(length)
    batch_valid = batch_user_valid.copy()

    for i, j in enumerate(batch_valid):
        batch_valid[i].extend(["-1"] * (max_len - len(j)))

    batch_valid = np.array(list(batch_valid))
    mask = batch_valid.copy()
    mask[mask != "-1"] = 1
    mask[mask == "-1"] = 0
    mask_softmax = mask.astype(int)  # mask for softmax
    batch_valid[batch_valid == "-1"] = "0"
    return batch_valid, mask_softmax


def process_batch_data(behavior_data, index):  # padding and masking
    data = behavior_data[index]
    history = data[:, 1]
    recent = data[:, 2]
    (
        batch_user_history,
        user_history_mask_selfattn,
        user_history_mask_attn,
    ) = pad_masking(history)
    (
        batch_user_short,
        user_short_mask_selfattn,
        user_short_mask_attn,
    ) = pad_masking(recent)
    batch_user_valid, mask_softmax = pad_validation_news(data[:, 3])
    batch_user_impressionID = np.array(list(data[:, 4])).squeeze(-1)
    return (
        batch_user_history,
        batch_user_short,
        user_history_mask_selfattn,
        user_history_mask_attn,
        user_short_mask_selfattn,
        user_short_mask_attn,
        batch_user_valid,
        mask_softmax,
        batch_user_impressionID,
    )


def evaluation(
    model,
    vali_data,
    evaluation_data,
    num_iterations_eval,
    batch_size,
    newsID_categoryID,
    newsID_subcategoryID,
    newsID_TitleWordID,
    newsID_AbstractWordID,
    newsID_titleEntityId_conf,
    newsID_abstractEntityId_conf,
    ranking_name,
):
    evaluate_data_index = torch.IntTensor(np.array(range(len(evaluation_data))))
    evaluate_data_index = Data.TensorDataset(
        data_tensor=evaluate_data_index, target_tensor=evaluate_data_index
    )
    loader = Data.DataLoader(
        dataset=evaluate_data_index,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        drop_last=False,
    )

    model.eval()
    ranking = []
    impressionIDs = []
    for step, (batch_x, batch_y) in enumerate(
        loader
    ):  # batch_x: index of batch data
        print(str(step) + "/" + str(num_iterations_eval))
        processed_data = process_batch_data(vali_data, batch_x.numpy())
        batch_user_history = processed_data[0]
        batch_user_short = processed_data[1]
        user_history_mask_selfattn = processed_data[2]
        user_history_mask_attn = processed_data[3]
        user_short_mask_selfattn = processed_data[4]
        user_short_mask_attn = processed_data[5]
        batch_user_valid = processed_data[6]
        batch_user_mask = processed_data[7]
        batch_user_impressionID = processed_data[8]

        # compute output
        batch_predict = model(
            batch_user_history,
            batch_user_short,
            user_history_mask_selfattn,
            user_history_mask_attn,
            user_short_mask_selfattn,
            user_short_mask_attn,
            batch_user_valid,
            newsID_categoryID,
            newsID_subcategoryID,
            newsID_TitleWordID,
            newsID_AbstractWordID,
            newsID_titleEntityId_conf,
            newsID_abstractEntityId_conf,
        ).cuda()
        batch_predict = F.softmax(batch_predict, dim=1)
        scores = batch_predict.cpu().data.numpy() * batch_user_mask
        for i in range(len(scores)):
            score = scores[i][: int(np.sum(batch_user_mask[i]))]
            rank = score.argsort().argsort() + 1
            rank = list(len(rank) + 1 - np.array(rank))
            ranking.append(rank)
        impressionIDs.extend(list(batch_user_impressionID))
    df = pd.DataFrame(
        list(zip(impressionIDs, ranking)), columns=["ImpressionID", "Rank"]
    )
    df = df.sort_values(by=["ImpressionID"], ascending=True)
    f = open(ranking_name, "a+")
    for i, j in df.iterrows():
        f.writelines(
            (
                str(j["ImpressionID"]),
                " ",
                str(list(j["Rank"])).replace(" ", ""),
                "\n",
            )
        )
    print("Ranking produced.")
    f.close()


def pad_masking(bat):
    "padding and masking news for batch"
    length = [len(i) if type(i) != float else 0 for i in bat]
    max_len = max(length)
    batch = bat.copy()
    if max_len == 0:
        for i, j in enumerate(batch):
            max_len = 1
            batch[i] = ["-1"] * max_len

    for i, j in enumerate(batch):
        if type(j) != float:
            batch[i].extend(["-1"] * (max_len - len(j)))
        else:
            batch[i] = ["-1"] * max_len

    batch = np.array(list(batch))
    mask = batch.copy()
    mask[mask != "-1"] = 1
    mask[mask == "-1"] = 0
    mask = mask.astype(int)  # mask for additive attention
    mask_attn = mask.copy()
    mask1 = mask[:, :, np.newaxis]  # mask for self attention
    mask2 = mask[:, np.newaxis, :]
    mask = np.matmul(mask1, mask2)
    batch[batch == "-1"] = "0"
    return batch, torch.IntTensor(mask), torch.IntTensor(mask_attn).cuda()


def process_batch_data(behavior_data, index):  # padding and masking
    data = behavior_data[index]
    history = data[:, 1]
    recent = data[:, 2]
    (
        batch_user_history,
        user_history_mask_selfattn,
        user_history_mask_attn,
    ) = pad_masking(history)
    (
        batch_user_short,
        user_short_mask_selfattn,
        user_short_mask_attn,
    ) = pad_masking(recent)
    batch_user_valid = np.array(list(data[:, 3]))
    batch_label = np.zeros(len(index), int)
    return (
        batch_user_history,
        batch_user_short,
        user_history_mask_selfattn,
        user_history_mask_attn,
        user_short_mask_selfattn,
        user_short_mask_attn,
        batch_user_valid,
        batch_label,
    )


def evaluate(
    num_iterations_vali,
    model,
    loader,
    vali_data,
    epoch,
    newsID_categoryID,
    newsID_subcategoryID,
    newsID_TitleWordID,
    newsID_AbstractWordID,
    newsID_titleEntityId_conf,
    newsID_abstractEntityId_conf,
):
    model.eval()
    summ = []
    for step, (batch_x, batch_y) in enumerate(
        loader
    ):  # batch_x: index of batch data
        processed_data = process_batch_data(vali_data, batch_x.numpy())
        batch_user_history = processed_data[0]
        batch_user_short = processed_data[1]
        user_history_mask_selfattn = processed_data[2]
        user_history_mask_attn = processed_data[3]
        user_short_mask_selfattn = processed_data[4]
        user_short_mask_attn = processed_data[5]
        batch_user_valid = processed_data[6]
        batch_label = processed_data[7]
        # compute output
        batch_predict = model(
            batch_user_history,
            batch_user_short,
            user_history_mask_selfattn,
            user_history_mask_attn,
            user_short_mask_selfattn,
            user_short_mask_attn,
            batch_user_valid,
            newsID_categoryID,
            newsID_subcategoryID,
            newsID_TitleWordID,
            newsID_AbstractWordID,
            newsID_titleEntityId_conf,
            newsID_abstractEntityId_conf,
        ).cuda()
        loss = model.loss(batch_predict, batch_label).cuda()
        print(
            "Epoch "
            + str(epoch)
            + ": the "
            + str(step + 1)
            + " /"
            + str(num_iterations_vali)
            + "-th validation: loss: "
            + str(loss.data[0])
            + "\n"
        )
        summ.append(loss.data[0])
    average_loss = np.average(summ)
    return average_loss


def train(
    model,
    num_iterations,
    optimizer,
    loader,
    train_data,
    epoch,
    newsID_categoryID,
    newsID_subcategoryID,
    newsID_TitleWordID,
    newsID_AbstractWordID,
    newsID_titleEntityId_conf,
    newsID_abstractEntityId_conf,
):
    model.train()
    summ = []
    for step, (batch_x, batch_y) in enumerate(
        loader
    ):  # batch_x: index of batch data
        print(
            "Epoch: ",
            epoch,
            " | Iteration: ",
            step + 1,
            "/" + str(num_iterations),
        )
        processed_data = process_batch_data(train_data, batch_x.numpy())
        batch_user_history = processed_data[0]
        batch_user_short = processed_data[1]
        user_history_mask_selfattn = processed_data[2]
        user_history_mask_attn = processed_data[3]
        user_short_mask_selfattn = processed_data[4]
        user_short_mask_attn = processed_data[5]
        batch_user_valid = processed_data[6]
        batch_label = processed_data[7]

        # clear gradients
        optimizer.zero_grad()
        batch_predict = model(
            batch_user_history,
            batch_user_short,
            user_history_mask_selfattn,
            user_history_mask_attn,
            user_short_mask_selfattn,
            user_short_mask_attn,
            batch_user_valid,
            newsID_categoryID,
            newsID_subcategoryID,
            newsID_TitleWordID,
            newsID_AbstractWordID,
            newsID_titleEntityId_conf,
            newsID_abstractEntityId_conf,
        ).cuda()
        loss = model.loss(batch_predict, batch_label).cuda()
        print(
            "Epoch "
            + str(epoch)
            + ": "
            + "The "
            + str(step + 1)
            + "/"
            + str(num_iterations)
            + "-th interation: loss: "
            + str(loss.data[0])
            + "\n"
        )
        loss.backward()
        optimizer.step()
        summ.append(loss.data[0])
    average_loss = np.mean(summ)
    return average_loss


def train_and_evaluate(
    batch_size,
    training_data,
    validation_data,
    news_final_embed_size,
    history_num_head,
    history_attn_vector_size,
    recent_num_head,
    recent_attn_vector_size,
    num_category,
    num_subcategory,
    title_embed_matrix,
    abstract_embed_matrix,
    entity_embed_matrix,
    num_head_text,
    num_head_entity,
    text_attn_vector_size,
    entity_attn_vector_size,
    news_final_attn_vector_size,
    final_attn_vector_size,
    newsID_categoryID,
    newsID_subcategoryID,
    newsID_TitleWordID,
    newsID_AbstractWordID,
    newsID_titleEntityId_conf,
    newsID_abstractEntityId_conf,
    num_iterations,
    num_iterations_vali,
    pack_loss,
    model_name,
):
    history_encoder = TextEncoder(
        news_final_embed_size, history_num_head, history_attn_vector_size
    )
    recent_encoder = TextEncoder(
        news_final_embed_size, recent_num_head, recent_attn_vector_size
    )

    news_encoder = NewsEncoder(
        num_category,
        num_subcategory,
        title_embed_matrix,
        abstract_embed_matrix,
        entity_embed_matrix,
        num_head_text,
        num_head_entity,
        text_attn_vector_size,
        entity_attn_vector_size,
        news_final_attn_vector_size,
        news_final_embed_size,
    )
    model = NRMS_new(
        history_encoder, recent_encoder, news_encoder, final_attn_vector_size
    )
    optimizer = torch.optim.Adam([{"params": model.parameters()}], lr=1e-3)

    best_vali_loss = 10000
    epoch = 0
    training_loss_epoch = []
    vali_loss_epoch = []

    train_data_index = torch.IntTensor(np.array(range(len(training_data))))
    train_data_index = Data.TensorDataset(
        data_tensor=train_data_index, target_tensor=train_data_index
    )
    train_loader = Data.DataLoader(
        dataset=train_data_index,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=False,
    )

    validate_data_index = torch.IntTensor(np.array(range(len(validation_data))))
    validate_data_index = Data.TensorDataset(
        data_tensor=validate_data_index, target_tensor=validate_data_index
    )
    vali_loader = Data.DataLoader(
        dataset=validate_data_index,
        batch_size=batch_size,
        shuffle=True,
        num_workers=16,
        drop_last=True,
    )

    while True:
        epoch += 1
        print("-----epoch " + str(epoch) + "------")
        print("set batches")
        training_loss = train(
            num_iterations=num_iterations,
            model=model,
            optimizer=optimizer,
            loader=train_loader,
            train_data=training_data,
            epoch=epoch,
            newsID_categoryID=newsID_categoryID,
            newsID_subcategoryID=newsID_subcategoryID,
            newsID_TitleWordID=newsID_TitleWordID,
            newsID_AbstractWordID=newsID_AbstractWordID,
            newsID_titleEntityId_conf=newsID_titleEntityId_conf,
            newsID_abstractEntityId_conf=newsID_abstractEntityId_conf,
        )
        training_loss_epoch.append(training_loss)
        print(
            "The average loss of training set for the first "
            + str(epoch)
            + " epochs: "
            + str(training_loss_epoch)
        )

        evaluation_loss = evaluate(
            num_iterations_vali=num_iterations_vali,
            model=model,
            loader=vali_loader,
            vali_data=validation_data,
            epoch=epoch,
            newsID_categoryID=newsID_categoryID,
            newsID_subcategoryID=newsID_subcategoryID,
            newsID_TitleWordID=newsID_TitleWordID,
            newsID_AbstractWordID=newsID_AbstractWordID,
            newsID_titleEntityId_conf=newsID_titleEntityId_conf,
            newsID_abstractEntityId_conf=newsID_abstractEntityId_conf,
        )
        vali_loss_epoch.append(evaluation_loss)
        print(
            "The average loss of validation set for the first "
            + str(epoch)
            + " epochs: "
            + str(vali_loss_epoch)
        )

        if evaluation_loss < best_vali_loss:
            best_vali_loss = evaluation_loss
            torch.save(model, model_name)
        if epoch >= 5:
            "ealry stopping"
            near_loss = vali_loss_epoch[-5:]
            if near_loss == sorted(
                near_loss
            ):  # loss increases for 5 consecutive epochs
                print("Best model found! Stop training, saving loss!")
                loss_train_vali = {
                    "training loss": training_loss_epoch,
                    "testing loss": vali_loss_epoch,
                }
                f = open(pack_loss, "wb")
                pickle.dump(loss_train_vali, f)
                f.close()
                break
