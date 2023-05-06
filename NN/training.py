from tqdm import trange
import random
from transformers import AdamW, BertTokenizer, get_linear_schedule_with_warmup
from torch import nn
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from datasetPreProcess import *
from createFeatures import *
from modelBERT import *
import torch 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler


# get model and tokenizer 
# model, tokenizer = get_model_and_tokenizer()
BATCH_SIZE = 32


def _get_dataloader(_train_dataset, _collate_fn):
    train_sampler = RandomSampler(_train_dataset)

    return DataLoader(
        _train_dataset,
        sampler=train_sampler,
        batch_size=BATCH_SIZE,
        collate_fn=_collate_fn
    )



def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train(model, tokenizer, train_dataloader, num_epochss):
    optimizer = AdamW(model.parameters(), lr=5e-5, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=len(train_dataloader) * num_epochss
    )

    global_step = 0
    epochs_trained = 0

    tr_loss = 0.0
    model.zero_grad()
    train_iterator = trange(
        epochs_trained, int(num_epochss), desc="Epoch"
    )
    for _ in train_iterator:
        epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        for step, batch in enumerate(epoch_iterator):
            model.train()
            loss = model.forward_gloss_selection(batch)[0]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            tr_loss += loss.item()
            optimizer.step()
            scheduler.step() 
            model.zero_grad()
            global_step += 1
            if global_step % 100 == 0:
                print("loss:", loss.item())
    return global_step, tr_loss / global_step


def evaulate(model, tokenizer, eval_dataloader):
    model.eval()
    eval_loss = 0
    nb_eval_steps = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            tmp_eval_loss = model.forward_gloss_selection(batch)[0]
            eval_loss += tmp_eval_loss.mean().item()
        nb_eval_steps += 1
    eval_loss = eval_loss / nb_eval_steps
    return eval_loss


def main():
    set_seed(42)
    model, tokenizer = get_model_and_tokenizer()
    train_dataset = load_dataset("./SemCor/semcor_data.csv", tokenizer, 128)
    train_dataloader = _get_dataloader(train_dataset, collate_batch)
    global_step, tr_loss = train(model, tokenizer, train_dataloader, 3)
    print("global_step = %s, average loss = %s" % (global_step, tr_loss))
    model.save_pretrained("./model_save")