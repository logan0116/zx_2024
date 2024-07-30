from transformers import AutoTokenizer, AutoModel
import torch
import json
import pandas as pd


def load_source_list(time_span):
    # keywords_multiple.xlsx
    df = pd.read_excel(f'data/topmine/keywords_multiple_{time_span}.xlsx')
    word_list = df['word'].values.tolist()
    word_list = [word.replace(' ', '') for word in word_list]
    return word_list


def load_target_list(time_span):
    with open(f'data/word_base_{time_span}.json', 'r', encoding='utf-8') as f:
        word_base = json.load(f)
    return word_base


def get_embedding(sentences, tokenizer, model):
    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt', max_length=24)
    input_ids = encoded_input['input_ids'].cuda()
    attention_mask = encoded_input['attention_mask'].cuda()
    token_type_ids = encoded_input['token_type_ids'].cuda()

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        # Perform pooling. In this case, cls pooling.
        sentence_embeddings = model_output[0][:, 0]
    # normalize embeddings../data_process/topmine/
    sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
    # print("Sentence embeddings:", sentence_embeddings)
    print(sentence_embeddings.shape)
    return sentence_embeddings


def get_top50(tokenizer, model, time_span):
    """
    :return:
    """
    source_list = load_source_list(time_span)
    c2target_list = load_target_list(time_span)
    # embed
    source_embed = get_embedding(source_list, tokenizer, model)
    c2target_embed = {}
    for c, target_list in c2target_list.items():
        target_embed = get_embedding(target_list, tokenizer, model)
        c2target_embed[c] = target_embed

    # c2top_50
    c2top50 = {}

    # compute sim
    for c in c2target_embed:
        target_embed = c2target_embed[c]
        # sim: [source_num, target_num]
        sim = torch.mm(source_embed, target_embed.T)
        # sim: [source_num]
        sim = torch.mean(sim, dim=1)
        # top 50 index
        sim_index = torch.argsort(sim, descending=True)
        sim_index = sim_index[:50]
        sim_index = sim_index.tolist()
        print(sim_index)
        top50 = [source_list[i] for i in sim_index]
        # save to excel
        c2top50[c] = top50

    # save by pandas
    df = pd.DataFrame(c2top50)
    df.to_excel(f'data/战新词表_topmine_top50_{time_span}.xlsx', index=False)


if __name__ == '__main__':
    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-zh-v1.5')
    model = AutoModel.from_pretrained('BAAI/bge-large-zh-v1.5')
    model.eval()
    model.cuda()
    time_span = '145'
    get_top50(tokenizer, model, time_span)
