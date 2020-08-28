"""Download and pre-process SQuAD and GloVe.

Usage:
    > source activate squad
    > python setup.py

Pre-processing code adapted from:
    > https://github.com/HKUST-KnowComp/R-Net/blob/master/prepro.py

Author:
    Chris Chute (chute@stanford.edu)
"""

import numpy as np
import os
import spacy
import ujson as json
import urllib.request

from args import get_setup_args
from codecs import open
from collections import Counter
from subprocess import run
from tqdm import tqdm
from zipfile import ZipFile
import requests
import time
import copy

def download_url(url, output_path, show_progress=True):
    class DownloadProgressBar(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if show_progress:
        # Download with a progress bar
        with DownloadProgressBar(unit='B', unit_scale=True,
                                 miniters=1, desc=url.split('/')[-1]) as t:
            urllib.request.urlretrieve(url,
                                       filename=output_path,
                                       reporthook=t.update_to)
    else:
        # Simple download with no progress bar
        urllib.request.urlretrieve(url, output_path)


def url_to_data_path(url):
    return os.path.join('./data/', url.split('/')[-1])


def download(args):
    downloads = [
        # Can add other downloads here (e.g., other word vectors)
        ('GloVe word vectors', args.glove_url),
    ]

    for name, url in downloads:
        output_path = url_to_data_path(url)
        if not os.path.exists(output_path):
            print(f'Downloading {name}...')
            download_url(url, output_path)

        if os.path.exists(output_path) and output_path.endswith('.zip'):
            extracted_path = output_path.replace('.zip', '')
            if not os.path.exists(extracted_path):
                print(f'Unzipping {name}...')
                with ZipFile(output_path, 'r') as zip_fh:
                    zip_fh.extractall(extracted_path)

    print('Downloading spacy language model...')
    run(['python', '-m', 'spacy', 'download', 'en'])

def word_tokenize(sent):
    doc = nlp(sent)
    return [token.text for token in doc]


def convert_idx(text, tokens):
    current = 0
    spans = []
    for token in tokens:
        current = text.find(token, current)
        if current < 0:
            print(f"Token {token} cannot be found")
            raise Exception()
        spans.append((current, current + len(token)))
        current += len(token)
    return spans


def process_file(filename, data_type, word_counter, char_counter):
    print(f"Pre-processing {data_type} examples...")
    examples = []
    eval_examples = {}
    total = 0
    with open(filename, "r") as fh:
        source = json.load(fh)
        for article in tqdm(source["data"]):
            for para in article["paragraphs"]:
                context = para["context"].replace(
                    "''", '" ').replace("``", '" ')
                context_tokens = word_tokenize(context)
                context_chars = [list(token) for token in context_tokens]
                spans = convert_idx(context, context_tokens)
                for token in context_tokens:
                    word_counter[token] += len(para["qas"])
                    for char in token:
                        char_counter[char] += len(para["qas"])
                for qa in para["qas"]:
                    total += 1
                    ques = qa["question"].replace(
                        "''", '" ').replace("``", '" ')
                    ques_tokens = word_tokenize(ques)
                    ques_chars = [list(token) for token in ques_tokens]
                    for token in ques_tokens:
                        word_counter[token] += 1
                        for char in token:
                            char_counter[char] += 1
                    y1s, y2s = [], []
                    answer_texts = []
                    for answer in qa["answers"]:
                        answer_text = answer["text"]
                        answer_start = answer['answer_start']
                        answer_end = answer_start + len(answer_text)
                        answer_texts.append(answer_text)
                        answer_span = []
                        for idx, span in enumerate(spans):
                            if not (answer_end <= span[0] or answer_start >= span[1]):
                                answer_span.append(idx)
                        y1, y2 = answer_span[0], answer_span[-1]
                        y1s.append(y1)
                        y2s.append(y2)
                    example = {"context_tokens": context_tokens,
                               "context_chars": context_chars,
                               "ques_tokens": ques_tokens,
                               "ques_chars": ques_chars,
                               "y1s": y1s,
                               "y2s": y2s,
                               "id": total}
                    examples.append(example)
                    eval_examples[str(total)] = {"context": context,
                                                 "question": ques,
                                                 "spans": spans,
                                                 "answers": answer_texts,
                                                 "uuid": qa["id"]}
        print(f"{len(examples)} questions in total")
    return examples, eval_examples


def get_embedding(counter, data_type, limit=-1, emb_file=None, vec_size=None, num_vectors=None):
    print(f"Pre-processing {data_type} vectors...")
    embedding_dict = {}
    filtered_elements = [k for k, v in counter.items() if v > limit]
    if emb_file is not None:
        assert vec_size is not None
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in tqdm(fh, total=num_vectors):
                array = line.split()
                word = "".join(array[0:-vec_size])
                vector = list(map(float, array[-vec_size:]))
                if word in counter and counter[word] > limit:
                    embedding_dict[word] = vector
        print(f"{len(embedding_dict)} / {len(filtered_elements)} tokens have corresponding {data_type} embedding vector")
    else:
        assert vec_size is not None
        for token in filtered_elements:
            embedding_dict[token] = [np.random.normal(
                scale=0.1) for _ in range(vec_size)]
        print(f"{len(filtered_elements)} tokens have corresponding {data_type} embedding vector")

    NULL = "--NULL--"
    OOV = "--OOV--"
    token2idx_dict = {token: idx for idx, token in enumerate(embedding_dict.keys(), 2)}
    token2idx_dict[NULL] = 0
    token2idx_dict[OOV] = 1
    embedding_dict[NULL] = [0. for _ in range(vec_size)]
    embedding_dict[OOV] = [0. for _ in range(vec_size)]
    idx2emb_dict = {idx: embedding_dict[token]
                    for token, idx in token2idx_dict.items()}
    emb_mat = [idx2emb_dict[idx] for idx in range(len(idx2emb_dict))]
    return emb_mat, token2idx_dict


def convert_to_features(args, data, word2idx_dict, char2idx_dict, is_test):
    example = {}
    context, question = data
    context = context.replace("''", '" ').replace("``", '" ')
    question = question.replace("''", '" ').replace("``", '" ')
    example['context_tokens'] = word_tokenize(context)
    example['ques_tokens'] = word_tokenize(question)
    example['context_chars'] = [list(token) for token in example['context_tokens']]
    example['ques_chars'] = [list(token) for token in example['ques_tokens']]

    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    char_limit = args.char_limit

    def filter_func(example):
        return len(example["context_tokens"]) > para_limit or \
               len(example["ques_tokens"]) > ques_limit

    if filter_func(example):
        raise ValueError("Context/Questions lengths are over the limit")

    context_idxs = np.zeros([para_limit], dtype=np.int32)
    context_char_idxs = np.zeros([para_limit, char_limit], dtype=np.int32)
    ques_idxs = np.zeros([ques_limit], dtype=np.int32)
    ques_char_idxs = np.zeros([ques_limit, char_limit], dtype=np.int32)

    def _get_word(word):
        for each in (word, word.lower(), word.capitalize(), word.upper()):
            if each in word2idx_dict:
                return word2idx_dict[each]
        return 1

    def _get_char(char):
        if char in char2idx_dict:
            return char2idx_dict[char]
        return 1

    for i, token in enumerate(example["context_tokens"]):
        context_idxs[i] = _get_word(token)

    for i, token in enumerate(example["ques_tokens"]):
        ques_idxs[i] = _get_word(token)

    for i, token in enumerate(example["context_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            context_char_idxs[i, j] = _get_char(char)

    for i, token in enumerate(example["ques_chars"]):
        for j, char in enumerate(token):
            if j == char_limit:
                break
            ques_char_idxs[i, j] = _get_char(char)

    return context_idxs, context_char_idxs, ques_idxs, ques_char_idxs


def is_answerable(example):
    return len(example['y2s']) > 0 and len(example['y1s']) > 0


def build_features(args, examples, data_type, out_file, word2idx_dict, char2idx_dict, edge_dict,is_test=False):
    graph=False if edge_dict is None else True
    para_limit = args.test_para_limit if is_test else args.para_limit
    ques_limit = args.test_ques_limit if is_test else args.ques_limit
    ans_limit = args.ans_limit
    char_limit = args.char_limit


    def drop_example(ex, is_test_=False):
        if is_test_:
            drop = False
        else:
            drop = len(ex["context_tokens"]) > para_limit or \
                   len(ex["ques_tokens"]) > ques_limit or \
                   (is_answerable(ex) and
                    ex["y2s"][0] - ex["y1s"][0] > ans_limit)

        return drop

    print(f"Converting {data_type} examples to indices...")
    total = 0
    total_ = 0
    meta = {}
    context_idxs = []
    context_char_idxs = []
    ques_idxs = []
    ques_char_idxs = []
    y1s = []
    y2s = []
    ids = []
    if(graph):
        concept_limit=args.test_concept_limit if is_test else args.concept_limit
        concept_idxs=[]
        edges_lists=[]
    for n, example in tqdm(enumerate(examples)):
        total_ += 1

        if drop_example(example, is_test):
            continue

        total += 1

        def _get_word(word):
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in word2idx_dict:
                    return word2idx_dict[each]
            return 1

        def _get_char(char):
            if char in char2idx_dict:
                return char2idx_dict[char]
            return 1

        #construct sub graph for each example
        def construct_subgraph(example,edge_dict):
            concept_idx=np.zeros([concept_limit],dtype=np.int32)
            edges_list=[]
            node_count=0
            node_dict={}
            for i, token in enumerate(example["context_tokens"]):
                id=_get_word(token)
                if node_dict.get(str(id)) is None:
                    concept_idx[node_count]=id
                    node_dict[str(id)]=node_count
                    node_count+=1

            originalnode_dict=copy.deepcopy(node_dict)
            for start, start_index in originalnode_dict.items():
                edges=edge_dict[start]
                for edge in edges:
                    end=str(edge[0])
                    end_index=node_dict.get(end)
                    if end_index is not None:
                        edges_list.append([start_index,end_index,edge[1]])
                    else:
                        if node_count<concept_limit:
                            concept_idx[node_count] = int(end)
                            edges_list.append([start_index, node_count, edge[1]])
                            node_dict[end]=node_count
                            node_count += 1

            return concept_idx,edges_list

        context_idx = np.zeros([para_limit], dtype=np.int32)
        context_char_idx = np.zeros([para_limit, char_limit], dtype=np.int32)
        ques_idx = np.zeros([ques_limit], dtype=np.int32)
        ques_char_idx = np.zeros([ques_limit, char_limit], dtype=np.int32)

        for i, token in enumerate(example["context_tokens"]):
            context_idx[i] = _get_word(token)
        context_idxs.append(context_idx)

        for i, token in enumerate(example["ques_tokens"]):
            ques_idx[i] = _get_word(token)
        ques_idxs.append(ques_idx)

        for i, token in enumerate(example["context_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                context_char_idx[i, j] = _get_char(char)
        context_char_idxs.append(context_char_idx)

        for i, token in enumerate(example["ques_chars"]):
            for j, char in enumerate(token):
                if j == char_limit:
                    break
                ques_char_idx[i, j] = _get_char(char)
        ques_char_idxs.append(ques_char_idx)
        if(graph):
            concept_idx,edges_list=construct_subgraph(example,edge_dict)
            concept_idxs.append(concept_idx)
            edges_lists.append(edges_list)

        if is_answerable(example):
            start, end = example["y1s"][-1], example["y2s"][-1]
        else:
            start, end = -1, -1

        y1s.append(start)
        y2s.append(end)
        ids.append(example["id"])

    if(graph):
        np.savez(out_file,
                 context_idxs=np.array(context_idxs),
                 context_char_idxs=np.array(context_char_idxs),
                 ques_idxs=np.array(ques_idxs),
                 ques_char_idxs=np.array(ques_char_idxs),
                 concept_idxs=np.array(concept_idxs),
                 edges_lists=edges_lists,
                 y1s=np.array(y1s),
                 y2s=np.array(y2s),
                 ids=np.array(ids))
    else:
        np.savez(out_file,
                 context_idxs=np.array(context_idxs),
                 context_char_idxs=np.array(context_char_idxs),
                 ques_idxs=np.array(ques_idxs),
                 ques_char_idxs=np.array(ques_char_idxs),
                 y1s=np.array(y1s),
                 y2s=np.array(y2s),
                 ids=np.array(ids))
    print(f"Built {total} / {total_} instances of features in total")
    meta["total"] = total
    return meta

def get_knowledge_edges(word2idx_dict):
    '''
    get edges data from Commense conceptNet by Restful API. Only keep edges that in the word vocabulary.

    parameters
    word2idx_dict: a dict indicate word vocabulary and its index (word, idx)

    Returns
    edge_dict: a dict indicate all the edges related to a words and its weight
    '''
    print("Get edges information from ConceptNet......")
    word_list=[k for k in word2idx_dict.keys()]
    edge_dict={}
    for word, idx in word2idx_dict.items():
        if idx == 1 or idx == 0:
            edge_dict[str(idx)] = []
            continue
        try:
            url = 'http://api.conceptnet.io/c/en/' + word
            obj = requests.get(url)
            while obj.__str__() == "<Response [429]>":
                print("too many request, process sleep for 600 seconds...")
                time.sleep(600)
                obj = requests.get(url)
            obj = obj.json()
            edges = obj["edges"]
            edge_list = []
            for i in range(len(edges)):
                if edges[i]["start"]["label"] == word:
                    end_word = edges[i]["end"]["label"]
                    if end_word in word_list and end_word != word:
                        weight = edges[i]["weight"]
                        e = (word2idx_dict.get(end_word), weight)  # (end_word_index,weight)
                        edge_list.append(e)
            edge_dict[idx] = edge_list
            if (idx % 1000 == 0):
                save("./squad/data/knowledge_edges.json", edge_dict, f"save knowledge edges after {idx} iterations")
        except Exception as e:
            edge_dict[idx] = []
            print(str(e))
            save("./squad/data/knowledge_edges.json", edge_dict, f"save knowledge edges after error occur on {idx} iterations")

    return edge_dict

def save(filename, obj, message=None):
    if message is not None:
        print(f"Saving {message}...")
        with open(filename, "w") as fh:
            json.dump(obj, fh)


def pre_process(args):
    # Process training set and use it to decide on the word/character vocabularies
    word_counter, char_counter = Counter(), Counter()
    train_examples, train_eval = process_file(args.train_file, "train", word_counter, char_counter)
    word_emb_mat, word2idx_dict = get_embedding(
        word_counter, 'word', emb_file=args.glove_file, vec_size=args.glove_dim, num_vectors=args.glove_num_vecs)
    char_emb_mat, char2idx_dict = get_embedding(
        char_counter, 'char', emb_file=None, vec_size=args.char_dim)


    if(args.add_knowledge_graph):
        #edges_dict=get_knowledge_edges(word2idx_dict)
        with open("./data/knowledge_edges.json", "r") as fh:
            edge_dict = json.load(fh)
        # Process dev and test sets
        dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter)
        build_features(args, train_examples, "train", args.train_record_file_graph, word2idx_dict, char2idx_dict,edge_dict)
        dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file_graph, word2idx_dict,char2idx_dict,edge_dict)
    else:
        dev_examples, dev_eval = process_file(args.dev_file, "dev", word_counter, char_counter)
        build_features(args, train_examples, "train", args.train_record_file, word2idx_dict, char2idx_dict,None)
        dev_meta = build_features(args, dev_examples, "dev", args.dev_record_file, word2idx_dict,char2idx_dict,None)

    if args.include_test_examples:
        test_examples, test_eval = process_file(args.test_file, "test", word_counter, char_counter)
        save(args.test_eval_file, test_eval, message="test eval")
        if (args.add_knowledge_graph):
            test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file_graph, word2idx_dict, char2idx_dict,edge_dict, is_test=True)
        else:
            test_meta = build_features(args, test_examples, "test",
                                   args.test_record_file, word2idx_dict, char2idx_dict,None, is_test=True)
        save(args.test_meta_file, test_meta, message="test meta")

    save(args.word_emb_file, word_emb_mat, message="word embedding")
    save(args.char_emb_file, char_emb_mat, message="char embedding")
    save(args.train_eval_file, train_eval, message="train eval")
    save(args.dev_eval_file, dev_eval, message="dev eval")
    save(args.word2idx_file, word2idx_dict, message="word dictionary")
    save(args.char2idx_file, char2idx_dict, message="char dictionary")
    save(args.dev_meta_file, dev_meta, message="dev meta")
    #save(args.knowledge_edges_file,edges_dict,message="knowledge graph edges")



if __name__ == '__main__':
    # Get command-line args
    args_ = get_setup_args()

    # Download resources
    download(args_)

    # Import spacy language model
    nlp = spacy.blank("en")

    # Preprocess dataset
    args_.train_file = url_to_data_path(args_.train_url)
    args_.dev_file = url_to_data_path(args_.dev_url)
    if args_.include_test_examples:
        args_.test_file = url_to_data_path(args_.test_url)
    glove_dir = url_to_data_path(args_.glove_url.replace('.zip', ''))
    glove_ext = f'.txt' if glove_dir.endswith('d') else f'.{args_.glove_dim}d.txt'
    args_.glove_file = os.path.join(glove_dir, os.path.basename(glove_dir) + glove_ext)
    pre_process(args_)

