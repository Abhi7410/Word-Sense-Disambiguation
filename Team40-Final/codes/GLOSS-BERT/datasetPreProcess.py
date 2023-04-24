import argparse
import csv
import random
import re
from pathlib import Path
from xml.etree.ElementTree import ElementTree
import nltk
from tqdm import tqdm


from nltk.corpus import wordnet as wn
POS = {'NOUN': wn.NOUN, 'VERB': wn.VERB, 'ADJ': wn.ADJ, 'ADV': wn.ADV}


def getInfo(type, pos, lemma):
    res = dict()
    word_pos = POS[pos] if pos is not None else None
    morpho = wn._morphy(lemma, pos=word_pos) if pos is not None else []

    for synset in tqdm(set(wn.synsets(lemma, pos=word_pos))):
        key = None
        for lem in synset.lemmas():
            if lem.name().lower() == lemma.lower():
                key = lem.key()
                break
            elif lem.name().lower() in morpho:
                key = lem.key()

        assert key is not None
        res[key] = synset.definition() if type == 'def' else synset.examples()

    return res


def get_glosses(pos, lemma):
    return getInfo('def', pos, lemma)


def getexample(pos, lemma):
    return getInfo('ex', pos, lemma)


def getAllWordnetLemmaNames():
    res = []
    for pos, pos_name in POS.items():
        for synset in wn.synsets(pos=pos_name):
            res.append((pos, wn.all_lemma_names(pos=pos_name)))

    return res


xml_file = './SemCor/semcor.data.xml'
gold_txt_file = './SemCor/semcor.gold.key.txt'
output_file = './SemCor/semcor_data.csv'
max_glossKey = 4

print("Creating dataset...")
root = ElementTree(file=xml_file).getroot()
with open(output_file, 'w', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['id', 'sentence', 'sense_keys',
                    'glosses', 'target_words'])

    def write_to_csv(id_, sentence_, lemma_, pos_, gold_keys_):
        sense_i = get_glosses(pos_, lemma_)
        # print(sense_i)
        gloss_sense_pairs = list()
        for i in gold_keys_:
            gloss_sense_pairs.append((i, sense_i[i]))
            del sense_i[i]
        rem = max_glossKey - len(gloss_sense_pairs)
        if len(sense_i) > rem:
            gloss_sense_pairs.extend(random.sample(list(sense_i.items()), rem))
        elif len(sense_i) > 0:
            gloss_sense_pairs.extend(list(sense_i.items()))

        random.shuffle(gloss_sense_pairs)
        glosses = [i[1] for i in gloss_sense_pairs]
        sense_keys = [i[0] for i in gloss_sense_pairs]

        target_words = [sense_keys.index(i) for i in gold_keys_]
        writer.writerow([id_, sentence_, sense_keys, glosses, target_words])

    with open(gold_txt_file, 'r', encoding='utf-8') as g:
        for dc in tqdm(root):
            for sentence in dc:
                instances = list()
                tokens = list()
                for token in sentence:
                    tokens.append(token.text)
                    if token.tag == 'instance':
                        strt_index = len(tokens) - 1
                        end_index = strt_index + 1
                        instances.append(
                            (token.attrib['id'], strt_index, end_index, token.attrib['lemma'], token.attrib['pos']))
                # print(instances)

                for id_, start, end, lemma, pos in instances:
                    gold_key = g.readline().strip().split()
                    gold = gold_key[1:]
                    assert id_ == gold_key[0]
                    sentence_ = ' '.join(
                        tokens[:start] + ['<target>'] +
                        tokens[start:end] + ['</target>'] + tokens[end:]
                    )
                    write_to_csv(id_, sentence_, lemma, pos, gold)


print("Done!")




# Path: NN/datasetPreProcess.py
# This file creates desired dataset csv file using SemCor corpus (2.26 lakh) sentences.
# csv format - id, sentence, sense_keys, glosses, target_words