def load_json(basedir):
    import pickle
    import json

    with open("{}/data/raw/splits/train_split.json".format(basedir), "r") as f:
      train_split = json.load(f)
    with open("{}/data/raw/splits/dev_split.json".format(basedir), "r") as f:
        dev_split = json.load(f)
    with open("{}/data/raw/splits/test_split.json".format(basedir), "r") as f:
        test_split = json.load(f)
    with open("{}/data/raw/paragraphs_topics_v1.pickle".format(basedir), "rb") as f:
        paragraph_topics = pickle.load(f)
    with open("{}/data/raw/paragraphs_v1.json".format(basedir), "r") as f:
        paragraph_json = json.load(f)
    return train_split, dev_split, test_split, paragraph_topics, paragraph_json

def parse_data(train_split, dev_split, test_split, paragraph_topics, paragraph_json, progress_bar=True):
    from tqdm import tqdm
    import nltk
    import spacy
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    nlp = spacy.load('en')

    train_data = []
    dev_data = []
    test_data = []
    image_ids = {}
    topic_set = set()
    for i, json in enumerate(tqdm(paragraph_json) if progress_bar else paragraph_json):
        topic_to_seq = {}
        for j, sentence in enumerate(sent_detector.tokenize(json['paragraph'])):
            sentence = sentence.strip().lower()
            t = nlp(sentence)
            image_id = json['image_id']
            image_ids[image_id] = i

            if 'perfect_match' in paragraph_topics[i][j]:
                topic_list = set([topic[0] for topic in paragraph_topics[i][j]['perfect_match']])
                for topic in topic_list:
                    topic_set.add(topic)
                    if topic not in topic_to_seq:
                        topic_to_seq[topic] = []
                    topic_to_seq[topic].extend(t)
            for topic in topic_to_seq:
                sentence = [word.text for word in topic_to_seq[topic]]
                if image_id in train_split:
                    train_data.append((image_id, topic, sentence))
                elif image_id in dev_split:
                    dev_data.append((image_id, topic, sentence))
                elif image_id in test_split:
                    test_data.append((image_id, topic, sentence))
    return train_data, dev_data, test_data, image_ids, topic_set

def save_data(train_data, dev_data, test_data, image_ids, topic_set, basedir):
    import pickle

    base = "{}/data/datasets".format(basedir)
    with open("{}/train_data.pkl".format(base), "wb") as f:
        pickle.dump(train_data, f)
    with open("{}/dev_data.pkl".format(base), "wb") as f:
        pickle.dump(dev_data, f)
    with open("{}/test_data.pkl".format(base), "wb") as f:
        pickle.dump(test_data, f)
    with open("{}/image_ids.pkl".format(base), "wb") as f:
        pickle.dump(image_ids, f)
    with open("{}/topic_set.pkl".format(base), "wb") as f:
        pickle.dump(topic_set, f)
    return base

def load_data(basedir):
    import pickle

    base = "{}/data/datasets".format(basedir)
    with open("{}/train_data.pkl".format(base), "rb") as f:
        train_data = pickle.load(f)
    with open("{}/dev_data.pkl".format(base), "rb") as f:
        dev_data = pickle.load(f)
    with open("{}/test_data.pkl".format(base), "rb") as f:
        test_data = pickle.load(f)
    with open("{}/image_ids.pkl".format(base), "rb") as f:
        image_ids = pickle.load(f)
    with open("{}/topic_set.pkl".format(base), "rb") as f:
        topic_set = pickle.load(f)
    return train_data, dev_data, test_data, image_ids, topic_set
