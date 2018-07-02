import argparse
import _init_paths
import lib.utils.process_data as process_data
import lib.utils.vocabulary as vocabulary

def main(args):
    # Saving the data in splits for faster load time
    train_split, dev_split, test_split, paragraph_topics, paragraph_json = process_data.load_json(args.data_dir)
    train_data, dev_data, test_data, image_ids, topic_set = process_data.parse_data(train_split, dev_split, test_split, 
                                                                                    paragraph_topics, paragraph_json, args.disable_progress_bar)
    base = process_data.save_data(train_data, dev_data, test_data, image_ids, topic_set, args.data_dir)
    print("Processed and saved data at {}".format(base))

    # Building and saving the vocabulary
    word_vocab = vocabulary.build_vocab([x[2] for x in train_data], args.min_occurrences)
    topic_vocab = vocabulary.Vocabulary()
    for topic in topic_set:
        topic_vocab.add_word(topic)
    vocab = {'word_vocab': word_vocab, 'topic_vocab': topic_vocab}
    save_file = vocabulary.save_vocab(vocab, args.data_dir, min_occurrences=args.min_occurrences)
    print("Built and saved vocabularies at {}".format(save_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='./data',
                        help='The data directory. Default value of ./data')
    parser.add_argument('--min_occurrences', type=int,
                        default=5,
                        help='The minimum number of times a word must appear in the train data to be included \
                              in the vocabulary. Default value of 5.')
    parser.add_argument('--disable_progress_bar', action='store_false',
                        help='Set to disable the progress bar.')
    args = parser.parse_args()
    main(args)
