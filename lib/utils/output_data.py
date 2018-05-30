def load_data(basedir, model_name, num_epochs):
    import pickle

    if basedir is not None:
        with open("{}/data/outputs/{}/train_{}.pkl".format(basedir, model_name, num_epochs), "rb") as f:
            output_train_data = pickle.load(f)
        with open("{}/data/outputs/{}/val_{}.pkl".format(basedir, model_name, num_epochs), "rb") as f:
            output_val_data = pickle.load(f)
        with open("{}/data/outputs/{}/bleu_{}.pkl".format(basedir, model_name, num_epochs), "rb") as f:
            bleu_score_data = pickle.load(f)
        with open("{}/data/outputs/{}/rouge_{}.pkl".format(basedir, model_name, num_epochs), "rb") as f:
            rouge_score_data = pickle.load(f)
        with open("{}/data/outputs/{}/cider_{}.pkl".format(basedir, model_name, num_epochs), "rb") as f:
            cider_score_data = pickle.load(f)
        with open("{}/data/outputs/{}/attention_{}.pkl".format(basedir, model_name, num_epochs), "rb") as f:
            attention_val_data = pickle.load(f)
    else:
        output_train_data = {}
        output_val_data = {}
        bleu_score_data = {}
        rouge_score_data = {}
        cider_score_data = {}
        attention_val_data = {}
    return output_train_data, output_val_data, bleu_score_data, 
            rouge_score_data, cider_score_data, attention_score_data

def save_data(output_train_data, output_val_data, bleu_score_data, rouge_score_data, 
              cider_score_data, attention_score_data, basedir, model_name, num_epochs):
    import pickle

    with open("{}/data/outputs/{}/train_{}.pkl".format(basedir, model_name, num_epochs), "wb") as f:
        pickle.dump(output_train_data, f)
    with open("{}/data/outputs/{}/val_{}.pkl".format(basedir, model_name, num_epochs), "wb") as f:
        pickle.dump(output_val_data, f)
    with open("{}/data/outputs/{}/bleu_{}.pkl".format(basedir, model_name, num_epochs), "wb") as f:
        pickle.dump(bleu_score_data, f)
    with open("{}/data/outputs/{}/rouge_{}.pkl".format(basedir, model_name, num_epochs), "wb") as f:
        pickle.dump(rouge_score_data, f)
    with open("{}/data/outputs/{}/cider_{}.pkl".format(basedir, model_name, num_epochs), "wb") as f:
        pickle.dump(cider_score_data, f)
    with open("{}/data/outputs/{}/attention_{}.pkl".format(basedir, model_name, num_epochs), "wb") as f:
        pickle.dump(attention_score_data, f)
