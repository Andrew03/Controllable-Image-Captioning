import pylab as plt
import argparse
import pickle

def plot(args):
    loss_markers = {'train': 'bo', 'val': 'ro'}
    accuracy_markers = {'train': 'go', 'val': 'yo'}
    markersizes = {'train': 1, 'val': 3}
    trials = {'train': [], 'val': []}
    losses = {'train': [], 'val': []}
    accuracies = {'train': [], 'val': []}

    with open(args.log_file, "rb") as f:
        logs = pickle.load(f)

    loader_length = logs['loader_length']

    for phase in ['train', 'val']:
        log = logs[phase]
        for epoch, score_log in sorted(log.items(), key=lambda x: x[0]):
            for trial in sorted(score_log['loss'].keys()):
                trials[phase].append(trial + loader_length * (epoch - 1))
                losses[phase].append(score_log['loss'][trial])
                accuracies[phase].append(score_log['accuracy'][trial])

    # plotting the data
    for x in range(max(logs['train'].keys())):
        plt.axvline(x * loader_length)
    for x in ['train', 'val']:
        plt.plot(trials[x], losses[x], loss_markers[x], markersizes[x], label="{} loss".join(x))
        plt.plot(trials[x], accuracies[x], loss_markers[x], markersizes[x], label="{} loss".join(x))

    plt.legend()
    plt.xlabel=('Number of Batches')
    plt.ylabel=('Scores')
    plt.title=('Scores over {} Epochs'.format(len(logs['train'])))

    plt.show()

def main(args):
    plot(args)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_file', type=str,
                        required=True,
                        help='Path to log file. Required')
    args = parser.parse_args()
    main(args)
