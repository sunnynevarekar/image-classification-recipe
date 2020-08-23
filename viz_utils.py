import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def print_table(*arrays, header):
    #print header
    assert len(arrays) == len(header)
    num_rows = len(arrays[0])
    for column in header:
        print(f'{column:<20}', end='')
    print()
    
    
    for j in range(num_rows):
        for i in range(len(header)):
            ele = arrays[i][j]
            if isinstance(ele, str) or isinstance(ele, int):
                print(f'{ele:<20}', end='')
            else:
                print(f'{ele:<20.4f}', end='')
        print()    

def imshow(inp, mean, std, title=None, filename=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated
    if filename:
        plt.savefig(filename)


def plot_confusion_matrix(conf_mat, labels):        
    plt.title('Confusion matrix')
    sns.heatmap(conf_mat, annot=True, cmap='Blues', fmt='', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()    