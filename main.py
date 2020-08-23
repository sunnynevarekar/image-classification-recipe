import numpy as np
import torch
import torchvision
from torchvision import transforms as T


from datasets import ImageClassificationDataset, load_annotations_from_folder, split_data
from engine import train_one_epoch, evaluate, fit
from callbacks import TrainMetricRecorder

import viz_utils

import argparse


        


def main():
    
    #Folder for images
    DATA_PATH = '/home/sunny/work/fridgeObjects'
    IMAGE_SIZE = 300
    BATCH_SIZE = 16
    LR = 0.0001
    EPOCHS = 10
    RANDOM_STATE = 19

    #define argument parser
    parser = argparse.ArgumentParser(description='Image classfication.')
    parser.add_argument('-fp', '--filepath', help='Base directory for the dataset')
    parser.add_argument('-i', '--image_size', help='Image size', type=int)
    parser.add_argument('-b', '--batch_size', help='Batch size', type=int)
    parser.add_argument('-lr', '--lr', help='Learning rate', type=float)
    parser.add_argument('-e', '--epochs', help='Number of epochs', type=int)
    parser.add_argument('-rs', '--random_state', help='Random state for datasplit')

    args = vars(parser.parse_args())

    if args['filepath']:
        DATA_PATH = args['filepath']

    if args['image_size']:
        IMAGE_SIZE = args['image_size']

    if args['batch_size']:
        BATCH_SIZE = args['batch_size']

    if args['lr']:
        LR = args['lr']

    if args['epochs']:
        EPOCHS = args['epochs']

    if args['random_state']:
        RANDOM_STATE = args['random_state']



    #load annotations
    image_ids, labels, class_names = load_annotations_from_folder(DATA_PATH)
    assert len(image_ids) == len(labels)

    print(f'Total images: {len(image_ids)}')
    print(f'Classes: {class_names}')

    #split data into train and validation set
    train_img_ids, val_img_ids, train_labels, val_labels = split_data(image_ids, labels, test_size=0.2, 
                                                                       shuffle=True, stratify=labels, random_state=RANDOM_STATE)

    assert len(train_img_ids) == len(train_labels)
    assert len(val_img_ids) == len(val_labels)
    
    print(f'Train samples: {len(train_img_ids)}')
    print(f'Validation samples: {len(val_img_ids)}')
    

    #calculate and display number of samples per class in train and test set
    print()
    print('Class distribution:')

    print('Train set:')
    train_classes, train_counts = np.unique(train_labels, return_counts=True)
    assert len(train_classes) == len(class_names)
    viz_utils.print_table(class_names, train_counts, header=['class', 'count'])
    print()
    print('validation set:') 
    val_classes, val_counts = np.unique(val_labels, return_counts=True)
    assert len(val_classes) == len(class_names)
    viz_utils.print_table(class_names, val_counts, header=['class', 'count'])

    #define transforms for train and val dataset
    #imagenet stats
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    train_transforms = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean, std)])
    val_transforms = T.Compose([T.Resize((IMAGE_SIZE, IMAGE_SIZE)), T.ToTensor(), T.Normalize(mean, std)])

    #create train and validation dataset
    train_dataset = ImageClassificationDataset(DATA_PATH, train_img_ids, train_labels, train_transforms)
    val_dataset = ImageClassificationDataset(DATA_PATH, val_img_ids, val_labels, val_transforms)

    #checks
    assert len(train_dataset) == len(train_img_ids)
    assert len(val_dataset) == len(val_img_ids)

    ti, tl = train_dataset[0]
    assert ti.shape == torch.Size((3, IMAGE_SIZE, IMAGE_SIZE))
    assert type(tl) == int

    #create dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader =  torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    batch_inputs, batch_labels = next(iter(train_loader))
    assert batch_inputs.shape == torch.Size((BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE))
    assert batch_labels.shape == torch.Size((BATCH_SIZE,))

    out = torchvision.utils.make_grid(batch_inputs)
    viz_utils.imshow(out, mean, std, title=[class_names[x] for x in batch_labels], filename='logs/train_img.png')

    batch_inputs, batch_labels = next(iter(val_loader))
    out = torchvision.utils.make_grid(batch_inputs)
    viz_utils.imshow(out, mean, std, title=[class_names[x] for x in batch_labels], filename='logs/val_img.png')


    #create model
    model = torchvision.models.resnet18(pretrained=True)
    in_features = model.fc.in_features
    model.fc = torch.nn.Linear(in_features, len(class_names))

    #create loss criterion
    criterion = torch.nn.CrossEntropyLoss()

    #create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    #select gpu device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #move model to right device
    model.to(device)

    #create recorder to record metrics during training
    recorder = TrainMetricRecorder(['accuracy', 'precision', 'recall', 'f1_score'])

    print("Training:")
   
    fit(model, train_loader, val_loader, optimizer, criterion, EPOCHS, device, recorder)
    
    history = recorder.history

    header = ['train_loss', 'val_loss', 'train_accuracy', 'val_accuracy']

    viz_utils.print_table(history[header[0]], history[header[1]], history[header[2]], history[header[3]], header=header)
    



if __name__ == '__main__':
    main()


