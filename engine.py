import torch
from tqdm import tqdm

def train_one_epoch(model, dataloader, optimizer, criterion, device=None, recorder=None):
    #set model in train mode
    model.train()
    
    #accumulate running loss
    running_loss = 0

    for inputs, labels in tqdm(dataloader):
        if device:
            #move data to right device
            inputs = inputs.to(device)
            labels = labels.to(device)

        #forward pass
        logits = model(inputs)

        #calculate loss
        loss = criterion(logits, labels)

        #set gredients to zero
        optimizer.zero_grad()

        #backward pass
        loss.backward()

        #update weights
        optimizer.step()

        running_loss = running_loss + loss.item()
        #records batch labels and targets for computing metrics
        if recorder:
            with torch.no_grad():
                _, preds = torch.max(logits, 1)
        
            recorder.on_train_batch_end(labels.cpu().numpy(), preds.cpu().numpy(), loss.item())
    
    return running_loss/len(dataloader)


    

def evaluate(model, dataloader, criterion, device=None, recorder=None):
    #set model in eval mode
    model.eval()
    #accumulate loss
    running_loss = 0

    for inputs, labels in tqdm(dataloader):
        if device:
            #move data to right device
            inputs = inputs.to(device)
            labels = labels.to(device)    
        #forward pass to get logits
        #we dont need gradients for evaluation
        with torch.no_grad():
            logits = model(inputs)

        #calculate loss
        loss = criterion(logits, labels)
        running_loss += loss.item()

        if recorder:
            _, preds = torch.max(logits, 1)
            recorder.on_val_batch_end(labels.cpu().numpy(), preds.cpu().numpy(), loss.item())
    
    return running_loss/len(dataloader)      
        

def fit(model, train_loader, va_loader, optimizer, criterion, epochs, device=None, recorder=None):
    for epoch in range(epochs):
        if recorder:
            recorder.on_epoch_start()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, recorder)
        val_loss = evaluate(model, va_loader, criterion, device, recorder)
        if recorder:
            recorder.on_epoch_end()

        print()
        print(f'Epoch {epoch+1}: train_loss: {train_loss:.4f} val_loss: {val_loss: .4f}')
        print()


def get_predictions(model, dataloader, device=None):
    #set model in eval mode
    model.eval()
    #lists to accumulate predicted classes and corresponding probabilities
    predictions = []
    probabilities = []
    predictions = []

    for batch in tqdm(dataloader):
        if len(batch) > 1:
            inputs = batch[0]
        else:
            inputs = batch
        
        if device:
            inputs = inputs.to(device)

        with torch.no_grad():
            logits = model(inputs)
        
        probs, preds = torch.max(torch.nn.functional.softmax(logits, 1), 1)

        #predictions
        predictions.extend(preds.cpu().numpy())
        #probabilities
        probabilities.extend(probs.cpu().numpy())

    return predictions,  probabilities      