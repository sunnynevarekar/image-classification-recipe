from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class TrainMetricRecorder:
    #train loss
    #val loss
    #train accuracy
    #val accuracy
    METRICS = {'accuracy': accuracy_score, 'precision': precision_score, 'recall': recall_score, 'f1_score': f1_score}
    def __init__(self, metrics):
        self.history = {}
        self.history['train_loss'] = []
        self.history['val_loss'] = []
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions =[]
        self.val_targets = []
        self.train_loss = []
        self.val_loss = []
        for metric in metrics:
            if metric not in self.METRICS:
                raise ValueError(f'{metric} is not a valid metric. Valid metrics are: {TrainMetricRecorder.METRICS}')
            else:
                self.history['train_'+metric] = []
                self.history['val_'+metric] = []
        self.metrics = metrics        

    def on_train_batch_end(self, y_true, y_preds, loss):
        self.train_targets.extend(y_true)
        self.train_predictions.extend(y_preds)
        self.train_loss.append(loss)

    def on_val_batch_end(self, y_true, y_preds, loss):
        self.val_targets.extend(y_true)
        self.val_predictions.extend(y_preds)
        self.val_loss.append(loss)                

    def on_epoch_start(self):
        self.train_predictions = []
        self.train_targets = []
        self.val_predictions =[]
        self.val_targets = []
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self):
        for metric_name in self.metrics:
            if metric_name != 'accuracy':
                self.history['train_'+metric_name].append(self.METRICS[metric_name](self.train_targets, self.train_predictions, average='macro'))
                self.history['val_'+metric_name].append(self.METRICS[metric_name](self.val_targets, self.val_predictions, average='macro'))
            else:
                self.history['train_'+metric_name].append(self.METRICS[metric_name](self.train_targets, self.train_predictions))
                self.history['val_'+metric_name].append(self.METRICS[metric_name](self.val_targets, self.val_predictions)  )

        #calculate average loss
        if len(self.train_loss) > 0:
            self.history['train_loss'].append(sum(self.train_loss)/len(self.train_loss))

        if len(self.val_loss) > 0:
            self.history['val_loss'].append(sum(self.val_loss)/len(self.val_loss))    

