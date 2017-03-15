from keras.callbacks import CSVLogger


class RefCSVLogger(CSVLogger):
    """Refined CSVLogger.

    Also record results at arbitrary batches.
    """
    def __init__(self, filename, separator=',', append=False, batch_epochs=[]):
        super().__init__(filename, separator=separator, append=append)
        self.batch_epochs = batch_epochs
        self.selected_epoch_batches = []

    def on_train_begin(self, logs=None):
        super().on_train_begin()
        self.batch_frac = self.params['batch_size'] / self.params['samples']
        for batch_epoch in self.batch_epochs:
            epoch = int(batch_epoch)
            batch = round((batch_epoch - epoch) / self.batch_frac)
            self.selected_epoch_batches += [(epoch, batch)]

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch = epoch
        self.batch_logs_collection = []

    def on_batch_end(self, batch, logs=None):
        # Collect logs for selected batches.
        if (self.epoch, batch) in self.selected_epoch_batches:
            outputs = self.model.test_on_batch(*self.validation_data)
            metrics = self.params['metrics']
            # The latter half of metrics are supposed to be 'val_*'.
            metrics = metrics[len(metrics) // 2:]
            test_logs = dict(zip(metrics, outputs))
            self.batch_logs_collection += [(batch, {**logs, **test_logs})]

    def on_epoch_end(self, epoch, logs=None):
        for batch, batch_logs in self.batch_logs_collection:
            batch_epoch = epoch - 1 + (batch + 1) * self.batch_frac
            batch_logs = {k: batch_logs[k] for k in logs.keys()}
            super().on_epoch_end(batch_epoch, logs=batch_logs)
        super().on_epoch_end(epoch, logs=logs)
