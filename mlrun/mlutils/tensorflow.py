from tensorflow import keras
from ..execution import MLClientCtx


class MLRunTFTrainLogger(keras.callbacks.Callback):
    def __init__(self, context: MLClientCtx):
        """An MLRun Callback to collect training statistics
        per epoch.

        Each epoch's results will be tracked individually and
        the final epoch will be tracked as the final ('best') result.

        Parameters
        ----------
        context : MLClientCtx
            an MLRun context.

        Example
        --------
        ```
        callbacks = []
        # When training with horovod please add the tracker only on rank 0
        # if hvd.rank() == 0:
        callbacks = callbacks.append(MLRunTrainLogger(mlctx))

        # Train with the selected callbacks
        history = model.fit(
            <training_dataset>,
            callbacks=callbacks,
        )

        ```
        """
        self.context = context
        self.context.logger.debug("Tracking training results with MLRunTFTrainLogger")

    def on_train_begin(self, logs=None):
        """Create a base for new iterations as the training starts

        Parameters
        ----------
        logs : dict, optional
            available training logs with collected metrics, by default None
        """
        # self.context.logger.info("Train start callback")
        pass

    def on_train_end(self, logs=None):
        """Finishes the training phase and sets the `best_iteration` for the run

        Parameters
        ----------
        logs : dict, optional
            available training logs with collected metrics, by default None
        """
        # self.context.logger.info("Train end callback")
        self.context.log_result("best_iteration", len(self.context._child))

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        """This callback collects the given metrics of the current epoch and
        logs them in the MLRun DB.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        logs : dict, optional
            a Dictionary containing the collected epoch metric results.
            by default None
        """
        # Create child context to hold the current epoch's results
        child_ctx = self.context.get_child()

        # Go over the metrics and log them to the context
        for metric, result in logs.items():
            child_ctx.log_result(metric, result)

        # Commit and commit children for MLRun flag bug
        self.context.commit_children()
        self.context.commit()

    def on_test_begin(self, logs=None):
        # self.context.logger.info("Test started callback")
        pass

    def on_test_end(self, logs=None):
        # self.context.logger.info("Test ended callback")
        pass
