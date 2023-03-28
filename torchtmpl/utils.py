# coding: utf-8

# Standard imports
import os

# External imports
import torch
import torch.nn
import tqdm


def generate_unique_logpath(logdir, raw_run_name):
    """
    Generate a unique directory name
    Argument:
        logdir: the prefix directory
        raw_run_name(str): the base name
    Returns:
        log_path: a non-existent path like logdir/raw_run_name_xxxx
                  where xxxx is an int
    """
    i = 0
    while True:
        run_name = raw_run_name + "_" + str(i)
        log_path = os.path.join(logdir, run_name)
        if not os.path.isdir(log_path):
            return log_path
        i = i + 1


class ModelCheckpoint(object):
    """
    Early stopping callback
    """

    def __init__(
        self,
        model: torch.nn.Module,
        savepath_pt,
        savepath_onnx,
        input_size,
        device,
        min_is_best: bool = True,
    ) -> None:
        self.model = model
        self.savepath_pt = savepath_pt
        self.savepath_onnx = savepath_onnx
        self.input_size = input_size
        self.device = device
        self.best_score = None
        if min_is_best:
            self.is_better = self.lower_is_better
        else:
            self.is_better = self.higher_is_better

    def lower_is_better(self, score):
        return self.best_score is None or score < self.best_score

    def higher_is_better(self, score):
        return self.best_score is None or score > self.best_score

    def update(self, score):
        if self.is_better(score):
            torch.save(self.model.state_dict(), self.savepath_pt)
            self.model.eval()
            export_input_size = (1,) + self.input_size
            torch.onnx.export(
                self.model,
                torch.zeros(export_input_size, device=self.device),
                self.savepath_onnx,
                verbose=False,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
            )
            self.best_score = score
            return True
        return False


def train(model, loader, f_loss, optimizer, device, fn_metrics, dynamic_display=True):
    """
    Train a model for one epoch, iterating over the loader
    using the f_loss to compute the loss and the optimizer
    to update the parameters of the model.
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    optimizer -- A torch.optim.Optimzer object
    device    -- A torch.device
    Returns :
    The averaged train metrics computed over a sliding window
    """

    # We enter train mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.train()

    total_loss = 0
    num_samples = 0
    for _, fn_metrics_i in fn_metrics.items():
        fn_metrics_i.reset()

    for i, (inputs, targets) in (pbar := tqdm.tqdm(enumerate(loader), file=sys.stdout)):

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

        for _, fn_metrics_i in fn_metrics.items():
            fn_metrics_i.update(outputs, targets)

        metrics_str = " | ".join(
            f"{name}: {metric.value():.2f}" for name, metric in fn_metrics.items()
        )
        pbar.set_description(
            f"Train loss : {total_loss/num_samples:.2f} | {metrics_str}"
        )

    return total_loss / num_samples


def test(model, loader, f_loss, device, fn_metrics):
    """
    Test a model over the loader
    using the f_loss as metrics
    Arguments :
    model     -- A torch.nn.Module object
    loader    -- A torch.utils.data.DataLoader
    f_loss    -- The loss function, i.e. a loss Module
    device    -- A torch.device
    Returns :
    """

    # We enter eval mode.
    # This is important for layers such as dropout, batchnorm, ...
    model.eval()

    total_loss = 0
    num_samples = 0
    for _, fn_metrics_i in fn_metrics.items():
        fn_metrics_i.reset()
    for (inputs, targets) in loader:

        inputs, targets = inputs.to(device), targets.to(device)

        # Compute the forward propagation
        outputs = model(inputs)

        loss = f_loss(outputs, targets)

        for _, fn_metrics_i in fn_metrics.items():
            fn_metrics_i.update(outputs, targets)

        # Update the metrics
        # We here consider the loss is batch normalized
        total_loss += inputs.shape[0] * loss.item()
        num_samples += inputs.shape[0]

    return total_loss / num_samples
