# coding: utf-8

# Standard imports
import logging
import sys
import os

# External imports
import yaml
import torch
import torch.nn as nn
from torchvision import transforms
import onnxruntime as ort

from captum.insights import AttributionVisualizer, Batch
from captum.insights.attr_vis.features import ImageFeature

# Local imports
from . import data
from . import models


# class ONNXWrapper:
#     def __init__(self, use_cuda, modelpath):
#         self.device = torch.device("cuda") if use_cuda else torch.device("cpu")
#         providers = []
#         if use_cuda:
#             providers.append("CUDAExecutionProvider")
#         providers.append("CPUExecutionProvider")
#         self.inference_session = ort.InferenceSession(modelpath, providers=providers)

#     def eval(self):
#         # ONNX model cannot be switched from train to test
#         pass

#     def train(self):
#         # ONNX model cannot be switch from test to train
#         pass

#     def __call__(self, torchX):
#         output = self.inference_session.run(
#             None, {self.inference_session.get_inputs()[0].name: torchX.cpu().numpy()}
#         )[0]

#         return torch.from_numpy(output).to(self.device)


def baseline_func(inp):
    return inp * 0


def visualize(rundir, config):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    data_config = config["data"]
    # Hand tune the batch size to get a reasonnable number of
    # inputs
    data_config["batch_size"] = 16
    train_loader, valid_loader, input_size, classes = data.get_dataloaders(
        data_config, use_cuda
    )
    logging.info(classes)

    def formatted_data_iter():
        it_valid = iter(valid_loader)
        while True:
            inputs, labels = next(it_valid)
            inputs, labels = inputs.to(device), labels.to(device)
            yield Batch(inputs=inputs, labels=labels)

    num_classes = len(classes)
    model_config = config["model"]
    model = models.build_model(model_config, input_size, num_classes)
    model.to(device)

    modelpath = os.path.join(rundir, "best_model.pt")
    model.load_state_dict(torch.load(modelpath))
    # Switch the model to evaluation mode for the BN and dropout
    model.eval()

    base_transform = transforms.Compose(
        [
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            data.to_color,
        ]
    )
    train_loader.dataset.dataset.transform = base_transform

    input_transform = transforms.Compose(
        [
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    visualizer = AttributionVisualizer(
        models=[model],
        score_func=lambda o: nn.functional.softmax(o, 1),
        classes=classes,
        dataset=formatted_data_iter(),
        features=[
            ImageFeature(
                "Photo",
                baseline_transforms=[baseline_func],
                input_transforms=[input_transform],
            )
        ],
    )
    visualizer.serve(debug=True)


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    if len(sys.argv) != 2:
        logging.error(f"Usage : {sys.argv[0]} <rundir>")
        sys.exit(-1)

    rundir = sys.argv[1]

    logging.info("Loading {}".format(sys.argv[1]))
    config = yaml.safe_load(open(os.path.join(rundir, "config.yaml"), "r"))
    visualize(rundir, config)
