# Experiments with captum for visualizing deep learning models

## Initial setup

```
python3 -m virtualenv venv
source venv/bin/activate
python -m pip install .
```

## Training a model

After sourcing the virtual environmnet

```
python -m torchtmpl.main config.yml train
```

With the sample configuration file, with a resnet18, you should get around 77% of validation accuracy after 100 epochs.

## Visualiation with captum

Once a model is trained, you can run the captum insights visualization tool.

The trained model is saved in the `logs` subdirectory. You need to provide the specific run you want to visualize. For example, for visualizing the run saved in `logs/resnet18_0` :

```
python -m torchtmpl.visualize logs/resnet18_0/
```

That should start the flask application to which you can connect with your browser and then experiment with the visualization algorithms. An example is displayed below.

![alt text](https://github.com/jeremyfix/test_captum/blob/main/images/occlusion.png?raw=true)


