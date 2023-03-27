# SaliencyMapsXAIVA
Code repository for the 2022 seminar on Explainable AI by Visual Analytics at Hasso-Plattner-Institute for Digital Engineering.

This project aims to develop a Visual Analytics system which allows Machine Learning practitioners and domain experts to interactively monitor and influence the training process of Deep Neural Networks (DNNs), more specifically Convolutional Neural Networks (CNNs), to solve a multiclass classification problem. Motivated by current advances in generating precise visual explanations using different Saliency Map methods, this web-based system uses different gradient-based and activation-based methods to produce visual explanations for images. This makes it more possible for humans to assess and interpret the prediction behavior and evolution of a model across the training phase. Additionally, this opens up a variety of means for humans to intervene in to the training process through methods like early stopping, penalizing the loss for undesired prediction behavior etc.

# Overview
The rest of this README is strucutured in the following way: 
1. [**Project Files and Structure**](#project-files-and-structure)
2. [**Running the Dashboard inside a Miniconda3 environment**](#reproducing-the-dashboard)
    - [**Requirements & Installation**](#miniconda-installation)
    - [**Setup and Configurations**](#setup-and-configurations)
3. [**References**](#references)

# Project Files and Structure
Our Deep Learning approaches were implemented using PyTorch [Paszke et al., 2019] as main framework and SKLearn to evaluate the performance of our trained models. Furthermore, we used the following libraries to implement main features of our Visual Analytics dashboard:
- [Streamlit](https://streamlit.io
) for the web-based application
- [Plotly](https://plotly.com/python/) to generate interactive plots
- [torchvision](https://pytorch.org/vision/stable/index.html) to apply image transformations for training
- [PIL](https://pillow.readthedocs.io/en/stable/) for general image processing and creation
- [numpy](https://numpy.org/) for numerical data processing (e.g. prediction comparison, search etc.)
- [pandas](https://pandas.pydata.org/) for tabular data processing and analysis
- [torchcam](https://frgfm.github.io/torch-cam/latest/index.html) for Saliency Map generation

In the following, we give a brief introduction of the main files in the. For more details, please refer to the comments in the code.

- `home.py`: The main Python file of the dashboard. Contains the Streamlit application. This is the file that needs to be run via  the `streamlit` command to launch the dashboard. 

- `train.py`: The main file for the neural network training process. Contains the main functions to train the neural network and to evaluate its performance on the validation and test sets.

- `saliency_maps.py`: Contains the main functions to generate Saliency Maps for a given image and a given model. The functions are based on the `torchcam` library. The functions are called by the `home.py` file to generate the Saliency Maps to be shown in the dashboard.

- `project_config.py`: Contains the main configurations for the dashboard. Here, you can specify where to store the datasets that will be downloaded when running the dashboard. Additionally, this file contains mappings for the `CIFAR10`, `CIFAR100` and `CUB200` datasets to map the class labels to the corresponding class names. This is used to display the class names in the dashboard instead of the class labels, which are not very intuitive for humans.

- `models/custom_resnet18.py`: Contains the architecture implementation of the ResNet18 model. The model is based on the original ResNet18 architecture, but with a different number of output classes. The model is implemented as a subclass of the `torch.nn.Module` class. Also, the forward pass function is implemented in the `forward` method and is extended to allow the demonstration of the input variance fallacy. The model is used in the `train.py` file to train the neural network.

- `datasets/cifar.py`: Contains the implementation of the `CIFAR10` and `CIFAR100` datasets. The datasets are implemented as a subclass of the `torch.utils.data.Dataset` class. The dataset is used in the `train.py` file to train the neural network.

- `datasets/cub200.py`: Contains the implementation of the `CUB200` dataset. The dataset is implemented as a subclass of the `torch.utils.data.Dataset` class. The dataset is used in the `train.py` file to train the neural network.

- `datasets/get_datasets.py`: Contains the main functions to download the `CIFAR10`, `CIFAR100` and `CUB200` datasets. The datasets are downloaded to the paths that are specified in the `project_config.py` script. The functions are called by the `home.py` file to download the datasets when the dashboard is launched.

- `augmentations/randaugment.py`: Contains the main functions to apply image augmentations to the `CIFAR10`, `CIFAR100` and `CUB200` datasets. The functions are called by the `home.py` file to apply the augmentations when the datasets are downloaded.

- `utils/utils.py`: Contains the main utility functions to keep track of the training process and to update the dashboard. The functions are called by the `home.py` file to update the dashboard during the training process.

- `visualization/generate_plots.py`: Contains the main functions to generate the plots that are shown in the dashboard. The functions are called by the `home.py` file to generate a variety of plots ranging from confusion matrices to loss and accuracy plots.

# Running the Dashboard inside a Miniconda3 environment

To run the dashboard without training, you need to have `Python 3.8` installed on your machine. We recommend using a virtual environment to run the dashboard. We provide an `environment.yml` file that can be used to recreate the same virtual environment that was used for development using the `conda` command. The following instructions assume that you have `conda` installed on your machine. If you do not have `conda` installed, please follow the instructions on the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) to install it.

- Run the following conda command to create a new environment and install all required packages from the provided environment file:

```bash
conda env create -f environment.yml
```

- It is highly recommended to launch the dashboard with CUDA support to ensure that all training/inference processes for the PyTorch models are stable and executed on the GPU. To do so, you need to have a CUDA-enabled GPU and the corresponding CUDA drivers installed on your machine. If you do not have a CUDA-enabled GPU, you can still run the dashboard, but the training/inference processes will be executed on the CPU. To launch the dashboard with CUDA support, you need to activate the newly created environment and then run the dashboard with the `CUDA_VISIBLE_DEVICES` environment variable set to `0`:

- Activate the newly created environment:
```bash
conda activate saliency
```

- Launch the dashboard with CUDA support:

```bash
CUDA_VISIBLE_DEVICES=0 streamlit run home.py
```

- As the entire prototypical and iterative development process was conducted with CUDA support on the DeLab, we cannot guarantee that the dashboard will work without CUDA support. However, if you do not have a CUDA-enabled GPU, you can still run the dashboard without CUDA support and e.g. view the Saliency Maps in the "Fallacy Demonstration" tab. 

- Furthermore, even with CUDA support, starting a training may cause a "CUDA out of memory" error. This is often caused by a too high batch size. To avoid this error, you can reduce the batch size in the sidebar of the dashboard and restart the training.

- After the dashboard has been launched, you can open it in your browser by clicking on the link that is shown in the terminal. The dashboard will be available locally.

# References

[Paszke et al., 2019]: Paszke, A., Gross, S., Massa, F., Lerer, A., Bradbury, J., Chanan, G., Killeen, T., Lin, Z., Gimelshein, N., Antiga, L., Desmaison, A., Köpf, A., Yang, E., DeVito, Z., Raison, M., Tejani, A., Chilamkurthy, S., Steiner, B., Fang, L., Bai, J., & Chintala, S.. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. https://doi.org/10.48550/arXiv.1912.01703
