import argparse
import os
import random
import shutil
import time
from io import BytesIO

import altair as alt
import IPython
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly as py
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch
import torchvision.transforms as T
from augmentations import get_transform
from datasets.get_datasets import (get_dataloader, get_dataset_setting,
                                   get_datasets)
from models.custom_resnet18 import BasicBlock, ResNet, weights_init
from PIL import Image
from project_config import class_names, explanation_methods
from smexplainers import explain
from streamlit_train import test, train
from torch.utils.data import DataLoader
from utils.utils import MetricsStorage, load_model
from visualization.generate_plots import (generate_initial_plots,
                                          get_confusion_matrix, get_metrics,
                                          get_precision_recall_curve,
                                          get_roc_curve, update_plots)

model_save_dir = os.path.join('/xaiva_dev', 'saved_models')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sm_viz', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--seed', default=1, type=int)
    
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    # ----------------------
    # UI SETUP
    # ----------------------
    st.set_page_config(layout="wide")
    st.title('Visual Analytics Dashboard for ResNet-18 Training')
    
    training_mon, pred_anal, local_expl, fallacy_demo = st.tabs(["Training Monitoring", "Prediction Analysis", "Local Explanations", "Fallacy Demonstration"])

    sidebar = st.sidebar
    sidebar.title("Training Hyperparameters")
    args.dataset_name = sidebar.selectbox('Select the Dataset', ('cub200', 'cifar10', 'cifar100', 'iNat21'))
    args.epochs = sidebar.slider(label='Total Epochs', min_value=0, max_value=200, value=10, step=1)
    args.epochs_per_checkpoint = sidebar.slider(label='Checkpoint after how many epochs ?', min_value=0, max_value=10, value=1, step=1)
    args.batch_size = sidebar.slider(label='Batch Size', min_value=16, max_value=256, value=64, step=8)
    args.lr = sidebar.slider(label='Initial Learning Rate', max_value=0.5, value=0.1, step=0.01, format='%f')
    args.momentum = sidebar.slider(label='Momentum', max_value=1.0, value=0.9, step=0.1)
    args.weight_decay = sidebar.slider(label='Weight Decay', min_value=0.1e-4, max_value=1e-2, value=1e-4, step=1e-5, format='%f')
    start_training = sidebar.button("Start / Continue Training")
    reset_training = sidebar.button("Reset Training")
    args = get_dataset_setting(args)

    # ----------------------
    # STATE VARIABLES
    # ----------------------
    if 'prediction_analysis' not in st.session_state:
        st.session_state['prediction_analysis'] = {
            'progress_bar': 0,
        }

    if 'training_monitoring' not in st.session_state:
        st.session_state['training_monitoring'] = {
            'epochs_trained': 0,
            'progress_bar': 0,
            'datasets': {},
            'dataloaders': {},
            'metrics_storage': MetricsStorage(),
            'training_plots': generate_initial_plots(),
            'model_paths': [],
            'predictions': {},
            'labels': {}
        }
    if 'local_explanations' not in st.session_state:
        st.session_state['local_explanations'] = {
            'ordering': 'descending',
            'options': None
        }
    
    if 'fallacy_demonstration' not in st.session_state:
        st.session_state['fallacy_demonstration'] = {
            'dataset': None
        }

    # ----------------------
    # CREATE FOLDERS FOR THE MODELS
    # ----------------------
    if st.session_state.training_monitoring['epochs_trained'] == 0:
        # TODO: Development/Debugging
        shutil.rmtree(model_save_dir) 
        os.mkdir(model_save_dir)
        os.mkdir(os.path.join(model_save_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_save_dir, 'best'))
        for i in range(args.epochs):
            os.mkdir(os.path.join(model_save_dir, 'checkpoints', str(i))) 
 
    with training_mon:
        bar = st.progress(st.session_state.training_monitoring['progress_bar'])
        step = 1 / args.epochs
        cols_mon = st.columns(spec=2, gap='medium')
        cols_mon[0].markdown(f"<div align=center> <span style=color:white;font-weight:600;font-size:calc(4vw/{len(cols_mon)});text-align:center;>Training Monitoring</span>", unsafe_allow_html=True)
        cols_mon[1].markdown(f"<div align=center> <span style=color:white;font-weight:600;font-size:calc(4vw/{len(cols_mon)});text-align:center;>Validation Monitoring</span>", unsafe_allow_html=True)
        
        if start_training:
            if st.session_state.training_monitoring['epochs_trained'] == args.epochs:
                st.error("Reached end of training, please reset the training", icon="🚨")
            else:
                if len(st.session_state.training_monitoring['datasets']) == 0:
                    with st.spinner(f"Loading {args.dataset_name} dataset..."):
                        
                        # get datasets transforms and datasets
                        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
                        train_dataset, val_dataset, test_dataset = get_datasets(dataset_name=args.dataset_name, train_transform=train_transform, test_transform=train_transform)
                        st.session_state.training_monitoring['datasets'] = {'train_dataset': train_dataset, 'val_dataset': val_dataset,'test_dataset': test_dataset}

                with st.spinner(f"Updating Dataloaders..."):
                    time.sleep(0.1)
                    # get dataloaders for the datasets
                    train_loader, val_loader, test_loader = get_dataloader(st.session_state.training_monitoring['datasets']['train_dataset'], st.session_state.training_monitoring['datasets']['val_dataset'], st.session_state.training_monitoring['datasets']['test_dataset'], args)
                    st.session_state.training_monitoring['dataloaders'] = {'train_loader': train_loader,'val_loader': val_loader,'test_loader': test_loader}
                    
                with st.spinner("Loading Model..."):
                    time.sleep(0.1)
                    # continue training with the state_dict of the last trained epoch
                    if st.session_state.training_monitoring['epochs_trained'] > 0:
                        model = load_model(st.session_state.training_monitoring['model_paths'][st.session_state.training_monitoring['epochs_trained']-1], args.device)
                    else:
                        # else load a new ResNet-18 model
                        model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(args.device)

                with st.spinner("Training..."):
                    for i in range(args.epochs_per_checkpoint):
                        st.session_state.training_monitoring['model_paths'].append(os.path.join(model_save_dir, 'checkpoints', str(st.session_state.training_monitoring['epochs_trained']), 'model.pt'))
                        st.session_state.training_monitoring['epochs_trained'] += 1

                        time.sleep(0.1)

                        # TRAIN FOR ONE EPOCH
                        train(
                            model=model, 
                            train_loader=st.session_state.training_monitoring['dataloaders']['train_loader'], 
                            val_loader=st.session_state.training_monitoring['dataloaders']['val_loader'], 
                            test_loader=st.session_state.training_monitoring['dataloaders']['test_loader'],
                            current_epoch=st.session_state.training_monitoring['epochs_trained']-1, 
                            metrics_storage=st.session_state.training_monitoring['metrics_storage'], 
                            args=args,
                            session_state=st.session_state
                        )

                        # UPDATE TRAINING PLOTS
                        update_plots(mode='training_monitoring', session_state=st.session_state)

                        # SAVE MODEL TO CHECKPOINT
                        save_path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(st.session_state.training_monitoring['epochs_trained']-1), 'model.pt')
                        torch.save(model, save_path)

                        # UPDATE PROGRESS BAR
                        st.session_state.training_monitoring['progress_bar'] += step
                        if st.session_state.training_monitoring['progress_bar'] >= 1.0 or st.session_state.training_monitoring['epochs_trained'] == args.epochs:
                            bar.progress(1.0)
                        else:
                            bar.progress(st.session_state.training_monitoring['progress_bar'])

                        if st.session_state.training_monitoring['epochs_trained'] == args.epochs:
                            st.success('Training completed!', icon="✅")
                            break

                    # PLOT THE UPDATED DATA
                    for elem in st.session_state.training_monitoring['training_plots'].keys():
                        if "train" in elem:
                            cols_mon[0].plotly_chart(st.session_state.training_monitoring['training_plots'][elem], use_container_width = False, theme='streamlit')
                        else:
                            cols_mon[1].plotly_chart(st.session_state.training_monitoring['training_plots'][elem], use_container_width = False, theme='streamlit')
                        time.sleep(0.1)
                    
                    if st.session_state.training_monitoring['epochs_trained'] < args.epochs:
                        st.info('Checkpoint reached! Please assess models and continue/end training')

        if reset_training:
            for key in st.session_state.keys():
                del st.session_state[key]
            time.sleep(1)
            st.experimental_rerun()

    with pred_anal:
        epochs_to_compare = st.multiselect(label='Select the epochs for which you want to compare the corresponding models', options=[i for i in range(st.session_state.training_monitoring['epochs_trained'])])
        classes_to_compare = st.multiselect(label='Select the image classes for which you want to compare the models', options=list(class_names[args.dataset_name].values()))
        
        if len(epochs_to_compare) != 0:
            compute_plots = st.button("Plot Charts", type='primary')
            st.session_state.prediction_analysis['progress_bar'] = 0
            bar = st.progress(st.session_state.prediction_analysis['progress_bar'])
            step = 1 / len(epochs_to_compare)
            if compute_plots:
                with st.spinner('Evaluating models on the test dataset...'):
                    # first, calulate the predictions of the model on the full test dataset
                    for i in epochs_to_compare:
                        
                        # retrieve the predictions and labels for the model at the epochs to compare
                        preds_probs, preds = st.session_state.training_monitoring['predictions'][i]
                        labels = st.session_state.training_monitoring['labels'][i] 

                        st.session_state.prediction_analysis['progress_bar'] += step
                        if st.session_state.prediction_analysis['progress_bar'] >= 1:
                            bar.progress(1.0)
                        else:
                            bar.progress(st.session_state.prediction_analysis['progress_bar'])

                        support_per_class = int(len(st.session_state.training_monitoring['dataloaders']['test_loader'].dataset) / len(list(class_names[args.dataset_name].keys())))
                        fig = get_confusion_matrix(preds, labels, classes_to_compare, i, support_per_class, args)
                        st.plotly_chart(fig, use_container_width = False)

                    # then, using the predictions, generate the plots
                    # precision_recall_curve
                    fig = get_precision_recall_curve(preds=st.session_state.training_monitoring['predictions'],labels=st.session_state.training_monitoring['labels'],classes_to_compare=classes_to_compare,args=args)
                    st.plotly_chart(fig, use_container_width = False)

                    # roc curve
                    fig = get_roc_curve(preds=st.session_state.training_monitoring['predictions'],labels=st.session_state.training_monitoring['labels'],classes_to_compare=classes_to_compare,args=args)
                    st.plotly_chart(fig, use_container_width = False)

                    # get metrics for each model as dataframe
                    df = get_metrics(preds=st.session_state.training_monitoring['predictions'], labels=st.session_state.training_monitoring['labels'], classes_to_compare=classes_to_compare, args=args)

                    # hide the index column
                    df = df.style.hide(axis="index")
                    st.write(df.to_html(), unsafe_allow_html=True)

    with local_expl:
        mapping = {}
        if args.dataset_name == 'cifar10':
            classes = list(class_names[args.dataset_name].keys())
        else:
            # Use the preselected classes from the bigger datasets (first, we only select the first 20 classes)
            classes = list(class_names[args.dataset_name].keys())[:20]

        if 'test_dataset' in list(st.session_state.training_monitoring['datasets'].keys()):   
            # for each class, select X (currently: the first 2) samples
            targets_np = np.array(st.session_state.training_monitoring['datasets']['test_dataset'].targets, dtype=np.int64)
            for cls in classes:
                idxs = np.where(targets_np == cls)[0].tolist()[:2]
                mapping[idxs[0]] = (cls, class_names[args.dataset_name][cls])
                mapping[idxs[1]] = (cls, class_names[args.dataset_name][cls])
            
            st.header("Saliency Maps for correct and wrong classifications:")
            st.subheader("The following visualizations use SmoothGradCam++ method on the most recent model")

            preds = st.session_state.training_monitoring['predictions'][st.session_state.training_monitoring['epochs_trained']-1][1]
            labels = st.session_state.training_monitoring['labels'][st.session_state.training_monitoring['epochs_trained']-1]
            path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(st.session_state.training_monitoring['epochs_trained']-1), 'model.pt')
                        
            st.markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(9vw/4);> Correct Classifications: </span>", unsafe_allow_html=True)

            # randomly chose at most 4 samples that were classified correctly
            cols = st.columns(spec=4, gap='medium')

            # this dictionary stores predictions for correct classifications. The predicted class is stored as key and the corresponding value corresponds to the index of the image.
            image_idxs_correct = {}
            for i in range(len(preds)):
                if preds[i] == labels[i] and preds[i] not in image_idxs_correct.keys():
                    image_idxs_correct[preds[i]] = i
                    if len(image_idxs_correct) == 4:
                        break
            
            images_to_visualize = []
            for elem in image_idxs_correct:
                image_pil, _ = st.session_state.training_monitoring['datasets']['test_dataset'].get_pil_image(image_idxs_correct[elem])
                image_tensor = st.session_state.training_monitoring['datasets']['test_dataset'][image_idxs_correct[elem]][0].to(args.device)
                explanations = explain(image_tensor=image_tensor.to(args.device), image_pil=image_pil, methods=['SmoothGradCAMpp'], model_paths=[path], device=args.device, epochs_to_compare=[0]) 
                
                # remove the digit picture used for the later manual evaluation of each epoch
                images_to_visualize += [explanations[1]]

            for i in range(len(images_to_visualize)):
                cols[i].markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/{len(cols)});> Class: {class_names[args.dataset_name][list(image_idxs_correct.keys())[i]]}</span>", unsafe_allow_html=True)
                cols[i].image(images_to_visualize[i], clamp=True, use_column_width=True, width=300)
            
            # randomly chose at most 4 samples that were classified incorrectly
            #st.subheader("Wrong classifications:")
            st.markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(9vw/{len(cols)});> Wrong Classifications: </span>", unsafe_allow_html=True)

            cols = st.columns(spec=4, gap='medium')

            image_idxs_incorrect = {}
            for i in range(len(preds)):
                if preds[i] != labels[i]:
                    image_idxs_incorrect[preds[i]] = i
                    if len(image_idxs_incorrect) == 4:
                        break            
            
            images_to_visualize = []
            for elem in image_idxs_incorrect:
                image_pil, _ = st.session_state.training_monitoring['datasets']['test_dataset'].get_pil_image(image_idxs_incorrect[elem])
                image_tensor = st.session_state.training_monitoring['datasets']['test_dataset'][image_idxs_incorrect[elem]][0].to(args.device)
                explanations = explain(image_tensor=image_tensor.to(args.device), image_pil=image_pil, methods=['SmoothGradCAMpp'], model_paths=[path], device=args.device, epochs_to_compare=[0]) 
                
                # remove the digit picture used for the later manual evaluation of each epoch
                images_to_visualize += [explanations[1]]
            
            for i in range(len(images_to_visualize)):
                cols[i].markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(4vw/{len(cols)});> Predicted Class: {class_names[args.dataset_name][list(image_idxs_incorrect.keys())[i]]}, True Class: {class_names[args.dataset_name][labels[image_idxs_incorrect[list(image_idxs_incorrect.keys())[i]]]]} </span>", unsafe_allow_html=True)
                cols[i].image(images_to_visualize[i], clamp=True, use_column_width=True, width=300)

            st.subheader("The following visualizations use SmoothGradCam++ method and show examples where the model always predicted the same class so far")
            
            cols = st.columns(spec=4, gap='medium')

            # take all predictions from all epochs and find the positions where the prediction was always the same, regardless of whether it was correct or not
            predictions_all_epochs = [st.session_state.training_monitoring['predictions'][i][1] for i in st.session_state.training_monitoring['predictions']]
            solution = np.where(np.all(np.array(predictions_all_epochs) == predictions_all_epochs[0], axis=0))[0]
            solution = np.random.choice(solution, replace=False, size=4).tolist()

            # then, take only the positions, where the prediction does not match the label. Those are wrong predictions that did not improve at all.
            for i, elem in enumerate(solution):
                if preds[elem] != labels[elem]:
                    image_pil, _ = st.session_state.training_monitoring['datasets']['test_dataset'].get_pil_image(elem)
                    image_tensor = st.session_state.training_monitoring['datasets']['test_dataset'][elem][0].to(args.device)
                    explanations = explain(image_tensor=image_tensor.to(args.device), image_pil=image_pil, methods=['SmoothGradCAMpp'], model_paths=[path], device=args.device, epochs_to_compare=[0])                
                    cols[i].markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(4vw/{len(cols)});> Predicted Class: {class_names[args.dataset_name][preds[elem]]}, True Class: {class_names[args.dataset_name][labels[elem]]} </span>", unsafe_allow_html=True)
                    cols[i].image(explanations[1], clamp=True, use_column_width=True, width=300)

            st.session_state.local_explanations['options'] = [f'{mapping[i][1]}_{i}' for i in list(mapping.keys())]
            selected_image = st.selectbox(label='Select an image from the pre-selected images', options=st.session_state.local_explanations['options'])
            
            # load the selected image as preview
            idx = int(selected_image.split('_')[-1])
            image_pil, _ = st.session_state.training_monitoring['datasets']['test_dataset'].get_pil_image(idx)
            image_tensor = st.session_state.training_monitoring['datasets']['test_dataset'][idx][0].to(args.device)

            st.header("Preview:")
            st.image(image_pil, clamp=True, width=300)

            epochs_to_compare = st.multiselect(label='Select the epochs for which you want to compute saliency maps', options=[i for i in range(st.session_state.training_monitoring['epochs_trained'])])
            methods_to_compare = st.multiselect(label='Select the saliency map methods', options=explanation_methods)
            compute_saliency_maps = st.button('Generate visual explanations', type='primary')
            
            if compute_saliency_maps:
                explanations = explain(
                    image_tensor=image_tensor.to(args.device), 
                    image_pil=image_pil, 
                    methods=methods_to_compare, 
                    model_paths=[st.session_state.training_monitoring['model_paths'][i] for i in epochs_to_compare],
                    device=args.device,
                    epochs_to_compare=epochs_to_compare
                ) 
                cols = st.columns(spec=len(methods_to_compare)+1, gap='medium')

                # write the header for the model epoch column
                cols[0].markdown(f"<div align=center> <span style=color:white;font-weight:300;font-size:calc(6vw/{len(cols)});text-align:center;>Epoch</span>", unsafe_allow_html=True)

                # write the headers for the different methods
                for i in range(len(cols)-1):
                    cols[i+1].markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/{len(cols)});>{methods_to_compare[i]}</span>", unsafe_allow_html=True)

                # for each model visualize the selected saliency map methods
                image_count = 0
                for epoch in epochs_to_compare:
                    for i in range(len(cols)):
                        cols[i].image(explanations[image_count], clamp=True, use_column_width=True, width=300)
                        image_count+=1

                st.markdown(f"<div align=center> <span style=color:white;text-align:center;>f source code for saliency maps: <a href='https://github.com/frgfm/torch-cam'>https://github.com/frgfm/torch-cam<a></span>", unsafe_allow_html=True)

        else:
            st.info('Please train the model for at least one epoch before generating local explanations!')
   
    with fallacy_demo:
    
        dataset_name = 'cub200'
        mapping = {}

        # check if we got our target dataset and if not, load it 
        if st.session_state['fallacy_demonstration']['dataset'] is None:
            image_size = 224
            num_classes = 200
            transform = 'imagenet'
            _, test_transform = get_transform(transform, image_size=image_size)
            _, _, test_dataset = get_datasets(dataset_name=dataset_name, train_transform=None, test_transform=test_transform)
            st.session_state['fallacy_demonstration']['dataset'] = test_dataset
        
        st.subheader("SmoothGradCam++ is an improved version of Class Activation Mapping (CAM) because it addresses some of the limitations of CAM. \
            This demonstration aims to show these limitations and how SmoothGradCam++ improves them")

        st.markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/5);>CAM uses the network's activations and weights to highlight the regions in an image that are most important for a given classification decision. Due it's simplicity, the output map can be calculated quicker than SmoothGradCam++ and for certain types of images, the output maps of both methods are fairly similar: </span>", unsafe_allow_html=True)
        classes = list(class_names[dataset_name].keys())
        targets_np = np.array(st.session_state['fallacy_demonstration']['dataset'].targets, dtype=np.int64)

        for cls in classes:
            idxs = np.where(targets_np == cls)[0].tolist()

            for idx in idxs:
                mapping[idx] = (cls, class_names[dataset_name][cls])

        #options = [f'{mapping[i][1]}_{i}' for i in list(mapping.keys())]
        #selected_image = st.selectbox(label='Select an image from the pre-selected images', options=options)
        
        images_to_visualize = ['spotted_catbbird_310', 'Ruby_throated_Hummingbird_1324', 'laysan_albatross_2', 'laysan_albatross_5']
        model = ResNet(BasicBlock, [2,2,2,2], 200).to(args.device)
        descriptions = ['Original Image', 'CAM', 'SmoothGradCAMpp']
        cols = st.columns(spec=3, gap='medium')
        for i in range(len(cols)):
            cols[i].markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/{len(cols)});>{descriptions[i]}</span>", unsafe_allow_html=True)
        
        for step, image in enumerate(images_to_visualize):
            idx = int(image.split('_')[-1])
            image_pil, _ = st.session_state['fallacy_demonstration']['dataset'].get_pil_image(idx)
            image_tensor = st.session_state['fallacy_demonstration']['dataset'][idx][0].to(args.device)
            explanations = explain(image_tensor=image_tensor.to(args.device), image_pil=image_pil, methods=['CAM', 'SmoothGradCAMpp'], model_paths=None, device=args.device, epochs_to_compare=[0], single_model=model)   
            explanations[0] = image_pil.resize((300,300))

            for i in range(len(cols)):
                cols[i].image(explanations[i], clamp=True, use_column_width=True, width=300)

            if step == 1:
                st.markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/5);>However, one main limitation of CAM is that it tends to generate noisy explanations. This can be seen especially for image examples with many diverse background elements. Also, CAM often fails to recognize the overall pattern of the main image element, because it highlights activations in a point-wise fashion. By adding small perturbations to the image, SmoothGradCam++ calculates multiple gradient-based visualizations for a single input and averages them to yield the final output. This has shown to be more robust for images that contain noise and this method also performs better in recognizing overall patterns. For instance: </span>", unsafe_allow_html=True)
                cols = st.columns(spec=3, gap='medium')


        st.markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/5);>Another experiment considers the robustness of the saliency map method. This was shown by Adebayo et al. (2020) <a href='https://www.semanticscholar.org/paper/Sanity-Checks-for-Saliency-Maps-Adebayo-Gilmer/8dc8f3e0127adc6985d4695e9b69d04717b2fde8'>Adebayo et al. (2020)<a> for instance by checking whether the explanation method is insensitive to small variance added to the input and removed directly after the first hidden layer of the network, causing no numerical difference in the actual computation of the forward pass, but a potential difference in the computation of the explanation. Use the following slider to set the scaling value for the noise and observe how CAM produces slightly different explanations while SmoothGradCam++ shows no sensitivity to this variance. </span>", unsafe_allow_html=True)
        slider = st.slider(label='Choose the scaling factor for the input variance', min_value=0.0, max_value=5.0, value=0.0, step=0.1)
        cols = st.columns(spec=3, gap='medium')
        for i in range(len(cols)):
            cols[i].markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/{len(cols)});>{descriptions[i]}</span>", unsafe_allow_html=True)
        
        for step, image in enumerate(images_to_visualize):
            idx = int(image.split('_')[-1])
            image_pil, _ = st.session_state['fallacy_demonstration']['dataset'].get_pil_image(idx)
            image_tensor = st.session_state['fallacy_demonstration']['dataset'][idx][0].to(args.device)
            explanations_with_input_variance = explain(image_tensor=image_tensor.to(args.device), image_pil=image_pil, methods=['CAM', 'SmoothGradCAMpp'], model_paths=None, device=args.device, epochs_to_compare=[0], single_model=model, use_input_variance=True, variance=slider)   
            explanations_with_input_variance[0] = image_pil.resize((300,300))
            for i in range(len(cols)):
                cols[i].image(explanations_with_input_variance[i], clamp=True, use_column_width=True, width=300)

        st.markdown(f"<div align=center> <span style=color:white;font-weight:300;text-align:center;font-size:calc(6vw/5);>In conclusion, even though CAM can be used for fast explanation generation, it should be used with care, as perturbations and the image structure can weaken it's visualization quality. Instead, smoothing-based methods such as SmoothGradCam++ can be used to provide a more comprehensive understanding of the model's decision-making process which allows for a more thorough analysis of the model's behavior.</span>", unsafe_allow_html=True)

