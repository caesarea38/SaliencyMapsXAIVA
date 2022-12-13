import argparse
import os
import shutil
import time
from io import BytesIO

import altair as alt
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
                                          get_roc_curve, line_plot,
                                          plot_charts, update_plots)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sm_viz', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--seed', default=1, type=int)
    
    args = parser.parse_args()
    args.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # ----------------------
    # UI SETUP
    # ----------------------
    st.set_page_config(layout="wide")
    st.title('Visual Analytics Dashboard for ResNet-18 Training')
    
    training_mon, pred_anal, local_expl, global_expl = st.tabs(["Training Monitoring", "Prediction Analysis", "Local Explanations", "Global Explanations"])

    sidebar = st.sidebar
    sidebar.title("Training Hyperparameters")
    args.dataset_name = sidebar.selectbox('Select the Dataset', ('cub200', 'cifar10', 'cifar100', 'iNat21'))
    args.epochs = sidebar.slider(label='Total Epochs', min_value=0, max_value=200, value=10, step=1)
    args.epochs_per_checkpoint = sidebar.slider(label='Checkpoint after how many epochs ?', min_value=0, max_value=10, value=1, step=1)
    args.batch_size = sidebar.slider(label='Batch Size', min_value=16, max_value=128, value=64, step=8)
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
            'predictions': {},
            'labels': None
        }

    if 'training_monitoring' not in st.session_state:
        st.session_state['training_monitoring'] = {
            'epochs_trained': 0,
            'progress_bar': 0,
            'datasets': {},
            'dataloaders': {},
            'metrics_storage': MetricsStorage(),
            'training_plots': generate_initial_plots()
        }
    # ----------------------
    # CREATE FOLDERS FOR THE MODELS
    # ----------------------
    if st.session_state.training_monitoring['epochs_trained'] == 0:
        # TODO: Development/Debugging
        model_save_dir = os.path.join('/xaiva_dev', 'saved_models')
        shutil.rmtree(model_save_dir) 
        os.mkdir(model_save_dir)
        os.mkdir(os.path.join(model_save_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_save_dir, 'best'))
        for i in range(1,args.epochs+1):
            os.mkdir(os.path.join(model_save_dir, 'checkpoints', str(i))) 
 
    with training_mon:
        bar = st.progress(st.session_state.training_monitoring['progress_bar'])
        step = 1 / args.epochs
        if start_training:
            if st.session_state.training_monitoring['epochs_trained'] == args.epochs:
                st.error("Reached end of training, please reset the training", icon="🚨")
            else:
                if len(st.session_state.training_monitoring['datasets']) == 0:
                    with st.spinner(f"Loading {args.dataset_name} dataset..."):
                        
                        # get datasets transforms and datasets
                        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
                        train_dataset, val_dataset, test_dataset = get_datasets(dataset_name=args.dataset_name, train_transform=train_transform, test_transform=train_transform, args=args)
                        st.session_state.training_monitoring['datasets'] = {'train_dataset': train_dataset, 'val_dataset': val_dataset,'test_dataset': test_dataset}

                        # get dataloaders for the datasets
                        train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, args)
                        st.session_state.training_monitoring['dataloaders'] = {'train_loader': train_loader,'val_loader': val_loader,'test_loader': test_loader}
                        
                        # load a new ResNet-18 model
                        model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(args.device)

                with st.spinner("Loading Model..."):
                    # continue training with the state_dict of the last trained epoch
                    if st.session_state.training_monitoring['epochs_trained'] > 0:
                        path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(st.session_state.training_monitoring['epochs_trained']), 'model.pt')
                        model = load_model(path, args.device)

                with st.spinner("Training..."):
                    for i in range(args.epochs_per_checkpoint):
                        st.session_state.training_monitoring['epochs_trained'] += 1
                        time.sleep(0.1)

                        # TRAIN FOR ONE EPOCH
                        train(model=model, train_loader=st.session_state.training_monitoring['dataloaders']['train_loader'], val_loader=st.session_state.training_monitoring['dataloaders']['val_loader'], current_epoch=st.session_state.training_monitoring['epochs_trained'], metrics_storage=st.session_state.training_monitoring['metrics_storage'], args=args)

                        # UPDATE PLOTS
                        update_plots(mode='training_monitoring', session_state=st.session_state)

                        # SAVE MODEL TO CHECKPOINT
                        save_path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(st.session_state.training_monitoring['epochs_trained']), 'model.pt')
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
                    plot_charts(mode='training_monitoring', session_state=st.session_state, figures=None)
                    if st.session_state.training_monitoring['epochs_trained'] < args.epochs:
                        st.info('Checkpoint reached! Please assess models and continue/end training')

        if reset_training:
            for key in st.session_state.keys():
                del st.session_state[key]
            time.sleep(1)
            st.experimental_rerun()

    with pred_anal:
        epochs_to_compare = st.multiselect(label='Select the epochs for which you want to compare the corresponding models', options=[i for i in range(1, st.session_state.training_monitoring['epochs_trained']+1)])
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
                        
                        # calculate the predictions and store them in the session state
                        path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(i), 'model.pt')
                        pred_probs, labels, preds = test(model = load_model(path, args.device), test_loader=st.session_state.training_monitoring['dataloaders']['test_loader'], args=args)

                        if st.session_state.prediction_analysis['labels'] is None:
                            st.session_state.prediction_analysis['labels'] = labels
                        st.session_state.prediction_analysis['predictions'][i] = (pred_probs, preds)

                        st.session_state.prediction_analysis['progress_bar'] += step
                        if st.session_state.prediction_analysis['progress_bar'] >= 1:
                            bar.progress(1.0)
                        else:
                            bar.progress(st.session_state.prediction_analysis['progress_bar'])
                    
                        fig = get_confusion_matrix(preds, labels, classes_to_compare, i, args)
                        st.plotly_chart(fig, use_container_width = False)

                    # then, using the predictions, generate the plots
                    # precision_recall_curve
                    fig = get_precision_recall_curve(preds=st.session_state.prediction_analysis['predictions'],labels=st.session_state.prediction_analysis['labels'],classes_to_compare=classes_to_compare,args=args)
                    st.plotly_chart(fig, use_container_width = False)

                    # roc curve
                    fig = get_roc_curve(preds=st.session_state.prediction_analysis['predictions'],labels=st.session_state.prediction_analysis['labels'],classes_to_compare=classes_to_compare,args=args)
                    st.plotly_chart(fig, use_container_width = False)

                    # get metrics for each model as dataframe
                    df = get_metrics(preds=st.session_state.prediction_analysis['predictions'], labels=st.session_state.prediction_analysis['labels'], classes_to_compare=classes_to_compare, args=args)
                    st.dataframe(df, use_container_width=True)
    
   #with local_expl:
        #all_images = []
        #epochs_to_compare = st.multiselect(label='Select the epochs for which you want to compute saliency maps', options=[i for i in range(1, st.session_state.training_monitoring['epochs_trained']+1)])
        #if st.session_state.training_monitoring['epochs_trained'] > 0:
        #    dataset = st.session_state.training_monitoring['datasets']['test_dataset']
        #    for i in range(10):
        #        all_images.append(dataset[i][0].numpy().swapaxes(0,1).swapaxes(1,2))
        #    st.header("Pre-selected images from the test dataset")
        #    st.image(all_images, clamp=True, channels='RGB', width=150)
        #    selected_image = st.selectbox(label='Select the image for which you want to generate visual explanations. The numbers represent the order of the images starting from the top left', options=[i for i in range(1,11)])
        #    if selected_image is not None:
        #        smaps_to_generate = st.multiselect(label='Select the saliency map methods to use', options=['CAM', 'Grad-CAM',])
    
    with global_expl:
        from datasets.cifar import CUSTOMCIFAR10
        from datasets.cub import CUSTOMCUB2011
        from project_config import cifar10_root, cub_root
        from torchvision.transforms import transforms
        from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
        from torchcam.methods import CAM, GradCAM, ScoreCAM, SSCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM
        from torchcam.utils import overlay_mask
        import torchvision.transforms as T


        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
        num_classes = 200
        torch.cuda.manual_seed(0)
        test_transform = transforms.Compose([transforms.Resize((224, 224)),transforms.ToTensor(),transforms.Normalize(mean=mean, std=std)])
        test_dataset = CUSTOMCUB2011(root=cub_root, transform=test_transform, train=False, download=True)
        image_numpy = test_dataset[25][0].numpy().swapaxes(0,1).swapaxes(1,2)
        image_tensor = test_dataset[25][0].to(args.device)
        cls_idx = test_dataset[25][1]
        st.image(image_numpy, clamp=True, width=300)
        model = ResNet(BasicBlock, [2,2,2,2], num_classes).to(args.device).eval()
        #model.apply(weights_init)

        img = image_tensor.cpu().numpy().swapaxes(0,1).swapaxes(1,2)

        cam_extractor = CAM(model, 'layer4', 'linear')
        with torch.no_grad():
            _, out = model(image_tensor.unsqueeze(0))
        cls_idx = out.squeeze(0).argmax().item()
        act_maps = cam_extractor(class_idx=cls_idx)
        activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
        
        #result = overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map_fused.squeeze(0), mode="F"), alpha=0.5)
        
        st.image(to_pil_image(image_numpy, mode='RGB'), clamp=True, width=300)
        # st.image(result, clamp=True, width=300)
        """
        epochs_to_compare = st.multiselect(label='Select the epochs for which you want to compute saliency maps', options=[i for i in range(10)])
        methods_to_compare = st.multiselect(label='Select the saliency map methods', options=explanation_methods)
        compute_saliency_maps = st.button('Generate visual explanations', type='secondary')
        #if compute_saliency_maps:
        methods_to_compare = ['CAM']
        epochs_to_compare = [0]
        explanations = explain(image_tensor=image_tensor, image_numpy=image_numpy, methods=methods_to_compare, models=[model]) 
        cols = st.columns(spec=len(methods_to_compare), gap='medium')

        # write the headers for the different methods
        for i in range(len(cols)):
            cols[i].header(methods_to_compare[i])

        # for each model visualize the selected saliency map methods
        method_per_model = 0
        for epoch in epochs_to_compare:
            for _ in range(len(cols)):

                #cols[method_per_model].pyplot(explanations[method_per_model])
                cols[method_per_model].image(explanations[method_per_model], clamp=True, use_column_width=False, width=300, caption=f'Model Epoch: {epoch}')
                method_per_model+=1
        """