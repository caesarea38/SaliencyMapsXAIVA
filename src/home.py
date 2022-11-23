import argparse
import os
import shutil
import time
from io import BytesIO

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import torch
from augmentations import get_transform
from datasets.get_datasets import get_dataset_setting, get_datasets, get_dataloader
from models.custom_resnet18 import BasicBlock, ResNet
from PIL import Image
from torch.utils.data import DataLoader
from streamlit_train import train
import plotly.express as px
from visualization.generate_plots import line_chart, update_plots, plot_charts
import plotly as py
import plotly.graph_objects as go

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='sm_viz', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--num_workers', default=16, type=int)
    parser.add_argument('--model_name', type=str, default='resnet18')
    parser.add_argument('--grad_from_block', type=int, default=11)
    parser.add_argument('--transform', type=str, default='pytorch-cifar', choices=['imagenet', 'pytorch-cifar'])
    parser.add_argument('--seed', default=1, type=int)
    
    args = parser.parse_args()
    args.device = torch.device('cuda:0')

    # ----------------------
    # UI SETUP
    # ----------------------
    st.set_page_config(layout="wide")
    st.title('Visual Analytics Dashboard for ResNet-18 Training')
    
    training_conf, pred_anal, local_expl, global_expl = st.tabs(["Training Monitoring", "Prediction Analysis", "Local Explanations", "Global Explanations"])

    sidebar = st.sidebar
    sidebar.title("Training Hyperparameters")
    args.dataset_name = sidebar.selectbox('Select the Dataset', ('cub200', 'cifar10', 'cifar100', 'iNat21'))
    args.epochs = sidebar.slider(label='Total Epochs', min_value=0, max_value=200, value=10, step=1)
    args.epochs_per_checkpoint = sidebar.slider(label='Checkpoint after how many epochs ?', min_value=0, max_value=10, value=2, step=1)
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
    if 'epochs_trained' not in st.session_state:
        st.session_state['epochs_trained'] = 0

    if 'progress_bar' not in st.session_state:
        st.session_state['progress_bar'] = 0 

    if 'datasets' not in st.session_state:
        st.session_state['datasets'] = {}
    
    if 'dataloaders' not in st.session_state:
        st.session_state['dataloaders'] = {}

    if 'metrics' not in st.session_state:
        st.session_state['metrics'] = {'train_loss': [],'train_acc': [],'val_loss': [],'val_acc': []}

    if 'training_plots' not in st.session_state:
        df_loss, df_acc = (pd.DataFrame(dict(epoch = [],loss = [])), pd.DataFrame(dict(epoch = [], acc = [])))
        st.session_state['training_plots'] = {'train_loss': None, 'train_acc': None, 'val_loss': None,'val_acc': None}

        # training loss and acc charts
        st.session_state['training_plots']['train_loss'] = go.FigureWidget(px.line(df_loss, x=df_loss.columns[0], y=df_loss.columns[1], title = "Train Loss", markers=True).update_xaxes(showgrid=False).update_yaxes(showgrid=False))
        st.session_state['training_plots']['train_acc'] = go.FigureWidget(px.line(df_acc, x=df_acc.columns[0], y=df_acc.columns[1], title = "Train Acc", markers=True).update_xaxes(showgrid=False).update_yaxes(showgrid=False))

        # validation loss and acc charts
        st.session_state['training_plots']['val_loss'] = go.FigureWidget(px.line(df_loss, x=df_loss.columns[0], y=df_loss.columns[1], title = "Val Loss", markers=True).update_xaxes(showgrid=False).update_yaxes(showgrid=False))
        st.session_state['training_plots']['val_acc'] = go.FigureWidget(px.line(df_acc, x=df_acc.columns[0], y=df_acc.columns[1], title = "Val Acc", markers=True).update_xaxes(showgrid=False).update_yaxes(showgrid=False))

    # ----------------------
    # CREATE FOLDERS FOR THE MODELS
    # ----------------------
    if st.session_state.epochs_trained == 0:
        # TODO: Development/Debugging
        model_save_dir = os.path.join('/xaiva_dev', 'saved_models')
        shutil.rmtree(model_save_dir) 
        os.mkdir(model_save_dir)
        os.mkdir(os.path.join(model_save_dir, 'checkpoints'))
        os.mkdir(os.path.join(model_save_dir, 'best'))
        for i in range(1,args.epochs+1):
            os.mkdir(os.path.join(model_save_dir, 'checkpoints', str(i))) 
 
    with training_conf:
        bar = st.progress(st.session_state.progress_bar)
        step = 1 / args.epochs
        if start_training:

            if st.session_state.epochs_trained == args.epochs:
                st.error("Reached end of training, please reset the training", icon="🚨")
                plot_charts(mode='training_monitoring', session_state=st.session_state, figures=None)
            else:
                if len(st.session_state.datasets) == 0:
                    with st.spinner(f"Loading {args.dataset_name} dataset..."):
                        
                        # GET DATASET TRANSFORMS AND DATASETS
                        train_transform, test_transform = get_transform(args.transform, image_size=args.image_size, args=args)
                        train_dataset, val_dataset, test_dataset = get_datasets(dataset_name=args.dataset_name, train_transform=train_transform, test_transform=train_transform, args=args)
                        st.session_state.datasets = {'train_dataset': train_dataset, 'val_dataset': val_dataset,'test_dataset': test_dataset}

                        # GET DATALOADER
                        train_loader, val_loader, test_loader = get_dataloader(train_dataset, val_dataset, test_dataset, args)
                        st.session_state.dataloaders = {'train_loader': train_loader,'val_loader': val_loader,'test_loader': test_loader}
                        
                        # load a new ResNet-18 model
                        model = ResNet(BasicBlock, [2,2,2,2], args.num_classes).to(args.device)

                with st.spinner("Loading Model..."):
                    # continue training with the state_dict of the last trained epoch
                    if st.session_state.epochs_trained > 0:
                        path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(st.session_state.epochs_trained), 'model.pt')
                        model = torch.load(path, map_location=args.device)

                with st.spinner("Training..."):
                    
                    for i in range(args.epochs_per_checkpoint):
                        st.session_state.epochs_trained += 1
                        time.sleep(0.1)

                        # TRAIN FOR ONE EPOCH
                        metrics = train(model=model, train_loader=st.session_state.dataloaders['train_loader'], val_loader=st.session_state.dataloaders['val_loader'], current_epoch=st.session_state.epochs_trained, args=args)
                        
                        # ADD NEW METRICS
                        for elem in metrics.keys():
                            st.session_state.metrics[elem].append(metrics[elem])

                        # UPDATE PLOTS
                        update_plots(mode='training_monitoring', session_state=st.session_state)

                        # SAVE MODEL TO CHECKPOINT
                        save_path = os.path.join('/xaiva_dev/saved_models/checkpoints', str(st.session_state.epochs_trained), 'model.pt')
                        torch.save(model, save_path)

                        # UPDATE PROGRESS BAR
                        st.session_state.progress_bar += step
                        if st.session_state.progress_bar >= 1.0 or st.session_state.epochs_trained == args.epochs:
                            bar.progress(1.0)
                        else:
                            bar.progress(st.session_state.progress_bar)

                        if st.session_state.epochs_trained == args.epochs:
                            st.success('Training completed!', icon="✅")
                            break

                    # PLOT THE UPDATED DATA
                    plot_charts(mode='training_monitoring', session_state=st.session_state, figures=None)

                    if st.session_state.epochs_trained < args.epochs:
                        st.info('Checkpoint reached! Please assess models and continue/end training')

        if reset_training:
            for key in st.session_state.keys():
                del st.session_state[key]
            time.sleep(1)
            st.experimental_rerun()

    with pred_anal:
        st.header("A catt")
        sidebar = None
        
    

