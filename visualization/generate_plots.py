import functools
import operator
import random
import time

from itertools import chain
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import streamlit as st
import torch
from project_config import class_names
from sklearn.metrics import (confusion_matrix, f1_score,
                             precision_recall_curve,
                             precision_recall_fscore_support, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize
from math import atan, pi

from sklearn.preprocessing import MinMaxScaler

def generate_initial_plots():
    return {
        'train_loss': line_plot(x_label='Epoch', y_label='Loss', title='Train Loss (Min-Max scaled)', showgrid=False),
        'train_acc': line_plot(x_label='Epoch', y_label='Acc', title='Train Acc (Min-Max scaled)', showgrid=False),
        'train_rec': line_plot(x_label='Epoch', y_label='Recall', title='Train Recall', showgrid=False),
        'train_prec': line_plot(x_label='Epoch', y_label='Precision', title='Train Precision', showgrid=False),
        'train_f1': line_plot(x_label='Epoch', y_label='F1-Score', title='Train F1-Score', showgrid=False),
        'val_loss': line_plot(x_label='Epoch', y_label='Loss', title='Val Loss', showgrid=False),
        'val_acc': line_plot(x_label='Epoch', y_label='Acc', title='Val Acc', showgrid=False),
        'val_rec': line_plot(x_label='Epoch', y_label='Recall', title='Val Recall', showgrid=False),
        'val_prec': line_plot(x_label='Epoch', y_label='Precision', title='Val Precision', showgrid=False),
        'val_f1': line_plot(x_label='Epoch', y_label='F1-Score', title='Val F1-Score', showgrid=False)
    }
def line_plot(x_label, y_label, title, showgrid):
    df = pd.DataFrame(columns=[x_label, y_label])
    #fig = go.FigureWidget(px.bar(df, x=df.columns[0], y=df.columns[1], title=title, width=300, height=300, marker={'color': '#ff95ed', 'pattern': {'shape': ''}}))
    fig = go.FigureWidget(px.bar(df, x=df.columns[0], y=df.columns[1], title=title, width=300, height=300))

    fig.update_xaxes(showgrid=showgrid)
    fig.update_yaxes(showgrid=showgrid)
    fig.update_layout(
        xaxis=dict(
            tickmode = 'linear',
        ),
        yaxis=dict(
            tickmode='array',
            tickvals=np.arange(0, 1.1, 0.1),
            range=[0,1]
        )
    )
    return fig

def update_plots(mode, session_state, legend=None, fig=None, x=None, y=None):
    if mode == 'training_monitoring':
        for elem in session_state.training_monitoring['training_plots'].keys():
            session_state.training_monitoring['training_plots'][elem].data[0].x = [i for i in range(session_state.training_monitoring['epochs_trained'])]
            if 'loss' in elem: 
                # normalize loss between 0 and 1 for fixed line chart axis ticks
                #print(session_state.training_monitoring['metrics_storage'].metrics[elem].all_values_avg)
                
                # Create an instance of the scaler
                scaler = MinMaxScaler(feature_range=(0,1))

                # Fit the scaler to your data
                data = session_state.training_monitoring['metrics_storage'].metrics[elem].all_values_avg
                data = np.array(data).reshape(1,-1) if len(data) == 1 else np.array(data).reshape(-1,1)
                scaler.fit(data)

                # Transform the data
                losses_scaled = scaler.transform(data)
                losses_scaled = list(chain.from_iterable(losses_scaled.tolist()))
                #losses_normalized = [((2 * atan(loss)) / pi) for loss in session_state.training_monitoring['metrics_storage'].metrics[elem].all_values_avg]
                session_state.training_monitoring['training_plots'][elem].data[0].y = losses_scaled
            else:
                session_state.training_monitoring['training_plots'][elem].data[0].y = session_state.training_monitoring['metrics_storage'].metrics[elem].all_values_avg


        return
    elif mode == 'prediction_analysis':
        fig.data[0].x = x
        fig.data[0].y = y
        fig.data[0].name = legend
        fig.data[0].showlegend = True
        fig.data[0].line.color = ["#"+''.join([random.choice('ABCDEF0123456789') for i in range(6)])][0]
        return fig

def get_precision_recall_curve(preds, labels, classes_to_compare, args):    
    df = pd.DataFrame(columns=['Recall', 'Precision'])
    all_figures = []
    for elem in list(preds.keys()):
        precision = dict()
        recall = dict()
        for cls in classes_to_compare:
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title='Precision vs. Recall Curve', markers=False, width=300, height=300)

            # get id from class name
            cls_idx = list(class_names[args.dataset_name].keys())[list(class_names[args.dataset_name].values()).index(cls)]
            labels_binarized = [1 if i == cls_idx else 0 for i in labels[elem]]
            predictions_for_cls = [i[cls_idx] for i in preds[elem][0]]
            precision[cls], recall[cls], _ = precision_recall_curve(labels_binarized, predictions_for_cls)
            
            fig = update_plots(mode='prediction_analysis', legend=f'model_{elem}_class_{cls}', session_state=None, fig=fig, x=list(recall[cls]), y=list(precision[cls]))                            
            all_figures.append(fig)

    fig_final = go.FigureWidget(data=functools.reduce(operator.add, [_.data for _ in all_figures]))
    fig_final.layout.title = {'text': 'Precision vs. Recall Curve'}
    fig_final.layout.xaxis = {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'Recall'}}
    fig_final.layout.yaxis = {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'Precision'}}
    return fig_final

def get_roc_curve(preds, labels, classes_to_compare, args):    
    df = pd.DataFrame(columns=['False Positive Rate (FPR)', 'True Positive Rate (TPR)'])
    all_figures = []
    for elem in list(preds.keys()):
        fpr = dict()
        tpr = dict()
        for cls in classes_to_compare:
            fig = px.line(df, x=df.columns[0], y=df.columns[1], title='ROC Curve', markers=False, width=300, height=300)

            # get id from class name
            cls_idx = list(class_names[args.dataset_name].keys())[list(class_names[args.dataset_name].values()).index(cls)]
            labels_binarized = [1 if i == cls_idx else 0 for i in labels[elem]]
            predictions_for_cls = [i[cls_idx] for i in preds[elem][0]]
            fpr[cls], tpr[cls], _ = roc_curve(labels_binarized, predictions_for_cls)
            
            fig = update_plots(mode='prediction_analysis', legend=f'model_{elem}_class_{cls}', session_state=None, fig=fig, x=list(fpr[cls]), y=list(tpr[cls]))                            
            all_figures.append(fig)

    fig_final = go.FigureWidget(data=functools.reduce(operator.add, [_.data for _ in all_figures]))
    fig_final.layout.title = {'text': 'ROC Curve'}
    fig_final.layout.xaxis = {'anchor': 'y', 'domain': [0.0, 1.0], 'title': {'text': 'False Positive Rate (FPR)'}}
    fig_final.layout.yaxis = {'anchor': 'x', 'domain': [0.0, 1.0], 'title': {'text': 'True Positive Rate (TPR)'}}
    return fig_final

def get_metrics(preds, labels, classes_to_compare, args):
    df = pd.DataFrame(columns=['Model Epoch', 'Class', 'Precision', 'Recall', 'F1-Score', 'AUC'])
    for elem in list(preds.keys()):
        # average = None makes sklearn calculate the metrics for each class
        precision, recall, f1, _ = precision_recall_fscore_support(y_true=labels[elem], y_pred=[i for i in preds[elem][1]], average=None, zero_division = 0)
        for cls in classes_to_compare:
            cls_idx = list(class_names[args.dataset_name].keys())[list(class_names[args.dataset_name].values()).index(cls)]
            
            labels_binarized = [1 if i == cls_idx else 0 for i in labels[elem]]
            predictions_for_cls = [i[cls_idx] for i in preds[elem][0]]

            # compute the AUC score for the class
            auc = roc_auc_score(y_true=labels_binarized, y_score=predictions_for_cls, average='macro')
            
            df = pd.concat([df, pd.DataFrame([{'Model Epoch': elem, 'Class': cls,'Precision': precision[cls_idx], 'Recall': recall[cls_idx], 'F1-Score': f1[cls_idx], 'AUC': auc}])], ignore_index=True)
            #df = df.append({'Model Epoch': elem, 'Class': cls,'Precision': precision[cls_idx], 'Recall': recall[cls_idx], 'F1-Score': f1[cls_idx], 'AUC': auc}, ignore_index=True)
            df = df.reset_index(drop=True)
    return df

def get_confusion_matrix(preds, labels, classes_to_compare, model_epoch, support_per_class, args):
    cls_idxs = []
    for cls in classes_to_compare:
        cls_idxs.append(list(class_names[args.dataset_name].keys())[list(class_names[args.dataset_name].values()).index(cls)])
    
    cls_idxs.sort()

    # This reduction using filtering allows us to construct a confusion matrix which only contains our desired classes
    # for which we want to perform a comparison
    idxs = [i for i in range(len(labels)) if labels[i] in cls_idxs and preds[i] in cls_idxs]
    preds = [preds[i] for i in idxs]
    labels = [labels[i] for i in idxs]
    conf_matrix = confusion_matrix(y_true=labels, y_pred=preds)

    
    # if the returned matrix is empty, this means, that the model never predicted a class that is in the set of desired classes or
    # the model did predict a desired class, but the corresponding label does not represent one of the desired classes
    # for this case, construct a matrix with zeros for all entries
    if len(conf_matrix) == 0:
         conf_matrix = np.zeros((len(cls_idxs), len(cls_idxs)), dtype=float)
    else:
        conf_matrix = np.array(conf_matrix, dtype=float)

        for i in range(len(conf_matrix)):
            for j in range(len(conf_matrix[i])):
                conf_matrix[i][j] = round(float(conf_matrix[i][j] / support_per_class) , 2)


    x = [class_names[args.dataset_name][i] for i in cls_idxs]
    y = list(reversed(x))
    # change each element of z to type string for annotations
    z_text = [[str(y) for y in x] for x in conf_matrix]
    fig = ff.create_annotated_heatmap(conf_matrix, x=x, y=y, annotation_text=z_text, colorscale='Jet', zmin=0, zmax=1)
    fig.update_layout(title_text=f'Confusion Matrix for model at epoch: {model_epoch}, support for all classes: {support_per_class}')
    fig.add_annotation(dict(font=dict(color="white",size=14), x=0.5, y=-0.15, showarrow=False, text="Scaled predicted values (predicted values (TPs) / support)", xref="paper", yref="paper"))
    fig.add_annotation(dict(font=dict(color="white",size=14), x=-0.35, y=0.5, showarrow=False, text="Real value", textangle=-90, xref="paper", yref="paper"))
    fig['data'][0]['showscale'] = True
    return fig
