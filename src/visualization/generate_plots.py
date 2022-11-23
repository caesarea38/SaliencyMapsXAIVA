import plotly.express as px
import torch
import streamlit as st

def line_chart(df, title):
    fig = px.line(df, x=df.columns[0], y=df.columns[1], title=title, markers=True)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    return fig

def update_plots(mode, session_state, fig_name=None, x=None, y=None):
    if mode == 'training_monitoring':
        for elem in session_state['training_plots'].keys():
            session_state['training_plots'][elem].data[0].x = [i for i in range(session_state.epochs_trained)]
            session_state['training_plots'][elem].data[0].y = session_state.metrics[elem]

def plot_charts(mode, session_state, figures=None):
    if mode == 'training_monitoring':
        for elem in session_state['training_plots'].keys():
            st.plotly_chart(session_state['training_plots'][elem], use_container_width = False, theme='streamlit')
