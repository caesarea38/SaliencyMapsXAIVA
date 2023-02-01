import random
from copy import deepcopy
import time
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
from models.custom_resnet18 import BasicBlock, ResNet, weights_init
from PIL import Image, ImageDraw, ImageFont
from torchcam.methods import (CAM, SSCAM, GradCAM, GradCAMpp, LayerCAM,
                              ScoreCAM, SmoothGradCAMpp, XGradCAM)
from torchcam.utils import overlay_mask
from torchvision.transforms import ToPILImage
from torchvision.transforms.functional import (normalize, resize, to_pil_image,
                                               to_tensor)
from utils.utils import load_model


# draws a number (the epoch) as pillow image and returns it
def get_image_from_digit(epoch):
    img = Image.new('RGB', (300,300), (14,17,23))
    draw = ImageDraw.Draw(img)
    font = ImageFont.truetype("/fonts/OpenSans-Regular.ttf", 150)
    draw.text((105, 30), str(epoch), font=font, fill="#FFF")
    return img

def explain(image_tensor, image_pil, methods, model_paths, device, epochs_to_compare, single_model=None):
    figs = []
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    np.random.seed(1)
    random.seed(1)

    if single_model is not None:
        model = single_model
        figs.append(get_image_from_digit(epochs_to_compare[0]))
        for method in methods:
            if method == 'CAM':
                cam_extractor = CAM(model, 'layer4', 'linear')
                with torch.no_grad():
                    out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                result = overlay_mask(image_pil, to_pil_image(activation_map_fused.squeeze(0), mode="F"), alpha=0.5)
                figs.append(result)

            elif method in ['GradCAM', 'GradCAMpp']:
                cam_extractor = GradCAM(model, 'layer4') if method == 'GradCAM' else GradCAMpp(model, 'layer4')
                out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                cam_extractor.remove_hooks()

            elif method in ['ScoreCAM', 'SmoothScoreCAM']:
                cam_extractor = ScoreCAM(model, 'layer4') if method == 'ScoreCAM' else SSCAM(model, 'layer4', num_samples=4)
                with torch.no_grad(): 
                    out = model(image_tensor.unsqueeze(0).requires_grad_())
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
        
            elif method == 'SmoothGradCAMpp':
                cam_extractor = SmoothGradCAMpp(model=model, target_layer='layer4')
                out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                cam_extractor.remove_hooks()
            
            elif method == 'XGradCAM':
                cam_extractor = XGradCAM(model, 'layer4')
                out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                cam_extractor.remove_hooks()

            elif method == 'LayerCAM':
                cam_extractor = LayerCAM(model, 'layer4')
                out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                cam_extractor.remove_hooks()
    else:
        for i, model_path in enumerate(model_paths):
            model = load_model(model_path, device=device)
            figs.append(get_image_from_digit(epochs_to_compare[i]))
            for method in methods:
                if method == 'CAM':
                    cam_extractor = CAM(model, 'layer4', 'linear')
                    with torch.no_grad():
                        out = model(image_tensor.unsqueeze(0))
                    cls_idx = out.squeeze(0).argmax().item()
                    act_maps = cam_extractor(class_idx=cls_idx)
                    activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                    result = overlay_mask(image_pil, to_pil_image(activation_map_fused.squeeze(0), mode="F"), alpha=0.5)
                    figs.append(result)

                elif method in ['GradCAM', 'GradCAMpp']:
                    cam_extractor = GradCAM(model, 'layer4') if method == 'GradCAM' else GradCAMpp(model, 'layer4')
                    out = model(image_tensor.unsqueeze(0))
                    cls_idx = out.squeeze(0).argmax().item()
                    act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                    activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                    figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                    cam_extractor.remove_hooks()

                elif method in ['ScoreCAM', 'SmoothScoreCAM']:
                    cam_extractor = ScoreCAM(model, 'layer4') if method == 'ScoreCAM' else SSCAM(model, 'layer4', num_samples=4)
                    with torch.no_grad(): 
                        out = model(image_tensor.unsqueeze(0).requires_grad_())
                    cls_idx = out.squeeze(0).argmax().item()
                    act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                    activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                    figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
            
                elif method == 'SmoothGradCAMpp':
                    cam_extractor = SmoothGradCAMpp(model=model, target_layer='layer4')
                    out = model(image_tensor.unsqueeze(0))
                    cls_idx = out.squeeze(0).argmax().item()
                    act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                    activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                    figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                    cam_extractor.remove_hooks()
                
                elif method == 'XGradCAM':
                    cam_extractor = XGradCAM(model, 'layer4')
                    out = model(image_tensor.unsqueeze(0))
                    cls_idx = out.squeeze(0).argmax().item()
                    act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                    activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                    figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                    cam_extractor.remove_hooks()

                elif method == 'LayerCAM':
                    cam_extractor = LayerCAM(model, 'layer4')
                    out = model(image_tensor.unsqueeze(0))
                    cls_idx = out.squeeze(0).argmax().item()
                    act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                    activation_map_fused = cam_extractor.fuse_cams(act_maps)
                    figs.append(overlay_mask(image_pil, to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                    cam_extractor.remove_hooks()
    for i in range(len(figs)):
        figs[i] = figs[i].resize((300,300))
    return figs