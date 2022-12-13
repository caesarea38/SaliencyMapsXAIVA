from torchcam.methods import CAM, GradCAM, ScoreCAM, SSCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM
from torchcam.utils import overlay_mask
from torchvision.transforms.functional import normalize, resize, to_pil_image, to_tensor
from torchvision.transforms import ToPILImage
import torch
import matplotlib.pyplot as plt

def explain(image_tensor, image_numpy, methods, models):
    figs = []
    for model in models:
        for method in methods:
            if method == 'CAM':
                cam_extractor = CAM(model, 'layer4', 'linear')
                with torch.no_grad():
                    _, out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)

                result = overlay_mask(ToPILImage(image_tensor), to_pil_image(activation_map_fused.squeeze(0), mode="F"), alpha=0.5)
                
                figs.append(result)

            elif method in ['GradCAM', 'GradCAMpp']:
                cam_extractor = GradCAM(model, 'layer4') if method == 'GradCAM' else GradCAMpp(model, 'layer4')
                _, out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
            
            elif method in ['ScoreCAM', 'SmoothScoreCAM']:
                cam_extractor = ScoreCAM(model, 'layer4') if method == 'ScoreCAM' else SSCAM(model, 'layer4')
                with torch.no_grad(): 
                    _, out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out.detach())
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
        
            elif method == 'SmoothGradCAMpp':
                cam_extractor = SmoothGradCAMpp(model, 'layer4')
                out2, out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()

                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
            
            elif method == 'XGradCAM':
                cam_extractor = XGradCAM(model, 'layer4')
                _, out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = act_maps[0] if len(act_maps) == 1 else cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
            
            elif method == 'LayerCAM':
                cam_extractor = LayerCAM(model, 'layer4')
                _, out = model(image_tensor.unsqueeze(0))
                cls_idx = out.squeeze(0).argmax().item()
                act_maps = cam_extractor(class_idx=cls_idx, scores=out)
                activation_map_fused = cam_extractor.fuse_cams(act_maps)
                figs.append(overlay_mask(to_pil_image(image_tensor), to_pil_image(activation_map_fused, mode='F'), alpha=0.5))
                
    return figs