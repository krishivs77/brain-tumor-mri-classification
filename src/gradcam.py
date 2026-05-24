import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.forward_handle = self.target_layer.register_forward_hook(
            self._save_activations
        )

        self.backward_handle = self.target_layer.register_full_backward_hook(
            self._save_gradients
        )
    
    def _save_activations(self, module, inputs, output):
        self.activations = output.detach()
    
    def _save_gradients(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class):
        self.model.zero_grad()

        outputs = self.model(input_tensor)
        class_score = outputs[0, target_class]
        class_score.backward()

        weights = self.gradients.mean(
            dim=(2, 3),
            keepdim=True
        )

        gradcam = (weights * self.activations).sum(
            dim=1,
            keepdim=True
        )

        gradcam = F.relu(gradcam)

        gradcam = F.interpolate(
            gradcam,
            size=(224, 224),
            mode="bilinear",
            align_corners=False
        )

        gradcam = gradcam.squeeze().cpu().numpy()

        gradcam = (
            (gradcam - gradcam.min())
            / (gradcam.max() - gradcam.min() + 1e-8)
        )

        return gradcam, outputs
    
    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def create_gradcam_overlay(original_image, heatmap, alpha=0.45):
    original_image = original_image.convert("L").resize((224, 224))
    original_array = np.array(original_image) / 255.0

    heatmap_rgba = plt_colormap(heatmap)

    overlay = (
        (1 - alpha) * np.stack([original_array] * 3, axis=-1)
        + alpha * heatmap_rgba[:, :, :3]
    )

    overlay = np.clip(overlay, 0, 1)

    return Image.fromarray((overlay * 255).astype(np.uint8))

def plt_colormap(heatmap):
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("jet")
    return cmap(heatmap)