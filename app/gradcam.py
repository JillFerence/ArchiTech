# Grad-CAM implementation for front-end and back-end communication
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

"""
Used Kazuto Nakashima's Grad-CAM with PyTorch implementation 
https://github.com/kazuto1011/grad-cam-pytorch
"""
class _BaseWrapper(object):
    def __init__(self, model):
        super(_BaseWrapper, self).__init__()
        self.device = next(model.parameters()).device
        self.model = model
        self.handlers = []  # A set of hook function handlers

    def _encode_one_hot(self, ids):
        one_hot = torch.zeros_like(self.logits).to(self.device)
        one_hot.scatter_(1, ids, 1.0)
        return one_hot

    def forward(self, image):
        self.image_shape = image.shape[2:]
        self.logits = self.model(image)
        self.probs = F.softmax(self.logits, dim=1)
        return self.probs.sort(dim=1, descending=True)  # Ordered results

    def backward(self, ids):
        """
        Class-specific backpropagation
        """
        one_hot = self._encode_one_hot(ids)
        self.model.zero_grad()
        self.logits.backward(gradient=one_hot, retain_graph=True)

    def generate(self):
        raise NotImplementedError

    def remove_hook(self):
        """
        Remove all the forward/backward hook functions
        """
        for handle in self.handlers:
            handle.remove()

"""
Used Kazuto Nakashima's Grad-CAM with PyTorch implementation 
https://github.com/kazuto1011/grad-cam-pytorch
"""
class GradCAM(_BaseWrapper):
    """
    "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization"
    https://arxiv.org/pdf/1610.02391.pdf
    Look at Figure 2 on page 4
    """

    def __init__(self, model, candidate_layers=None):
        super(GradCAM, self).__init__(model)
        self.fmap_pool = {} # Stores the feature maps (activations) from the forward pass through the target layer
        self.grad_pool = {} # Stores the gradients of the class score w/ respect to the activtions of the target layer
        self.candidate_layers = candidate_layers # Only used last layer for our implementation

        def save_fmaps(key):
            def forward_hook(module, input, output):
                self.fmap_pool[key] = output.detach()

            return forward_hook

        def save_grads(key):
            def backward_hook(module, grad_in, grad_out):
                self.grad_pool[key] = grad_out[0].detach()

            return backward_hook

        for name, module in self.model.named_modules():
          self.handlers.append(module.register_forward_hook(save_fmaps(name)))
          self.handlers.append(module.register_backward_hook(save_grads(name)))

    def _find(self, pool, target_layer):
        if target_layer in pool.keys():
            return pool[target_layer]
        else:
            raise ValueError("Invalid layer name: {}".format(target_layer))

    # Generates the class activation map
    def generate_heatmap(self, target_layer):
        fmaps = self._find(self.fmap_pool, target_layer)
        grads = self._find(self.grad_pool, target_layer)
        weights = F.adaptive_avg_pool2d(grads, 1)

        gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)

        # Added to solve missing heatmaps 
        # Balances between positive and negative contributions towards target class
        max_value = gcam.max()
        if max_value > 0:
            gcam = F.relu(gcam) # Apply ReLU if there are positive contributions
        # Don't apply ReLU if there are only negative contributions
        print(f"Maximum Grad-CAM score for the predicted class: {max_value}")

        gcam = F.interpolate(
            gcam, self.image_shape, mode="bilinear", align_corners=False
        )

        B, C, H, W = gcam.shape
        gcam = gcam.view(B, -1)
        gcam -= gcam.min(dim=1, keepdim=True)[0]
        gcam /= gcam.max(dim=1, keepdim=True)[0]
        gcam = gcam.view(B, C, H, W)

        return gcam


    def __call__(self, image, class_idx, target_layer):
        """Performs forward and backward pass to compute Grad-CAM."""
        self.image_shape = image.shape[2:]  # Get the spatial size of the input image
        output = self.model(image)

        # Zero gradients
        self.model.zero_grad()

        # Perform backward pass for the specified class
        output[:, class_idx].backward()

        # Generate and return the heatmap
        heatmap = self.generate_heatmap(target_layer)
        return heatmap

def overlay_heatmap(heatmap, img, alpha=0.5):
    # Resize the heatmap to match the original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((224, 224), Image.LANCZOS)
    heatmap = np.array(heatmap)

    # Load the original image
    min_val = img.min()
    max_val = img.max()
    img = (img - min_val) / (max_val - min_val)

    original_img = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    original_img = np.uint8(original_img * 255)

    # Overlay the heatmap
    overlayed_img = original_img * (1 - alpha) + np.array(plt.cm.jet(heatmap)[:,:,:3]) * 255 * alpha
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)

    return overlayed_img