# gradCAM
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import matplotlib.pyplot as plt
from PIL import Image

def load_image_not_path(img):
    # Transformation for passing image into the network
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
    ])

    
    image = img.convert('RGB')
    image = transform(image).unsqueeze(0)
    print("SHAPE")
    print(image.shape)
    return image


# GradCAM
# Decodes importance of each feature map for a specific class by analyzing gradients in the last convolutional layer
class GradCAM:
    def __init__(self, model, last_layer):
        self.model = model # Pre-trained ResNet152 Model
        self.last_layer = last_layer # Last convolutional layer
        self.gradients = None # Stores the gradients of the class score w/ respect to the activtions of the target layer
        self.activations = None # Stores the feature maps (activations) from the forward pass through the target layer
        self.hook_layers()

    # Gets the activations and the gradients
    def hook_layers(self):
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        self.last_layer.register_forward_hook(forward_hook)
        self.last_layer.register_backward_hook(backward_hook)

    # Generates the class activation map
    def generate_heatmap(self, class_idx):
        # Calculate gradients of the class score w/ respect to the activations of the target layer
        weights = torch.mean(self.gradients, dim=(2, 3))  # Global average pooling of gradients
        grad_cam = torch.sum(weights[:, :, None, None] * self.activations, dim=1) # Weighted sum of feature maps
        grad_cam = F.relu(grad_cam)  # Apply ReLU
        grad_cam = grad_cam.squeeze().cpu().detach().numpy()

        # Normalize the heatmap to a range [0, 1]
        grad_cam -= grad_cam.min()
        grad_cam /= grad_cam.max()
        return grad_cam

    def __call__(self, image, class_idx):
        # Forward pass
        output = self.model(image)

        # Zero gradients
        self.model.zero_grad()

        # Backward pass
        output[:, class_idx].backward()

        # Generate heatmap
        heatmap = self.generate_heatmap(class_idx)
        return heatmap

def overlay_heatmap_not_path(heatmap, img, alpha=0.6):
    # Resize the heatmap to match the original image size
    heatmap = np.uint8(255 * heatmap)
    heatmap = Image.fromarray(heatmap).resize((224, 224), Image.LANCZOS)
    heatmap = np.array(heatmap)

    # Load the original image
    #original_img = img.resize((224, 224))
    print("SHAPE")
    print(img.shape)
    original_img = np.array(img)
    print("SHAPE")
    print(original_img.shape)
    #original_img = np.resize(original_img,(224,224))
    original_img = np.squeeze(original_img)
    original_img = np.transpose(original_img, (1,2,0))
    print("SHAPE final")
    print(original_img.shape, heatmap.shape)

    # Overlay the heatmap
    overlayed_img = original_img * 0.4 + np.array(plt.cm.jet(heatmap)[:,:,:3]) * 255 * alpha
    overlayed_img = np.clip(overlayed_img, 0, 255).astype(np.uint8)

    plt.imshow(original_img)
    plt.axis('off')
    plt.show()

    plt.imshow(heatmap)
    plt.axis('off')
    plt.show()

    plt.imshow(overlayed_img)
    plt.axis('off')
    plt.show()
    return overlayed_img