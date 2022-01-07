import torch
import numpy as np
from torchray.attribution.grad_cam import gradient_to_grad_cam_saliency
from torchray.attribution.guided_backprop import GuidedBackpropReLU
from torchray.attribution.common import Probe, get_module
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms
import cmapy
from explainability_metrics import imshow

def deprocess(image):
    transform = transforms.Compose(
        [
            transforms.Normalize(mean=[0, 0, 0], std=[4.3668, 4.4643, 4.4444]),
            transforms.Normalize(mean=[-0.485, -0.456, -0.406], std=[1, 1, 1]),
            transforms.ToPILImage(),
        ]
    )
    return transform(image)


"""
Saliency maps
"""


def batch_saliency(model, inputs):
    x = inputs.to(0)
    x.requires_grad_()
    scores = model(x)
    score_max_index = scores.argmax(dim=1)
    score_max = scores[:, score_max_index]
    score_max.backward(torch.ones_like(score_max))
    saliency, _ = torch.max(x.grad.data.abs(), dim=1)
    return saliency, score_max_index


def prepare_saliency(sal_map, index):
    return sal_map[index].cpu().numpy()


def saving_saliency_map(sal, index, x, input_filename, main_folder, label, pred_res):
    sal = np.uint8((255 * sal) / np.max(sal))
    plt.figure()
    plt.imshow(sal, cmap=plt.cm.hot, alpha=0.9)
    plt.imshow(deprocess(x[index].cpu()), alpha=0.6)
    plt.axis("off")
    plt.savefig(str(main_folder / f"{label}/{input_filename}_{pred_res}.png"))
    plt.close()


"""
Grad CAM maps
"""


def grad_cam_batch(model, inputs, gb_cam=False):
    x = inputs.to(0)
    x.requires_grad_()
    saliency_layer = get_module(model, model.layer4)
    probe = Probe(saliency_layer, target="output")
    y = model(x)
    score_max_index = y.argmax(dim=1)
    z = y[:, score_max_index]
    z.backward(torch.ones_like(z))
    grad_cam = gradient_to_grad_cam_saliency(probe.data[0])
    if not gb_cam:
        return grad_cam, score_max_index
    else:
        return grad_cam, score_max_index, x


def preparing_grad_cam(batch_grad_cam, index):
    heatmap = np.float32(batch_grad_cam[index, 0].cpu().detach())
    final_map = cv2.resize(heatmap, (224, 224))
    return np.uint8((255 * final_map) / np.max(final_map))


def saving_grad_cam_map(sal, index, x, input_filename, main_folder, label, pred_res):
    plt.figure()
    img = np.array(deprocess(x[index].cpu().detach()))
    plt.imshow(sal, alpha=0.9, cmap=plt.cm.hot)
    plt.imshow(img, alpha=0.6)
    plt.axis("off")
    plt.savefig(str(main_folder / f"{label}/{input_filename}_{pred_res}.png"))
    plt.close()


"""
Guided Back propagation Grad Cam maps
"""


class GuidedBackpropReLUModel:
    def __init__(self, model):
        self.model = model

        self.model.eval()

        def recursive_relu_apply(module_top):
            for idx, module in module_top._modules.items():
                recursive_relu_apply(module)
                if module.__class__.__name__ == "ReLU":
                    module_top._modules[idx] = GuidedBackpropReLU.apply

        # replace ReLU with GuidedBackpropReLU
        recursive_relu_apply(self.model)

    def forward(self, input_img):
        return self.model(input_img)

    def __call__(self, input_img, target_category=None):
        input_img = input_img.requires_grad_(True)
        output = self.forward(input_img)

        if target_category == None:
            target_category = np.argmax(output.cpu().data.numpy())

        one_hot = np.zeros((1, output.size()[-1]), dtype=np.float32)
        one_hot[0][target_category] = 1
        one_hot = torch.from_numpy(one_hot).requires_grad_(True).to(0)

        one_hot = torch.sum(one_hot * output)
        one_hot.backward(retain_graph=True)
        output = input_img.grad.cpu().data.numpy()
        output = output[0, :, :, :]

        return output


def deprocess_image_gb(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255 / np.max(img))
    return np.abs(img)

def default_deprocess_image_gb(img):
    """ see https://github.com/jacobgil/keras-grad-cam/blob/master/grad-cam.py#L65 """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    img = np.uint8(img * 255)
    return np.abs(img - int(0.5 * 255))

def preparing_gb_grad_cam(batch_grad_cam, index, guided_backprop_model, x, labels):
    heatmap = np.float32(batch_grad_cam[index, 0].cpu().detach())
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    cam_mask = cv2.merge([heatmap, heatmap, heatmap])
    gb = guided_backprop_model(
        x[index].unsqueeze(0).to(0), target_category=labels[index]
    )
    gb = gb.transpose((1, 2, 0))
    final_map = deprocess_image_gb(cam_mask * gb)
    return cv2.cvtColor(final_map, cv2.COLOR_RGB2GRAY)


def saving_gb_grad_cam(gb_grad_map, input_filename, main_folder, label, pred_res, x, index):
    plt.figure()
    img = np.array(deprocess(x[index].cpu().detach()))
    plt.imshow(gb_grad_map, alpha=0.9, cmap=plt.cm.hot)
    plt.imshow(img, alpha=0.6)
    plt.axis("off")
    plt.savefig(str(main_folder / f"{label}/{input_filename}_{pred_res}.png"))
    plt.close()
