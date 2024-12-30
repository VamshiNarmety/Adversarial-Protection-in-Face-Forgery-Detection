import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import os
from src.utils.trigger_generator import TriggerGenerator
import dlib
import numpy as np
import cv2
from imutils import face_utils
from src.utils.trigger_generator import embed_trigger
from pytorch_msssim import ssim

def load_and_preprocess_image(image_path):
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    image = Image.open(image_path).convert("RGB")
    image = transform(image)*255.0
    return image


def calculate_psnr(original, processed):
    mse = torch.mean((original - processed)**2)
    if mse==0:
        return float('inf')#perfect match
    max_pixel = 255.0 #Our Images are 8-bit images
    psnr = 10*torch.log10((max_pixel**2)/mse)
    return psnr.item()


def calculate_l_infinity(original, processed):
    return torch.max(torch.abs(original-processed)).item()


def calculate_ssim(original, processed):
    ssim_value = ssim(original, processed, data_range=255, size_average=True).item()
    return ssim_value


def detect_landmarks(image_path):
    image = cv2.imread(image_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_image = cv2.resize(rgb_image, (224, 224))
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'src/preprocess/shape_predictor_81_face_landmarks.dat'
    faces = detector(rgb_image, 1)
    predictor = dlib.shape_predictor(predictor_path)
    landmarks = []
    for face in faces:
        # Get the landmarks for each detected face
        shape = predictor(rgb_image, face)
        landmarks.append(face_utils.shape_to_np(shape))
    return landmarks

def create_mask(image_size, landmarks):
    mask = np.zeros(image_size, dtype=np.float32)
    all_points = np.vstack(landmarks)
    # Calculate the convex hull of all points
    hull = cv2.convexHull(all_points)
    # Fill the mask based on the convex hull
    cv2.fillConvexPoly(mask, hull, 1)
    return mask


def plot_results(results, image_paths):
    for image_idx, image_path in enumerate(image_paths):
        psnr_values = []
        ssim_values = []
        l_infinity_values = []
        kernel_sizes = []
        for model_result in results:
            image_result = model_result['Image Results'][image_idx]
            psnr_values.append(image_result['PSNR'])
            ssim_values.append(image_result['SSIM'])
            l_infinity_values.append(image_result['L'])
            kernel_sizes.append(model_result['Model'].split('_')[-1].replace(".pth", ""))
        
        plt.figure(figsize=(6, 3))   
        plt.subplot(1, 3, 1)
        plt.plot(kernel_sizes, psnr_values, marker='o')
        plt.title('PSNR v/s Kernel size')
        plt.xlabel('Kernel size')
        plt.ylabel('PSNR(dB)')
        plt.subplot(1, 3, 2)
        plt.plot(kernel_sizes, ssim_values, marker='o')
        plt.title("SSIM v/s Kernel size")
        plt.xlabel('Kernel size')
        plt.ylabel('SSIM')
        plt.subplot(1, 3, 3)
        plt.plot(kernel_sizes, l_infinity_values, marker='o')
        plt.title("L∞ Norm v/s kernel size")
        plt.xlabel('Kernel size')
        plt.ylabel('L∞ Norm')
        plt.tight_layout()
        plt.savefig(f'{image_idx}_exp.png', bbox_inches='tight', pad_inches=0)
        plt.close() 

  


if __name__=='__main__':
    models_paths = ["src/utils/generator_3.pth", "src/utils/generator_5.pth", "src/utils/generator_7.pth", "src/utils/generator_9.pth", "src/utils/generator_11.pth", "src/utils/generator_13.pth"]
    image_paths = ["sample_images/samplefromceleb.jpg", "sample_images/samplefromFF++.jpg", "sample_images/samplefromrealvsfake.jpg", "sample_images/my_image.jpg"]
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    results = []
    for model_id, model_path in enumerate(models_paths):
        print(f'\n Evaluating Trigger generator {model_id+1}/{len(models_paths)}')
        model = TriggerGenerator().to(device)
        model.load_state_dict(torch.load(model_path, weights_only=True))
        model.eval()
        
        model_results = {"Model":model_path, "Image Results":[]}
        for image_id, image_path in enumerate(image_paths):
            original_image = load_and_preprocess_image(image_path)
            original_image = original_image.unsqueeze(0).to(device)
            z = torch.randn(1, 3, 224, 224)
            delta = model(z)
            landmarks = detect_landmarks(image_path)
            mask = np.zeros((224, 224), dtype=np.float32)
            mask = mask + create_mask((224, 224), landmarks)
            mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(device)
            modified_image = embed_trigger(original_image, delta, mask_tensor, a=0.05)
            psnr = calculate_psnr(original_image, modified_image)
            ssim_value = calculate_ssim(original_image, modified_image)
            l_infinity = calculate_l_infinity(original_image, modified_image)

            print(f"{image_id}, PSNR: {psnr:.2f}, SSIM: {ssim_value:.4f}, L∞: {l_infinity:.2f}")

            model_results["Image Results"].append({
                "Image Path": image_path,
                "PSNR": psnr,
                "SSIM": ssim_value,
                "L": l_infinity,
            })

        results.append(model_results)
    
    plot_results(results, image_paths)
    
    


