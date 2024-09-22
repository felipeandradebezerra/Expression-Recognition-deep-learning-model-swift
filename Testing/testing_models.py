#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: testing_models.py
Description:    This script is used to test the models created in the main.py script.
                It compares the results of the models created in PyTorch and CoreML.
                I've created this to find out why the models were not giving the same results.
Author: Felipe Andrade
Date Created: 21-09-2024
Usage: python3 main.py
License: MIT License
"""

import torch
import numpy as np
from PIL import Image
import coremltools as ct
from torchvision import transforms

model_version = "v006"
models = ["resnet152", "efficientnet_b0"]
model_architecture = models[0] #
model_filename = f"Model_{model_architecture}_{model_version}"

img = Image.open('../Data/AffectNet/train/3/image0000045.jpg')
class_names = ['Raiva', 'Nojo', 'Medo', 'Felicidade', 'Tristeza', 'Surpresa', 'Neutro', 'Desprezo']

# Carregamento dos modelos
def load_models():
    print("Loading models")
    coreml_model = ct.models.MLModel(f"../Models/{model_filename}.mlmodel")
    pytorch_model = torch.jit.load(f"../Models/{model_filename}.pt")
    return coreml_model, pytorch_model

# Roda a avaliação do modelo CoreML
def run_coreml_evaluation():
    print("Running coreml evaluation")
    img_converted = img.convert("L") # Converte para escala de cinza
    img_converted = img_converted.convert("RGB")
    img_converted = img_converted.resize((48, 48))

    input_data = {'x_1': img_converted}
    predictions = coreml_model.predict(input_data)

    output_array = predictions["linear_0"][0]
    max_index = np.argmax(output_array)
    predicted_expression = class_names[max_index]
    return predicted_expression

# Roda a avaliação do modelo PyTorch
def run_evaluation():
    print("Running evaluation")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Define as transformações utilizadas durante o treinamento
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(48),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    new_image_tensor = transform(img)
    new_image_tensor = new_image_tensor.unsqueeze(0)
    new_image_tensor = new_image_tensor.to(device)

    model = pytorch_model.to(torch.float32).to(device)
    model.eval()

    with torch.no_grad():
        outputs = model(new_image_tensor)
        _, predicted_class = torch.max(outputs, 1)
        
        probabilities = torch.softmax(outputs, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1)
        predicted_expression = class_names[predicted_class.item()]

        return predicted_expression

coreml_model, pytorch_model = load_models()
expression_pt = run_evaluation()
expression_coreml = run_coreml_evaluation()

print(f"Predicted expression (pt-coreml): {expression_pt} - {expression_coreml}")