#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script: main.py
Description: This script is used to train a model to detect facial expressions using the AffectNet dataset.

Author: Felipe Andrade
Date Created: 21-09-2024
Usage: python3 main.py
License: MIT License
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy
from models import get_models
from torch.utils.tensorboard import SummaryWriter
import torch.nn.utils.prune as prune
import coremltools as ct

# Função de treinamento do modelo
def train_model(model, criterion, optimizer, dataloaders, device, num_epochs=20):
    model.train()
    for epoch in range(num_epochs):
        print(f'Época {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            print(f'Running phase: {phase}')

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device).to(torch.float32)
                labels = labels.to(device).long()
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    preds = torch.argmax(outputs, dim=1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        # Aplica clipping do gradiente para evitar o problema do exploding gradient
                        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.float() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Adiciona as métricas ao TensorBoard
            writer.add_scalar(f'{phase} Loss', epoch_loss, epoch)
            writer.add_scalar(f'{phase} Accuracy', epoch_acc, epoch)

    return model

 # Seta o modelo para o modo de avaliação e avalia o modelo
def evaluate_model(model, criterion, test_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device).to(torch.float32)
            labels = labels.to(device).long()

            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.float() / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    return test_loss, test_acc

# Inicializa o treinamento do modelo
if __name__ == '__main__':
    # Define o dispositivo de execução do modelo como 
    # MPS (Máquina de Aprendizado de Precisão Mista) disponível no Macbook Air M3
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Definição das transformações para melhorar a generalização do modelo
    data_transforms = {
        'train': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.RandomResizedCrop(48),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(48),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Definição dos diretórios dos datasets
    data_dirs = {
        'train': './Data/AffectNet/train',
        'val': './Data/AffectNet/val',
        'test': './Data/AffectNet/test'
    }

    # Carregamento dos datasets
    image_datasets = {x: datasets.ImageFolder(data_dirs[x], data_transforms[x])
                    for x in ['train', 'val', 'test']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=32, shuffle=True, num_workers=4)
                for x in ['train', 'val']}
    test_loader = DataLoader(image_datasets['test'], batch_size=32, shuffle=False, num_workers=4)
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val', 'test']}
    class_names = image_datasets['train'].classes

    print(f"Torch version: {torch.__version__}")
    print(f"Dispositivo: {device}")
    print(f"Classes: {class_names}")
    print(f"Tamanho dos datasets - Treino: {dataset_sizes['train']}, Validação: {dataset_sizes['val']}, Teste: {dataset_sizes['test']}")

    # Definição do modelo a ser utilizado
    model_name = "resnet152"
    num_epochs = 100
    model_version = "v006"
    model_filename = f"Model_{model_name}_{model_version}"

    # Carregamento do modelo
    model = get_models(model_name, output_size=8, input_size=dataset_sizes['val'], pretrained=True)
    model = model.to(torch.float32).to(device)

    # Define se o modelo será treinado por completo ou apenas o classificador
    fine_tune_classifier=True
    if fine_tune_classifier:
        for param in model.parameters():
            param.requires_grad = True
    else:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

    # Define os pesos das classes para evitar o desbalanceamento
    class_counts = [5000, 3803, 5000, 5000, 5000, 5000, 5000, 3750]
    class_weights = 1. / torch.tensor(class_counts, dtype=torch.float32)
    class_weights = class_weights.to(device)

    # Definição do otimizador e função de custo para o treinamento com objetivo de minimizar a CrossEntropyLoss
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
    criterion = criterion.to(torch.float32).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.1)

    # Inicializa o treinamento do modelo
    # SummaryWriter é utilizado para visualizar o treinamento no TensorBoard
    writer = SummaryWriter(log_dir="./Runs/fine_tuning_experiment")
    model_ft = train_model(model, criterion, optimizer, dataloaders, device, num_epochs=num_epochs)
    
    # Depois do treino, avalia o modelo
    test_loss, test_acc = evaluate_model(model_ft, criterion, test_loader, device)
    print(f'Final test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

    # Quantização (não disponível no Macbook Air M3)
    # model_ft = torch.quantization.quantize_dynamic(
    #     model_ft, {torch.nn.Linear}, dtype=torch.qint8
    # )

    # Define a quantidade de pruning
    for module in model_ft.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=0.2)
    
    # Remove os reparametrizados de pruning
    for module in model_ft.modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            prune.remove(module, 'weight')

    # Converte para TorchScript
    input = torch.randn(1, 3, 48, 48).to(device)
    mlmodel_traced = torch.jit.trace(model_ft, input)
    torch.jit.save(mlmodel_traced, f"./Models/{model_filename}.pt")
    
    # Converte para CoreML com entrada de uma imagem com 48x48 pixels e 3 canais (RGB)
    image_input = ct.ImageType(shape=(1, 3, 48, 48), scale=1/255.0, bias=[-0.485, -0.456, -0.406])
    coreml_model = ct.convert(mlmodel_traced,
                            inputs=[image_input],
                            minimum_deployment_target=ct.target.iOS14)

    # Define as informações do modelo para exibição no XCode
    coreml_model.type = "imageClassifier"
    coreml_model.author = "Felipe Andrade"
    coreml_model.short_description = "Detects the face expression"
    coreml_model.license = "MIT"
    coreml_model.version = model_version
    coreml_model.input_description["x_1"] = "Input image to be classified"
    coreml_model.user_defined_metadata["com.apple.coreml.model.preview.type"] = "imageClassifier"
    coreml_model.save(f"./Models/{model_filename}.mlmodel")

    # Encerra o tensorboard
    writer.close()
