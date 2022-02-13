#!/usr/bin/env python
# coding: utf-8

import torch


def get_encoder(output_dim):
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)  # Chaim permitted pretrained ResNet50 (on ImageNet)
        mlp_layer0 = torch.nn.Linear(model.fc.in_features, model.fc.in_features)
        mlp_layer1 = torch.nn.ReLU()
        mlp_layer2 = torch.nn.Linear(model.fc.in_features, output_dim)
        model.fc = torch.nn.Sequential(mlp_layer0, mlp_layer1, mlp_layer2)  # Replace head for MLP for MoCo v2

        return model

# Test feature extractor
# model = get_encoder()
# model(torch.arange(64*3*7*7).reshape((64,3,7,7)).float()).dtype
