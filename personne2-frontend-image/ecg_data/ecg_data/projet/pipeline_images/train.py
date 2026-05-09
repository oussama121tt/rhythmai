#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
   PIPELINE IMAGES — Classification ECG par images (EfficientNet-B0)
================================================================================

USAGE :
  python -m pipeline_images.train --config config.yaml
  python -m pipeline_images.train --epochs 100 --batch_size 32
================================================================================
"""

import sys
import os

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from common.utils import (
    set_seed, setup_logging, load_config,
    CLASS_NAMES, CLASS_TO_IDX,
)
from pipeline_images.dataset import ECGImageDataset, load_image_dataset
from pipeline_images.model import ECGClassifier
from pipeline_images.trainer import ImageTrainer


def main():
    parser = argparse.ArgumentParser(
        description='ECG Image Classification Training')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--export_onnx', action='store_true')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    logger = setup_logging('logs/train_images.log', logger_name="ECG_Images")
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")

    logger.info("=" * 70)
    logger.info("   PIPELINE IMAGES — EfficientNet-B3 + CWT Scalograms")
    logger.info("=" * 70)
    logger.info("Approche: CSV → Labels | Images → EfficientNet → Classification")

    # Chemins
    csv_path = os.path.join(
        config['paths']['ptbxl_signals'], 'ptbxl_database.csv')
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Charger le dataset
    df = load_image_dataset(csv_path, config['paths']['ptbxl_signals'], logger)

    # Distribution des classes
    logger.info("\n--- Distribution des classes ---")
    class_counts = df['label'].value_counts()
    total = len(df)
    for cls in CLASS_NAMES:
        count = class_counts.get(cls, 0)
        pct = 100 * count / total
        logger.info(f"  {cls}: {count:5d} ({pct:5.1f}%)")
    logger.info(f"  TOTAL: {total}")
    logger.info(f"  Ratio max/min: {class_counts.max()/class_counts.min():.1f}x")

    # Splits (patient-wise via strat_fold)
    train_df = df[df['strat_fold'].isin(range(1, 9))]
    val_df = df[df['strat_fold'] == 9]
    test_df = df[df['strat_fold'] == 10]

    logger.info(f"\nSplits (patient-wise via strat_fold):")
    logger.info(f"  Train: {len(train_df)} ({100*len(train_df)/total:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({100*len(val_df)/total:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({100*len(test_df)/total:.1f}%)")

    # Datasets
    train_dataset = ECGImageDataset(train_df, augment=True)
    val_dataset = ECGImageDataset(val_df, augment=False)
    test_dataset = ECGImageDataset(test_df, augment=False)

    # DataLoaders
    class_sample_counts = np.stack(train_df['target'].values).sum(axis=0)

    nw = 4 if device.type == 'cuda' else 0
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=nw,
        pin_memory=device.type == 'cuda', drop_last=True)
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=nw,
        pin_memory=device.type == 'cuda')
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=nw,
        pin_memory=device.type == 'cuda')

    # Modèle
    model = ECGClassifier(num_classes=5, pretrained=True, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"\nModèle: ECGClassifier (EfficientNet-B0)")
    logger.info(f"  Paramètres: {n_params:,}")

    # Config & Trainer
    trainer_config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'patience': 15,
    }
    trainer = ImageTrainer(model, trainer_config, device, logger,
                           len(train_loader))
    trainer.setup_loss(class_sample_counts, loss_type='logit_adj', tau=1.0)

    # Entraînement
    history = trainer.train(train_loader, val_loader, args.epochs,
                            checkpoint_dir)

    # Charger le meilleur modèle
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, 'best_image_model.pth'),
        map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # Évaluation finale sur test
    test_metrics, preds, labels = trainer.validate(test_loader)

    logger.info("\n" + "=" * 70)
    logger.info("   RÉSULTATS FINAUX (TEST)")
    logger.info("=" * 70)
    logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
    logger.info(f"F1-macro: {test_metrics['f1_macro']:.4f}")
    logger.info(f"F1/classe: {test_metrics['f1_per_class']}")
    logger.info(f"Sensibilité/classe: {test_metrics['recall_per_class']}")
    logger.info("\n" + classification_report(labels, preds,
                                             target_names=CLASS_NAMES))

    # Matrices de confusion multilabel par classe
    mcm = multilabel_confusion_matrix(labels, preds)
    logger.info("\nMatrices de confusion multilabel:")
    for i, mat in enumerate(mcm):
        logger.info(f"{CLASS_NAMES[i]} -> TN={mat[0,0]}, FP={mat[0,1]}, FN={mat[1,0]}, TP={mat[1,1]}")

    # Vérification objectifs
    logger.info("\n--- Vérification des objectifs ---")
    f1_macro = test_metrics['f1_macro']
    mi_recall = test_metrics['recall_per_class'].get('MI', 0)
    arr_recall = test_metrics['recall_per_class'].get('ARR', 0)
    logger.info(f"  F1-macro:        {f1_macro:.4f} "
                f"{'✓' if f1_macro >= 0.80 else '✗'} (objectif: ≥0.80)")
    logger.info(f"  Sensibilité MI:  {mi_recall:.4f} "
                f"{'✓' if mi_recall >= 0.95 else '✗'} (objectif: ≥0.95)")
    logger.info(f"  Sensibilité ARR: {arr_recall:.4f} "
                f"{'✓' if arr_recall >= 0.92 else '✗'} (objectif: ≥0.92)")

    # Export ONNX
    if args.export_onnx:
        model.eval()
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        onnx_path = 'models/ecg_image_classifier.onnx'
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=12,
            do_constant_folding=True,
            input_names=['image'],
            output_names=['logits'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            })
        logger.info(f"\nModèle exporté: {onnx_path}")

    logger.info("\n" + "=" * 70)
    logger.info("   ENTRAÎNEMENT TERMINÉ !")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
