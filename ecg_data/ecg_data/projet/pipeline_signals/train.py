#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
   PIPELINE SIGNAUX — Classification ECG par signaux 1D (ResNet1D)
================================================================================

USAGE :
  python -m pipeline_signals.train --config config.yaml
  python -m pipeline_signals.train --epochs 50 --batch_size 32
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
    CLASS_NAMES, NUM_CLASSES, CLASS_TO_IDX,
)
from common.preprocessing import HAS_SCIPY
from pipeline_signals.dataset import ECGSignalDataset, load_signal_dataset
from pipeline_signals.model import ECGSignalClassifier
from pipeline_signals.trainer import SignalTrainer


def main():
    parser = argparse.ArgumentParser(
        description='ECG Signal Classification Training (1D)')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--export_onnx', action='store_true')
    args = parser.parse_args()

    # Setup
    set_seed(args.seed)
    logger = setup_logging('logs/train_signals.log',
                           logger_name="ECG_Signals")
    config = load_config(args.config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    logger.info(f"Seed: {args.seed}")
    logger.info(f"scipy disponible: {HAS_SCIPY}")

    logger.info("=" * 70)
    logger.info("   PIPELINE SIGNAUX — ResNet1D + Lead Attention + Transformer")
    logger.info("=" * 70)
    logger.info("Approche: CSV → Labels | Signaux 12-lead → ResNet1D"
                " → Classification")

    # Chemins
    ptbxl_dir = config['paths']['ptbxl_signals']
    csv_path = os.path.join(ptbxl_dir, 'ptbxl_database.csv')
    signals_dir = ptbxl_dir
    checkpoint_dir = 'models/checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Charger le dataset
    df = load_signal_dataset(csv_path, signals_dir, logger)

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
    train_dataset = ECGSignalDataset(train_df, augment=True)
    val_dataset = ECGSignalDataset(val_df, augment=False)
    test_dataset = ECGSignalDataset(test_df, augment=False)

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
    model = ECGSignalClassifier(
        num_classes=NUM_CLASSES, embed_dim=256, dropout=0.3)
    n_params = sum(p.numel() for p in model.parameters())
    n_backbone = sum(p.numel() for p in model.backbone.parameters())
    n_head = sum(p.numel() for p in model.classifier.parameters())
    logger.info(f"\nModèle: ECGSignalClassifier (ResNet1D + SE + AttnPool)")
    logger.info(f"  Total params:  {n_params:,}")
    logger.info(f"  Backbone:      {n_backbone:,}")
    logger.info(f"  Head:          {n_head:,}")

    # Config & Trainer
    trainer_config = {
        'learning_rate': args.lr,
        'weight_decay': 1e-4,
        'epochs': args.epochs,
        'patience': 12,
    }
    trainer = SignalTrainer(model, trainer_config, device, logger)
    trainer.setup_loss(class_sample_counts, loss_type='logit_adj', tau=1.0)

    # Entraînement
    history = trainer.train(train_loader, val_loader, args.epochs,
                            checkpoint_dir)

    # Charger le meilleur modèle
    checkpoint = torch.load(
        os.path.join(checkpoint_dir, 'best_signal_model.pth'),
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
                f"{'✓' if f1_macro >= 0.75 else '✗'} (objectif: ≥0.75)")
    logger.info(f"  Sensibilité MI:  {mi_recall:.4f} "
                f"{'✓' if mi_recall >= 0.85 else '✗'} (objectif: ≥0.85)")
    logger.info(f"  Sensibilité ARR: {arr_recall:.4f} "
                f"{'✓' if arr_recall >= 0.85 else '✗'} (objectif: ≥0.85)")

    # Export ONNX
    if args.export_onnx:
        model.eval()
        dummy_input = torch.randn(1, 12, 5000).to(device)
        onnx_path = 'models/ecg_signal_classifier.onnx'
        torch.onnx.export(
            model, dummy_input, onnx_path,
            export_params=True, opset_version=12,
            do_constant_folding=True,
            input_names=['signal'],
            output_names=['logits'],
            dynamic_axes={
                'signal': {0: 'batch_size'},
                'logits': {0: 'batch_size'}
            })
        logger.info(f"\nModèle exporté: {onnx_path}")

    logger.info("\n" + "=" * 70)
    logger.info("   ENTRAÎNEMENT TERMINÉ !")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
