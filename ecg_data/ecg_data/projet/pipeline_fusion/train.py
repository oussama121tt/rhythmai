#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
   PIPELINE FUSION — Signaux 1D + Images 2D (GatedFusion v6)
================================================================================

USAGE :
  python -m pipeline_fusion.train --config config.yaml
  python -m pipeline_fusion.train --loss_type logit_adj --tau 1.0
  python -m pipeline_fusion.train --resume
================================================================================
"""

import sys
import os

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import multilabel_confusion_matrix, classification_report

from common.utils import (
    set_seed, setup_logging, load_config,
    CLASS_NAMES, NUM_CLASSES,
)
from common.preprocessing import HAS_SCIPY
from pipeline_fusion.dataset import ECGDualDataset, load_fusion_dataset
from pipeline_fusion.model import ECGFusionModel
from pipeline_fusion.trainer import FusionTrainer


def main():
    parser = argparse.ArgumentParser(
        description='ECG Fusion Training v6 — Clean Baseline')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    # [ANCIEN] parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_batches', type=int, default=None,
                        help='Limite le nombre de batches par epoch pour un smoke test')
    parser.add_argument('--seed', type=int, default=42)
    # [ANCIEN] parser.add_argument('--loss_type', type=str, default='sqrt',
    # [ANCIEN]                     choices=['sqrt', 'logit_adj'],
    # V4: Par défaut Focal Loss (meilleure performance)
    parser.add_argument('--loss_type', type=str, default='focal',
                        choices=['focal', 'logit_adj'],
                        help='focal = FocalLoss, logit_adj = Logit Adjustment')
    parser.add_argument('--tau', type=float, default=1.0,
                        help='τ pour logit adjustment (default: 1.0)')
    parser.add_argument('--resume', action='store_true',
                        help='Reprendre depuis le dernier checkpoint')
    parser.add_argument('--resume_epoch', type=int, default=None,
                        help='Époque de départ à utiliser si le checkpoint ne contient pas de métadonnées')
    parser.add_argument('--export_onnx', action='store_true')
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='auto: utilise CUDA si disponible, sinon CPU')
    parser.add_argument('--num_workers', type=int, default=-1,
                        help='-1 = auto selon machine')
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help='Nombre de batches prefetched par worker')
    parser.add_argument('--persistent_workers', action='store_true',
                        help='Garder les workers DataLoader vivants entre epochs')
    parser.add_argument('--amp', action='store_true',
                        help='Active mixed precision (GPU uniquement)')
    parser.add_argument('--compile', action='store_true',
                        help='Active torch.compile (PyTorch 2+, peut accelerer sur GPU)')
    parser.add_argument('--channels_last', action='store_true',
                        help='Active memory format channels_last (branche image)')
    parser.add_argument('--disable_hw_autotune', action='store_true',
                        help='Désactive l\'auto-ajustement pour GPU 8 Go / Windows')
    args = parser.parse_args()

    set_seed(args.seed)
    logger = setup_logging('logs/train_fusion.log', logger_name="ECG_Fusion")
    config = load_config(args.config)

    if args.device == 'cpu':
        device = torch.device('cpu')
    elif args.device == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda demande mais CUDA indisponible.")
        device = torch.device('cuda')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    gpu_mem_gb = None
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision('high')

    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        logger.info(f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB)")

    if not args.disable_hw_autotune:
        # Profil matériel ciblé: Windows + GPU 8 Go + RAM modérée.
        if device.type == 'cuda' and gpu_mem_gb is not None and gpu_mem_gb <= 8.5:
            if args.batch_size > 8:
                logger.info(
                    f"AutoTune HW: batch_size réduit {args.batch_size} -> 8 (GPU <= 8.5 Go)")
                args.batch_size = 8
            if not args.amp:
                logger.info("AutoTune HW: AMP activé automatiquement (CUDA)")
                args.amp = True
            if args.prefetch_factor > 2:
                logger.info(
                    f"AutoTune HW: prefetch_factor réduit {args.prefetch_factor} -> 2")
                args.prefetch_factor = 2

        if device.type == 'cpu' and args.batch_size > 8:
            logger.info(
                f"AutoTune HW: batch_size réduit {args.batch_size} -> 8 (CPU)")
            args.batch_size = 8
    logger.info(f"Seed: {args.seed}")
    logger.info(f"scipy disponible: {HAS_SCIPY}")
    logger.info("=" * 70)
    logger.info("   PIPELINE FUSION — Signaux 1D + Images 2D")
    logger.info("   Cross-Attention + MI Head + CD Head + Metadata")
    logger.info("=" * 70)

    # ── Chemins ──
    ptbxl_dir = config['paths']['ptbxl_signals']
    csv_path = os.path.join(ptbxl_dir, 'ptbxl_database.csv')
    signals_dir = ptbxl_dir
    ckpt_dir = 'models/checkpoints'

    # ── Dataset ──
    df = load_fusion_dataset(csv_path, signals_dir, config['paths']['ptbxl_images'], logger)

    # Distribution
    logger.info("\n--- Distribution des classes ---")
    cc = df['label'].value_counts()
    total = len(df)
    for c in CLASS_NAMES:
        n = cc.get(c, 0)
        logger.info(f"  {c}: {n:5d} ({100*n/total:5.1f}%)")
    logger.info(f"  TOTAL: {total}")
    logger.info(f"  Ratio max/min: {cc.max()/cc.min():.1f}x")

    # Modality stats
    n_both = len(df)
    n_sig_only = 0
    n_img_only = 0
    logger.info("\n--- Modalités disponibles ---")
    logger.info(f"  Signal + Image: {n_both}")
    logger.info(f"  Signal seul:    {n_sig_only}")
    logger.info(f"  Image seule:    {n_img_only}")

    # ── Splits ──
    train_df = df[df['strat_fold'].isin(range(1, 9))]
    val_df = df[df['strat_fold'] == 9]
    test_df = df[df['strat_fold'] == 10]

    logger.info(f"\nSplits (patient-wise via strat_fold):")
    logger.info(f"  Train: {len(train_df)} ({100*len(train_df)/total:.1f}%)")
    logger.info(f"  Val:   {len(val_df)} ({100*len(val_df)/total:.1f}%)")
    logger.info(f"  Test:  {len(test_df)} ({100*len(test_df)/total:.1f}%)")

    # ── Datasets & Loaders ──
    train_ds = ECGDualDataset(train_df, augment=True)
    val_ds = ECGDualDataset(val_df, augment=False)
    test_ds = ECGDualDataset(test_df, augment=False)

    class_sample_counts = np.stack(train_df['target'].values).sum(axis=0)

    if args.num_workers >= 0:
        nw = args.num_workers
    else:
        cpu_count = os.cpu_count() or 4
        if os.name == 'nt':
            nw = min(4, max(2, cpu_count // 3))
        else:
            nw = min(8, max(2, cpu_count // 2))
        if (device.type == 'cuda' and gpu_mem_gb is not None and gpu_mem_gb <= 8.5) or device.type == 'cpu':
            nw = min(nw, 4)
    pm = device.type == 'cuda'
    persistent_workers = bool(args.persistent_workers) and nw > 0
    loader_kwargs = {
        'num_workers': nw,
        'pin_memory': pm,
        'persistent_workers': persistent_workers,
    }
    if nw > 0:
        loader_kwargs['prefetch_factor'] = max(2, args.prefetch_factor)

    # Calcul des poids par sample pour rééquilibrer ARR
    arr_idx = CLASS_NAMES.index('ARR')
    sample_weights = []
    for target in train_df['target'].values:
        if target[arr_idx] == 1:
            # [V1] sample_weights.append(3.0)  # ARR surpondéré x3
            sample_weights.append(2.0)  # ARR surpondéré x2
        else:
            sample_weights.append(1.0)
    sample_weights = torch.tensor(sample_weights, dtype=torch.float)
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    # [ANCIEN] train_loader = DataLoader(train_ds, batch_size=args.batch_size,
    # [ANCIEN]                           shuffle=True, drop_last=True,
    # [ANCIEN]                           **loader_kwargs)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=sampler, drop_last=True,
                              **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            **loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, **loader_kwargs)

    logger.info(f"DataLoader: workers={nw}, pin_memory={pm}, prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}, persistent_workers={persistent_workers}")

    # ── Modèle ──
    model = ECGFusionModel(
        num_classes=NUM_CLASSES, embed_dim=256,
        pretrained=True, dropout=0.3, modality_drop_p=0.15)

    if args.channels_last and device.type == 'cuda':
        model = model.to(memory_format=torch.channels_last)
        logger.info("Memory format: channels_last activé")

    if args.compile:
        if hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("torch.compile activé")
            except Exception as exc:
                logger.warning(f"torch.compile indisponible: {exc}")
        else:
            logger.warning("torch.compile non disponible dans cette version de PyTorch")

    n_params = sum(p.numel() for p in model.parameters())
    n_sig = sum(p.numel() for p in model.signal_branch.parameters())
    n_img = sum(p.numel() for p in model.image_branch.parameters())
    n_fhead = sum(p.numel() for p in model.fusion_head.parameters())
    n_aux = (sum(p.numel() for p in model.aux_head_1d.parameters())
             + sum(p.numel() for p in model.aux_head_2d.parameters()))
    n_cd = sum(p.numel() for p in model.cd_head.parameters())
    n_mi = sum(p.numel() for p in model.mi_head.parameters())

    logger.info(f"\nModèle: ECGFusionModel v6 (Clean)")
    logger.info(f"  Total params:      {n_params:,}")
    logger.info(f"  Signal branch:     {n_sig:,}  (ResNet1D + SE + AttnPool)")
    logger.info(f"  Image branch:      {n_img:,}  (EfficientNet-B3)")
    logger.info(f"  Fusion head:       {n_fhead:,}")
    logger.info(f"  Aux heads (1D+2D): {n_aux:,}")
    logger.info(f"  CD head:           {n_cd:,}")
    logger.info(f"  MI head:           {n_mi:,}")
    logger.info(f"  Modality dropout:  p={model.modality_drop_p}")

    # ── Config ──
    trainer_config = {
        'learning_rate': args.lr,
        # [V1] 'weight_decay': 1e-4,
        'weight_decay': 5e-4,
        'epochs': args.epochs,
        'patience': 10,
        'use_amp': bool(args.amp),
    }

    if args.amp and device.type != 'cuda':
        logger.warning("AMP demandé mais device CPU: AMP sera ignoré")

    # ── Trainer ──
    trainer = FusionTrainer(model, trainer_config, device, logger)
    trainer.setup_loss(class_sample_counts, loss_type=args.loss_type,
                       tau=args.tau)
    logger.info("WeightedRandomSampler activé : ARR x2.0")
    logger.info("Loss par défaut : logit_adj (tau=1.0)")
    logger.info(f"Optimizer LR (initial): {trainer.optimizer.param_groups[0]['lr']:.2e}")

    # ── Resume ──
    start_epoch = 1
    ckpt_scheduler_state = None
    if args.resume:
        ckpt_path = os.path.join(ckpt_dir, 'best_fusion_model.pth')
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location=device,
                              weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model_state_dict = ckpt['model_state_dict']
            else:
                model_state_dict = ckpt

            missing, unexpected = model.load_state_dict(
                model_state_dict, strict=False)
            if missing:
                logger.info(f"  ⚠ Nouvelles couches (random init): "
                            f"{[k.split('.')[0] for k in missing]}")
            trainer.model = model.to(device)
            if isinstance(ckpt, dict) and 'optimizer_state_dict' in ckpt and ckpt['optimizer_state_dict']:
                try:
                    trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                    logger.info("Optimizer state restauré depuis checkpoint")
                except Exception as exc:
                    logger.warning(f"Impossible de restaurer l'optimizer: {exc}")
            else:
                logger.warning("Checkpoint sans optimizer_state_dict: reprise avec optimizer réinitialisé")

            if isinstance(ckpt, dict) and 'scheduler_state_dict' in ckpt and ckpt['scheduler_state_dict']:
                ckpt_scheduler_state = ckpt['scheduler_state_dict']
            else:
                ckpt_scheduler_state = None

            trainer.best_f1 = ckpt.get('best_f1', 0.0) if isinstance(ckpt, dict) else 0.0
            trainer.patience_counter = ckpt.get('patience_counter', 0) if isinstance(ckpt, dict) else 0
            saved_epoch = ckpt.get('epoch', args.resume_epoch or 0) if isinstance(ckpt, dict) else (args.resume_epoch or 0)
            if saved_epoch <= 0 and args.resume_epoch is not None:
                saved_epoch = args.resume_epoch
            start_epoch = saved_epoch + 1
            logger.info(
                f"\n*** REPRISE depuis checkpoint epoch {saved_epoch} ***")
            logger.info(f"    Best F1: {trainer.best_f1:.4f}")
            logger.info(f"    Patience: {trainer.patience_counter}/10")
            logger.info(
                f"    LR actuel: "
                f"{trainer.optimizer.param_groups[0]['lr']:.2e}")
        else:
            logger.info(
                "Aucun checkpoint trouvé, démarrage from scratch")

    # ── Train ──
    logger.info(f"Loss type: {args.loss_type}"
                + (f" (τ={args.tau})" if args.loss_type == 'logit_adj'
                   else ""))
    history = trainer.train_loop(train_loader, val_loader, args.epochs,
                                 ckpt_dir, start_epoch=start_epoch,
                                 max_batches=args.max_batches,
                                 scheduler_state_dict=ckpt_scheduler_state)
    manifest = {
        'args': vars(args),
        'trainer_config': trainer_config,
        'device': str(device),
        'loss_type': args.loss_type,
        'tau': args.tau if args.loss_type == 'logit_adj' else None,
        'class_sample_counts': class_sample_counts.tolist(),
        'n_train': len(train_df),
        'n_val': len(val_df),
        'n_test': len(test_df),
        'n_params': n_params,
        'scipy_available': HAS_SCIPY,
    }
    manifest_path = os.path.join(ckpt_dir, 'run_manifest.json')
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, default=str)
    logger.info(f"Config manifest sauvegardé: {manifest_path}")

    # ── Évaluation finale sur test ──
    best_ckpt = torch.load(os.path.join(ckpt_dir, 'best_fusion_model.pth'),
                           map_location=device, weights_only=False)
    model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
    trainer.model = model.to(device)

    logger.info("\nRecherche des seuils optimaux par classe (validation)...")
    optimal_thresholds = trainer.find_optimal_thresholds(val_loader, step=0.01)
    thresholds_path = os.path.join(ckpt_dir, 'optimal_thresholds.json')
    with open(thresholds_path, 'w', encoding='utf-8') as f:
        json.dump(optimal_thresholds, f, indent=2)
    logger.info(f"Seuils optimaux sauvegardés: {thresholds_path}")

    logger.info("Comparaison val F1: seuil 0.5 vs seuils optimisés")
    val_default_m, _, _ = trainer.validate(val_loader)
    val_opt_m, _, _ = trainer.validate(val_loader, thresholds=optimal_thresholds)
    logger.info(f"  Val F1 @0.5: {val_default_m['f1_macro']:.4f}")
    logger.info(f"  Val F1 @thr_opt: {val_opt_m['f1_macro']:.4f}")

    test_m, preds, labels = trainer.validate(test_loader, thresholds=optimal_thresholds)

    logger.info("\n" + "=" * 70)
    logger.info("   RÉSULTATS FINAUX (TEST)")
    logger.info("=" * 70)
    logger.info(f"Accuracy:  {test_m['accuracy']:.4f}")
    logger.info(f"F1-macro:  {test_m['f1_macro']:.4f}")
    logger.info(f"F1/classe: {test_m['f1_per_class']}")
    logger.info(f"Recall:    {test_m['recall_per_class']}")
    logger.info("\n" + classification_report(labels, preds,
                                             target_names=CLASS_NAMES))

    # ── Matrices de confusion multilabel ──
    mcm = multilabel_confusion_matrix(labels, preds)
    logger.info("\nMatrices de confusion multilabel:")
    for i, mat in enumerate(mcm):
        logger.info(f"{CLASS_NAMES[i]} -> TN={mat[0,0]}, FP={mat[0,1]}, FN={mat[1,0]}, TP={mat[1,1]}")

    # ── Vérification objectifs cliniques ──
    logger.info("\n--- Vérification des objectifs ---")
    f1 = test_m['f1_macro']
    mi_r = test_m['recall_per_class'].get('MI', 0)
    arr_r = test_m['recall_per_class'].get('ARR', 0)
    cd_f1 = test_m['f1_per_class'].get('CD', 0)
    logger.info(f"  F1-macro:        {f1:.4f} "
                f"{'✓' if f1 >= 0.70 else '✗'} (objectif: >=0.70)")
    logger.info(f"  Sensibilité MI:  {mi_r:.4f} "
                f"{'✓' if mi_r >= 0.80 else '✗'} (objectif: >=0.80)")
    logger.info(f"  Sensibilité ARR: {arr_r:.4f} "
                f"{'✓' if arr_r >= 0.80 else '✗'} (objectif: >=0.80)")
    logger.info(f"  F1 CD:           {cd_f1:.4f} "
                f"{'✓' if cd_f1 >= 0.60 else '✗'} (objectif: >=0.60)")

    # ── Sauvegarder résultats finaux ──
    final_results = {
        'test_metrics': test_m,
        'best_val_f1': history['best_f1'],
        'loss_type': args.loss_type,
        'tau': args.tau if args.loss_type == 'logit_adj' else None,
        'optimal_thresholds': optimal_thresholds,
        'epochs_trained': len(history['history']['train_loss']),
    }
    results_path = os.path.join(ckpt_dir, 'final_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(final_results, f, indent=2, default=str)
    logger.info(f"\nRésultats sauvegardés: {results_path}")

    # ── Export ONNX ──
    if args.export_onnx:
        model.eval()
        dummy_sig = torch.randn(1, 12, 5000).to(device)
        dummy_img = torch.randn(1, 3, 300, 300).to(device)
        dummy_mask = torch.ones(1, 2).to(device)
        dummy_meta = torch.zeros(1, 3).to(device)
        onnx_path = 'models/ecg_fusion.onnx'
        torch.onnx.export(
            model, (dummy_sig, dummy_img, dummy_mask, dummy_meta), onnx_path,
            export_params=True, opset_version=14,
            input_names=['signal', 'image', 'mask', 'metadata'],
            output_names=['logits'],
            dynamic_axes={'signal': {0: 'B'}, 'image': {0: 'B'},
                          'mask': {0: 'B'}, 'metadata': {0: 'B'},
                          'logits': {0: 'B'}})
        logger.info(f"\nModèle ONNX exporté: {onnx_path}")

    logger.info("\n" + "=" * 70)
    logger.info("   ENTRAÎNEMENT TERMINÉ !")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
