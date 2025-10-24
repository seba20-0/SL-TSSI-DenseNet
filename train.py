import argparse
from config import RANDOM_SEED
from dataset import Dataset
import numpy as np
import tensorflow as tf
from model import build_densenet121_model
from optimizer import build_sgd_optimizer
from utils import str2bool
import os
import json
import time

dataset = None


def run_experiment(config=None, log_to_wandb=True, verbose=0):
    global dataset

    tf.keras.backend.clear_session()
    tf.keras.utils.set_random_seed(RANDOM_SEED)

    # check if config was provided
    if config is None:
        raise Exception("Not config provided.")
    print("[INFO] Configuration:", config, "\n")

    # check if dataset was provided
    if dataset is None:
        raise Exception("Dataset not provided.")

    # generate train dataset
    deterministic = config['augmentation']
    train_dataset = dataset.get_training_set(
        batch_size=config['batch_size'],
        buffer_size=dataset.num_train_examples,
        repeat=False,
        deterministic=deterministic,
        augmentation=config['augmentation'],
        pipeline=config['pipeline'])

    # generate val dataset
    if config['validate']:
        validation_dataset = dataset.get_validation_set(
            batch_size=config['batch_size'],
            pipeline=config['pipeline'])
    else:
        validation_dataset = None

    # describe dataset distribution
    print("[INFO] Dataset Total examples:", dataset.num_total_examples)
    print("[INFO] Dataset Training examples:", dataset.num_train_examples)
    print("[INFO] Dataset Validation examples:", dataset.num_val_examples)

    # describe input shape
    input_shape = [dataset.input_height, dataset.input_width, 3]
    print("[INFO] Input Shape:", input_shape)

    # setup optimizer
    optimizer = build_sgd_optimizer(initial_learning_rate=config['initial_learning_rate'],
                                    maximal_learning_rate=config['maximal_learning_rate'],
                                    momentum=config['momentum'],
                                    nesterov=config['nesterov'],
                                    step_size=config['step_size'],
                                    weight_decay=config['weight_decay'])
    
    # setup model
    model = build_densenet121_model(input_shape=input_shape,
                                    dropout=config['dropout'],
                                    optimizer=optimizer,
                                    pretraining=config['pretraining'],
                                    num_classes=dataset.num_classes)

    # print summary of the model
    model.summary()

    # setup callbacks
    callbacks = []
    if log_to_wandb:
        # Import lazily to allow running without wandb installed
        import wandb
        from wandb.keras import WandbCallback, WandbModelCheckpoint
        wandb_callback = WandbCallback(
            monitor="val_top_1",
            mode="max",
            save_model=False
        )
        callbacks.append(wandb_callback)
        
        if config['save_weights']:
            wandb_model_checkpoint = WandbModelCheckpoint(
                f"artifacts/{wandb.run.id}/weights",
                save_weights_only=True,
                save_freq=config['save_freq'],
                verbose=1
            )
            callbacks.append(wandb_model_checkpoint)
        
    # compute model params and size
    total_params = model.count_params()
    bytes_total = 0
    for v in model.weights:
        try:
            bytes_total += v.shape.num_elements() * tf.as_dtype(v.dtype).size
        except Exception:
            pass
    size_mb = bytes_total / (1024 * 1024)
    print(f"[INFO] Model parameters: {total_params}")
    print(f"[INFO] Approx model size (MB): {size_mb:.2f}")

    # train model
    t0 = time.time()
    hist = model.fit(train_dataset,
                     epochs=config['num_epochs'],
                     verbose=verbose,
                     validation_data=validation_dataset,
                     callbacks=callbacks)
    t1 = time.time()
    train_time_sec = t1 - t0
    sec_per_epoch = train_time_sec / config['num_epochs'] if config['num_epochs'] > 0 else train_time_sec
    print(f"[INFO] Training time (s): {train_time_sec:.2f} | per epoch: {sec_per_epoch:.2f}")

    # optionally save weights when not using W&B
    if config['save_weights'] and not log_to_wandb:
        try:
            out_dir = os.path.join("artifacts", "local", "weights")
            os.makedirs(out_dir, exist_ok=True)
            out_path = os.path.join(out_dir, "ckpt")
            model.save_weights(out_path)
            print(f"[INFO] Saved weights to {out_path}")
        except Exception as e:
            print(f"[WARN] Could not save weights locally: {e}")

    # save local report
    try:
        os.makedirs("artifacts/reports", exist_ok=True)
        report = {
            'config': dict(config),
            'model': {
                'total_params': int(total_params),
                'approx_size_mb': float(f"{size_mb:.4f}")
            },
            'training': {
                'total_time_sec': float(f"{train_time_sec:.4f}"),
                'sec_per_epoch': float(f"{sec_per_epoch:.4f}")
            }
        }
        with open("artifacts/reports/train_report.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        print("[INFO] Saved training report to artifacts/reports/train_report.json")
    except Exception as e:
        print(f"[WARN] Could not save training report: {e}")

    # get the logs of the model
    return hist


def agent_fn(config, project, entity, verbose=0):
    import wandb
    wandb.init(entity=entity, project=project, config=config,
               reinit=True, settings=wandb.Settings(code_dir="."))
    _ = run_experiment(config=wandb.config, log_to_wandb=True, verbose=verbose)
    wandb.finish()


def main(args):
    global dataset
    
    if args.concat_validation_to_train:
        validate = False
    else:
        validate = True

    dataset = Dataset(args.dataset, args.concat_validation_to_train)
    steps_per_epoch = np.ceil(dataset.num_train_examples / args.batch_size)
    config = {
        'pretraining': args.pretraining,
        'dropout': args.dropout,

        'initial_learning_rate': args.lr_min,
        'maximal_learning_rate': args.lr_max,
        'momentum': 0.9,
        'nesterov': True,
        'weight_decay': args.weight_decay,
        'step_size': int(args.num_epochs / 2) * steps_per_epoch,

        'augmentation': args.augmentation,
        'batch_size': args.batch_size,
        'pipeline': args.pipeline,
        'num_epochs': args.num_epochs,
        
        'save_weights': args.save_weights,
        'save_freq': int(args.num_epochs * steps_per_epoch),
        'validate': validate
    }

    project_name = args.dataset + "_" + args.project
    if args.use_wandb:
        agent_fn(config=config, entity=args.entity, project=project_name, verbose=2)
    else:
        _ = run_experiment(config=config, log_to_wandb=False, verbose=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--entity', type=str,
                        help='Entity', default='davidlainesv')
    parser.add_argument('--project', type=str,
                        help='Project name', default='training')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset', default='wlasl100_tssi')
    parser.add_argument('--use_wandb', type=str2bool,
                        help='Log to Weights & Biases', default=True)
    parser.add_argument('--concat_validation_to_train', type=str2bool,
                        help='Add validation set to training set', default=False)
    parser.add_argument('--save_weights', type=str2bool,
                        help='Save weights at last epoch', default=False)

    parser.add_argument('--pretraining', type=str2bool,
                        help='Add pretraining', default=False)
    parser.add_argument('--dropout', type=float,
                        help='Dropout at the final layer', default=0)
    
    parser.add_argument('--lr_min', type=float,
                        help='Minimum learning rate', default=0.001)
    parser.add_argument('--lr_max', type=float,
                        help='Minimum learning rate', default=0.01)
    parser.add_argument('--weight_decay', type=float,
                        help='Weight decay', default=0)
    
    parser.add_argument('--augmentation', type=str2bool,
                        help='Add augmentation', default=False)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size of training and testing', default=64)
    parser.add_argument('--pipeline', type=str,
                        help='Pipeline', default="default")
    parser.add_argument('--num_epochs', type=int,
                        help='Number of epochs', default=24)

    args = parser.parse_args()

    print(args)

    main(args)
