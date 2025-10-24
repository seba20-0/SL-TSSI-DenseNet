import argparse
import time
import json
import os
from config import RANDOM_SEED
from dataset import Dataset
import numpy as np
import tensorflow as tf
from model import build_densenet121_model
from optimizer import build_sgd_optimizer
from utils import str2bool

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

    # generate testing dataset
    testing_dataset = dataset.get_testing_set(
        batch_size=config['batch_size'],
        pipeline=config['pipeline'])

    # describe dataset distribution
    print("[INFO] Dataset Testing examples:", dataset.num_test_examples)

    # describe input shape
    input_shape = [dataset.input_height, dataset.input_width, 3]
    print("[INFO] Input Shape:", input_shape)

    # placeholder optimizer
    optimizer = build_sgd_optimizer()
    
    # placeholder model
    model = build_densenet121_model(input_shape=input_shape,
                                    dropout=0,
                                    optimizer=optimizer,
                                    pretraining=False,
                                    num_classes=dataset.num_classes)
    
    # load weights
    model.load_weights(config['weights_dir'])

    # print summary of the model
    model.summary()

    # Option A: full test evaluate
    if config.get('per_user_eval') is None:
        scores = model.evaluate(testing_dataset, return_dict=True)
        return scores
    else:
        # Option B: per-user evaluation (mlr511_npz only)
        users = config['per_user_eval']
        results = {}
        for u in users:
            ds_user = dataset.get_testing_set_for_users([u], batch_size=config['batch_size'], pipeline=config['pipeline'])
            total = 0
            correct = 0
            t0 = time.time()
            for batch_x, batch_y in ds_user:
                preds = model.predict_on_batch(batch_x)
                y_true = np.argmax(batch_y.numpy(), axis=1)
                y_pred = np.argmax(preds, axis=1)
                correct += int((y_true == y_pred).sum())
                total += int(y_true.shape[0])
            t1 = time.time()
            acc = correct / total if total > 0 else 0.0
            inf_time_per_sample = (t1 - t0) / total if total > 0 else 0.0
            results[str(u)] = {
                'samples': total,
                'accuracy_top1': acc,
                'inference_time_sec_per_sample': inf_time_per_sample
            }
            print(f"[RESULT] user {u}: N={total}, acc={acc:.4f}, infer_time={inf_time_per_sample*1000:.2f} ms/sample")

        # persist report
        try:
            os.makedirs("artifacts/reports", exist_ok=True)
            with open("artifacts/reports/test_per_user_report.json", "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
            print("[INFO] Saved per-user test report to artifacts/reports/test_per_user_report.json")
        except Exception as e:
            print(f"[WARN] Could not save per-user report: {e}")

        return results


def agent_fn(config, project, entity, verbose=0):
    import wandb
    wandb.init(entity=entity, project=project, config=config,
               reinit=True, settings=wandb.Settings(code_dir="."))
    _ = run_experiment(config=wandb.config, log_to_wandb=True, verbose=verbose)
    wandb.finish()


def main(args):
    global dataset

    dataset = Dataset(args.dataset)
    config = {
        'batch_size': args.batch_size,
        'pipeline': args.pipeline,
        'weights_dir': args.weights_dir
    }
    if args.test_users:
        # Support passing comma separated ids like "1,8,11" or userXX list
        raw = [s.strip() for s in args.test_users.split(',') if s.strip()]
        norm = []
        for r in raw:
            if r.startswith('user'):
                norm.append(r)
            else:
                try:
                    norm.append(int(r))
                except Exception:
                    norm.append(r)
        config['per_user_eval'] = norm

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
                        help='Project name', default='testing')
    parser.add_argument('--dataset', type=str,
                        help='Name of dataset', default='wlasl100_tssi')
    parser.add_argument('--use_wandb', type=str2bool,
                        help='Log to Weights & Biases', default=True)
    parser.add_argument('--weights_dir', type=str,
                        help='Dir to weights', required=True)

    parser.add_argument('--batch_size', type=int,
                        help='Batch size of training and testing', default=64)
    parser.add_argument('--pipeline', type=str,
                        help='Pipeline', default="default")
    parser.add_argument('--test_users', type=str,
                        help='Comma-separated test user ids (e.g., 1,8,11 or user01,user08,user11). Only for mlr511_npz.', default='')

    args = parser.parse_args()

    print(args)

    main(args)
