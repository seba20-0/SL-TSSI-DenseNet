from enum import IntEnum
import os
import json
import numpy as np
import tensorflow as tf
from config import AUTSL_INPUT_HEIGHT, MEJIAPEREZ_INPUT_HEIGHT, WLASL100_INPUT_HEIGHT, MLR511_INPUT_HEIGHT
from data_augmentation import RandomFlip, RandomScale, RandomShift, RandomRotation, RandomSpeed
from preprocessing import Center, CenterAtFirstFrame2D, FillBlueWithAngle, PadIfLessThan, RemoveZ, ResizeIfMoreThan, TranslationScaleInvariant
import tensorflow_datasets as tfds


def _read_lines(path):
    with open(path, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def _load_npz_sample(path):
    data = np.load(path, allow_pickle=True)
    pose = data['pose'].astype(np.float32)
    # label can be provided or inferred from filename later
    label = int(data['label']) if 'label' in data else -1
    return pose, label


def _build_npz_split(paths, label_map=None):
    # Create tensors of file paths and map to (pose, label)
    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    def _py_loader(p):
        p = p.decode('utf-8')
        pose, label = _load_npz_sample(p)
        # infer label from filename if needed
        if label == -1 and label_map is not None:
            base = os.path.basename(p)
            # expected like user01_G01_R01.npz or contains GXX
            gesture_token = None
            for k in label_map.keys():
                if k in base:
                    gesture_token = k
                    break
            if gesture_token is not None:
                label = int(label_map[gesture_token])
            else:
                raise ValueError(f"Cannot infer label for {p}. Provide label in npz or ensure filename contains a known gesture code.")
        return pose, np.int32(label)

    def _tf_loader(p):
        pose, label = tf.numpy_function(_py_loader, [p], [tf.float32, tf.int32])
        pose.set_shape([None, None, 2])  # (frames, joints, 2)
        label.set_shape([])
        return pose, label

    ds = path_ds.map(_tf_loader, num_parallel_calls=tf.data.AUTOTUNE)
    return ds


class LayerType(IntEnum):
    Augmentation = 1
    Normalization = 2
    Data = 3


LayerDict = {
    'random_speed': {
        'type': LayerType.Augmentation,
        'layer': RandomSpeed(min_frames=40, max_frames=128, seed=5),
    },
    'random_rotation': {
        'type': LayerType.Augmentation,
        'layer': RandomRotation(factor=15.0, min_value=0.0, max_value=1.0, seed=4),
    },
    'random_flip': {
        'type': LayerType.Augmentation,
        'layer': RandomFlip("horizontal", min_value=0.0, max_value=1.0, seed=3),
    },
    'random_scale': {
        'type': LayerType.Augmentation,
        'layer': RandomScale(min_value=0.0, max_value=1.0, seed=1),
    },
    'random_shift': {
        'type': LayerType.Augmentation,
        'layer': RandomShift(min_value=0.0, max_value=1.0, seed=2)
    },
    'invariant_frame': {
        'type': LayerType.Normalization,
        'layer': TranslationScaleInvariant(level="frame")
    },
    'invariant_joint': {
        'type': LayerType.Normalization,
        'layer': TranslationScaleInvariant(level="joint")
    },
    'center': {
        'type': LayerType.Normalization,
        'layer': Center(around_index=0)
    },
    'center_at_first': {
        'type': LayerType.Normalization,
        'layer': CenterAtFirstFrame2D(around_index=0)
    },
    'train_resize': {
        'type': LayerType.Normalization,
        'layer': ResizeIfMoreThan(frames=100)
    },
    'test_resize': {
        'type': LayerType.Normalization,
        'layer': ResizeIfMoreThan(frames=100)
    },
    'pad': {
        'type': LayerType.Normalization,
        'layer': PadIfLessThan(frames=100)
    },
    'angle': {
        'type': LayerType.Data,
        'layer': FillBlueWithAngle(x_channel=0, y_channel=1, scale_to=[0, 1]),
    },
    'norm_imagenet': {
        'type': LayerType.Normalization,
        'layer': tf.keras.layers.Normalization(axis=-1,
                                               mean=[0.485, 0.456, 0.406],
                                               variance=[0.052441, 0.050176, 0.050625]),
    },
    # placeholder for layer, mean and variance are obtained dinamically
    'norm': {
        'type': LayerType.Normalization,
        'layer': tf.keras.layers.Normalization(axis=-1,
                                               mean=[248.08896, 246.56985, 0.],
                                               variance=[9022.948, 17438.518, 0.])
    }
}


# Augmentation Order = ['speed', 'rotation', 'flip', 'scale', 'shift']
PipelineDict = {
    'default': {
        'train': ['random_speed', 'random_flip', 'random_scale', 'train_resize', 'pad'],
        'test': ['test_resize', 'pad']
    },
    'default_center_at_0': {
        'train': ['random_speed', 'random_flip', 'random_scale', 'center_at_first', 'train_resize', 'pad'],
        'test': ['center_at_first', 'test_resize', 'pad']
    },
    'default_center': {
        'train': ['center', 'random_speed', 'train_resize', 'pad'],
        'test': ['center', 'test_resize', 'pad']
    },
    'default_angle': {
        'train': ['angle', 'random_speed', 'train_resize', 'pad'],
        'test': ['angle', 'test_resize', 'pad']
    },
    'invariant_frame': {
        'train': ['random_speed', 'train_resize', 'invariant_frame', 'pad'],
        'test': ['test_resize', 'invariant_frame', 'pad']
    }
}


def generate_train_dataset(dataset,
                           train_map_fn,
                           repeat=False,
                           batch_size=32,
                           buffer_size=5000,
                           deterministic=False):
    # shuffle, map and batch dataset
    if deterministic:
        train_dataset = dataset \
            .shuffle(buffer_size) \
            .map(train_map_fn) \
            .batch(batch_size)
    else:
        train_dataset = dataset \
            .shuffle(buffer_size) \
            .map(train_map_fn,
                 num_parallel_calls=tf.data.AUTOTUNE,
                 deterministic=False) \
            .batch(batch_size) \
            .prefetch(tf.data.AUTOTUNE)

    if repeat:
        train_dataset = train_dataset.repeat()

    return train_dataset


def generate_test_dataset(dataset,
                          test_map_fn,
                          batch_size=32):
    # batch dataset
    max_element_length = 200
    bucket_boundaries = list(range(1, max_element_length))
    bucket_batch_sizes = [batch_size] * max_element_length
    ds = dataset.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[0],
        bucket_boundaries=bucket_boundaries,
        bucket_batch_sizes=bucket_batch_sizes,
        no_padding=True)

    # map dataset
    dataset = ds \
        .map(test_map_fn,
             num_parallel_calls=tf.data.AUTOTUNE,
             deterministic=False) \
        .cache()

    return dataset


def build_pipeline(pipeline, exclude_augmentation=False, split="train"):
    # normalization: None, str or list
    if pipeline == None:
        layers = []
    elif type(pipeline) is str:
        items = [LayerDict[name] for name in PipelineDict[pipeline][split]]
        if exclude_augmentation:
            items = [item for item in items if
                     item["type"] != LayerType.Augmentation]
        layers = [item["layer"] for item in items]
    else:
        raise Exception("Pipeline " +
                        str(pipeline) + " not found")
    pipeline = tf.keras.Sequential(layers, name="normalization")
    return pipeline


class Dataset():
    def __init__(self, name: str, concat_validation_to_train=False):
        global LayerDict

        tfds_names = {"autsl_tssi", "mejiaperez_tssi", "wlasl100_tssi"}

        if name in tfds_names:
            # obtain dataset from TFDS-like archives in ./datasets
            ds, info = tfds.load(name, data_dir="datasets", with_info=True)

            # generate train dataset
            if concat_validation_to_train:
                ds["train"] = ds["train"].concatenate(ds["validation"])

            # preprocess labels -> (pose, one_hot_label)
            @tf.function
            def label_to_one_hot(item):
                one_hot_label = tf.one_hot(item["label"],
                                           info.features['label'].num_classes)
                return item["pose"], one_hot_label
            ds["train"] = ds["train"].map(label_to_one_hot).cache()
            ds["validation"] = ds["validation"].map(label_to_one_hot)

            # obtain characteristics of the dataset
            num_train_examples = ds["train"].cardinality()
            num_val_examples = ds["validation"].cardinality()
            if "test" in ds.keys():
                ds["test"] = ds["test"].map(label_to_one_hot)
                num_test_examples = ds["test"].cardinality()
            else:
                num_test_examples = tf.constant(0, dtype=tf.int64)
            num_total_examples = num_train_examples + num_val_examples + num_test_examples

            if name == "autsl_tssi":
                input_height = AUTSL_INPUT_HEIGHT
            elif name == "mejiaperez_tssi":
                input_height = MEJIAPEREZ_INPUT_HEIGHT
            elif name == "wlasl100_tssi":
                input_height = WLASL100_INPUT_HEIGHT

            LayerDict["train_resize"]["layer"] = ResizeIfMoreThan(frames=input_height)
            LayerDict["test_resize"]["layer"] = ResizeIfMoreThan(frames=input_height)
            LayerDict["pad"]["layer"] = PadIfLessThan(frames=input_height)

            self.ds = ds
            self.name = name
            self.num_train_examples = int(num_train_examples.numpy()) if hasattr(num_train_examples, 'numpy') else int(num_train_examples)
            self.num_val_examples = int(num_val_examples.numpy()) if hasattr(num_val_examples, 'numpy') else int(num_val_examples)
            self.num_test_examples = int(num_test_examples.numpy()) if hasattr(num_test_examples, 'numpy') else int(num_test_examples)
            self.num_total_examples = int(num_total_examples.numpy()) if hasattr(num_total_examples, 'numpy') else int(num_total_examples)
            self.input_height = input_height
            self.input_width = info.features['pose'].shape[1]
            self.num_classes = info.features['label'].num_classes

        elif name == "mlr511_npz":
            # Expect prepared NPZ dataset under ./datasets/mlr511_npz
            root = os.path.join("datasets", "mlr511_npz")
            splits_dir = os.path.join(root, "splits")
            samples_dir = os.path.join(root, "samples")
            if not os.path.isdir(root) or not os.path.isdir(splits_dir) or not os.path.isdir(samples_dir):
                raise Exception("Expected NPZ dataset at 'datasets/mlr511_npz' with 'samples' and 'splits' subfolders.")

            # label map is optional if labels are embedded in npz
            label_map_path = os.path.join(root, "label_map.json")
            label_map = None
            if os.path.isfile(label_map_path):
                with open(label_map_path, 'r', encoding='utf-8') as f:
                    label_map = json.load(f)

            def to_abs(p):
                return p if os.path.isabs(p) else os.path.join(root, p)

            train_list = [to_abs(p) for p in _read_lines(os.path.join(splits_dir, "train.txt"))]
            val_list = [to_abs(p) for p in _read_lines(os.path.join(splits_dir, "validation.txt"))]
            test_txt = os.path.join(splits_dir, "test.txt")
            test_list = [to_abs(p) for p in _read_lines(test_txt)] if os.path.isfile(test_txt) else []

            # build datasets yielding (pose, one_hot_label)
            raw_train = _build_npz_split(train_list, label_map)
            raw_val = _build_npz_split(val_list, label_map)
            raw_test = _build_npz_split(test_list, label_map) if len(test_list) > 0 else None

            # infer counts and classes
            num_classes = 0
            if label_map is not None:
                num_classes = int(max(label_map.values())) + 1
            else:
                # scan labels from the training set once
                labels = []
                for p in train_list:
                    _, lab = _load_npz_sample(p)
                    if lab != -1:
                        labels.append(int(lab))
                if len(labels) == 0:
                    raise Exception("Could not determine number of classes. Provide label_map.json or include 'label' in each npz.")
                num_classes = max(labels) + 1

            def to_one_hot_ds(ds):
                if ds is None:
                    return None
                return ds.map(lambda pose, y: (pose, tf.one_hot(y, num_classes)),
                              num_parallel_calls=tf.data.AUTOTUNE)

            ds = {
                "train": to_one_hot_ds(raw_train),
                "validation": to_one_hot_ds(raw_val)
            }
            if raw_test is not None:
                ds["test"] = to_one_hot_ds(raw_test)

            # input metadata
            # read one example to determine joints width
            sample_pose, _ = _load_npz_sample(train_list[0])
            input_width = int(sample_pose.shape[1])

            input_height = MLR511_INPUT_HEIGHT

            LayerDict["train_resize"]["layer"] = ResizeIfMoreThan(frames=input_height)
            LayerDict["test_resize"]["layer"] = ResizeIfMoreThan(frames=input_height)
            LayerDict["pad"]["layer"] = PadIfLessThan(frames=input_height)

            # set instance fields
            self.ds = ds
            self.name = name
            self.num_train_examples = len(train_list)
            self.num_val_examples = len(val_list)
            self.num_test_examples = len(test_list)
            self.num_total_examples = self.num_train_examples + self.num_val_examples + self.num_test_examples
            self.input_height = input_height
            self.input_width = input_width
            self.num_classes = num_classes
            # keep meta for filtered evaluation
            self._npz_meta = {
                'root': root,
                'label_map': label_map,
                'train_list': train_list,
                'val_list': val_list,
                'test_list': test_list,
            }

        else:
            raise Exception("Dataset " + name + " not found.")

    def get_testing_set_for_users(self,
                                  user_ids,
                                  batch_size=32,
                                  pipeline="default"):
        if self.name != "mlr511_npz":
            raise Exception("get_testing_set_for_users is only available for 'mlr511_npz'.")

        # normalize user ids to 'userXX' strings
        norm_users = []
        for u in user_ids:
            if isinstance(u, int):
                norm_users.append(f"user{u:02d}")
            else:
                s = str(u)
                if s.startswith("user"):
                    norm_users.append(s)
                else:
                    try:
                        norm_users.append(f"user{int(s):02d}")
                    except Exception:
                        raise Exception(f"Unrecognized user id: {u}")

        # filter paths by prefix
        test_list = self._npz_meta.get('test_list', [])
        if not test_list:
            raise Exception("No test split available. Re-run dataset preparation with a test split.")

        sel = []
        for p in test_list:
            base = os.path.basename(p)
            # expected format userXX_...
            for nu in norm_users:
                if base.startswith(nu + "_"):
                    sel.append(p)
                    break
        if not sel:
            raise Exception(f"No test samples found for users: {norm_users}")

        # rebuild dataset from selected files
        label_map = self._npz_meta.get('label_map')
        raw = _build_npz_split(sel, label_map)

        def to_one_hot(pose, y):
            return pose, tf.one_hot(y, self.num_classes)

        ds = raw.map(to_one_hot, num_parallel_calls=tf.data.AUTOTUNE)

        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        @tf.function
        def test_map_fn(batch_x, batch_y):
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        return generate_test_dataset(ds, test_map_fn, batch_size=batch_size)

    def get_training_set(self,
                         batch_size=32,
                         buffer_size=5000,
                         repeat=False,
                         deterministic=False,
                         augmentation=True,
                         pipeline="default"):
        # define pipeline
        exclude_augmentation = not augmentation
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation, "train")

        # define the train map function
        @tf.function
        def train_map_fn(x, y):
            batch = tf.expand_dims(x, axis=0)
            # batch = RemoveZ()(batch)
            batch = preprocessing_pipeline(batch, training=True)
            x = tf.ensure_shape(
                batch[0], [self.input_height, self.input_width, 3])
            return x, y

        train_ds = self.ds["train"]
        dataset = generate_train_dataset(train_ds,
                                         train_map_fn,
                                         repeat=repeat,
                                         batch_size=batch_size,
                                         buffer_size=buffer_size,
                                         deterministic=deterministic)

        return dataset

    def get_validation_set(self,
                           batch_size=32,
                           pipeline="default"):
        # define pipeline
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            # batch_x = RemoveZ()(batch_x)
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        val_ds = self.ds["validation"]
        dataset = generate_test_dataset(val_ds,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset

    def get_testing_set(self,
                        batch_size=32,
                        pipeline="default"):
        if "test" not in self.ds.keys():
            return None

        # define pipeline
        preprocessing_pipeline = build_pipeline(
            pipeline, exclude_augmentation=True, split="test")

        # define the val map function
        @tf.function
        def test_map_fn(batch_x, batch_y):
            # batch_x = RemoveZ()(batch_x)
            batch_x = preprocessing_pipeline(batch_x)
            return batch_x, batch_y

        test_ds = self.ds["test"]
        dataset = generate_test_dataset(test_ds,
                                        test_map_fn,
                                        batch_size=batch_size)

        return dataset
