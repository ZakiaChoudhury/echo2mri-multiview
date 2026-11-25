"""
Multi-view echocardiography to cardiac MRI synthesis (2D).

Four echo views (A2CH, A3CH, A4CH, PSAX) are stacked as a 4-channel input.
A CNN encoder maps the echo input to the latent space of a pretrained
MRI autoencoder, and the pretrained MRI decoder generates the PSAX MRI.

Author: Zakia Zinat Choudhury, University of Otago
"""

import argparse
import csv
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Input,
                                     MaxPooling2D, UpSampling2D)
from tensorflow.keras.losses import MeanAbsoluteError
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam

IMG_SIZE = 128


# ---------------------------------------------------------
# Data loading and preprocessing
# ---------------------------------------------------------

def load_image_npy_folder(folder):
    """Load all .npy files in a folder, sorted by file name."""
    file_names = sorted(f for f in os.listdir(folder) if f.endswith(".npy"))
    images = np.array([np.load(os.path.join(folder, f)) for f in file_names])
    print(f"Loaded {len(file_names)} images from {folder}, shape: {images.shape}")
    return images, file_names


def reshape_images_gray(images, new_size=(IMG_SIZE, IMG_SIZE)):
    """Convert to single channel, min-max normalise, and resize."""
    out = []
    for img in images:
        if img.ndim == 2:
            img = img[..., np.newaxis]
        if img.shape[-1] > 1:
            img = np.mean(img, axis=-1, keepdims=True)

        lo, hi = img.min(), img.max()
        img = (img - lo) / (hi - lo) if hi > lo else np.zeros_like(img)

        img_uint8 = (img * 255).astype(np.uint8)
        pil = Image.fromarray(img_uint8.squeeze(), mode="L").resize(new_size)
        out.append(np.array(pil))
    out = np.array(out)
    print("Reshaped images shape:", out.shape)
    return out


def data_preprocessor_gray(images):
    """(N, H, W) uint8 -> (N, H, W, 1) float32 in [-1, 1]."""
    images = images.reshape((images.shape[0], IMG_SIZE, IMG_SIZE, 1)).astype("float32")
    return (images - 127.5) / 127.5


def split_data_two_modalities(data_a, data_b, split=0.2, seed=42):
    """Apply the same random train/validation split to paired echo and MRI."""
    assert len(data_a) == len(data_b), "Echo and MRI sample counts must match."
    rng = random.Random(seed)
    idx = list(range(len(data_a)))
    rng.shuffle(idx)
    n_val = int(len(data_a) * split)
    val_idx, train_idx = idx[:n_val], idx[n_val:]
    return data_a[train_idx], data_a[val_idx], data_b[train_idx], data_b[val_idx]


def load_4view_echo(folder_list, new_size=(IMG_SIZE, IMG_SIZE)):
    """
    Load four echo views and stack them as channels.
    All folders must contain the same number of files, with matching
    sorted order per patient/slice. Output: (N, 128, 128, 4).
    """
    assert len(folder_list) == 4, "Need exactly 4 echo folders (A2CH, A3CH, A4CH, PSAX)."
    views, base_names = [], None
    for i, folder in enumerate(folder_list):
        raw, names = load_image_npy_folder(folder)
        views.append(data_preprocessor_gray(reshape_images_gray(raw, new_size)))
        if i == 0:
            base_names = names
        elif len(names) != len(base_names):
            raise ValueError("Echo view folders have different numbers of files.")
    echo = np.concatenate(views, axis=-1)
    print("Multi-view echo shape:", echo.shape)
    return echo, base_names


# ---------------------------------------------------------
# Model
# ---------------------------------------------------------

def echo_encoder_block(echo_input):
    """CNN encoder for 4-channel echo. 128 -> 64 -> 32 -> 16 -> 8 (512 channels)."""
    x = echo_input
    for filters in (64, 128, 256, 512):
        x = Conv2D(filters, 3, activation="relu", padding="same")(x)
        x = BatchNormalization()(x)
        x = MaxPooling2D(2, padding="same")(x)
        x = Conv2D(filters, 3, activation="relu", padding="same")(x)
    return x


def build_decoder_from_ae(ae_model, verbose=True):
    """
    Extract the decoder from a pretrained MRI autoencoder.
    The bottleneck is the last MaxPooling2D before the first UpSampling2D.
    """
    first_up = next((i for i, l in enumerate(ae_model.layers)
                     if isinstance(l, UpSampling2D)), None)
    if first_up is None:
        raise ValueError("No UpSampling2D layer found in the autoencoder.")

    pools = [j for j in range(first_up) if isinstance(ae_model.layers[j], MaxPooling2D)]
    if not pools:
        raise ValueError("No MaxPooling2D layer found before the decoder.")

    bottleneck = ae_model.layers[pools[-1]]
    if verbose:
        print(f"Bottleneck layer: {bottleneck.name}, shape: {tuple(bottleneck.output.shape)}")

    dec_in = Input(shape=tuple(bottleneck.output.shape[1:]), name="decoder_input")
    x = dec_in
    for layer in ae_model.layers[pools[-1] + 1:]:
        if not isinstance(layer, tf.keras.layers.InputLayer):
            x = layer(x)
    return Model(dec_in, x, name="mri_decoder")


def build_echo2mri_model(decoder):
    echo_input = Input(shape=(IMG_SIZE, IMG_SIZE, 4), name="echo_input_4view")
    encoded = echo_encoder_block(echo_input)

    enc_shape = tuple(encoded.shape[1:])
    dec_shape = tuple(decoder.inputs[0].shape[1:])
    if enc_shape != dec_shape:
        raise ValueError(f"Encoder output {enc_shape} does not match decoder input "
                         f"{dec_shape}. Adjust echo_encoder_block or the pretrained AE.")

    return Model(echo_input, decoder(encoded), name="echo4view_to_psax_mri")


# ---------------------------------------------------------
# Loss: MAE + image gradient loss
# ---------------------------------------------------------

def gradient_loss(y_true, y_pred):
    dy_t, dx_t = tf.image.image_gradients(y_true)
    dy_p, dx_p = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.abs(dy_t - dy_p) + tf.abs(dx_t - dx_p))


def combined_loss(y_true, y_pred):
    mae = tf.reduce_mean(tf.abs(y_true - y_pred))
    return 0.9 * mae + 0.1 * gradient_loss(y_true, y_pred)


# ---------------------------------------------------------
# Callback: log test MAE and save predictions
# ---------------------------------------------------------

class Echo2MRICallback(tf.keras.callbacks.Callback):
    def __init__(self, echo_samples, mri_samples, save_dir, log_path, file_names,
                 save_every=100):
        super().__init__()
        self.echo_samples = echo_samples
        self.mri_samples = mri_samples
        self.save_dir = save_dir
        self.log_path = log_path
        self.file_names = file_names
        self.save_every = save_every
        self.mae = MeanAbsoluteError()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        pred = self.model.predict(self.echo_samples, verbose=0)
        avg_mae = float(np.mean([self.mae(self.mri_samples[i:i + 1], pred[i:i + 1]).numpy()
                                 for i in range(len(self.mri_samples))]))

        with open(self.log_path, "a", newline="") as f:
            csv.writer(f, delimiter="\t").writerow(
                [epoch + 1, f"{logs.get('loss', 0.0):.4f}",
                 f"{logs.get('val_loss', 0.0):.4f}", f"{avg_mae:.4f}"])

        if (epoch + 1) % self.save_every:
            return

        out_dir = os.path.join(self.save_dir, f"epoch_{epoch + 1}")
        npy_dir = os.path.join(out_dir, "npy_predictions")
        os.makedirs(npy_dir, exist_ok=True)

        for j in range(len(self.echo_samples)):
            fig, axs = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
            for ax, img, title in zip(
                    axs,
                    (self.echo_samples[j, :, :, 0], self.mri_samples[j], pred[j]),
                    ("Echo (view 1)", "Ground-truth PSAX MRI", "Predicted PSAX MRI")):
                ax.imshow(img.squeeze(), cmap="gray")
                ax.set_title(title)
                ax.axis("off")
            plt.tight_layout()
            name = os.path.splitext(os.path.basename(self.file_names[j]))[0]
            plt.savefig(os.path.join(out_dir, f"{name}.png"))
            plt.close(fig)
            np.save(os.path.join(npy_dir, f"{name}.npy"), pred[j].squeeze())


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

VIEWS = ("A2ch", "A3ch", "A4ch", "PSAX")


def parse_args():
    p = argparse.ArgumentParser(description="Train multi-view echo -> MRI model.")
    p.add_argument("--data_dir", required=True,
                   help="Folder with Echo_<view>, Echo_<view>_Test, MRI_Train, MRI_Test.")
    p.add_argument("--pretrained_ae", required=True,
                   help="Path to the pretrained MRI autoencoder (.h5).")
    p.add_argument("--out_dir", default="results")
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val_split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_every", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, "loss.txt")
    with open(log_path, "a", newline="") as f:
        csv.writer(f, delimiter="\t").writerow(["Epoch", "Train_Loss", "Val_Loss", "MAE_Test"])

    d = args.data_dir
    echo_train, _ = load_4view_echo([os.path.join(d, f"Echo_{v}") for v in VIEWS])
    echo_test, test_names = load_4view_echo([os.path.join(d, f"Echo_{v}_Test") for v in VIEWS])

    mri_train = data_preprocessor_gray(reshape_images_gray(load_image_npy_folder(os.path.join(d, "MRI_Train"))[0]))
    mri_test = data_preprocessor_gray(reshape_images_gray(load_image_npy_folder(os.path.join(d, "MRI_Test"))[0]))

    x_tr, x_val, y_tr, y_val = split_data_two_modalities(
        echo_train, mri_train, split=args.val_split, seed=args.seed)
    print("Train:", x_tr.shape, y_tr.shape)
    print("Val:  ", x_val.shape, y_val.shape)
    print("Test: ", echo_test.shape, mri_test.shape)

    decoder = build_decoder_from_ae(load_model(args.pretrained_ae, compile=False))
    model = build_echo2mri_model(decoder)
    model.compile(optimizer=Adam(learning_rate=args.lr), loss=combined_loss)
    model.summary()

    cb = Echo2MRICallback(echo_test, mri_test, args.out_dir, log_path, test_names,
                          save_every=args.save_every)
    model.fit(x_tr, y_tr, batch_size=args.batch_size, epochs=args.epochs, shuffle=True,
              validation_data=(x_val, y_val), callbacks=[cb])
    weights_path = os.path.join(args.out_dir, "echo2mri.weights.h5")
    model.save_weights(weights_path)
    print("Saved model weights to", weights_path)


if __name__ == "__main__":
    main()
