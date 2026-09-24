import argparse
import copy
import json
import math
import os
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the AL-RNN tutorial model on one fixed Halvorsen trajectory."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="/export/home/klenkeit/training_data/single_halvorsen_constparams/just14_noisy_trajectories",
        help="Directory containing trajectories.npy.",
    )
    parser.add_argument("--trajectory-id", type=int, default=0)
    parser.add_argument("--split-idx", type=int, default=80000)
    parser.add_argument("--latent-dim", "-M", type=int, default=20)
    parser.add_argument("--pwl-units", "-P", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sequence-length", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batches-per-epoch", type=int, default=50)
    parser.add_argument("--teacher-forcing-interval", type=int, default=16)
    parser.add_argument("--start-lr", type=float, default=1e-3)
    parser.add_argument("--end-lr", type=float, default=1e-5)
    parser.add_argument("--ssi", type=int, default=20)
    parser.add_argument("--t-gen", type=int, default=10000)
    parser.add_argument("--t-transient", type=int, default=1000)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/halvorsen_just14_plain",
        help="Directory for model, metrics, and plots.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even if it is available.",
    )
    return parser.parse_args()


args = parse_args()

os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
os.environ["NUMEXPR_NUM_THREADS"] = str(args.cpu_threads)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange

from dataset import TimeSeriesDataset
from metrics import power_spectrum_error, state_space_divergence_binning


class AL_RNN(nn.Module):
    def __init__(self, M, P, N, phi_dim=0):
        super(AL_RNN, self).__init__()
        self.M = M
        self.P = P
        self.N = N
        self.phi_dim = phi_dim
        self.A, self.W, self.h = self.initialize_AWh_random()
        self.B = self.init_uniform((self.N, self.M))
        if self.phi_dim > 0:
            self.C = nn.Parameter(torch.zeros(self.M, self.phi_dim))
        else:
            self.register_parameter("C", None)

    def forward(self, z, phi_t=None):
        z_unactivated = torch.clone(z)
        if self.P > 0:
            z[:, -self.P :] = F.relu(z[:, -self.P :])
        out = self.A * z_unactivated + z @ self.W.t() + self.h
        if self.C is not None:
            if phi_t is None:
                raise ValueError("phi_t must be provided when phi_dim > 0")
            out = out + phi_t @ self.C.t()
        return out

    def initialize_AWh_random(self):
        A = nn.Parameter(torch.diagonal(self.normalized_positive_definite(self.M), 0))
        W = nn.Parameter(torch.randn(self.M, self.M) * 0.01)
        h = nn.Parameter(torch.zeros(self.M))
        return A, W, h

    def normalized_positive_definite(self, M):
        R = np.random.randn(M, M).astype(np.float32)
        K = np.matmul(R.T, R) / M + np.eye(M)
        lambda_max = np.max(np.abs(np.linalg.eigvals(K)))
        return torch.tensor(K / lambda_max).float()

    def init_uniform(self, shape):
        tensor = torch.empty(*shape)
        r = 1 / math.sqrt(shape[0])
        torch.nn.init.uniform_(tensor, -r, r)
        return nn.Parameter(tensor, requires_grad=True)


@torch.no_grad()
def predict_free_sequence(model, x, T, phi=None):
    b, N = x.size()
    Z = torch.empty(size=(T, b, model.M), device=x.device)
    z = x @ model.B
    z[:, 0:N] = x
    if model.phi_dim > 0 and phi is None:
        raise ValueError("phi must be provided when model.phi_dim > 0")
    if phi is not None and phi.dim() == 2:
        phi = phi.unsqueeze(1)
    for t in range(0, T):
        phi_t = phi[t] if phi is not None else None
        z = model(z, phi_t)
        Z[t] = z
    return Z.permute(1, 0, 2)


def predict_sequence_using_gtf(model, x, alpha, n_interleave, phi=None):
    x_ = x.permute(1, 0, 2)
    T, b, _ = x_.size()
    Z = torch.empty(size=(T, b, model.M), device=x.device)
    if model.phi_dim > 0 and phi is None:
        raise ValueError("phi must be provided when model.phi_dim > 0")
    phi_ = phi.permute(1, 0, 2) if phi is not None else None
    z = x_[0] @ model.B
    z = teacher_force(z, x_[0], alpha=1, N=model.N)
    for t in range(0, T):
        if (t % n_interleave == 0) and (t > 0):
            z = teacher_force(z, x_[t], alpha, N=model.N)
        phi_t = phi_[t] if phi_ is not None else None
        z = model(z, phi_t)
        Z[t] = z
    return Z.permute(1, 0, 2)


def teacher_force(z, x, alpha, N):
    z[:, :N] = alpha * x + (1 - alpha) * z[:, :N]
    return z


def train_sh(
    model,
    dataset,
    optimizer,
    scheduler,
    loss_fn,
    num_epochs,
    alpha,
    n_interleave,
    device,
    batches_per_epoch=50,
    ssi=25,
    use_best_model=True,
):
    model.train()
    best_model = copy.deepcopy(model)
    losses = []
    klx = []
    dh = []

    with trange(num_epochs, desc="Training Progress") as epochs:
        for e in epochs:
            epoch_losses = []
            for _ in range(batches_per_epoch):
                optimizer.zero_grad()
                x, y, s = dataset.sample_batch()
                x = x.to(device)
                y = y.to(device)
                s = s.to(device) if s is not None else None
                z_hat = predict_sequence_using_gtf(model, x, alpha, n_interleave, s)
                loss = loss_fn(z_hat[:, :, : model.N], y)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())

            scheduler.step()
            average_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            epochs.set_postfix(loss=average_epoch_loss)
            losses.append(average_epoch_loss)

            if e % ssi == 0:
                with torch.no_grad():
                    s_test = (
                        dataset.S.clone().detach()[0:10000, :].unsqueeze(1).to(device)
                        if dataset.S is not None
                        else None
                    )
                    x0_test = dataset.X.clone().detach()[0:1, :].to(device)
                    z_test = predict_free_sequence(model, x0_test, 10000, s_test)
                    z_test_obs = z_test[0, :, 0 : model.N].detach().cpu()
                    x_train_cpu = dataset.X.clone().detach().cpu()
                    klx.append(state_space_divergence_binning(z_test_obs, x_train_cpu))
                    dh.append(
                        power_spectrum_error(z_test_obs, x_train_cpu[0:10000, :])
                    )
                    if torch.argmin(torch.tensor(klx)) + 1 == len(torch.tensor(klx)):
                        best_model = copy.deepcopy(model)

    if use_best_model:
        model.load_state_dict(best_model.state_dict())
    return {"losses": losses, "dstsp": klx, "dh": dh}


def main():
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(args.cpu_threads)

    if torch.cuda.is_available() and not args.force_cpu:
        torch.cuda.set_device(args.gpu_id)
        device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")
    print("Using device:", device)
    print("CPU threads:", args.cpu_threads)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trajectories = np.load(Path(args.data_dir) / "trajectories.npy").astype(np.float32)
    X = trajectories[args.trajectory_id]
    X_train = X[: args.split_idx]
    X_test = X[args.split_idx :]
    phi_train = None
    phi_test = None
    phi_dim = 0

    T_train, N = X_train.shape
    T_test = X_test.shape[0]
    print("X_train:", X_train.shape, "X_test:", X_test.shape)

    model = AL_RNN(M=args.latent_dim, P=args.pwl_units, N=N, phi_dim=phi_dim).to(
        device
    )
    dataset = TimeSeriesDataset(
        X_train,
        external_inputs=phi_train,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
    )

    optimizer = torch.optim.RAdam(model.parameters(), lr=args.start_lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(
        optimizer, gamma=np.exp(np.log(args.end_lr / args.start_lr) / args.epochs)
    )
    loss_fn = nn.MSELoss()

    metrics = train_sh(
        model,
        dataset,
        optimizer,
        scheduler,
        loss_fn,
        num_epochs=args.epochs,
        alpha=1,
        n_interleave=args.teacher_forcing_interval,
        device=device,
        batches_per_epoch=args.batches_per_epoch,
        ssi=args.ssi,
    )

    model_path = output_dir / "model.pt"
    torch.save(model.state_dict(), model_path)

    X_test_torch = torch.tensor(X_test[:], device=device).unsqueeze(0)
    phi_test_torch = (
        torch.tensor(phi_test[: args.t_gen + args.t_transient], device=device).unsqueeze(
            1
        )
        if phi_test is not None
        else None
    )
    orbit = (
        predict_free_sequence(
            model,
            X_test_torch[:, 0, :],
            args.t_gen + args.t_transient,
            phi_test_torch,
        )
        .detach()
        .cpu()
        .numpy()[0][args.t_transient :, :]
    )

    dstsp = state_space_divergence_binning(
        torch.tensor(orbit[:, 0 : model.N]), X_test_torch[0, :, :].detach().cpu()
    )
    dh = power_spectrum_error(
        torch.tensor(orbit[:, 0 : model.N]),
        X_test_torch[0, 0 : args.t_gen, :].detach().cpu(),
    )
    orbit_finite = bool(np.isfinite(orbit).all())
    print("State space distance (Dstsp):", dstsp)
    print("Hellinger Distance (DH):", dh)
    print(
        "Orbit finite/min/max:",
        orbit_finite,
        float(np.nanmin(orbit)),
        float(np.nanmax(orbit)),
    )

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "args": vars(args),
                "T_train": int(T_train),
                "T_test": int(T_test),
                "N": int(N),
                "final_dstsp": float(dstsp),
                "final_dh": float(dh),
                "orbit_finite": orbit_finite,
                "orbit_min": float(np.nanmin(orbit)),
                "orbit_max": float(np.nanmax(orbit)),
                "training": metrics,
            },
            f,
            indent=2,
        )

    plt.figure(figsize=(7, 4))
    plt.plot(metrics["losses"], lw=2)
    plt.yscale("log")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png", dpi=180)
    plt.close()

    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        X_test[: args.t_gen, 0],
        X_test[: args.t_gen, 1],
        X_test[: args.t_gen, 2],
        label="Ground Truth",
        linewidth=1.0,
    )
    ax.plot(
        orbit[:, 0],
        orbit[:, 1],
        orbit[:, 2],
        label="Freely Generated",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_title(f"Dstsp={dstsp:.3f}, DH={dh:.3f}")
    ax.legend(loc="upper left")
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(output_dir / "free_rollout_3d.png", dpi=180)
    plt.close()

    np.save(output_dir / "orbit.npy", orbit)
    np.save(output_dir / "x_test.npy", X_test[: args.t_gen])
    print("Wrote outputs to:", output_dir)
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
