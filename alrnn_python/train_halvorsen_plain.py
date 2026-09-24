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
    parser.add_argument(
        "--multi-trajectory",
        action="store_true",
        help="Use all trajectories in trajectories.npy instead of one trajectory.",
    )
    parser.add_argument(
        "--use-phi",
        action="store_true",
        help="Load trajectory_phi.npy and feed phi through the C phi_t term.",
    )
    parser.add_argument(
        "--no-balanced-phi",
        action="store_true",
        help="Disable balanced per-phi sampling for multi-trajectory data.",
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
        "--eval-only",
        action="store_true",
        help="Skip training and evaluate output-dir/model.pt.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Disable CUDA even if it is available.",
    )
    return parser.parse_args()


def json_ready(value):
    if isinstance(value, dict):
        return {key: json_ready(val) for key, val in value.items()}
    if isinstance(value, list):
        return [json_ready(val) for val in value]
    if isinstance(value, tuple):
        return [json_ready(val) for val in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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

from metrics import power_spectrum_error, state_space_divergence_binning


class SingleTrajectoryDataset:
    def __init__(self, data, external_inputs=None, sequence_length=200, batch_size=16):
        self.X = torch.tensor(data, dtype=torch.float32)
        self.total_time_steps = self.X.shape[0]
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        if external_inputs is not None:
            self.S = torch.tensor(external_inputs, dtype=torch.float32)
            assert self.X.size(0) == self.S.size(0)
        else:
            self.S = None

    def __len__(self):
        return self.total_time_steps - self.sequence_length - 1

    def __getitem__(self, t):
        x = self.X[t : t + self.sequence_length, :]
        y = self.X[t + 1 : t + self.sequence_length + 1, :]
        if self.S is None:
            return x, y, None
        s = self.S[t : t + self.sequence_length, :]
        return x, y, s

    def sample_batch(self):
        X, Y, S = [], [], []
        for _ in range(self.batch_size):
            idx = np.random.randint(0, len(self))
            x, y, s = self[idx]
            X.append(x)
            Y.append(y)
            S.append(s)
        if S[0] is None:
            return torch.stack(X), torch.stack(Y), None
        return torch.stack(X), torch.stack(Y), torch.stack(S)

    def eval_reference(self, length=10000, trajectory_index=0):
        x_ref = self.X[:length]
        if self.S is None:
            return x_ref, None
        return x_ref, self.S[:length]


class MultiTrajectoryDataset:
    def __init__(
        self,
        data,
        external_inputs=None,
        phi_constants=None,
        sequence_length=200,
        batch_size=20,
        split_idx=80000,
        balanced_phi=True,
    ):
        self.X = torch.tensor(data[:, :split_idx, :], dtype=torch.float32)
        self.X_test = torch.tensor(data[:, split_idx:, :], dtype=torch.float32)
        self.num_trajectories, self.total_time_steps, _ = self.X.shape
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.balanced_phi = balanced_phi

        if external_inputs is not None:
            self.S = torch.tensor(external_inputs[:, :split_idx, :], dtype=torch.float32)
            self.S_test = torch.tensor(
                external_inputs[:, split_idx:, :], dtype=torch.float32
            )
            assert self.X.shape[:2] == self.S.shape[:2]
        else:
            self.S = None
            self.S_test = None

        if phi_constants is None:
            self.phi_constants = np.arange(self.num_trajectories)
        else:
            self.phi_constants = np.asarray(phi_constants)
        self.unique_phi = np.unique(self.phi_constants)
        self.indices_by_phi = {
            phi: np.where(self.phi_constants == phi)[0] for phi in self.unique_phi
        }

    def __len__(self):
        return self.num_trajectories * (self.total_time_steps - self.sequence_length - 1)

    def _sample_one(self, trajectory_index):
        max_start = self.total_time_steps - self.sequence_length - 1
        start = np.random.randint(0, max_start)
        x = self.X[trajectory_index, start : start + self.sequence_length, :]
        y = self.X[
            trajectory_index, start + 1 : start + self.sequence_length + 1, :
        ]
        if self.S is None:
            return x, y, None
        s = self.S[trajectory_index, start : start + self.sequence_length, :]
        return x, y, s

    def _balanced_trajectory_indices(self):
        phis = list(self.unique_phi)
        base = self.batch_size // len(phis)
        remainder = self.batch_size % len(phis)
        selected = []
        shuffled_phi_positions = np.random.permutation(len(phis))
        counts = {phi: base for phi in phis}
        for position in shuffled_phi_positions[:remainder]:
            counts[phis[position]] += 1
        for phi in phis:
            pool = self.indices_by_phi[phi]
            selected.extend(np.random.choice(pool, size=counts[phi], replace=True))
        np.random.shuffle(selected)
        return selected

    def sample_batch(self):
        if self.balanced_phi and len(self.unique_phi) > 1:
            trajectory_indices = self._balanced_trajectory_indices()
        else:
            trajectory_indices = np.random.randint(
                0, self.num_trajectories, size=self.batch_size
            )
        X, Y, S = [], [], []
        for trajectory_index in trajectory_indices:
            x, y, s = self._sample_one(int(trajectory_index))
            X.append(x)
            Y.append(y)
            S.append(s)
        if S[0] is None:
            return torch.stack(X), torch.stack(Y), None
        return torch.stack(X), torch.stack(Y), torch.stack(S)

    def eval_reference(self, length=10000, trajectory_index=0):
        x_ref = self.X[int(trajectory_index), :length, :]
        if self.S is None:
            return x_ref, None
        return x_ref, self.S[int(trajectory_index), :length, :]

    def test_trajectory(self, trajectory_index, length=None):
        x = self.X_test[int(trajectory_index)]
        if length is not None:
            x = x[:length]
        if self.S_test is None:
            return x, None
        s = self.S_test[int(trajectory_index)]
        if length is not None:
            s = s[:length]
        return x, s


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
    last_dstsp = float("nan")

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
            epochs.set_postfix(loss=average_epoch_loss, dstsp=last_dstsp)
            losses.append(average_epoch_loss)

            if e % ssi == 0:
                with torch.no_grad():
                    x_ref, s_ref = dataset.eval_reference(length=10000)
                    s_test = s_ref.unsqueeze(1).to(device) if s_ref is not None else None
                    x0_test = x_ref[0:1, :].to(device)
                    z_test = predict_free_sequence(
                        model, x0_test, x_ref.shape[0], s_test
                    )
                    z_test_obs = z_test[0, :, 0 : model.N].detach().cpu()
                    x_ref_cpu = x_ref.detach().cpu()
                    klx.append(state_space_divergence_binning(z_test_obs, x_ref_cpu))
                    last_dstsp = klx[-1]
                    dh.append(power_spectrum_error(z_test_obs, x_ref_cpu))
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

    data_dir = Path(args.data_dir)
    trajectories = np.load(data_dir / "trajectories.npy").astype(np.float32)
    all_phi = (
        np.load(data_dir / "trajectory_phi.npy").astype(np.float32)
        if args.use_phi
        else None
    )
    phi_constants_path = data_dir / "trajectory_phi_constants.npy"
    phi_constants = (
        np.load(phi_constants_path).astype(np.float32)
        if phi_constants_path.exists()
        else None
    )

    if args.multi_trajectory:
        X_train_shape = trajectories[:, : args.split_idx, :].shape
        X_test_shape = trajectories[:, args.split_idx :, :].shape
        phi_dim = 0 if all_phi is None else all_phi.shape[-1]
        dataset = MultiTrajectoryDataset(
            trajectories,
            external_inputs=all_phi,
            phi_constants=phi_constants,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            split_idx=args.split_idx,
            balanced_phi=not args.no_balanced_phi,
        )
        T_train = X_train_shape[1]
        T_test = X_test_shape[1]
        N = X_train_shape[-1]
        print("Multi-trajectory mode")
        print("X_train:", X_train_shape, "X_test:", X_test_shape)
        print("unique phi:", dataset.unique_phi)
        print("balanced phi sampling:", dataset.balanced_phi)
    else:
        X = trajectories[args.trajectory_id]
        X_train = X[: args.split_idx]
        X_test = X[args.split_idx :]
        phi_single = all_phi[args.trajectory_id] if all_phi is not None else None
        phi_train = phi_single[: args.split_idx] if phi_single is not None else None
        phi_test = phi_single[args.split_idx :] if phi_single is not None else None
        phi_dim = 0 if phi_train is None else phi_train.shape[-1]
        dataset = SingleTrajectoryDataset(
            X_train,
            external_inputs=phi_train,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
        )
        T_train, N = X_train.shape
        T_test = X_test.shape[0]
        print("Single-trajectory mode")
        print("X_train:", X_train.shape, "X_test:", X_test.shape)

    model = AL_RNN(M=args.latent_dim, P=args.pwl_units, N=N, phi_dim=phi_dim).to(
        device
    )
    model_path = output_dir / "model.pt"
    if args.eval_only:
        print("Eval-only mode: loading model from", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        metrics = {"losses": [], "dstsp": [], "dh": []}
    else:
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

        torch.save(model.state_dict(), model_path)

    def evaluate_test_trajectory(x_test_tensor, phi_test_tensor, label, file_stem):
        horizon = min(args.t_gen, x_test_tensor.shape[0] - args.t_transient)
        total_steps = horizon + args.t_transient
        x_test_torch = x_test_tensor[:total_steps].to(device).unsqueeze(0)
        phi_test_torch = (
            phi_test_tensor[:total_steps].to(device).unsqueeze(1)
            if phi_test_tensor is not None
            else None
        )
        orbit = (
            predict_free_sequence(
                model,
                x_test_torch[:, 0, :],
                total_steps,
                phi_test_torch,
            )
            .detach()
            .cpu()
            .numpy()[0][args.t_transient :, :]
        )
        x_truth = x_test_tensor[args.t_transient : args.t_transient + horizon].cpu()
        dstsp = state_space_divergence_binning(
            torch.tensor(orbit[:, 0 : model.N]), x_truth
        )
        dh = power_spectrum_error(torch.tensor(orbit[:, 0 : model.N]), x_truth)
        orbit_finite = bool(np.isfinite(orbit).all())
        print(
            f"{label} | Dstsp={dstsp:.6f} | DH={dh:.6f} | "
            f"finite={orbit_finite} | min={float(np.nanmin(orbit)):.6f} | "
            f"max={float(np.nanmax(orbit)):.6f}"
        )

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111, projection="3d")
        x_truth_np = x_truth.numpy()
        ax.plot(
            x_truth_np[:, 0],
            x_truth_np[:, 1],
            x_truth_np[:, 2],
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
        ax.set_title(f"{label}\nDstsp={dstsp:.3f}, DH={dh:.3f}")
        ax.legend(loc="upper left")
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(output_dir / f"{file_stem}_free_rollout_3d.png", dpi=180)
        plt.close()
        np.save(output_dir / f"{file_stem}_orbit.npy", orbit)
        np.save(output_dir / f"{file_stem}_x_test.npy", x_truth_np)
        return {
            "label": label,
            "file_stem": file_stem,
            "dstsp": float(dstsp),
            "dh": float(dh),
            "orbit_finite": orbit_finite,
            "orbit_min": float(np.nanmin(orbit)),
            "orbit_max": float(np.nanmax(orbit)),
        }

    eval_results = []
    if args.multi_trajectory:
        for phi_value in dataset.unique_phi:
            trajectory_index = int(dataset.indices_by_phi[phi_value][0])
            x_test_eval, phi_test_eval = dataset.test_trajectory(
                trajectory_index, length=args.t_gen + args.t_transient
            )
            safe_phi = str(float(phi_value)).replace("-", "m").replace(".", "p")
            eval_results.append(
                evaluate_test_trajectory(
                    x_test_eval,
                    phi_test_eval,
                    label=f"phi={float(phi_value):+.3f}, traj={trajectory_index}",
                    file_stem=f"phi_{safe_phi}_traj_{trajectory_index}",
                )
            )
    else:
        x_test_eval = torch.tensor(X_test, dtype=torch.float32)
        phi_test_eval = (
            torch.tensor(phi_test, dtype=torch.float32) if phi_test is not None else None
        )
        eval_results.append(
            evaluate_test_trajectory(
                x_test_eval,
                phi_test_eval,
                label=f"trajectory={args.trajectory_id}",
                file_stem="single",
            )
        )

    primary_result = eval_results[0]

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(
            json_ready(
                {
                    "args": vars(args),
                    "T_train": int(T_train),
                    "T_test": int(T_test),
                    "N": int(N),
                    "phi_dim": int(phi_dim),
                    "primary_dstsp": primary_result["dstsp"],
                    "primary_dh": primary_result["dh"],
                    "evaluation": eval_results,
                    "training": metrics,
                }
            ),
            f,
            indent=2,
        )

    if metrics["losses"]:
        plt.figure(figsize=(7, 4))
        plt.plot(metrics["losses"], lw=2)
        plt.yscale("log")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.title("Training loss")
        plt.tight_layout()
        plt.savefig(output_dir / "training_loss.png", dpi=180)
        plt.close()

    if len(eval_results) > 1:
        labels = [result["label"] for result in eval_results]
        x_pos = np.arange(len(eval_results))
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(eval_results) * 1.4), 4))
        axes[0].bar(x_pos, [result["dstsp"] for result in eval_results])
        axes[0].set_title("Dstsp")
        axes[0].set_xticks(x_pos)
        axes[0].set_xticklabels(labels, rotation=45, ha="right")
        axes[1].bar(x_pos, [result["dh"] for result in eval_results])
        axes[1].set_title("DH")
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig(output_dir / "per_phi_metrics.png", dpi=180)
        plt.close()

    print("Wrote outputs to:", output_dir)
    print("Saved model:", model_path)


if __name__ == "__main__":
    main()
