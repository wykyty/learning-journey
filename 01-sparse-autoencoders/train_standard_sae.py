# Training Standard SAE (ReLU + L1) on T5 Decoder with TransformerLens
#
# Standard SAE uses ReLU activation with L1 regularization to enforce sparsity.
# The L1 penalty encourages the model to learn sparse representations by penalizing
# the sum of absolute activations.
#
# Loss = MSE + L1_coefficient * ||feature_acts||_1

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from tqdm.auto import tqdm
from safetensors.torch import save_file, load_file
from transformer_lens import HookedEncoderDecoder
from datasets import load_dataset
import matplotlib.pyplot as plt


# ========== Setup ==========
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ========== Config ==========
@dataclass
class SAEConfig:
    d_in: int = 1024        # T5-large d_model
    d_sae: int = 16384      # SAE bottleneck width
    lr: float = 1e-4
    batch_size: int = 4096
    context_size: int = 128
    target_size: int = 64
    total_steps: int = 50_000
    l1_coefficient: float = 5.0
    l1_warm_up_steps: int = 2_500
    decoder_init_norm: float = 0.1
    apply_b_dec_to_input: bool = False
    normalize_activations: bool = True
    n_batches_for_norm_estimate: int = 100
    target_block: int = 12
    hook_name: str = "decoder.12.hook_mlp_out"
    log_every: int = 100
    device: str = device
    dtype: str = "float32"


# ========== Standard SAE (ReLU + L1) ==========
class SparseAutoencoder(nn.Module):
    """Standard SAE with ReLU activation and L1 regularization.

    Architecture:
        encode: x -> ReLU(x @ W_enc + b_enc)
        decode: acts -> acts @ W_dec + b_dec

    Sparsity is enforced via L1 penalty on the feature activations.
    """

    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.dtype)

        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in, dtype=self.dtype))
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae, dtype=self.dtype))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, dtype=self.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, dtype=self.dtype))

        self.register_buffer("scaling_factor", torch.tensor(1.0))
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.W_dec.data)
        with torch.no_grad():
            self.W_dec.data /= self.W_dec.norm(dim=-1, keepdim=True)
            self.W_dec.data *= self.cfg.decoder_init_norm
        self.W_enc.data = self.W_dec.data.T.clone().detach().contiguous()

    def estimate_scaling_factor(self, data_loader, n_batches=100):
        if not self.cfg.normalize_activations:
            return
        norms = []
        for i, batch in enumerate(data_loader):
            if i >= n_batches:
                break
            norms.append(batch.float().norm(dim=-1).mean().item())
        mean_norm = np.mean(norms)
        self.scaling_factor = torch.tensor(
            (self.cfg.d_in ** 0.5) / mean_norm, device=self.cfg.device
        )
        print(f"Estimated scaling factor: {self.scaling_factor.item():.4f}")
        print(f"  Mean activation L2 norm: {mean_norm:.4f}")
        print(f"  Target norm (sqrt(d_in)): {self.cfg.d_in ** 0.5:.4f}")

    def encode(self, x: torch.Tensor):
        if self.cfg.normalize_activations and self.scaling_factor > 0:
            x = x * self.scaling_factor
        if self.cfg.apply_b_dec_to_input:
            x = x - self.b_dec
        hidden_pre = x @ self.W_enc + self.b_enc
        feature_acts = F.relu(hidden_pre)
        return feature_acts, hidden_pre

    def decode(self, feature_acts: torch.Tensor):
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        feature_acts, hidden_pre = self.encode(x)
        reconstruction = self.decode(feature_acts)
        return reconstruction, feature_acts, hidden_pre

    def calculate_loss(self, x, reconstruction, feature_acts, l1_coefficient):
        per_item_mse = F.mse_loss(reconstruction, x, reduction="none")
        mse_loss = per_item_mse.sum(dim=-1).mean()
        weighted_feature_acts = feature_acts * self.W_dec.norm(dim=1)
        l1_loss = l1_coefficient * weighted_feature_acts.norm(p=1, dim=-1).mean()
        total_loss = mse_loss + l1_loss

        with torch.no_grad():
            per_token_l2_loss = per_item_mse.sum(dim=-1)
            total_variance = (x - x.mean(0)).pow(2).sum(-1)
            explained_variance = 1 - per_token_l2_loss.mean() / (total_variance.mean() + 1e-8)
            l0 = feature_acts.bool().float().sum(-1).mean()

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "l1_loss": l1_loss,
            "explained_variance": explained_variance,
            "l0": l0,
        }

    def save_model(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({
            "W_enc": self.W_enc.data,
            "W_dec": self.W_dec.data,
            "b_enc": self.b_enc.data,
            "b_dec": self.b_dec.data,
            "scaling_factor": self.scaling_factor,
        }, str(path / "sae_weights.safetensors"))
        with open(path / "sae_config.json", "w") as f:
            json.dump({
                "d_in": self.cfg.d_in,
                "d_sae": self.cfg.d_sae,
                "target_block": self.cfg.target_block,
                "hook_name": self.cfg.hook_name,
                "decoder_init_norm": self.cfg.decoder_init_norm,
                "normalize_activations": self.cfg.normalize_activations,
            }, f, indent=2)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, cfg: SAEConfig):
        path = Path(path)
        state_dict = load_file(str(path / "sae_weights.safetensors"))
        sae = cls(cfg)
        sae.W_enc.data = state_dict["W_enc"]
        sae.W_dec.data = state_dict["W_dec"]
        sae.b_enc.data = state_dict["b_enc"]
        sae.b_dec.data = state_dict["b_dec"]
        sae.scaling_factor = state_dict["scaling_factor"]
        print(f"Model loaded from {path}")
        return sae


# ========== Activation Collection ==========
class DecoderActivationCollector:
    """Collects decoder-side activations from T5 using TransformerLens."""

    def __init__(self, model: HookedEncoderDecoder, config: SAEConfig):
        self.model = model
        self.config = config
        self.hook_name = config.hook_name
        self.tokenizer = model.tokenizer

    @torch.no_grad()
    def collect_activations(self, source_texts: list[str], target_texts: list[str]) -> torch.Tensor:
        collected = []
        for src, tgt in zip(source_texts, target_texts):
            enc_tokens = self.tokenizer(
                src, return_tensors="pt", truncation=True,
                max_length=self.config.context_size, padding=False,
            )
            dec_tokens = self.tokenizer(
                tgt, return_tensors="pt", truncation=True,
                max_length=self.config.target_size, padding=False,
            )
            n_dec_tokens = dec_tokens.input_ids.shape[1]

            _, cache = self.model.run_with_cache(
                enc_tokens.input_ids,
                decoder_input=dec_tokens.input_ids,
                names_filter=lambda name: name == self.hook_name,
            )

            acts = cache[self.hook_name].squeeze(0).float()
            acts = acts[:n_dec_tokens]
            collected.append(acts)

        if not collected:
            return torch.empty(0, self.config.d_in, device=self.config.device)
        return torch.cat(collected, dim=0)

    def collect_batch(self, dataset_iter, target_tokens: int = 4096) -> torch.Tensor:
        collected = []
        total_tokens = 0

        while total_tokens < target_tokens:
            try:
                sample = next(dataset_iter)
            except StopIteration:
                break

            source = sample["document"]
            target = sample["summary"]

            if not source.strip() or not target.strip():
                continue

            acts = self.collect_activations([source], [target])
            if acts.shape[0] > 0:
                collected.append(acts)
                total_tokens += acts.shape[0]

        if not collected:
            return torch.empty(0, self.config.d_in, device=self.config.device)

        all_acts = torch.cat(collected, dim=0)
        if all_acts.shape[0] > target_tokens:
            indices = torch.randperm(all_acts.shape[0])[:target_tokens]
            all_acts = all_acts[indices]
        return all_acts


# ========== Training ==========
def train_sae(
    sae: SparseAutoencoder,
    collector: DecoderActivationCollector,
    dataset_iter,
    config: SAEConfig,
):
    sae.to(config.device)
    sae.train()

    optimizer = torch.optim.Adam(sae.parameters(), lr=config.lr, betas=(0.9, 0.999))

    if config.normalize_activations:
        print("Estimating activation scaling factor...")

        def temp_provider():
            while True:
                acts = collector.collect_batch(dataset_iter, target_tokens=4096)
                if acts.shape[0] > 0:
                    yield acts

        sae.estimate_scaling_factor(temp_provider(), n_batches=config.n_batches_for_norm_estimate)

    metrics_history = {
        "step": [], "total_loss": [], "mse_loss": [], "l1_loss": [],
        "explained_variance": [], "l0": [], "l1_coeff": [],
    }

    print(f"\nStarting training for {config.total_steps} steps...")
    print(f"  Target hook: {config.hook_name}")
    print(f"  d_in={config.d_in}, d_sae={config.d_sae}")
    print(f"  L1 coefficient: {config.l1_coefficient}")
    print(f"  Dataset: XSum (summarization)\n")

    pbar = tqdm(range(config.total_steps), desc="Training Standard SAE")

    for step in pbar:
        batch_acts = collector.collect_batch(dataset_iter, target_tokens=config.batch_size)
        if batch_acts.shape[0] == 0:
            continue

        reconstruction, feature_acts, _ = sae(batch_acts)

        if step < config.l1_warm_up_steps:
            l1_coeff = config.l1_coefficient * (step / config.l1_warm_up_steps)
        else:
            l1_coeff = config.l1_coefficient

        losses = sae.calculate_loss(batch_acts, reconstruction, feature_acts, l1_coeff)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()

        metrics_history["step"].append(step)
        metrics_history["total_loss"].append(losses["total_loss"].item())
        metrics_history["mse_loss"].append(losses["mse_loss"].item())
        metrics_history["l1_loss"].append(losses["l1_loss"].item())
        metrics_history["explained_variance"].append(losses["explained_variance"].item())
        metrics_history["l0"].append(losses["l0"].item())
        metrics_history["l1_coeff"].append(l1_coeff)

        if step % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "mse": f"{losses['mse_loss'].item():.4f}",
                "ev": f"{losses['explained_variance'].item():.4f}",
                "l0": f"{losses['l0'].item():.1f}",
            })

    pbar.close()
    return metrics_history


# ========== Evaluation ==========
@torch.no_grad()
def evaluate_sae(sae, collector, dataset_iter, model, n_batches=10):
    sae.eval()
    all_feature_acts, all_reconstructions, all_inputs = [], [], []

    for _ in range(n_batches):
        batch_acts = collector.collect_batch(dataset_iter, target_tokens=4096)
        if batch_acts.shape[0] == 0:
            continue
        reconstruction, feature_acts, _ = sae(batch_acts)
        all_feature_acts.append(feature_acts)
        all_reconstructions.append(reconstruction)
        all_inputs.append(batch_acts)

    if not all_feature_acts:
        print("No activations collected")
        return {}

    feature_acts = torch.cat(all_feature_acts, dim=0)
    reconstructions = torch.cat(all_reconstructions, dim=0)
    inputs = torch.cat(all_inputs, dim=0)

    per_token_mse = (reconstructions - inputs).pow(2).sum(dim=-1)
    total_variance = (inputs - inputs.mean(0)).pow(2).sum(dim=-1)
    explained_variance = 1 - per_token_mse.mean() / (total_variance.mean() + 1e-8)

    active_features = feature_acts.bool().float()
    l0 = active_features.sum(-1).mean()
    feature_density = active_features.mean(0)
    dead_features = (feature_density < 1e-6).sum().item()

    eval_metrics = {
        "explained_variance": explained_variance.item(),
        "l0": l0.item(),
        "feature_density_mean": feature_density.mean().item(),
        "dead_features": dead_features,
        "total_features": feature_acts.shape[-1],
        "dead_feature_pct": dead_features / feature_acts.shape[-1] * 100,
    }

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.hist(feature_density.cpu().numpy(), bins=50, edgecolor="black")
    plt.xlabel("Feature Activation Rate")
    plt.ylabel("Count")
    plt.title("Feature Density Distribution")
    plt.axvline(x=feature_density.mean().item(), color="r", linestyle="--")

    plt.subplot(1, 2, 2)
    plt.hist(l0.cpu().numpy(), bins=50, edgecolor="black")
    plt.xlabel("L0 (Active Features per Token)")
    plt.ylabel("Count")
    plt.title("L0 Distribution")
    plt.axvline(x=l0.item(), color="r", linestyle="--")

    plt.tight_layout()
    plt.show()
    return eval_metrics


@torch.no_grad()
def logit_lens_analysis(sae, model, n_features=10):
    """Project SAE decoder weights onto the unembedding matrix."""
    embeddings = model.W_E
    projection = sae.W_dec @ embeddings.T

    top_k = 5
    vals, inds = projection.topk(top_k, dim=1)
    random_indices = torch.randint(0, projection.shape[0], (n_features,))

    print(f"Top {top_k} tokens promoted by {n_features} random features:")
    print("-" * 80)
    for idx in random_indices:
        feat = idx.item()
        tokens = [model.to_string(i) for i in inds[feat]]
        probs = F.softmax(vals[feat], dim=0)
        token_strs = [f"'{t}' ({p:.3f})" for t, p in zip(tokens, probs)]
        print(f"Feature {feat:5d}: {', '.join(token_strs)}")
    return projection


# ========== Plotting ==========
def plot_training_metrics(metrics):
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle("Standard SAE Training Metrics", fontsize=14)

    axes[0, 0].plot(metrics["step"], metrics["total_loss"])
    axes[0, 0].set_title("Total Loss (MSE + L1)")
    axes[0, 0].set_xlabel("Step")

    axes[0, 1].plot(metrics["step"], metrics["mse_loss"])
    axes[0, 1].set_title("MSE Loss")
    axes[0, 1].set_xlabel("Step")

    axes[0, 2].plot(metrics["step"], metrics["l1_loss"])
    axes[0, 2].set_title("L1 Loss")
    axes[0, 2].set_xlabel("Step")

    axes[1, 0].plot(metrics["step"], metrics["explained_variance"])
    axes[1, 0].set_title("Explained Variance")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="Target (0.8)")
    axes[1, 0].legend()

    axes[1, 1].plot(metrics["step"], metrics["l0"])
    axes[1, 1].set_title("L0 (Active Features)")
    axes[1, 1].set_xlabel("Step")

    axes[1, 2].plot(metrics["step"], metrics["l1_coeff"])
    axes[1, 2].set_title("L1 Coefficient (with warmup)")
    axes[1, 2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


# ========== Main ==========
def main():
    # Load model
    print("Loading T5-large via TransformerLens...")
    model = HookedEncoderDecoder.from_pretrained("google-t5/t5-large")
    model.eval()
    print(f"Model loaded: google-t5/t5-large")
    print(f"  d_model = {model.cfg.d_model}")
    print(f"  d_mlp = {model.cfg.d_mlp}")
    print(f"  n_heads = {model.cfg.n_heads}")
    print(f"  n_layers = {model.cfg.n_layers}")

    # Load dataset
    print("\nLoading XSum dataset...")
    dataset = load_dataset("EdinburghNLP/xsum", split="train", streaming=True)
    dataset_iter = iter(dataset)

    # Config
    config = SAEConfig()
    print(f"\nStandard SAE Config:")
    print(f"  d_in={config.d_in}, d_sae={config.d_sae}")
    print(f"  Expansion ratio: {config.d_sae / config.d_in:.1f}x")
    print(f"  Total parameters: {config.d_in * config.d_sae * 2 + config.d_sae + config.d_in:,}")
    print(f"  L1 coefficient: {config.l1_coefficient}")
    print(f"  Target hook: {config.hook_name}")

    # Initialize SAE
    sae = SparseAutoencoder(config)
    print(f"\nSAE initialized with {sum(p.numel() for p in sae.parameters()):,} parameters")

    # Initialize collector
    collector = DecoderActivationCollector(model, config)

    # Train
    metrics = train_sae(sae, collector, dataset_iter, config)
    print("\nTraining complete!")

    # Plot metrics
    plot_training_metrics(metrics)

    # Evaluate
    eval_metrics = evaluate_sae(sae, collector, dataset_iter, model)
    print("\nEvaluation Metrics:")
    for key, value in eval_metrics.items():
        print(f"  {key}: {value}")

    # Logit lens analysis
    logit_lens_analysis(sae, model)

    # Save
    save_path = Path("checkpoints/standard_sae_t5_large_decoder_block12")
    sae.save_model(str(save_path))
    with open(save_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"\nAll artifacts saved to: {save_path}")


if __name__ == "__main__":
    main()
