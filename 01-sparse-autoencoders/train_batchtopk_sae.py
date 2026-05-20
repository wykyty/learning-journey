# Training BatchTopk SAE on T5 Decoder with TransformerLens
#
# Based on SAELens's BatchTopK implementation.
# Unlike standard SAE with ReLU + L1, BatchTopk uses top-k activation across the batch
# to directly control sparsity.
#
# Key features:
# - Top-k selection across the entire batch (not per-sample)
# - ReLU applied before top-k selection
# - Optional rescaling by decoder norms
# - Auxiliary loss for dead neuron recovery
# - Threshold tracking via EMA for inference-time conversion to JumpReLU
#
# Loss = MSE + Auxiliary Loss (dead neuron recovery)

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
class BatchTopkSAEConfig:
    d_in: int = 1024        # T5-large d_model
    d_sae: int = 16384      # SAE bottleneck width
    k: float = 100          # Average number of active features per sample (float for batch mode)
    lr: float = 1e-4
    batch_size: int = 4096
    context_size: int = 128
    target_size: int = 64
    total_steps: int = 50_000
    decoder_init_norm: float = 0.1
    normalize_activations: bool = True
    n_batches_for_norm_estimate: int = 100
    rescale_acts_by_decoder_norm: bool = True  # Rescale pre-activations by decoder norms
    aux_loss_coefficient: float = 1.0          # Coefficient for dead neuron auxiliary loss
    topk_threshold_lr: float = 0.01            # Learning rate for threshold EMA update
    target_block: int = 12
    hook_name: str = "decoder.12.hook_mlp_out"
    log_every: int = 100
    device: str = device
    dtype: str = "float32"


# ========== BatchTopk SAE ==========
class BatchTopkSAE(nn.Module):
    """BatchTopk SAE: uses top-k activation across the batch for sparsity control.

    Based on SAELens's BatchTopK implementation. Key features:
    - Top-k selection across the entire batch (not per-sample)
    - ReLU applied before top-k selection
    - Optional rescaling by decoder norms
    - Auxiliary loss for dead neuron recovery
    - Threshold tracking via EMA for inference-time conversion to JumpReLU
    """

    def __init__(self, cfg: BatchTopkSAEConfig):
        super().__init__()
        self.cfg = cfg
        self.dtype = getattr(torch, cfg.dtype)

        self.W_dec = nn.Parameter(torch.empty(cfg.d_sae, cfg.d_in, dtype=self.dtype))
        self.W_enc = nn.Parameter(torch.empty(cfg.d_in, cfg.d_sae, dtype=self.dtype))
        self.b_enc = nn.Parameter(torch.zeros(cfg.d_sae, dtype=self.dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg.d_in, dtype=self.dtype))

        self.register_buffer("scaling_factor", torch.tensor(1.0))
        # Track minimum positive activation for threshold estimation (EMA)
        self.register_buffer(
            "topk_threshold",
            torch.tensor(0.0, dtype=torch.double, device=cfg.device),
        )
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

    def encode(self, x: torch.Tensor):
        if self.cfg.normalize_activations and self.scaling_factor > 0:
            x = x * self.scaling_factor
        hidden_pre = x @ self.W_enc + self.b_enc
        return hidden_pre

    def batch_topk_activation(self, hidden_pre: torch.Tensor):
        """Apply batch top-k activation (SAELens style).

        1. Apply ReLU to pre-activations
        2. Flatten across batch dimension
        3. Select top-k values across the entire batch
        4. Scatter back to original positions
        """
        # Apply ReLU first (as in SAELens)
        acts = F.relu(hidden_pre)
        flat_acts = acts.flatten()

        # Calculate number of samples (batch dimension)
        num_samples = acts.shape[:-1].numel()

        # Select top-k values across the entire batch
        k_total = int(self.cfg.k * num_samples)
        topk_values, topk_indices = torch.topk(flat_acts, k_total)

        # Scatter back to original shape
        return (
            torch.zeros_like(flat_acts)
            .scatter(-1, topk_indices, topk_values)
            .reshape(acts.shape)
        )

    def decode(self, feature_acts: torch.Tensor):
        # Optionally rescale by decoder norms (as in SAELens)
        if self.cfg.rescale_acts_by_decoder_norm:
            feature_acts = feature_acts * (1 / self.W_dec.norm(dim=-1))
        return feature_acts @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor):
        hidden_pre = self.encode(x)

        # Optionally rescale pre-activations by decoder norms
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)

        feature_acts = self.batch_topk_activation(hidden_pre)
        reconstruction = self.decode(feature_acts)
        return reconstruction, feature_acts, hidden_pre

    def calculate_loss(self, x, reconstruction, feature_acts, hidden_pre):
        """Loss = MSE + auxiliary loss for dead neurons."""
        per_item_mse = F.mse_loss(reconstruction, x, reduction="none")
        mse_loss = per_item_mse.sum(dim=-1).mean()

        # Auxiliary loss for dead neuron recovery (from SAELens)
        aux_loss = self._calculate_aux_loss(x, reconstruction, hidden_pre)

        total_loss = mse_loss + aux_loss

        with torch.no_grad():
            per_token_l2_loss = per_item_mse.sum(dim=-1)
            total_variance = (x - x.mean(0)).pow(2).sum(-1)
            explained_variance = 1 - per_token_l2_loss.mean() / (total_variance.mean() + 1e-8)
            l0 = feature_acts.bool().float().sum(-1).mean()

        return {
            "total_loss": total_loss,
            "mse_loss": mse_loss,
            "aux_loss": aux_loss,
            "explained_variance": explained_variance,
            "l0": l0,
        }

    def _calculate_aux_loss(self, sae_in, sae_out, hidden_pre):
        """Auxiliary loss to recover dead neurons (from SAELens TopK).

        This loss encourages dead neurons to learn useful features by having
        them reconstruct the residual error from the live neurons.
        """
        # Detect dead neurons (haven't fired in this batch)
        with torch.no_grad():
            # Use the activation to detect which neurons fired
            if self.cfg.rescale_acts_by_decoder_norm:
                scaled_pre = hidden_pre * self.W_dec.norm(dim=-1)
            else:
                scaled_pre = hidden_pre
            acts = F.relu(scaled_pre)
            dead_mask = (acts.sum(dim=0) == 0)  # [d_sae]

        num_dead = dead_mask.sum().item()
        if num_dead == 0:
            return sae_out.new_tensor(0.0)

        # Residual from live neurons
        residual = (sae_in - sae_out).detach()

        # Heuristic: use ~50% of d_in as k_aux
        k_aux = min(sae_in.shape[-1] // 2, num_dead)
        scale = min(num_dead / (sae_in.shape[-1] // 2), 1.0)

        # Select top-k dead neurons by pre-activation
        auxk_latents = torch.where(dead_mask[None], hidden_pre, torch.tensor(-float('inf')))
        auxk_topk = auxk_latents.topk(k_aux, sorted=False)

        # Create sparse activations for dead neurons
        auxk_acts = torch.zeros_like(hidden_pre)
        auxk_acts.scatter_(-1, auxk_topk.indices, auxk_topk.values)

        # Decode without bias (bias already in residual)
        recons = auxk_acts @ self.W_dec
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()

        return self.cfg.aux_loss_coefficient * scale * auxk_loss

    @torch.no_grad()
    def update_topk_threshold(self, feature_acts: torch.Tensor):
        """Update threshold using EMA of minimum positive activation."""
        positive_mask = feature_acts > 0
        lr = self.cfg.topk_threshold_lr
        if positive_mask.any():
            min_positive = feature_acts[positive_mask].min().to(self.topk_threshold.dtype)
            self.topk_threshold = (1 - lr) * self.topk_threshold + lr * min_positive

    def save_model(self, path: str):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        save_file({
            "W_enc": self.W_enc.data,
            "W_dec": self.W_dec.data,
            "b_enc": self.b_enc.data,
            "b_dec": self.b_dec.data,
            "scaling_factor": self.scaling_factor,
            "topk_threshold": self.topk_threshold,
        }, str(path / "sae_weights.safetensors"))
        with open(path / "sae_config.json", "w") as f:
            json.dump({
                "d_in": self.cfg.d_in,
                "d_sae": self.cfg.d_sae,
                "k": self.cfg.k,
                "target_block": self.cfg.target_block,
                "hook_name": self.cfg.hook_name,
                "decoder_init_norm": self.cfg.decoder_init_norm,
                "normalize_activations": self.cfg.normalize_activations,
                "rescale_acts_by_decoder_norm": self.cfg.rescale_acts_by_decoder_norm,
                "aux_loss_coefficient": self.cfg.aux_loss_coefficient,
                "topk_threshold_lr": self.cfg.topk_threshold_lr,
                "sae_type": "batch_topk",
            }, f, indent=2)
        print(f"Model saved to {path}")

    @classmethod
    def load_model(cls, path: str, cfg: BatchTopkSAEConfig):
        path = Path(path)
        state_dict = load_file(str(path / "sae_weights.safetensors"))
        sae = cls(cfg)
        sae.W_enc.data = state_dict["W_enc"]
        sae.W_dec.data = state_dict["W_dec"]
        sae.b_enc.data = state_dict["b_enc"]
        sae.b_dec.data = state_dict["b_dec"]
        sae.scaling_factor = state_dict["scaling_factor"]
        if "topk_threshold" in state_dict:
            sae.topk_threshold = state_dict["topk_threshold"]
        print(f"Model loaded from {path}")
        return sae


# ========== Activation Collection ==========
class DecoderActivationCollector:
    """Collects decoder-side activations from T5 using TransformerLens."""

    def __init__(self, model: HookedEncoderDecoder, config: BatchTopkSAEConfig):
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
def train_batch_topk_sae(
    sae: BatchTopkSAE,
    collector: DecoderActivationCollector,
    dataset_iter,
    config: BatchTopkSAEConfig,
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
        "step": [], "total_loss": [], "mse_loss": [], "aux_loss": [],
        "explained_variance": [], "l0": [], "topk_threshold": [],
    }

    print(f"\nStarting BatchTopk SAE training for {config.total_steps} steps...")
    print(f"  Target hook: {config.hook_name}")
    print(f"  d_in={config.d_in}, d_sae={config.d_sae}, k={config.k}")
    print(f"  Sparsity: k={config.k} active features per sample")
    print(f"  Aux loss coefficient: {config.aux_loss_coefficient}")
    print(f"  Rescale by decoder norm: {config.rescale_acts_by_decoder_norm}")
    print(f"  Dataset: XSum (summarization)\n")

    pbar = tqdm(range(config.total_steps), desc="Training BatchTopk SAE")

    for step in pbar:
        batch_acts = collector.collect_batch(dataset_iter, target_tokens=config.batch_size)
        if batch_acts.shape[0] == 0:
            continue

        reconstruction, feature_acts, hidden_pre = sae(batch_acts)
        losses = sae.calculate_loss(batch_acts, reconstruction, feature_acts, hidden_pre)

        optimizer.zero_grad()
        losses["total_loss"].backward()
        torch.nn.utils.clip_grad_norm_(sae.parameters(), 1.0)
        optimizer.step()

        # Update topk threshold (EMA)
        sae.update_topk_threshold(feature_acts)

        metrics_history["step"].append(step)
        metrics_history["total_loss"].append(losses["total_loss"].item())
        metrics_history["mse_loss"].append(losses["mse_loss"].item())
        metrics_history["aux_loss"].append(losses["aux_loss"].item())
        metrics_history["explained_variance"].append(losses["explained_variance"].item())
        metrics_history["l0"].append(losses["l0"].item())
        metrics_history["topk_threshold"].append(sae.topk_threshold.item())

        if step % config.log_every == 0:
            pbar.set_postfix({
                "loss": f"{losses['total_loss'].item():.4f}",
                "mse": f"{losses['mse_loss'].item():.4f}",
                "aux": f"{losses['aux_loss'].item():.4f}",
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
    plt.title("Feature Density Distribution (BatchTopk)")
    plt.axvline(x=feature_density.mean().item(), color="r", linestyle="--")

    plt.subplot(1, 2, 2)
    plt.hist(l0.cpu().numpy(), bins=50, edgecolor="black")
    plt.xlabel("L0 (Active Features per Token)")
    plt.ylabel("Count")
    plt.title("L0 Distribution (BatchTopk)")
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
    fig.suptitle("BatchTopk SAE Training Metrics", fontsize=14)

    axes[0, 0].plot(metrics["step"], metrics["total_loss"])
    axes[0, 0].set_title("Total Loss (MSE + Aux)")
    axes[0, 0].set_xlabel("Step")

    axes[0, 1].plot(metrics["step"], metrics["mse_loss"])
    axes[0, 1].set_title("MSE Loss")
    axes[0, 1].set_xlabel("Step")

    axes[0, 2].plot(metrics["step"], metrics["aux_loss"])
    axes[0, 2].set_title("Auxiliary Loss (Dead Neuron)")
    axes[0, 2].set_xlabel("Step")

    axes[1, 0].plot(metrics["step"], metrics["explained_variance"])
    axes[1, 0].set_title("Explained Variance")
    axes[1, 0].set_xlabel("Step")
    axes[1, 0].axhline(y=0.8, color="r", linestyle="--", alpha=0.5, label="Target (0.8)")
    axes[1, 0].legend()

    axes[1, 1].plot(metrics["step"], metrics["l0"])
    axes[1, 1].set_title("L0 (Active Features)")
    axes[1, 1].set_xlabel("Step")

    axes[1, 2].plot(metrics["step"], metrics["topk_threshold"])
    axes[1, 2].set_title("TopK Threshold (EMA)")
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
    config = BatchTopkSAEConfig()
    print(f"\nBatchTopk SAE Config:")
    print(f"  d_in={config.d_in}, d_sae={config.d_sae}, k={config.k}")
    print(f"  Expansion ratio: {config.d_sae / config.d_in:.1f}x")
    print(f"  Sparsity: k={config.k} active features per sample")
    print(f"  Aux loss coefficient: {config.aux_loss_coefficient}")
    print(f"  Rescale by decoder norm: {config.rescale_acts_by_decoder_norm}")
    print(f"  Target hook: {config.hook_name}")

    # Initialize SAE
    sae = BatchTopkSAE(config)
    print(f"\nSAE initialized with {sum(p.numel() for p in sae.parameters()):,} parameters")

    # Initialize collector
    collector = DecoderActivationCollector(model, config)

    # Train
    metrics = train_batch_topk_sae(sae, collector, dataset_iter, config)
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
    save_path = Path("checkpoints/batchtopk_sae_t5_large_decoder_block12")
    sae.save_model(str(save_path))
    with open(save_path / "training_metrics.json", "w") as f:
        json.dump(metrics, f)
    print(f"\nAll artifacts saved to: {save_path}")


if __name__ == "__main__":
    main()
