import argparse
import os
import logging
from datetime import timedelta
import math

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

from model.smooth_cross_entropy import smooth_crossentropy, trades_loss
from utils.cifar import Cifar, Cifar100
from utils.log import Log
from utils.initialize import initialize
from utils.step_lr import StepLR
# from utils.Esam import ESAM
# from utils.mail import send_email
from utils.options import args, setup_model
from utils.MiscTools import count_parameters
from utils.dist_util import get_world_size

logger = logging.getLogger(__name__)


# ===================== LE-SAM Optimizer (Loss-Equated SAM, SAM-style two-step) =====================
class LESAM(torch.optim.Optimizer):

    def __init__(
        self,
        params,
        base_optimizer_ctor,
        sigma: float = 0.1,
        rho_max: float = 0.05,
        adaptive: bool = False,
        eps: float = 1e-12,
        **kwargs
    ):

        assert sigma >= 0.0, "sigma must be non-negative"
        defaults = dict(sigma=sigma, rho_max=rho_max, adaptive=adaptive, eps=eps, **kwargs)
        super().__init__(params, defaults)

        self.base_optimizer = base_optimizer_ctor(self.param_groups, **kwargs)
        self._saved_eps = None
        self._last_rho = None

    # --- helpers ---
    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:

        device = self.param_groups[0]["params"][0].device
        adaptive = self.param_groups[0]["adaptive"]
        eps = self.param_groups[0]["eps"]
        acc = None
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                scale = p.abs() if adaptive else 1.0
                g = (scale * p.grad).norm(p=2)
                acc = g * g if acc is None else acc + g * g
        if acc is None:
            return torch.zeros((), device=device)
        return (acc + eps).sqrt()

    @property
    def last_rho(self):
        return self._last_rho

    @torch.no_grad()
    def ascent_step(self, zero_grad: bool = True):

        grad_norm = self._grad_norm()
        if grad_norm.item() == 0.0:

            self._saved_eps = None
            self._last_rho = 0.0
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        sigma = self.param_groups[0]["sigma"]
        rho_max = self.param_groups[0]["rho_max"]
        adaptive = self.param_groups[0]["adaptive"]
        eps = self.param_groups[0]["eps"]

        rho_t = sigma / (grad_norm + eps)
        if rho_max is not None and rho_max > 0.0:
            rho_t = min(rho_t, rho_max)

        self._last_rho = float(rho_t)
        self._saved_eps = []

        scale_factor = rho_t / (grad_norm + eps)

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    self._saved_eps.append(None)
                    continue
                scale = p.abs() if adaptive else 1.0
                e = scale * p.grad * scale_factor
                p.add_(e)
                self._saved_eps.append(e)

        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def descent_step(self, zero_grad: bool = True):

        if self._saved_eps is None:
          
            self.base_optimizer.step()
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return

        k = 0
        for group in self.param_groups:
            for p in group["params"]:
                e = self._saved_eps[k]
                if e is not None:
                    p.sub_(e)
                k += 1

        self.base_optimizer.step()

        if zero_grad:
            self.zero_grad(set_to_none=True)

        self._saved_eps = None

    def step(self, *args, **kwargs):

        raise NotImplementedError("Use ascent_step() and descent_step() like SAM.")


# ====== Reproducibility ======
def set_global_seed(seed: int, deterministic: bool = True):
    import os, random
    import numpy as np
    import torch
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


# ====== Top-p helpers ======
def compute_topk_indices(loss_vec: torch.Tensor, topk_frac: float) -> torch.Tensor:
    assert loss_vec.dim() == 1, "loss_vec must be shape (B,)"
    B = loss_vec.size(0)
    topk_frac = float(topk_frac)
    assert 0.0 < topk_frac <= 1.0, "topk_frac must be in (0, 1]"
    k = max(1, int((B * topk_frac) + 0.999999))
    k = min(k, B)
    scores = loss_vec.detach()
    _, idx = torch.topk(scores, k=k, largest=True, sorted=False)
    return idx


def masked_mean_with_indices(loss_vec: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if indices.numel() == loss_vec.numel():
        return loss_vec.mean()
    return loss_vec.index_select(0, indices).mean()


# ====== NEW: sigma schedule helper with linear warmup ======
def schedule_lesam_sigma(
    epoch: int,
    sigma_init: float,
    sigma_min: float,
    decay_start: int,
    total_epochs: int,
    mode: str = "none",
    warmup_epochs: int = 0,
    warmup_start: float = 0.0,
) -> float:
    """
    Per-epoch sigma schedule for LE-SAM.

    Stage 1: optional linear warmup
        epoch = 0                       -> warmup_start
        epoch = warmup_epochs - 1       -> sigma_init

    Stage 2: optional decay
        keep sigma_init before decay_start, then decay to sigma_min until the last epoch.
        mode: "none" | "linear" | "cosine"

    Notes:
        - If warmup_epochs <= 0, warmup is disabled.
        - Decay starts after max(decay_start, warmup_epochs), so warmup and decay do not overlap.
    """
    mode = (mode or "none").lower()
    warmup_epochs = max(0, int(warmup_epochs))
    total_epochs = int(total_epochs)
    decay_start = int(decay_start)

    sigma_init = float(sigma_init)
    sigma_min = float(sigma_min)
    warmup_start = float(warmup_start)

    # ---------- Stage 1: linear warmup ----------
    if warmup_epochs > 0 and epoch < warmup_epochs:
        if warmup_epochs == 1:
            sigma = sigma_init
        else:
            # t: 0 -> 1 over the warmup window
            t = float(epoch) / float(warmup_epochs - 1)
            t = max(0.0, min(1.0, t))
            sigma = warmup_start + (sigma_init - warmup_start) * t

        # clamp inside [warmup_start, sigma_init]
        lo = min(warmup_start, sigma_init)
        hi = max(warmup_start, sigma_init)
        sigma = max(lo, min(hi, float(sigma)))
        return float(sigma)

    # ---------- Stage 2: optional decay ----------
    if mode == "none":
        return float(sigma_init)

    # Avoid decay overlapping with warmup.
    effective_decay_start = max(decay_start, warmup_epochs)

    if epoch < effective_decay_start:
        return float(sigma_init)

    # number of epochs in decay stage
    decay_len = total_epochs - int(effective_decay_start)
    if decay_len <= 1:
        return float(sigma_min)

    # progress t in [0,1], where epoch=effective_decay_start -> 0,
    # epoch=total_epochs-1 -> 1
    denom = max(1, decay_len - 1)
    t = float(epoch - effective_decay_start) / float(denom)
    t = max(0.0, min(1.0, t))

    if mode == "linear":
        sigma = sigma_init + (sigma_min - sigma_init) * t
    elif mode == "cosine":
        # cosine from sigma_init -> sigma_min
        sigma = sigma_min + 0.5 * (sigma_init - sigma_min) * (1.0 + math.cos(math.pi * t))
    else:
        # unknown mode -> fallback no schedule
        sigma = sigma_init

    # safety clamp
    lo = min(sigma_init, sigma_min)
    hi = max(sigma_init, sigma_min)
    sigma = max(lo, min(hi, float(sigma)))
    return float(sigma)


def train(args, model):
    device = args.device

    dataset = Cifar(args) if args.dataset == "cifar10" else Cifar100(args)
    log = Log(log_each=10)

    if getattr(args, "SCE_loss", "False") == "True":
        loss_fct = smooth_crossentropy
    else:
        def _ce(logits, targets):
            return torch.nn.functional.cross_entropy(logits, targets, reduction="none")
        loss_fct = _ce

    if args.local_rank in [-1, 0]:
        writer = SummaryWriter(log_dir=os.path.join("logs", args.name))
    else:
        writer = None

    # ====== Optim & LE-SAM config ======
    base_lr = float(getattr(args, "learning_rate", 0.1))
    weight_decay = float(getattr(args, "weight_decay", 0))
    momentum = float(getattr(args, "momentum", 0.9)) if hasattr(args, "momentum") else 0.9

    lesam_sigma = float(getattr(args, "lesam_sigma", 0.35))
    lesam_rho_max = float(getattr(args, "lesam_rho_max", 0.3))
    lesam_adaptive = bool(getattr(args, "lesam_adaptive", False))
    lesam_eps = float(getattr(args, "lesam_eps", 1e-12))

    lesam_topk = float(getattr(args, "lesam_topk", 1.0))
    lesam_topk_share_indices = bool(getattr(args, "lesam_topk_share_indices", False))
    lesam_rho_monitor = bool(getattr(args, "lesam_rho_monitor", True))

    lesam_alpha = float(getattr(args, "lesam_alpha", 0.0))

    # ====== sigma schedule args (all optional) ======

    lesam_sigma_decay = str(getattr(args, "lesam_sigma_decay", "cosine")).lower()
    lesam_sigma_decay_start = int(getattr(args, "lesam_sigma_decay_start", 160))  
    lesam_sigma_min = float(getattr(args, "lesam_sigma_min", 0.0))

    lesam_sigma_warmup_epochs = int(
        getattr(args, "lesam_sigma_warmup_epochs", getattr(args, "lesam_sigma_warmup", 10))
    )
    lesam_sigma_warmup_start = float(getattr(args, "lesam_sigma_warmup_start", 0.0))

    optimizer = LESAM(
        model.parameters(),
        base_optimizer_ctor=lambda pgs, **kw: torch.optim.SGD(
            pgs, lr=base_lr, momentum=momentum, weight_decay=weight_decay
        ),
        sigma=lesam_sigma,
        rho_max=lesam_rho_max,
        adaptive=lesam_adaptive,
        eps=lesam_eps,
        lr=base_lr, momentum=momentum, weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer.base_optimizer, T_max=args.epochs)

    if args.local_rank in [-1, 0]:
        logger.info(
            f"[LE-SAM cfg] sigma={lesam_sigma}, rho_max={lesam_rho_max}, "
            f"adaptive={lesam_adaptive}, eps={lesam_eps}, "
            f"topk={lesam_topk}, share_indices={lesam_topk_share_indices}, "
            f"rho_monitor={lesam_rho_monitor}, alpha_grad={lesam_alpha}"
        )
        logger.info(
            f"[LE-SAM sigma schedule] mode={lesam_sigma_decay}, "
            f"warmup_epochs={lesam_sigma_warmup_epochs}, "
            f"warmup_start={lesam_sigma_warmup_start}, "
            f"decay_start={lesam_sigma_decay_start}, sigma_min={lesam_sigma_min}"
        )

    scaler = None
    use_fp16 = bool(getattr(args, "fp16", False))
    use_bf16 = bool(getattr(args, "bf16", False))
    autocast_dtype = None
    if use_fp16:
        scaler = GradScaler()
        autocast_dtype = None
    elif use_bf16:
        scaler = None
        autocast_dtype = torch.bfloat16
    else:
        scaler = None
        autocast_dtype = None

    if args.local_rank != -1:
        model = DDP(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=False
        )

    best_acc = 0.0
    global_step = -1
    sampler = dataset.train.sampler

    rho_epoch_sum = 0.0
    rho_epoch_min = float("inf")
    rho_epoch_max = float("-inf")
    rho_epoch_cnt = 0

    for epoch in range(args.epochs):
        # ====== NEW: update sigma per-epoch (after epoch n: linear/cosine decay) ======
        sigma_now = schedule_lesam_sigma(
            epoch=epoch,
            sigma_init=lesam_sigma,
            sigma_min=lesam_sigma_min,
            decay_start=lesam_sigma_decay_start,
            total_epochs=args.epochs,
            mode=lesam_sigma_decay,
            warmup_epochs=lesam_sigma_warmup_epochs,
            warmup_start=lesam_sigma_warmup_start,
        )
        for g in optimizer.param_groups:
            g["sigma"] = float(sigma_now)

        if args.local_rank in [-1, 0]:
            logger.info(f"[LE-SAM] epoch={epoch} sigma_now={sigma_now:.6f}")
            if writer is not None:
                writer.add_scalar("train/lesam_sigma", float(sigma_now), global_step=epoch)

        if args.local_rank != -1 and sampler is not None:
            sampler.set_epoch(epoch)

        model.train()
        log.train(len_dataset=len(dataset.train))

        rho_epoch_sum = 0.0
        rho_epoch_min = float("inf")
        rho_epoch_max = float("-inf")
        rho_epoch_cnt = 0

        for batch in dataset.train:
            global_step += 1
            inputs, targets = (b.to(device, non_blocking=True) for b in batch)

            if use_fp16 or use_bf16:
                with autocast(dtype=autocast_dtype):
                    logits = model(inputs)
                    loss_vec = loss_fct(logits, targets)
                    loss_center_full = loss_vec.mean()
                    if lesam_topk < 1.0:
                        idx_top = compute_topk_indices(loss_vec, lesam_topk)
                        loss_center = masked_mean_with_indices(loss_vec, idx_top)
                    else:
                        idx_top = None
                        loss_center = loss_center_full
            else:
                logits = model(inputs)
                loss_vec = loss_fct(logits, targets)
                loss_center_full = loss_vec.mean()
                if lesam_topk < 1.0:
                    idx_top = compute_topk_indices(loss_vec, lesam_topk)
                    loss_center = masked_mean_with_indices(loss_vec, idx_top)
                else:
                    idx_top = None
                    loss_center = loss_center_full

            optimizer.zero_grad(set_to_none=True)

            g_center = None

            # backward at center + ascent
            if scaler is not None:
                scaler.scale(loss_center).backward()
                scaler.unscale_(optimizer.base_optimizer)

                if lesam_alpha != 0.0:
                    g_center = []
                    for p in model.parameters():
                        if p.grad is None:
                            g_center.append(None)
                        else:
                            g_center.append(p.grad.detach().clone())

                optimizer.ascent_step(zero_grad=True)
            else:
                loss_center.backward()

                if lesam_alpha != 0.0:
                    g_center = []
                    for p in model.parameters():
                        if p.grad is None:
                            g_center.append(None)
                        else:
                            g_center.append(p.grad.detach().clone())

                optimizer.ascent_step(zero_grad=True)

            if use_fp16 or use_bf16:
                with autocast(dtype=autocast_dtype):
                    logits_n = model(inputs)
                    loss_vec_n = loss_fct(logits_n, targets)
                    loss_neighbor_full = loss_vec_n.mean()
                    if lesam_topk < 1.0:
                        if lesam_topk_share_indices and (idx_top is not None):
                            idx_use = idx_top
                        else:
                            idx_use = compute_topk_indices(loss_vec_n, lesam_topk)
                        loss_neighbor = masked_mean_with_indices(loss_vec_n, idx_use)
                    else:
                        loss_neighbor = loss_neighbor_full
            else:
                logits_n = model(inputs)
                loss_vec_n = loss_fct(logits_n, targets)
                loss_neighbor_full = loss_vec_n.mean()
                if lesam_topk < 1.0:
                    if lesam_topk_share_indices and (idx_top is not None):
                        idx_use = idx_top
                    else:
                        idx_use = compute_topk_indices(loss_vec_n, lesam_topk)
                    loss_neighbor = masked_mean_with_indices(loss_vec_n, idx_use)
                else:
                    loss_neighbor = loss_neighbor_full

            if scaler is not None:
                scaler.scale(loss_neighbor).backward()
                scaler.unscale_(optimizer.base_optimizer)

                if lesam_alpha != 0.0 and g_center is not None:
                    idx = 0
                    for p in model.parameters():
                        g_c = g_center[idx]
                        if (p.grad is not None) and (g_c is not None):
                            p.grad.mul_(1.0)
                            p.grad.add_(-lesam_alpha * g_c)
                        idx += 1

                optimizer.descent_step(zero_grad=True)
                scaler.update()
            else:
                loss_neighbor.backward()

                if lesam_alpha != 0.0 and g_center is not None:
                    idx = 0
                    for p in model.parameters():
                        g_c = g_center[idx]
                        if (p.grad is not None) and (g_c is not None):
                            p.grad.mul_(1.0)
                            p.grad.add_(-lesam_alpha * g_c)
                        idx += 1

                optimizer.descent_step(zero_grad=True)

            with torch.no_grad():
                correct = (logits.argmax(1) == targets)
                curr_lr = optimizer.base_optimizer.param_groups[0]["lr"]
                log(model, loss_vec.detach().cpu(), correct.cpu(), curr_lr)
                acc = (correct.sum() + 0.01) / (len(targets) + 0.01)

            if lesam_rho_monitor and (args.local_rank in [-1, 0]):
                rho_now = optimizer.last_rho
                if rho_now is not None:
                    rho_epoch_sum += float(rho_now)
                    rho_epoch_cnt += 1
                    rho_epoch_min = min(rho_epoch_min, float(rho_now))
                    rho_epoch_max = max(rho_epoch_max, float(rho_now))
                    if writer is not None:
                        writer.add_scalar("train/rho_now", float(rho_now), global_step=global_step)

            if writer is not None and args.local_rank in [-1, 0]:
                writer.add_scalar(
                    "train/loss_center_full",
                    float(loss_center_full.detach().cpu()),
                    global_step=global_step,
                )
                writer.add_scalar(
                    "train/loss_neighbor_full",
                    float(loss_neighbor_full.detach().cpu()),
                    global_step=global_step,
                )
                writer.add_scalar(
                    "train/loss_center_top",
                    float(loss_center.detach().cpu()),
                    global_step=global_step,
                )
                writer.add_scalar(
                    "train/loss_neighbor_top",
                    float(loss_neighbor.detach().cpu()),
                    global_step=global_step,
                )
                writer.add_scalar("train/acc", float(acc), global_step=global_step)
                writer.add_scalar("train/lr", float(curr_lr), global_step=global_step)

        scheduler.step()

        if lesam_rho_monitor and (args.local_rank in [-1, 0]):
            if rho_epoch_cnt > 0:
                rho_mean = rho_epoch_sum / rho_epoch_cnt
                msg = (
                    f"[rho] epoch={epoch} "
                    f"mean={rho_mean:.6f} min={rho_epoch_min:.6f} "
                    f"max={rho_epoch_max:.6f} steps={rho_epoch_cnt}"
                )
                logger.info(msg)
                if writer is not None:
                    writer.add_scalar("train/rho_epoch_mean", rho_mean, global_step=epoch)
                    writer.add_scalar("train/rho_epoch_min", rho_epoch_min, global_step=epoch)
                    writer.add_scalar("train/rho_epoch_max", rho_epoch_max, global_step=epoch)
            else:
                logger.info(f"[rho] epoch={epoch} no steps collected.")

        if args.local_rank in [-1, 0]:
            model.eval()
            log.eval(len_dataset=len(dataset.test))
            with torch.no_grad():
                tol_cor = 0
                tol_len = 0
                for batch in dataset.test:
                    inputs, targets = (b.to(device, non_blocking=True) for b in batch)
                    if use_fp16 or use_bf16:
                        with autocast(dtype=autocast_dtype):
                            predictions = model(inputs)
                    else:
                        predictions = model(inputs)
                    loss_eval = smooth_crossentropy(predictions, targets)
                    correct = (predictions.argmax(1) == targets)
                    log(model, loss_eval.cpu(), correct.cpu())
                    tol_len += len(targets)
                    tol_cor += correct.sum()
                acc = tol_cor / (tol_len + 0.0)
                if acc > best_acc:
                    best_acc = acc
                    model_to_save = model.module if hasattr(model, "module") else model
                    os.makedirs("../output", exist_ok=True)
                    torch.save(
                        model_to_save.state_dict(),
                        "../output/" + f"{args.name}_checkpoint.bin",
                    )
                if writer is not None:
                    writer.add_scalar("test/acc", scalar_value=acc, global_step=global_step)


def main(args):
    seed = int(getattr(args, "seed", 7))
    if args.local_rank == -1:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", timeout=timedelta(minutes=60))
        args.n_gpu = 1
    args.device = device
    train_name = "train"
    log_path = args.name + "_" + train_name
    os.makedirs("../output/logs", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        filename="../output/logs/" + log_path,
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN,
    )
    if args.local_rank in [-1, 0]:
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logging.getLogger().addHandler(console)

    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, fp16: %s, bf16: %s",
        args.local_rank,
        args.device,
        args.n_gpu,
        bool(args.local_rank != -1),
        getattr(args, "fp16", False),
        getattr(args, "bf16", False),
    )

    model = setup_model(args).to(args.device)
    train(args, model)


if __name__ == "__main__":
    main(args)
