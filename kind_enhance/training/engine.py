import json
import inspect
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from kind_enhance.training.losses.base import compute_loss,refinement_loss
from kind_enhance.infrastructure.runtime import image_on_device,read_checkpoint,resource_snapshot,save_checkpoint,sha256,write_json

def train(model, training, validation, device, stage, config, epochs, directory, fingerprint, source=None,
          validation_scoring=None, loss_function=None):
    directory.mkdir(parents=True, exist_ok=True)
    joint = config.get("decomposition_lr")
    active_loss = loss_function or (refinement_loss if joint else compute_loss)
    config = {**config, "loss_source_sha256": sha256(Path(inspect.getfile(active_loss))),
              "metrics_source_sha256": sha256(Path(__file__).resolve().parents[1] / "evaluation" / "reference.py")}
    if joint:
        enhancement_parameters = [*model.restoration.parameters(), *model.adjustment.parameters()]
        decomposition_parameters = list(model.decomposition.parameters())
        parameters = enhancement_parameters + decomposition_parameters
        optimizer_groups = [
            {"params": enhancement_parameters},
            {"params": decomposition_parameters, "lr": joint},
        ]
    elif stage == "enhance":
        model.decomposition.load_state_dict(source["model"], strict=True)
        model.decomposition.requires_grad_(False)
        parameters = [param for param in model.parameters() if param.requires_grad]
        config = {**config, "decomp_sha256": source["sha256"]}
        optimizer_groups = parameters
    else:
        parameters = list(model.decomposition.parameters())
        optimizer_groups = parameters
    active = model.decomposition if stage == "decomp" else model
    optimizer = torch.optim.Adam(optimizer_groups, lr=config["lr"])
    start_epoch, best, history = 1, None, []
    last_path, best_path = directory / "last.pt", directory / "best.pt"
    if last_path.exists():
        saved = read_checkpoint(last_path, device, fingerprint, stage)
        if saved["config"] != config:
            raise ValueError(f"Cannot resume with different settings: {directory}")
        active.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        start_epoch, best, history = saved["epoch"] + 1, saved["best"], saved["history"]
        if best is not None and history[-1]["epoch"] == best["epoch"] and (
            not best_path.exists() or read_checkpoint(best_path, device, fingerprint, stage)["epoch"] != best["epoch"]
        ):
            save_checkpoint(best_path, {key: value for key, value in saved.items() if key != "optimizer"})
        write_json(directory / "history.json", history)
    training.seed = config["seed"]
    for epoch in range(start_epoch, epochs + 1):
        tick = time.perf_counter()
        model.train()
        if stage == "enhance" and not joint:
            model.decomposition.eval()
        training.epoch = epoch
        generator = torch.Generator().manual_seed(config["seed"] + epoch)
        loader = DataLoader(training, batch_size=config["batch"], shuffle=True, generator=generator)
        total = 0.0
        for low, high, _ in loader:
            low, high = image_on_device(low, device), image_on_device(high, device)
            optimizer.zero_grad(set_to_none=True)
            loss = (loss_function(model, low, high, config) if loss_function is not None
                    else refinement_loss(model, low, high, config) if joint
                    else compute_loss(model, low, high, stage, config.get("structure", 0)))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 5, error_if_nonfinite=True)
            optimizer.step()
            total += loss.item() * low.shape[0]
        row = {
            "epoch": epoch, "train_loss": total / len(training), "validation": None,
        }
        improved = False
        if epoch % config.get("validate_every", 1) == 0 or epoch == epochs:
            if validation_scoring is not None:
                result, criterion = validation_scoring(model, validation, device)
            else:
                raise ValueError("An explicit validation_scoring callback is required")
                criterion = (-result["loss"], 0) if stage == "decomp" else (result["psnr_db"], result["ssim"])
            row["validation"] = result
            improved = best is None or criterion > tuple(best["criterion"])
            if improved:
                best = {"epoch": epoch, "criterion": list(criterion), "validation": result}
        row.update(seconds=time.perf_counter() - tick, resources=resource_snapshot(device))
        history.append(row)
        state = {
            "stage": stage, "epoch": epoch, "config": config, "model": active.state_dict(),
            "manifest": fingerprint, "best": best, "history": history,
        }
        save_checkpoint(last_path, {**state, "optimizer": optimizer.state_dict()})
        if improved:
            save_checkpoint(best_path, state)
        write_json(directory / "history.json", history)
        print(json.dumps({"run": directory.name, **row}, allow_nan=False), flush=True)
    return {"best": best, "epochs": len(history), "config": config, "checkpoint": best_path.as_posix()}
