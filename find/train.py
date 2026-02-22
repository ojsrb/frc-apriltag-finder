from types import SimpleNamespace
import sys
import wandb
from ultralytics import YOLO
import json

model = YOLO("yolov8n.yaml")

def train_model(config):
    results = model.train(
        data="yolo/dataset/data.yaml",
        epochs=4,
        imgsz=640,
        device="mps",
        save_period=1,

        lr0=config.lr0,
        lrf=config.lrf,
        momentum=config.momentum,
        weight_decay=config.weight_decay,
        warmup_epochs=config.warmup_epochs,
    )
    print(model.metrics)
    return model, model.metrics

def sweep_run():
    with wandb.init("apriltags") as run:
        config = run.config
        model, metrics = train_model(config)
        run.log(metrics.results_dict)
        run.finish(0)

def wandb_sweep():
    config = json.load(open("config.json"))
    sweep_id = wandb.sweep(config, project="apriltags")
    wandb.agent(sweep_id, sweep_run)

def train_best():
    config_json = json.load(open("best_config.json"))
    config = SimpleNamespace(**config_json)

    model, metrics = train_model(config)

    model.save("find/apriltags.pt")

if __name__ == "__main__":
    action = sys.argv[1]
    if action == "train":
        train_best()
    elif action == "sweep":
        wandb_sweep()