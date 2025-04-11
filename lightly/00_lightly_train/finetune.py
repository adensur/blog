from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb
import argparse
import os


def main():
    parser = argparse.ArgumentParser(description="Train a YOLO model with specified parameters.")
    parser.add_argument("--model", type=str, help="Path to the YOLO model file.", default="yolo11n.pt")
    parser.add_argument("--data", type=str, help="Path to the dataset configuration file.", default="datasets/pidray_converted/data.yaml")
    parser.add_argument("--epochs", type=int, help="Number of epochs to train.", default=2)
    parser.add_argument("--output", type=str)
    args = parser.parse_args()

    # Load a model
    model = YOLO(args.model)
    wandb.init(project="ultralytics", job_type="training")
    add_wandb_callback(model)

    # Train the model
    results = model.train(data=args.data, epochs=args.epochs, imgsz=640, project="maksim_ultralytics")
    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        model.save(args.output)

    print("Results: \n", results)

if __name__ == "__main__":
    main()