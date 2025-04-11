from ultralytics import YOLO
import lightly_train
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-train a YOLO model with specified parameters.")
    parser.add_argument("--model", type=str, help="Path to the YOLO model file.", default="yolo11n.pt")
    parser.add_argument("--data", type=str, help="Path to the dataset directory.", default="datasets/brain-tumor/train/images")
    parser.add_argument("--epochs", type=int, help="Number of epochs to pre-train.", default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--num-gpus", type=int)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--method", type=str, default="distillation")
    args = parser.parse_args()

    model = YOLO(args.model)
    # model = YOLO("yolov8s.pt") # Uncomment this to start from a COCO checkpoint.

    # Pre-train with lightly-train.
    lightly_train.train(
        out=args.output,  # Output directory.
        data=args.data,                                   # Directory with images.
        model=model,                                      # Pass the YOLO model.
        method=args.method,                                    # Self-supervised learning method.
        epochs=args.epochs,                               # Adjust epochs for faster training.
        batch_size=args.batch_size,                                    # Adjust batch size based on hardware.
        loggers={"wandb": {"project": "my_project"}},
        loader_args={"shuffle": args.shuffle},
        num_nodes=args.num_nodes or "auto",
        devices=args.num_gpus or "auto",
        checkpoint=args.checkpoint,
    )
