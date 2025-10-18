import os
import sys
import argparse
import subprocess
from dotenv import load_dotenv
import wandb

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), '..', '.env'))

load_dotenv()

if "WANDB_API_KEY" in os.environ:
    wandb.login(key=os.environ["WANDB_API_KEY"])
    
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src_root = os.path.join(project_root, "src")
sys.path.insert(0, src_root)

def run_training():
    print("\nStarting training with .npy data...")
    train_script = os.path.join(src_root, "training", "train_by_npy.py")
    subprocess.run(["python", train_script], check=True)
    
def run_training_v2():
    print("\nStarting training with .npy data...")
    train_script = os.path.join(src_root, "training", "training_cbam_se_fgf.py")
    subprocess.run(["python", train_script], check=True)
    
def run_fine_tuning():
    print("\nStarting fine tuning with .npy data...")
    train_script = os.path.join(src_root, "training", "fine_tune.py")
    subprocess.run(["python", train_script], check=True)

def run_evaluation():
    print("\nRunning evaluation...")
    eval_script = os.path.join(src_root, "evaluation", "evaluate_by_npy.py")
    subprocess.run(["python", eval_script], check=True)

def run_prediction(video_path):
    print(f"\nPredicting on video: {video_path}")
    predict_script = os.path.join(src_root, "inference", "predict_by_npy.py")
    subprocess.run(["python", predict_script, "--video", video_path], check=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="🎬 Violence Detection 3D-CNN - Project Controller CLI"
    )
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train","train_v2", "evaluate", "predict", "fine_tune"],
        help="Select execution mode: train, evaluate, predict."
    )
    parser.add_argument(
        "--video",
        type=str,
        help="Path to a video file (required only for --mode predict)."
    )

    args = parser.parse_args()

    os.chdir(project_root)

    try:
        if args.mode == "train":
            run_training()
        elif args.mode == "train_v2":
            run_training_v2()
        elif args.mode == "fine_tune":
            run_fine_tuning()
        elif args.mode == "evaluate":
            run_evaluation()
        elif args.mode == "predict":
            if args.video:
                run_prediction(args.video)
            else:
                print("\nYou must provide a video path using --video for prediction mode.")
                sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"\nError occurred while executing subprocess: {e}")
        sys.exit(1)
