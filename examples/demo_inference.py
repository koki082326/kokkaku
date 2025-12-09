# examples/demo_inference.py
# example usage: python examples/demo_inference.py --model LSTM --model_path checkpoints/best_LSTM.pth --input samples/sample_pose.json
import argparse
from src.inference import main as inference_main


if __name__ == '__main__':
inference_main()
