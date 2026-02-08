import argparse

import torch.optim as optim

from big_deep import BigDeep, load_model
from alpharmada import AlphArmadaWorker, AlphArmadaTrainer
from configs import Config

def work(model : BigDeep, worker_id: int) -> None:
    worker = AlphArmadaWorker(model, worker_id)
    worker.self_play()

def train(model : BigDeep, current_iter : int) -> None:

    optimizer = optim.AdamW(
            model.parameters(), 
            lr=Config.LEARNING_RATE, 
            weight_decay=Config.L2_LAMBDA
            )
    
    trainer = AlphArmadaTrainer(model, optimizer)
    trainer.train_model(current_iter + 1)


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=False, default="worker", help="Mode: worker / trainer")
    parser.add_argument("--worker_id", type=int, required=False, default=0, help="Machine ID for multi-machine setup")
    args = parser.parse_args()

    model, current_iter = load_model()
    
    if args.mode == "worker":
        work(model, args.worker_id)

    elif args.mode == "trainer": 
        
        train(model, current_iter)

if __name__ == "__main__":
    main()