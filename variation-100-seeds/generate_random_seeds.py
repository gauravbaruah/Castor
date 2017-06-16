import argparse
import random

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="generates given number of random seeds")
    ap.add_argument("num_seeds", help="number of random seeds to be generated",\
        type=int)

    args = ap.parse_args()

    random.seed(1234567890)

    for i in range(args.num_seeds):
        print(random.randint(1, 1e6))

    