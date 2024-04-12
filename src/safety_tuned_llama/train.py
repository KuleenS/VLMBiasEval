import argparse

from config import SafetyTunedLLaMaConfig
from llama import SafetyTunedLLaMa


def main(args):

    config = args.config 

    config_class = SafetyTunedLLaMaConfig(**config)

    llama_model = SafetyTunedLLaMa(config_class)

    llama_model.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c")
    args = parser.parse_args()

    main(args)