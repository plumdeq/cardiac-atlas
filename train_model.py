# coding:utf8
"""
copyright Asan AGIBETOV <asan.agibetov@gmail.com>

Test training phases of the model

"""
# Third-party imports
import click


# Cross-library imports
import dataloader
import finetuned_convnet
import training


@click.command()
def main():
    """Training from scratch runs and terminates"""
    loader = dataloader.AmyloidDataloader()
    model = finetuned_convnet.FinetunedConvnet(model_config="new", use_cuda=True)
    trainer = training.Trainer(model, loader, use_cuda=True, num_epochs=2)

    model = trainer.train()



if __name__ == "__main__":
    main()
