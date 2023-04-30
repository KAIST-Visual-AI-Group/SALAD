import gc

import rich
from rich.tree import Tree
from rich.syntax import Syntax
import torch
from omegaconf import DictConfig, OmegaConf


def print_config(
    config,
    fields=(
        "trainer",
        "model",
        "callbacks",
        "logger",
        "seed",
        "name",
    ),
    resolve: bool = True,
    save_config: bool = False,
) -> None:
    """Prints content of DictConfig using Rich library and its tree structure.
    Args:
        config (DictConfig): Configuration composed by Hydra.
        fields (Sequence[str], optional): Determines which main fields from config will
        be printed and in what order.
        resolve (bool, optional): Whether to resolve reference fields of DictConfig.
    """

    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)

    for field in fields:
        branch = tree.add(field, style=style, guide_style=style)

        config_section = config.get(field)
        branch_content = str(config_section)
        if isinstance(config_section, DictConfig):
            branch_content = OmegaConf.to_yaml(config_section, resolve=resolve)

        branch.add(Syntax(branch_content, "yaml"))

    rich.print(tree)

    if save_config:
        with open("config_tree.log", "w") as fp:
            rich.print(tree, file=fp)


def clean_gpu():
    gc.collect()
    torch.cuda.empty_cache()
