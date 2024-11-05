import hydra
import pytorch_lightning as pl
import jutils
@hydra.main(config_path="configs", config_name="simclr.yaml")
def main(config):
    pl.seed_everything(63)
    jutils.sysutil.print_config(config, ("callbacks", "logger", "paths", "experiment", "debug", "data", "trainer", "model"))
    model = hydra.utils.instantiate(config.model, _recursive_=True)

    callbacks = []
    if config.get("callbacks"):
        for cb_name, cb_conf in config.callbacks.items():
            if config.get("debug") and cb_name == "model_checkpoint":
                continue
            callbacks.append(hydra.utils.instantiate(cb_conf))

    logger = []
    if config.get("logger"):
        for lg_name, lg_conf in config.logger.items():
            if config.get("debug") and lg_name == "wandb":
                continue
            logger.append(hydra.utils.instantiate(lg_conf))

    trainer = hydra.utils.instantiate(
        config.trainer,
        callbacks=callbacks,
        logger=logger,
        _convert_="partial",
        log_every_n_steps=200,
    )

    trainer.fit(model)

if __name__ == "__main__":
    main()
