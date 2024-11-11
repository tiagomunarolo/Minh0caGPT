import logging
from uuid import uuid4
import torch.utils.tensorboard
from datetime import datetime
from tokenizer import CustomTokenizer
from logger import CustomLogger
from config import Config, ModelType
from models.lstm import LSTMTextGenerator
from models.gpt2textgenerator import Gpt2TextGenerator
from dependency_injector.containers import DeclarativeContainer
from dependency_injector import providers


class Container(DeclarativeContainer):
    conf = Config()
    _model = conf.MODEL
    _batch = conf.CONFIG_GPT.BATCH_SIZE if _model == ModelType.GPT else conf.CONFIG_LSTM.BATCH_SIZE
    _epochs = conf.CONFIG_GPT.EPOCHS if _model == ModelType.GPT else conf.CONFIG_LSTM.EPOCHS
    _today = str(datetime.today()).replace(' ', '_')
    config = providers.Configuration(default={
        "model": _model.value,
        "mock_data": conf.MOCKED_DATA,
        "batch_size": _batch,
        "epochs": _epochs,
        "max_len": conf.MAX_LENGTH,
        "force_retrain": conf.FORCE_RETRAIN,
        "sampling": conf.SAMPLING,
        "device": conf.DEVICE,
        "tokenizer_type": conf.TOKENIZER_TYPE,
        "config_lstm": conf.CONFIG_LSTM,
        "config_gpt": conf.CONFIG_GPT
    })

    tensor_writer = providers.Singleton(
        torch.utils.tensorboard.SummaryWriter,
        log_dir=f'/Users/tiago_m2/profiler/{_today}'
    )

    custom_logger = providers.Singleton(
        CustomLogger,
        name='M_GPT',
        level=logging.DEBUG
    )

    logger = providers.Factory(
        custom_logger.provided.get_logger.call(),
    )

    tokenizer = providers.Singleton(
        CustomTokenizer,
        tokenizer_type=config.tokenizer_type(),
    )

    model = providers.Selector(
        config.model,
        lstm=providers.Singleton(
            LSTMTextGenerator,
            tokenizer=tokenizer.provided,
            embedding_dim=config.config_lstm().EMBEDDING_DIM,
            max_length=config.max_len() - 1,
            hidden_size=config.config_lstm().HIDDEN_LSTM_SIZE,
            num_layers=config.config_lstm().NUM_LSTM_LAYERS,
            device=config.device
        ),
        gpt=providers.Singleton(
            Gpt2TextGenerator,
            tokenizer=tokenizer.provided,
            embedding_dim=config.config_gpt().EMBEDDING_DIM,
            max_length=config.max_len() - 1,
            num_heads=config.config_gpt().NUM_HEADS,
            ff_dim=config.config_gpt().FF_DIM,
            num_layers=config.config_gpt().NUM_TRANSFORMER_LAYERS,
            device=config.device
        ),
    )
