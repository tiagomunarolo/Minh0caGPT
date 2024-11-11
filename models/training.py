import torch
from torch.utils.data import DataLoader, Dataset
from containers import Container
from dependency_injector.wiring import inject, Provide
from typing import Any


@inject
def train_model(
        data_loader: DataLoader[Dataset],
        model: Any = Provide[Container.model],
        epochs: int = Provide[Container.config.epochs],
        summary_writer: Any = Provide[Container.tensor_writer]
) -> Any:
    model = torch.compile(model)
    model = model.fit(data_loader=data_loader, epochs=epochs, summary_writer=summary_writer)
    return model
