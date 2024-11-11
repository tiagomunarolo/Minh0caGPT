from logging import Logger

from dependency_injector.wiring import inject, Provide
from containers import Container
from utils import wire_all_modules
from data_preparation import prepare_data
from models.training import train_model


@inject
def main(
        logger: Logger = Provide[Container.logger],
        retrain: bool = Provide[Container.config.force_retrain]
) -> None:
    """
    Main function to execute the text generation pipeline:
    1. Prepare data (tokenization, sequences generation).
    2. Train model or load pre-trained model.
    3. Generate text from the trained model.
    """
    logger.info("Starting text generation pipeline.")
    # Step 1: Prepare data (load and clean text data > tokenization > build data_loader)
    data_loader = prepare_data()
    logger.info("Data preparation completed.")
    # Step 3: Train or load trained model
    if retrain:
        model = train_model(data_loader=data_loader)
        logger.info("Model training completed.")
    else:
        raise NotImplementedError()

    logger.info("Pipeline completed.")
    # Step 4: Generate text using the trained model
    generated_text = model.generate_text(seed_text='<bos>', max_length=20)
    logger.info(f"Generated Text: {generated_text}")


if __name__ == '__main__':
    """
    Entry point for the text generation pipeline.
    """
    # Set up dependency injection container
    container = Container()
    modules = [__name__]
    modules.extend(wire_all_modules(__file__))
    # Wire container to the required modules
    container.wire(modules=modules)
    # Start the main function
    main()
