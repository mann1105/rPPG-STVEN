import logging
from config import get_config
from trainer import Trainer
from dataloader import PPGDataset
from torch.utils.data import DataLoader

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Load config
    config = get_config()
    
    # Setup data
    dataset = PPGDataset(config)
    dataloader = DataLoader(
        dataset,
        batch_size=config.DATASET.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATASET.NUM_WORKERS
    )
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Training loop
    for epoch in range(config.TRAIN.EPOCHS):
        loss = trainer.train_epoch(dataloader, epoch)
        logger.info(f'Epoch {epoch}: loss = {loss:.4f}')
        
        if (epoch + 1) % config.TRAIN.SAVE_INTERVAL == 0:
            trainer.save_checkpoint(epoch, loss)

if __name__ == '__main__':
    main()
