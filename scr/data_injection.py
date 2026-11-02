import pandas as pd
from scr.config import DataBaseUrl,Target_column,logger

class data_injection:
    def __init__(self):
        self.databaseurl = DataBaseUrl
    def load_data(self):
        logger.info("Starting to load the data")
        try:
            data = pd.read_csv(self.databaseurl)
            if data.empty:
                raise ValueError('Data is empty')
            if Target_column not in data.columns:
                raise ValueError('dependent variable is not available in data')
            if data.duplicated().sum() > 0:
                logger.warning(f"{data.duplicated().sum() } number of duplicates are there in data")
            logger.info(f"shape of the data is {data.shape}")
            return data
            logger.info("Data loaded sucessfully")
        except Exception as e:
            logger.exception("Failed to load the data")
        raise