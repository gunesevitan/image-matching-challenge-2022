from datetime import datetime
from pathlib import Path
import logging


ROOT = Path('/home/gunes/Desktop/Kaggle/image-matching-challenge-2022')
DATA = ROOT / 'data'
LOGS = ROOT / 'logs'
MODELS = ROOT / 'models'
EDA = ROOT / 'eda'

logging.basicConfig(
    filename=LOGS / f'{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
