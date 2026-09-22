import pandas as pd, numpy as np
import sys
from pathlib import Path

from typing import Literal

sys.path.append('../../finetuning')
from constants import *

class datasets:

    class load:

        @staticmethod
        def kaggle22(subset: Literal['train', 'test', 'both'] = 'train') -> pd.DataFrame:

            match subset:

                case 'train':

                    return pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV)
                
                case 'test':

                    return pd.read_csv(KAGGLE_HERBARIUM_22_TEST_CSV)

                case 'both':

                    return pd.concat(pd.read_csv(KAGGLE_HERBARIUM_22_TRAIN_CSV), pd.read_csv(KAGGLE_HERBARIUM_22_TEST_CSV))

                case _:

                    raise ValueError(f'Invalid arg for \'subset\' must be in [\'train\', \'test\', \'both\'] not {subset}')

        @staticmethod
        def kaggle21(subset: Literal['train', 'test', 'both'] = 'train') -> pd.DataFrame:

            match subset:

                case 'train':

                    return pd.read_csv(KAGGLE_HERBARIUM_21_TRAIN_CSV)
                
                case 'test':

                    return pd.read_csv(KAGGLE_HERBARIUM_21_TEST_CSV)

                case 'both':

                    return pd.concat(pd.read_csv(KAGGLE_HERBARIUM_21_TRAIN_CSV), pd.read_csv(KAGGLE_HERBARIUM_21_TEST_CSV))

                case _:

                    raise ValueError(f'Invalid arg for \'subset\' must be in [\'train\', \'test\', \'both\'] not {subset}')

    @staticmethod
    def unique_plants(df =  pd.DataFrame) -> list[str]:

        return (df["genus"] + "^^^" + df["species"]).unique().tolist()