import os
import sys
from os.path import join as pjoin
base_path = os.path.dirname(os.path.abspath("."))
sys.path.append(base_path)

from lib import ove6d

def load_codebooks(model_net, eval_dataset, codebook_path, cfg):
    codebook_saving_dir = pjoin(base_path,'Dataspace/object_codebooks', cfg.DATASET_NAME, 
        'zoom_{}'.format(cfg.ZOOM_DIST_FACTOR), 'views_{}'.format(str(cfg.RENDER_NUM_VIEWS)))

    object_codebooks = ove6d.OVE6D_codebook_generation(codebook_dir=codebook_saving_dir, 
                                                        model_func=model_net,
                                                        dataset=eval_dataset, 
                                                        config=cfg, 
                                                        device=DEVICE)

    print("Object codebooks have been loaded with id's:")
    print(object_codebooks.keys())

    return object_codebooks
