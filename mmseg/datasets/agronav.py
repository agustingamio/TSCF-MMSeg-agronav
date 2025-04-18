from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset

# define class and palette for agronav
classes = ('soil', 'sidewalk', 'vegetation', 'sky', 'human', 'vehicle', 'building', 'wall', 'others')
palette = [[128, 64, 128], [244, 35, 232], [107, 142, 35], [70, 130, 180], [220, 20, 60], [0, 0, 142], [70, 70, 70], [102, 102, 156], [0, 0, 0]]

# suffix: indica el sufijo de las imagenes y las etiquetas
# reduce_cero_label: inica si la clase 0 la queremos tomar en cuenta
#kwargs: son argumentos adicionales que el config puede pasar (img_dir, ann_dir, etc), mmseg se encarga de ellos
@DATASETS.register_module()
class AgroNavDataset(BaseSegDataset):

    METAINFO = dict(
        classes=classes,
        palette=palette)

    def __init__(self, **kwargs):
        super().__init__(img_suffix='.JPG', seg_map_suffix='.png', reduce_zero_label=False, **kwargs)