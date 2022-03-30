from datasets.HRSC_dataset import HRSCDataset
from datasets.SSDD_dataset import SSDDataset
from datasets.collater import Collater

if __name__ == '__main__':
    training_set = SSDDataset(root_path='/data/fzh/RSSDD/',
                               set_name='train',
                               augment=True,
                               classes=['ship'])

    """Check some outputs from custom collater.
       1. User can specify the test_idx manually. 
       2. User can visualize scale image result to cancel annotation line (57-65) in collater.py"""
    test_idxs = [0, 1, 2, 3, 4, 5, 6]
    batch = [training_set[idx] for idx in test_idxs]
    collater = Collater(scales=512, keep_ratio=False, multiple=32)
    result = collater(batch)
    print(result)
