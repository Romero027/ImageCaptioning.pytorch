from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import numpy as np
import pylab
import pickle

annFile='/home/ubuntu/coco/annotations/instances_val2014.json'
captionFile = '/home/ubuntu/coco/annotations/captions_val2014.json'


coco=COCO(annFile)
cocoCaption = COCO(captionFile)

cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['cat'])
imgIds = coco.getImgIds(catIds=catIds )


catsIds = list(filter(lambda x: x < 10000, imgIds))

gt_captions = {}
for imgId in catsIds:
    annIds = cocoCaption.getAnnIds(imgIds=imgId)
    anns = cocoCaption.loadAnns(annIds)
    captions = cocoCaption.showAnns(anns)
    gt_captions[imgId] = captions

print(gt_captions)
print(len(gt_captions))



#with open('cat_caption_10000.pkl', 'wb') as f:
#    pickle.dump(gt_captions, f)
