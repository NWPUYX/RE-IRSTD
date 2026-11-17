"""Exact plateau components for the decoder's 3x3 local-maximum mask."""
import numpy as np

import cv2


def representatives(score,candidates):
    # Adjacent 3x3 local maxima necessarily have equal scores: each maximum's
    # neighborhood contains the other. Hence binary 8-connected components
    # exactly equal the reference equal-score plateaus for this mask.
    count,labels=cv2.connectedComponents(candidates.astype(np.uint8),connectivity=8)
    if count==1:return []
    flat=np.flatnonzero(candidates)
    smallest=np.full(count,score.size,dtype=np.int32)
    np.minimum.at(smallest,labels.ravel()[flat],flat)
    positions=smallest[1:];y,x=np.divmod(positions,score.shape[1])
    order=np.lexsort((x,y,-score[y,x]))
    return [(int(x[i]),int(y[i])) for i in order]
