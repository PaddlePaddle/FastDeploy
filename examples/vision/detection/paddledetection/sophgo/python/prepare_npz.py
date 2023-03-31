import cv2
import numpy as np

def prepare(img_path, sz):
    im = cv2.imread(img_path)
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, sz)
    im = im.transpose((2,0,1))
    im = im[np.newaxis,...]
    im_scale_y = sz[0] / float(im.shape[2])
    im_scale_x = sz[1] / float(im.shape[3])
    inputs = {}
    inputs['image'] = np.array(im).astype('float32')
    # scale = np.array([im_scale_y, im_scale_x])
    # scale = scale[np.newaxis,...]
    inputs['scale_factor'] = np.array(([im_scale_y, im_scale_x], )).astype('float32')
    np.savez('inputs.npz', image=inputs['image'], scale_factor=inputs['scale_factor'])