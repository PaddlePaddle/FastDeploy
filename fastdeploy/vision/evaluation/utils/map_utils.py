from __future__ import absolute_import

import os


def draw_pr_curve(precision,
                  recall,
                  iou=0.5,
                  out_dir='pr_curve',
                  file_name='precision_recall_curve.jpg'):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    output_path = os.path.join(out_dir, file_name)
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        # logger.error('Matplotlib not found, plaese install matplotlib.'
        #              'for example: `pip install matplotlib`.')
        raise e
    plt.cla()
    plt.figure('P-R Curve')
    plt.title('Precision/Recall Curve(IoU={})'.format(iou))
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.plot(recall, precision)
    plt.savefig(output_path)
