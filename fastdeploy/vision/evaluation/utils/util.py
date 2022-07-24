import platform
import multiprocessing as mp


def is_pic(img_name):
    valid_suffix = ['JPEG', 'jpeg', 'JPG', 'jpg', 'BMP', 'bmp', 'PNG', 'png']
    suffix = img_name.split('.')[-1]
    if suffix not in valid_suffix:
        return False
    return True


def get_num_workers(num_workers):
    if not platform.system() == 'Linux':
        # Dataloader with multi-process model is not supported
        # on MacOS and Windows currently.
        return 0
    if num_workers == 'auto':
        num_workers = mp.cpu_count() // 2 if mp.cpu_count() // 2 < 2 else 2
    return num_workers
