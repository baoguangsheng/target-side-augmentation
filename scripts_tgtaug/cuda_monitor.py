import argparse
import logging
import os
import torch
import pynvml as nv
import time

logger = logging.getLogger()


class CudaMonitor:
    def __init__(self):
        self.visible_devices = os.environ["CUDA_VISIBLE_DEVICES"] if "CUDA_VISIBLE_DEVICES" in os.environ else None
        if self.visible_devices is not None and len(self.visible_devices) > 0:
            self.visible_devices = list(map(int, self.visible_devices.split(',')))
        else:
            self.visible_devices = [i for i in range(torch.cuda.device_count())]
        self.objs = []

    def __enter__(self):
        nv.nvmlInit()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nv.nvmlShutdown()

    def get_gpu_util(self, device_id):
        h = nv.nvmlDeviceGetHandleByIndex(self.visible_devices[device_id])
        info = nv.nvmlDeviceGetUtilizationRates(h)
        return info.gpu

    def get_mem_usage(self, device_id):
        h = nv.nvmlDeviceGetHandleByIndex(self.visible_devices[device_id])
        info = nv.nvmlDeviceGetMemoryInfo(h)
        return info.total, info.free, info.used

    def idle(self, device_id):
        total, free, used = self.get_mem_usage(device_id)
        # treat the GPU as idle if the mem usage is less than 512M
        return used < 512 * 1024 * 1024

    def all_idle(self, devices):
        return all(self.idle(device_id) for device_id in range(devices))


def print_device_info():
    devices = torch.cuda.device_count()
    logger.info('Running on %s devices.' % devices)

    for device_id in range(devices):
        logger.info('Device %s (%s) with capability(%s) and properties(%s).'
                    % (device_id, torch.cuda.get_device_name(device_id),
                       torch.cuda.get_device_capability(device_id),
                       torch.cuda.get_device_properties(device_id)))


def report_device_usage():
    devices = torch.cuda.device_count()
    with CudaMonitor() as monitor:
        items = []
        for device_id in range(devices):
            total, free, used = monitor.get_mem_usage(device_id)
            gpu = monitor.get_gpu_util(device_id)
            items.append('%s: %.0f%%, %.0fG/%.0fG;' % (device_id, gpu, used/1024/1024/1024, total/1024/1024/1024))
        print(' '.join(items))


def wait_for_device():
    logger.info('Waiting for GPU to complete ...')
    devices = torch.cuda.device_count()
    with CudaMonitor() as monitor:
        while True:
            if monitor.all_idle(devices):
                logger.info('All GPUs are available now.')
                break
            else:
                time.sleep(10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='check', choices=['check', 'wait'],
                        help="check - check the usage of the GPUs. "
                             "wait - wait for the GPUs to finish current job.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filename='./cuda_monitor.log', format="[%(asctime)s %(levelname)s] %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("[%(asctime)s %(levelname)s] %(message)s"))
    logger.addHandler(console_handler)

    if not torch.cuda.is_available():
        logger.exception('CUDA is not available.')
        exit(-1)

    if args.mode == 'check':
        report_device_usage()

    elif args.mode == 'wait':
        report_device_usage()
        wait_for_device()
        report_device_usage()

