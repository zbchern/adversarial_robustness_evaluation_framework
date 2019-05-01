import os
import logging


def set_best_gpu(top_k=1):
    best_gpu = _scan(top_k)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, best_gpu))
    return best_gpu


def _scan(top_k):
    CMD1 = 'nvidia-smi| grep MiB | grep -v Default | cut -c 4-8'
    CMD2 = 'nvidia-smi -L | wc -l'
    CMD3 = 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits'

    total_gpu = int(os.popen(CMD2).read())

    assert top_k <= total_gpu, 'top_k > total_gpu !'

    # first choose the free gpus
    gpu_usage = set(map(lambda x: int(x), os.popen(CMD1).read().split()))
    free_gpus = set(range(total_gpu)) - gpu_usage

    # then choose the most memory free gpus
    gpu_free_mem = list(map(lambda x: int(x), os.popen(CMD3).read().split()))
    gpu_sorted = list(sorted(range(total_gpu), key=lambda x: gpu_free_mem[x], reverse=True))[len(free_gpus):]

    res = list(free_gpus) + list(gpu_sorted)
    return res[:top_k]


def print_flags(FLAGS):
    logging.info("################ Hyper Settings ################")
    for key, value in FLAGS.flag_values_dict().items():
        logging.info(key + ": {0}".format(value))


def print_config(config):
    for key, value in config.items():
        if key != '__comment':
            logging.info(key + ": {0}".format(value))
    logging.info("################ Hyper Settings ################")


def init_logging(log_path):
    log = logging.getLogger()
    log.setLevel(logging.INFO)
    logFormatter = logging.Formatter('%(asctime)s [%(levelname)s]: %(message)s')

    fileHandler = logging.FileHandler(log_path)
    fileHandler.setFormatter(logFormatter)
    log.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    log.addHandler(consoleHandler)
