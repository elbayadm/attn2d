import subprocess
import os
import logging
from subprocess import STDOUT, check_output


def exec_cmd(command):
    # return stdout, stderr output of a command
    # return subprocess.Popen(command, shell=True,
                            # stdout=subprocess.PIPE,
                            # stderr=subprocess.PIPE).communicate()

    try:
        out = check_output(command, shell=True,
                            stderr=subprocess.PIPE,
                            timeout=100)
        return out
    except subprocess.TimeoutExpired:
        print('process ran too long')
        return 0


def get_gpu_memory(gpuid):
    # Get the current gpu usage ('cos sometimes oar mess up)
    result = exec_cmd('nvidia-smi -i %d --query-gpu=memory.free \
                      --format=csv,nounits,noheader' % int(gpuid))
    result = int(result.strip())
    return result


def set_env(jobname, manual_gpu_id):
    logger = logging.getLogger(jobname)
    # setup gpu
    try:
        # subprocess.call(['source', 'gpu_setVisibleDevices.sh'], shell=True)
        gpu_id = subprocess.check_output('gpu_getIDs.sh',
                                         shell=True).decode('UTF-8')
        gpu_ids = gpu_id.split()
        num_gpus = 1
        if len(gpu_ids) > 1:
            gpu_id = ",".join(gpu_ids)
            num_gpus = len(gpu_ids)
        else:
            gpu_id = str(gpu_ids[0])
        logger.warn('Get gpuIds output: %s', gpu_id)
        assert type(gpu_id) == str
    except:
        gpu_id = manual_gpu_id
        num_gpus = len(gpu_id.split(','))

    print('Using gpu ids:', gpu_id)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

    # logger.warn('GPU ID: %s | available memory: %dM'
                # % (os.environ['CUDA_VISIBLE_DEVICES'],
                   # get_gpu_memory(gpu_id)))
    return num_gpus

