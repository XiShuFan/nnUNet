import os

from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA

default_num_processes = 8 if 'nnUNet_def_n_proc' not in os.environ else int(os.environ['nnUNet_def_n_proc'])

ANISO_THRESHOLD = 3  # determines when a sample is considered anisotropic (3 means that the spacing in the low
# resolution axis must be 3x as large as the next largest spacing)

default_n_proc_DA = get_allowed_n_proc_DA()

os.environ['nnUNet_raw'] = 'D:\\xsf\\Dataset\\nnUNet_raw'
os.environ['nnUNet_preprocessed'] ='D:\\xsf\\Dataset\\nnUNet_preprocessed'
os.environ['nnUNet_results'] ='D:\\xsf\\Dataset\\nnUNet_results'