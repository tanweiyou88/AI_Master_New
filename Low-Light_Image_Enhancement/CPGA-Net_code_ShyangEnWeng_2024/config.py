from utils import str2bool
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--ori_data_path', type=str, default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/data/our485/high',  help='Origin image path') # normal light images for training
parser.add_argument('--haze_data_path', type=str, default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/data/our485/low',  help='Haze image path') # low light images for training

parser.add_argument('--val_ori_data_path', type=str, help='Validation origin image path', default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/data/eval15/high') # normal light images for testing
parser.add_argument('--val_haze_data_path', type=str,help='Validation haze image path', default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/data/eval15/low')  # low light images for testing

parser.add_argument('--dataset_type', type=str,  help='...', default='LOL-v1')
parser.add_argument('--ad_dataset_type', type=str,  help='...')

parser.add_argument('--net_name', type=str, default='nets')

parser.add_argument('--use_gpu', type=str2bool, default=False, help='Use GPU')
parser.add_argument('--gpu', type=int, default=-1, help='GPU id')

parser.add_argument('--lr', type=float, default=1e-4, help='Learning Rate. Default=1e-4')
# parser.add_argument('--num_workers', type=int, default=0, help='Number of threads for data loader, for window set to 0') # 0 is the original num_workers proposed by the authors
parser.add_argument('--num_workers', type=int, default=4, help='Number of threads for data loader, for window set to 0')
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--grad_clip_norm', type=float, default=0.1)
parser.add_argument('--print_gap', type=int, default=50, help='number of batches to print average loss ')

# parser.add_argument('--batch_size', type=int, default=16, help='Training batch size') # 16 is the original batch size proposed by the authors
parser.add_argument('--batch_size', type=int, default=24, help='Training batch size')
parser.add_argument('--val_batch_size', type=int, default=1, help='Validation batch size')
parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')


parser.add_argument('--output_name', type=str,  help='...')
parser.add_argument('--sample_output_folder', type=str, default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/data/samples',  help='Validation haze image path')
parser.add_argument('--model_dir', type=str, default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/model')
parser.add_argument('--log_dir', type=str, default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/log')
parser.add_argument('--ckpt', type=str, default='D:/AI_Master_New/Low-Light_Image_Enhancement/CPGA-Net_code_ShyangEnWeng_2024/weights/enhance_color-llie-ResCBAM_g.pkl')
parser.add_argument('--video_dir', type=str,  help='...')

parser.add_argument('--efficient', type=str2bool, default=False, help='Use efficient (DGF) version')



def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
