import argparse                  #导入模块用于解析命令行参数
#from solver import Solver               #从solver.py中导入Solver类，包含模型的主要逻辑
from solver_n_msg_conditional import SolverNMsgCond             #适用于处理有条件消息的情况
from solver_n_msg_deepsteg import SolverNMsgMultipleDecodersDeepSteg
from solver_freq_chop_baseline import SolverBaseline
import torch
torch.manual_seed(0)            #设置PyTorch的随机种子为0，以确保结果的可重复性。

def main(config):
    torch.set_num_threads(1000)           #设置PyTorch使用的线程数为1000（实际应用中这通常需要根据硬件资源调整）
    #根据config.model_type选择不同的解算器实例化
    if config.model_type == 'n_msg':
        solver = SolverNMsgMultipleDecodersDeepSteg(config)
    elif config.model_type == 'n_msg_cond':
        solver = SolverNMsgCond(config)
    elif config.model_type == 'baseline':
        solver = SolverBaseline(config)
    else:
        print("dataset type not supported!")
        return -1
    #根据config.mode执行相应的操作
    if config.mode == 'train':
        solver.train()
        solver.test()
    elif config.mode == 'test':
        solver.test()
    elif config.mode == 'sample':
        solver.eval_mode()             #首先将解算器设置为评估模式，然后调用sample_examples()生成样本。
        solver.sample_examples()

if __name__ == '__main__': #判断脚本是否作为主程序运行
    tunable = True             #未被使用（？）
    parser = argparse.ArgumentParser(description='Hide and Speak')
    parser.add_argument('--lr', type=float, default=0.001, help='')            #设置学习率
    parser.add_argument('--num_iters', type=int, default=100, help='number of epochs')     #表示迭代次数（epoch数）
    parser.add_argument('--loss_type', type=str, default='mse', choices=['mse', 'abs'], help='loss function used for training')  #指定使用的损失函数。
    parser.add_argument('--opt', type=str, default='adam', help='optimizer')     #指定优化器。
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test', 'sample'], help='`train` will initiate training, `test` should be used in conjunction with `load_ckpt` to run a test epoch, `sample` should be used in conjunction with `load_ckpt` to sample examples from dataset')
    #训练集路径
    parser.add_argument('--train_path', required=True, type=str, help='path to training set. should be a folder containing .wav files for training')
    #验证集路径
    parser.add_argument('--val_path', required=True, type=str, help='')
    #测试集路径
    parser.add_argument('--test_path', required=True, type=str, help='')
    #批次大小
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    #从.wav文件生成的训练样本数量
    parser.add_argument('--n_pairs', type=int, default=50000, help='number of training examples generated from wav files')
    #隐藏消息的数量
    parser.add_argument('--n_messages', type=int, default=1, help='number of hidden messages')
    #选择的数据集（'timit'or'yoho'）
    parser.add_argument('--dataset', type=str, default='timit', help='select dataset', choices=['timit', 'yoho'])
    #模型类型
    parser.add_argument('--model_type', type=str, default='n_msg', help='`n_msg` default model type, `n_msg_cond` conditional message decoding, `baseline` is the frequency-chop baseline', choices=['n_msg', 'n_msg_cond', 'baseline'])
    #梯度停止标志
    parser.add_argument('--carrier_detach', default=-1, type=int, help='flag that stops gradients from the generated carrier and back. if -1 will not be used, if set to k!=-1 then gradients will be stopped from the kth iteration (used for fine-tuning the message decoder)')
    #频谱图噪声添加
    parser.add_argument('--add_stft_noise', default=0, type=int, help='flag that trasforms the generated carrier spectrogram back to the time domain to simulate real-world conditions. if -1 will not be used, if set to k!=-1 will be used from the kth iteration')
    #载波噪声类型
    parser.add_argument('--add_carrier_noise', default=None, type=str, choices=['gaussian', 'snp', 'salt', 'pepper', 'speckle'], help='add different types of noise the the carrier spectrogram')
    #载波噪声强度
    parser.add_argument('--carrier_noise_norm', default=0.0, type=float, help='strength of carrier noise')
    #添加--adv参数，若在命令行中指定则存储为True，否则为False，默认值为False，用于指示是否使用对抗性训练。
    parser.add_argument('--adv', action='store_true',default=False, help='flag that indicates if adversarial training should be used')


    #编码器/解码器块类型。
    parser.add_argument('--block_type', type=str, default='normal', choices=['normal', 'skip', 'bn', 'in', 'relu'], help='type of block for encoder/decoder')
    #编码器层数
    parser.add_argument('--enc_n_layers', default=3, type=int, help='number of layers in encoder')
    #解码器层数
    parser.add_argument('--dec_c_n_layers', default=4, type=int, help='number of layers in decoder')
    #载波损失项的系数
    parser.add_argument('--lambda_carrier_loss', type=float, default=3.0, help='coefficient for carrier loss term')
    #消息损失项的系数
    parser.add_argument('--lambda_msg_loss', type=float, default=1.0, help='coefficient for message loss term')

    #数据加载的工作线程数
    parser.add_argument('--num_workers', type=int, default=20, help='number of data loading workers')
    #检查点路径
    parser.add_argument('--load_ckpt', type=str, default=None, help='path to checkpoint (used for test epoch or for sampling)')
    #运行目录
    parser.add_argument('--run_dir', type=str, default='.', help='output directory for logs, samples and checkpoints')
    #保存模型间隔
    parser.add_argument('--save_model_every', type=int, default=None, help='')
    #采样间隔
    parser.add_argument('--sample_every', type=int, default=None, help='')
    args = parser.parse_args()

    main(args)
