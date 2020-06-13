import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--log_info',
                        type=str,
                        help='info that will be displayed when logging',
                        default='Exp1')

    parser.add_argument('--cls',
                        type=str,
                        help='class of the surgical task',
                        choices=['Knot_Tying', 'Needle_Passing', 'Suturing'],
                        default='Knot_Tying')

    parser.add_argument('--std',
                        type=float,
                        help='standard deviation for gaussian distribution learning',
                        default=1)

    parser.add_argument('--save',
                        action='store_true',
                        help='if set true, save the best model',
                        default=False)

    parser.add_argument('--type',
                        type=str,
                        help='type of model: USDL or MUSDL',
                        choices=['USDL', 'MUSDL'],
                        default='USDL')

    parser.add_argument('--lr',
                        type=float,
                        help='learning rate',
                        default=1e-4)

    parser.add_argument('--weight_decay',
                        type=float,
                        help='L2 weight decay',
                        default=0)

    parser.add_argument('--seed',
                        type=int,
                        help='manual seed',
                        default=1)

    parser.add_argument('--num_workers',
                        type=int,
                        help='number of subprocesses for dataloader',
                        default=8)

    parser.add_argument('--gpu',
                        type=str,
                        help='id of gpu device(s) to be used',
                        default='0,1')

    parser.add_argument('--train_batch_size',
                        type=int,
                        help='batch size for training phase',
                        default=4)

    parser.add_argument('--test_batch_size',
                        type=int,
                        help='batch size for test phase',
                        default=20)

    parser.add_argument('--num_epochs',
                        type=int,
                        help='number of training epochs',
                        default=80)

    return parser


