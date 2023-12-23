import os
import warnings

from torch.utils.data import DataLoader
from torchinfo import summary

from app.train import *
from config import *
from data.dataset import *
from model import MossFormer
from utils import *


def main():
    
    args = get_config()
    args = args_dict(args)
    print(args.ex_name)
    print(vars(args))

    seed_init(1234)

    if args.action == 'train':

        train_dataset = LibriMixDataset(args.dataset['train'],
                                        segment=args.setting['segment'],
                                        sample_rate= args.setting['sample_rate'])

        train_loader  = DataLoader(train_dataset,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   num_workers=args.num_worker,
                                   collate_fn=collate_fn)
         
        val_dataset   = LibriMixDataset(args.dataset['val'],
                                        sample_rate= args.setting['sample_rate'])
        
        val_loader    = DataLoader(val_dataset,
                                   batch_size=1,
                                   shuffle=False,
                                   num_workers=args.num_worker,
                                   collate_fn=collate_fn)
        
        data_loader   = {'train':train_loader, 'val':val_loader}
        model = MossFormer(speaker_num=2)

        trainer = SeparateTrainer(model, data_loader, args)
        trainer.train()
    # else:
    #     val_dataset   = LibriMixDataset(args.dataset['val'],
    #                                     sample_rate= args.setting['sample_rate'],
    #                                     dynamic_mix=False)
    #     val_loader = DataLoader(val_dataset,
    #                             batch_size=1,
    #                             shuffle=False,
    #                             num_workers=args.num_worker,
    #                             collate_fn=collate_fn)
        
    #     data_loader   = {'val':val_loader}
    #     model  = MossFormer(speaker_num=2)
    #     tester = TimeSepaTest(model, data_loader, args = args)
    #     print('---Test score---')
    #     tester.test()

if __name__ == "__main__":
    main()
