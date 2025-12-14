'''
Purify adversarial images within l_inf <= 16/255
'''

import torch
import os
import argparse
from networks import *
from utils import *
import tqdm
import sys

parser = argparse.ArgumentParser(description='Purify Images')
parser.add_argument('--dir', default='../../help/adv/adv/Ours-TI-DIM', help='input image folder (can have subdirs)')
parser.add_argument('--purifier', type=str, default='NRP',  help='NRP, NRP_resG')
parser.add_argument('--dynamic', action='store_true', help='Dynamic inference (white-box defense)')
parser.add_argument('--output', type=str, default='../../help/purified', help='output folder')
parser.add_argument('--model_pth', type=str, default='../../help/pretrained_purifiers/NRP.pth', help='pretrained model path')
parser.add_argument('--GPU_ID', default='0', type=str, help='GPU_ID')
# 新增：输入控制与性能参数
parser.add_argument('--list_file', type=str, default='', help='text/csv file listing images (relative to --dir or absolute)')
parser.add_argument('--recursive', action='store_true', help='recursively scan --dir for images')
parser.add_argument('--extensions', type=str, default='.png,.jpg,.jpeg', help='comma separated image extensions to include')
parser.add_argument('--batchsize', type=int, default=10, help='DataLoader batch size')
parser.add_argument('--num_workers', type=int, default=1, help='DataLoader num_workers')
parser.add_argument('--eps', type=float, default=16/255, help='l_inf budget used in dynamic mode')
args = parser.parse_args()

def main():
    # 使用解析好的 args 运行主流程
    print(args)

    # Windows 下多进程需保护，且建议禁用 DataLoader 多进程
    if sys.platform.startswith('win') and args.num_workers != 0:
        print(f"Windows detected, forcing num_workers=0 (was {args.num_workers})")
        args.num_workers = 0

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU_ID
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    if args.purifier == 'NRP':
        netG = NRP(3, 3, 64, 23)
        netG.load_state_dict(torch.load(args.model_pth))
    elif args.purifier == 'NRP_resG':
        netG = NRP_resG(3, 3, 64, 23)
        netG.load_state_dict(torch.load(args.model_pth))
    else:
        raise ValueError(f"Unknown purifier: {args.purifier}")
    netG = netG.to(device)
    netG.eval()
    for p in netG.parameters():
        p.requires_grad = False

    print('Parameters (Millions):', sum(p.numel() for p in netG.parameters() if p.requires_grad)/1000000)

    # --------- 构建输入文件列表，支持 help 列表与递归扫描 ---------
    from typing import List, Tuple
    import csv
    import numpy as np

    # 将 utils.read_img 转为张量（CHW，float，[0,1]，RGB），与 custom_dataset 保持一致
    def load_img_tensor(path: str) -> torch.Tensor:
        img = read_img(path)  # numpy, HWC, BGR, [0,1]
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]  # BGR -> RGB
        img = torch.from_numpy(np.ascontiguousarray(np.transpose(img, (2, 0, 1)))).float()
        if img.size(0) == 1:
            img = torch.cat((img, img, img), dim=0)
        if img.size(0) == 4:
            img = img[:3, :, :]
        return img

    def _load_list_file(list_path: str) -> List[str]:
        paths: List[str] = []
        with open(list_path, 'r', encoding='utf-8') as f:
            # 尝试按 csv 读取，若不是 csv 也能逐行解析
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                p = row[0].strip()
                if p:
                    paths.append(p)
        return paths

    def _gather_images(root: str, recursive: bool, exts: List[str]) -> List[str]:
        files: List[str] = []
        if recursive:
            for dirpath, dirnames, filenames in os.walk(root):
                # 忽略 Jupyter 缓存目录
                dirnames[:] = [d for d in dirnames if d != '.ipynb_checkpoints']
                for fn in filenames:
                    if os.path.splitext(fn)[1].lower() in exts:
                        files.append(os.path.join(dirpath, fn))
        else:
            for fn in sorted(os.listdir(root)):
                fp = os.path.join(root, fn)
                if os.path.isfile(fp) and os.path.splitext(fn)[1].lower() in exts:
                    files.append(fp)
        return files

    # 解析扩展名
    exts = [e.strip().lower() for e in args.extensions.split(',') if e.strip()]

    if args.list_file:
        # 列表文件里的路径可相对 --dir 或绝对路径
        listed = _load_list_file(args.list_file)
        input_files = []
        for p in listed:
            fp = p if os.path.isabs(p) else os.path.join(args.dir, p)
            if os.path.isfile(fp):
                input_files.append(fp)
        if not input_files:
            raise FileNotFoundError(f"No valid files from list_file: {args.list_file}")
    else:
        input_files = _gather_images(args.dir, args.recursive, exts)
        if not input_files:
            raise FileNotFoundError(f"No images found in {args.dir} (recursive={args.recursive}, exts={exts})")

    # 使用简易 Dataset，保留相对路径以便输出保持子目录结构
    class FileDataset(torch.utils.data.Dataset):
        def __init__(self, files: List[str], root: str):
            self.files = files
            self.root = os.path.abspath(root)
        def __len__(self):
            return len(self.files)
        def __getitem__(self, idx):
            fp = self.files[idx]
            img = load_img_tensor(fp)
            rel = os.path.relpath(fp, self.root)
            return img, rel

    dataset = FileDataset(input_files, args.dir)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batchsize, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    # --------- 输出目录，按输入子目录结构保存 ---------
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    for i, (img, relpath) in tqdm.tqdm(enumerate(test_loader), total=len(test_loader)):
        # img: (B, C, H, W) or (C,H,W) if batchsize==1 after collation
        if isinstance(relpath, str):
            img = img.unsqueeze(0) if img.dim() == 3 else img
            rel_list = [relpath]
        else:
            rel_list = list(relpath)
        img = img.to(device)

        if args.dynamic:
            eps = args.eps
            img_m = img + torch.randn_like(img) * 0.05
            # Projection 到 l_inf 球
            img_m = torch.min(torch.max(img_m, img - eps), img + eps)
            img_m = torch.clamp(img_m, 0.0, 1.0)
        else:
            img_m = img

        purified = netG(img_m).detach()

        # 逐样本保存
        for b, rel in enumerate(rel_list):
            out_fp = os.path.join(args.output, rel)
            out_dir = os.path.dirname(out_fp)
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            save_img(tensor2img(purified[b]), out_fp)

if __name__ == '__main__':
    main()

