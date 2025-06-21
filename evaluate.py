from models.loca import build_model
from utils.data import FSC147Dataset
from utils.arg_parser import get_argparser

import argparse
import os

import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel
from torch import distributed as dist


@torch.no_grad()
def evaluate(args):

    if 'SLURM_PROCID' in os.environ:
        world_size = int(os.environ['SLURM_NTASKS'])
        rank = int(os.environ['SLURM_PROCID'])
        gpu = rank % torch.cuda.device_count()
        print("Running on SLURM", world_size, rank, gpu)
    else:
        world_size = int(os.environ['WORLD_SIZE'])
        rank = int(os.environ['RANK'])
        gpu = int(os.environ['LOCAL_RANK'])

    torch.cuda.set_device(gpu)
    device = torch.device(gpu)

    dist.init_process_group(
        backend='nccl', init_method='env://',
        world_size=world_size, rank=rank
    )

    model = DistributedDataParallel(
        build_model(args).to(device),
        device_ids=[gpu],
        output_device=gpu
    )
    state_dict = torch.load(os.path.join(args.model_path, f'{args.model_name}.pt'))['model']
    state_dict = {k if 'module.' in k else 'module.' + k: v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)

    img_list = []
    predicted_density_list = []
    
    for split in ['val', 'test']:
        test = FSC147Dataset(
            args.data_path,
            args.image_size,
            split=split,
            num_objects=args.num_objects,
            tiling_p=args.tiling_p,
        )
        test_loader = DataLoader(
            test,
            sampler=DistributedSampler(test),
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=args.num_workers
        )
        ae = torch.tensor(0.0).to(device)
        se = torch.tensor(0.0).to(device)
        model.eval()

        
        catch_index = 0


        for img, bboxes, density_map in test_loader:
            img = img.to(device)
            bboxes = bboxes.to(device)
            # density_map = density_map.to(device)

            out, _ = model(img, bboxes)
            # ae += torch.abs(
            #     density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            # ).sum()
            # se += ((
            #     density_map.flatten(1).sum(dim=1) - out.flatten(1).sum(dim=1)
            # ) ** 2).sum()

            if catch_index < 10 :
                catch_image = img[0,:,:,:].to("cpu")
                normalize_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1) # reshape for broadcasting
                normalize_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1) # reshape for broadcasting
                catch_image =  catch_image * normalize_std + normalize_mean
                catch_image = torch.clamp(catch_image, 0.0, 1.0)
                # catch_image = 
                
                # catch_image = catch_image / catch_image.max()
                # print(catch_image)
                img_list.append(catch_image.permute(1,2,0))
                catch_density_map =  out[0,:,:,:].to("cpu")
                catch_density_map = torch.abs(catch_density_map)
                catch_density_map = catch_density_map / catch_density_map.max()
                predicted_density_list.append(catch_density_map.permute(1,2,0))
                # print(img[0,:,:,:].shape)
                catch_index += 1

                print(f"{catch_image.min()} / {catch_image.max()} / {catch_density_map.min()} / {catch_density_map.max()}")
                
            
            else :
                break

        dist.all_reduce_multigpu([ae])
        dist.all_reduce_multigpu([se])

        # if rank == 0:
        #     print(
        #         f"{split.capitalize()} set",
        #         f"MAE: {ae.item() / len(test):.2f}",
        #         f"RMSE: {torch.sqrt(se / len(test)).item():.2f}",
        #     )


        # figs, axes = plt.subplots(5, 2, figsize=(10, 10))

        # 遍歷每一行 (row)
        # for i in range(5):
        #     # # 左邊的子圖 (第 i 行，第 0 列) 顯示 img
        #     # ax_left = axes[i, 0]
        #     # ax_left.imshow(img_list[2*i])
        #     # ax_left.axis('off') # 關閉座標軸

        #     # # 右邊的子圖 (第 i 行，第 1 列) 顯示 out
        #     # ax_right = axes[i, 1]
        #     # ax_left.imshow(img_list[2*i + 1])
        #     # ax_right.axis('off') # 關閉座標軸

        #     plt.imshow(img_list[i])
        #     plt.show()

        #     plt.imshow(predicted_density_list[i])
        #     plt.show()
        
        
            

        # # --- 調整佈局並顯示 ---
        # plt.tight_layout() # 自動調整子圖間距，避免重疊
        # plt.show() # 顯示所有圖片       

        # fig, axes = plt.subplots(5, 2, figsize=(12, 25)) # 調整 figsize 使圖片更大更清晰

        # for i in range(5):
        #     # 左邊的子圖 (第 i 行，第 0 列) 顯示 img_list[i]
        #     ax_left = axes[i, 0]
        #     ax_left.imshow(img_list[i])
        #     ax_left.set_title(f'Original Image {i+1}')
        #     ax_left.axis('off') # 關閉座標軸

        #     # 右邊的子圖 (第 i 行，第 1 列) 顯示 predicted_density_list[i]
        #     ax_right = axes[i, 1]
        #     ax_right.imshow(predicted_density_list[i], cmap='viridis') # 密度圖通常用熱力圖更直觀
        #     ax_right.set_title(f'Predicted Density {i+1}')
        #     ax_right.axis('off') # 關閉座標軸
        #     # 如果你想顯示密度圖的顏色條，可以在這裡添加，但可能需要調整布局
        #     # fig.colorbar(ax_right.imshow(predicted_density_list[i], cmap='viridis'), ax=ax_right, orientation='vertical')


        # # --- 最後一步：調整佈局並一次性顯示所有圖片 ---
        # plt.tight_layout() # 自動調整子圖間距，避免重疊
        # plt.show() # 只在所有圖片都繪製到子圖上之後，呼叫一次 plt.show()

    plt.imshow(img_list[0])
    plt.show()

    plt.imshow(predicted_density_list[0])
    plt.show()

    # --- 1. 定義資料夾名稱 ---
    gt_folder = "image_gt"
    pred_density_folder = "predicted_density"

    # --- 2. 創建資料夾（如果它們不存在） ---
    # os.makedirs(path, exist_ok=True) 會在路徑不存在時創建它，如果已存在則什麼也不做
    os.makedirs(gt_folder, exist_ok=True)
    os.makedirs(pred_density_folder, exist_ok=True)

    print(f"Saving images to '{gt_folder}' and '{pred_density_folder}'...")

    # --- 3. 儲存 img_list 中的圖片 ---
    for i, img_data in enumerate(img_list):
        # 創建一個新的 Figure 和 Axes 對象來繪製單張圖片
        # 這是因為 plt.savefig() 儲存的是當前 "活動" 的 Figure
        # 每次新建 Figure 可以確保每張圖片都是獨立的，不會互相影響
        fig_gt, ax_gt = plt.subplots(figsize=(img_data.shape[1]/100, img_data.shape[0]/100), dpi=100)
        # plt.figure() 也可以，但 subplots 更靈活，確保有 axes
        
        ax_gt.imshow(img_data)
        # ax_gt.set_title(f'Original Image {i+1}')
        ax_gt.axis('off')

        # 定義圖片的完整路徑
        file_path_gt = os.path.join(gt_folder, f'image_gt_{i+1:03d}.png') # 使用 :03d 格式化為三位數字
        
        # 儲存圖片
        plt.savefig(file_path_gt, bbox_inches='tight', pad_inches=0) # bbox_inches='tight' 移除多餘邊白
        plt.close(fig_gt) # 關閉當前 Figure，釋放記憶體

    # --- 4. 儲存 predicted_density_list 中的圖片 ---
    for i, density_data in enumerate(predicted_density_list):
        fig_pred, ax_pred = plt.subplots(figsize=(density_data.shape[1]/100, density_data.shape[0]/100), dpi=100)
        
        ax_pred.imshow(density_data, cmap='viridis') # 密度圖使用熱力圖
        # ax_pred.set_title(f'Predicted Density {i+1}')
        ax_pred.axis('off')
        
        file_path_pred = os.path.join(pred_density_folder, f'predicted_density_{i+1:03d}.png')
        
        plt.savefig(file_path_pred, bbox_inches='tight', pad_inches=0)
        plt.close(fig_pred) # 關閉當前 Figure

    print("All images saved successfully!")
        




    dist.destroy_process_group()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('LOCA', parents=[get_argparser()])
    args = parser.parse_args()
    evaluate(args)
