import torch
import random
import cv2
import os

PERMUTATIONS = [(2, 4, 3, 1, 0),
                      (4, 3, 1, 0, 2)]
MAX_THREADS = 76


def get_shuffled_img(real_img: torch.Tensor):
     # Split into top (rows 0..499) and bottom (rows 500..end)
    top_half = real_img[:, :500, :]      # [C, 500, W]
    bottom_half = real_img[:, 500:, :]   # [C, H-500, W]

    # Pick one hardcoded permutation for the 5 subimages in bottom_half
    chosen_perm = random.choice(PERMUTATIONS)
    # Each subimage is 500 columns wide: subimage i is at columns [i*500 : (i+1)*500]
    # We'll reorder these 5 subimages side by side
    reordered_subimgs = []
    for i in chosen_perm:
        sub = bottom_half[:, :, i*500 : (i+1)*500]  # shape: [C, H-500, 500]
        reordered_subimgs.append(sub)

    # Concatenate them horizontally in the permuted order
    new_bottom = torch.cat(reordered_subimgs, dim=2)  # [C, H-500, 5*500]

    # Rebuild the full image with top half + shuffled bottom
    fake_img = torch.cat([top_half, new_bottom], dim=1)  # [C, H, W] again

    return fake_img


def process_image(img_path: str, output_path: str):
    img = torch.tensor(cv2.imread(img_path), dtype=torch.float32).permute(2, 0, 1)
    img = get_shuffled_img(img)
    # permute back to [H, W, C] in order to make processing in dataloader work
    img = img.permute(1, 2, 0).numpy()
    img_name = os.path.basename(img_path)
    img_output_path = os.path.join(output_path, img_name)
    cv2.imwrite(img_output_path, img)

if __name__ == "__main__":
    import concurrent.futures
    import tqdm
    
    real_img_path = "/work/scratch/kurse/kurs00079/data/AVLips/balanced_train_stride_5/0_real"
    output_path = "/work/scratch/kurse/kurs00079/data/AVLips/balanced_train_stride_5/unsupervised_1_fake"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    # Get list of all image paths
    image_paths = [os.path.join(real_img_path, img_path) for img_path in os.listdir(real_img_path)]
    
    # Process images in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        # Create a list of futures
        futures = [executor.submit(process_image, img_path, output_path) 
                  for img_path in image_paths]
        
        # Show progress bar while processing
        for _ in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            pass
        