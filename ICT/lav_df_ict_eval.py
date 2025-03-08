import torch

from dataloader.LAV_DF import (
    create_lav_df_dataloader,
)
from localize_fakes import extract_fake_periods, run_eval, run_inference, store_preds
from models import ict

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA is not available"

    NUM_WORKERS = 1
    BATCH_SIZE = 1

    data_root = "../../data/LAV-DF"
    print("Loading fake data")
    data = create_lav_df_dataloader(
        data_root,
        split="test",
        video_type="only_video_fake",
        num_workers=NUM_WORKERS,
        batch_size=BATCH_SIZE,
    )
    print("DATASET ITEMS: ", len(data.dataset))

    model_path = "checkpoints/ICT_Base.pth"
    print("Loading model...")
    model = ict.combface_base_patch8_112()
    model.load_state_dict(torch.load(model_path)["model"])
    model.to(device)
    model.eval()
    print("Model loaded")

    print("Running eval...")
    auc, accuracy, best_tresholds = run_eval(model, data, device)
