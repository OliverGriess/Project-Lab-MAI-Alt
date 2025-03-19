from tqdm import tqdm
import torch
from eval.validate import validate
from .trainer import Trainer
from options.train_options import TrainOptions
from data.DataLoader import create_dataloader


def get_val_opt():
    val_opt = TrainOptions().parse(print_options=False)
    val_opt.isTrain = False
    val_opt.data_label = "val"
    val_opt.real_list_path = "./datasets/train/0_real"
    val_opt.fake_list_path = "./datasets/train/1_fake"
    return val_opt


def get_train_opt():
    train_opt = TrainOptions().parse(print_options=False)
    train_opt.isTrain = True
    train_opt.data_label = "train"
    train_opt.real_list_path = "./datasets/test/0_real"
    train_opt.fake_list_path = "./datasets/test/1_fake"
    return train_opt


if __name__ == "__main__":
    opt = get_train_opt()
    val_opt = get_val_opt()
    model = Trainer(opt)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = create_dataloader(opt, train=True)

    val_loader = create_dataloader(val_opt, train=False)

    for epoch in range(opt.epoch):
        model.train()
        for i, data in enumerate(tqdm(data_loader, desc="Training")):
            img = data["img"]
            crops = data["crops"]
            label = data["label"]
            model.total_steps += 1

            model.set_input((img, crops, label))
            model.forward()
            loss = model.get_loss()

            model.optimize_parameters()

            if model.total_steps % opt.loss_freq == 0:
                print(
                    "Train loss: {}\tstep: {}".format(
                        model.get_loss(), model.total_steps
                    ), flush=True
                )

        model.eval()
        ap, fpr, fnr, acc = validate(model.model, val_loader, opt.gpu_ids)
        print(
            "(Val @ epoch {}) acc: {} ap: {} fpr: {} fnr: {}".format(
                epoch + model.step_bias, acc, ap, fpr, fnr
            ), flush=True
        )

        if epoch % opt.save_epoch_freq == 0:
            print("saving the model at the end of epoch %d" %
                  (epoch + model.step_bias))
            model.save_networks("model_epoch_%s.pth" %
                                (epoch + model.step_bias))
