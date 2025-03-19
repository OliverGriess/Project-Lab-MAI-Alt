import os
import torch
import torch.nn as nn
from models import build_model, get_loss


class Trainer(nn.Module):
    def __init__(self, opt):
        super(Trainer, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = (
            torch.device("cuda:{}".format(opt.gpu_ids[0]))
            if opt.gpu_ids
            else torch.device("cpu")
        )
        self.model = build_model(opt.arch)

        self.step_bias = (
            0
            if not opt.fine_tune
            else int(opt.pretrained_model.split("_")[-1].split(".")[0]) + 1
        )
        if opt.fine_tune:
            state_dict = torch.load(opt.pretrained_model, map_location="cpu")
            self.model.load_state_dict(state_dict["model"])
            self.total_steps = state_dict["total_steps"]
            print(f"Model loaded @ {opt.pretrained_model.split('/')[-1]}")

        params = self.model.parameters()

        if opt.optim == "adam":
            print("Using AdamW optimizer")
            self.optimizer = torch.optim.AdamW(
                params,
                lr=opt.lr,
                betas=(opt.beta1, 0.999),
                weight_decay=opt.weight_decay,
            )
        elif opt.optim == "sgd":
            print("Using SGD optimizer")
            self.optimizer = torch.optim.SGD(
                params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay
            )
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.criterion = get_loss(k=opt.k).to(self.device)
        # self.criterion1 = nn.CrossEntropyLoss()
        self.criterion1 = nn.BCEWithLogitsLoss()

        print("MOVING MODEL TO DEVICE...")
        # self.model.to(opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        print("DONE")

    def adjust_learning_rate(self, min_lr=1e-8):
        for param_group in self.optimizer.param_groups:
            if param_group["lr"] < min_lr:
                return False
            param_group["lr"] /= 10.0
        return True

    def set_input(self, input):
        self.input = input[0].to(self.device)
        self.crops = [[t.to(self.device) for t in sublist]
                      for sublist in input[1]]
        self.label = input[2].to(self.device).float()

    def forward(self):
        self.get_features()
        self.output, self.weights_max, self.weights_org = self.model.forward(
            self.crops, self.features
        )
        self.output = self.output.view(-1)
        loss = self.criterion(
            self.weights_max, self.weights_org
        )
        loss1 = self.criterion1(self.output, self.label)
        self.loss = self.opt.omega * loss + loss1

    def get_loss(self):
        # loss = self.loss.data.tolist()
        # return loss[0] if isinstance(loss, type(list())) else loss
        return self.loss

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.loss.backward()
        self.optimizer.step()

    def get_features(self):
        self.features = self.model.get_features(self.input).to(
            self.device
        )  # shape: (batch_size

    def eval(self):
        self.model.eval()

    def test(self):
        with torch.no_grad():
            self.forward()

    def save_networks(self, save_filename):
        save_path = os.path.join(self.save_dir, save_filename)

        # serialize model and optimizer to dict
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
        }

        torch.save(state_dict, save_path)
