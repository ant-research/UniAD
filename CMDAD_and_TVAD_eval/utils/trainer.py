from .helpers import AverageMeter
import torch

# class EMA():
#     """
#         empirical moving average
#     """

#     def __init__(self, beta):
#         super().__init__()
#         self.beta = beta

#     def update_model_average(self, ma_model, current_model):
#         for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
#             old_weight, up_weight = ma_params.data, current_params.data
#             ma_params.data = self.update_average(old_weight, up_weight)

#     def update_average(self, old, new):
#         if old is None:
#             return new
#         return old * self.beta + (1 - self.beta) * new


class Trainer(object):
    def __init__(
            self,
            model,
            datasetloader,
            train_lr=1e-4,
            gradient_accumulate_every=1,
    ):
        super().__init__()
        self.model = model
        self.gradient_accumulate_every = gradient_accumulate_every

        self.dataloader = datasetloader
        # self.optimizer = torch.optim.AdamW(model.parameters(), lr=train_lr, weight_decay=0.0)
        self.optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=train_lr, weight_decay=0.0)

    # -----------------------------------------------------------------------------#
    # ------------------------------------ api ------------------------------------#
    # -----------------------------------------------------------------------------#

    def train(self, if_calculate_acc, scheduler):
        self.model.train()
        losses = AverageMeter()
        self.optimizer.zero_grad()

        for i, batch in enumerate(self.dataloader):  
            bs = batch.shape[0]         
            loss = self.model(text_input=batch)[0]
            # loss = loss / self.gradient_accumulate_every
            loss.backward()
            losses.update(loss.item(), bs)

            self.optimizer.step()
            self.optimizer.zero_grad()
            # scheduler.step()

        if if_calculate_acc:
            with torch.no_grad():
                pass
                # output = self.model(batch)
                # actions_pred = output[:, :, args.class_dim:args.class_dim+self.model.module.action_dim]\
                #     .contiguous().view(-1, self.model.module.action_dim)  # [bs*T, action_dim]

                # (acc1, acc5), trajectory_success_rate, MIoU1, MIoU2, a0_acc, aT_acc = \
                #     accuracy(actions_pred.cpu(), video_label.cpu(), topk=(1, 5),
                #              max_traj_len=self.model.module.horizon)

                # return torch.tensor(losses.avg), acc1, acc5, torch.tensor(trajectory_success_rate), \
                #        torch.tensor(MIoU1), torch.tensor(MIoU2), a0_acc, aT_acc

        else:
            return torch.tensor(losses.avg)