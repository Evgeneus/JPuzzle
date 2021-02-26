import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pl_bolts.optimizers.lars_scheduling import LARSWrapper

from models.nets import Encoder
from models.transformer import Transformer


class JPNet(pl.LightningModule):

    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.current_momentum = self.hparams.base_momentum

        # online encoder
        self.online_encoder = Encoder(
            arch=self.hparams.arch,
            low_res='CIFAR' in self.hparams.dataset)

        # regressor to predict correct position of a token
        self.feat_dim = self.online_encoder.encoder.layer4[-1].bn2.weight.shape[0]
        if 'CIFAR' in self.hparams.dataset:
            # number of classes (positions) for x, y coordinate in image grid
            # equal to square root of num_elements in the last cnn channel of encoder
            # depends on image resolution
            self.num_pos = 4
            self.num_tokens = self.num_pos**2
        else:
            raise NotImplementedError

        # define transformer
        depth, heads = 1, 1
        mlp_hidden_dim = 128
        self.transformer = Transformer(self.num_tokens, self.feat_dim, mlp_hidden_dim, depth, heads)

        # define coordinate predictor
        self.predictor_x = torch.nn.Linear(self.feat_dim, self.num_pos)
        self.predictor_y = torch.nn.Linear(self.feat_dim, self.num_pos)

        # linear layer for eval
        self.linear = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(self.feat_dim, self.hparams.num_classes)
        )

    def collect_params(self, models, exclude_bias_and_bn=True):
        param_list = []
        for model in models:
            for name, param in model.named_parameters():
                if exclude_bias_and_bn and any(
                    s in name for s in ['bn', 'downsample.1', 'bias']):
                    param_dict = {
                        'params': param,
                        'weight_decay': 0.,
                        'lars_exclude': True}
                    # NOTE: with the current pytorch lightning bolts
                    # implementation it is not possible to exclude 
                    # parameters from the LARS adaptation
                else:
                    param_dict = {'params': param}
                param_list.append(param_dict)
        return param_list

    def configure_optimizers(self):
        params = self.collect_params([
            self.online_encoder, self.linear])
        optimizer = LARSWrapper(torch.optim.SGD(
            params,
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay))
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr)
        return [optimizer], [scheduler]

    def forward(self, x):
        return self.linear(self.online_encoder.encoder(x))

    def permute_tokens(self, tokens):
        # create labels for  for jigsaw puzzle, i e x, y coordinates
        labels_x = torch.tensor([list(range(self.num_pos)) * self.num_pos] * self.hparams.batch_size, dtype=torch.long)
        labels_x = labels_x.to(self.device)
        labels_y = []
        for i in range(self.num_pos):
            labels_y += [i] * self.num_pos
        labels_y = torch.tensor([labels_y] * self.hparams.batch_size, dtype=torch.long).to(self.device)

        # feature permutation for jigsaw puzzle
        # divide minibatch into buckets and do permutation of tokens in each bucket
        rand_idx = torch.randperm(self.num_tokens)
        labels_x = labels_x[:, rand_idx]
        labels_y = labels_y[:, rand_idx]
        tokens = tokens[:, rand_idx, :]
        # bucket_size = 5
        # for i in range(0, self.hparams.batch_size, bucket_size):
        #     rand_idx = torch.randperm(self.num_tokens)
        #     labels_x[i: i + bucket_size] = labels_x[i: i + bucket_size, rand_idx]
        #     labels_y[i: i + bucket_size] = labels_y[i: i + bucket_size, rand_idx]
        #     tokens[i: i + bucket_size, :] = tokens[i: i + bucket_size, rand_idx, :]
        labels_x = labels_x.view(-1)
        labels_y = labels_y.view(-1)

        return tokens, labels_x, labels_y

    def training_step(self, batch, batch_idx):
        views, labels = batch

        # forward online encoder
        input_online = torch.cat(views, dim=0)
        # reshape cnn features for representing tokens
        features = self.online_encoder(input_online)
        tokens = features.view(self.hparams.batch_size, self.feat_dim, self.num_tokens)
        # each element in the feature grid is a token
        tokens = tokens.permute(0, 2, 1)
        # permute tokens and create labels for for jigsaw puzzle, i e x, y coordinates
        tokens_perm, labels_x, labels_y = self.permute_tokens(tokens.clone())

        # transformer forward pass
        tokens_perm = self.transformer(tokens_perm)

        # predict x,y coordinates for each token
        pred_x = self.predictor_x(tokens_perm).view(-1, self.num_pos)
        pred_y = self.predictor_y(tokens_perm).view(-1, self.num_pos)
        loss = F.cross_entropy(pred_x, labels_x) + F.cross_entropy(pred_y, labels_y)

        # compute accuracy for coordinate predictions
        _, pred_x_class = torch.max(pred_x.data, 1)
        jx_acc = (pred_x_class == labels_x).sum().item() / len(labels_x)
        _, pred_y_class = torch.max(pred_y.data, 1)
        jy_acc = (pred_y_class == labels_y).sum().item() / len(labels_y)

        # train linear layer
        preds_linear = self.linear(features.detach())
        loss_linear = F.cross_entropy(preds_linear, labels)

        # gather results and log stats
        logs = {
            'j_loss': loss.item() / 2,
            'loss_linear': loss_linear.item(),
            'jx_acc': jx_acc,
            'jy_acc': jy_acc,
            'lr': self.trainer.optimizers[0].param_groups[0]['lr']}
        self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)
        return loss + loss_linear * self.hparams.linear_loss_weight

    def on_train_batch_end(self, outputs, batch, batch_idx, dataloader_idx):
        pass

    # def validation_step(self, batch, batch_idx):
    #     images, labels = batch
    #
    #     # predict using online encoder
    #     preds = self(images)
    #
    #     # calculate accuracy @k
    #     acc1, acc5 = self.accuracy(preds, labels)
    #
    #     # gather results and log
    #     logs = {'val/acc@1': acc1, 'val/acc@5': acc5}
    #     self.log_dict(logs, on_step=False, on_epoch=True, sync_dist=True)

    @torch.no_grad()
    def accuracy(self, preds, targets, k=(1,5)):
        preds = preds.topk(max(k), 1, True, True)[1].t()
        correct = preds.eq(targets.view(1, -1).expand_as(preds))

        res = []
        for k_i in k:
            correct_k = correct[:k_i].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / targets.size(0)))
        return res
