import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from torch import nn


class Model(pl.LightningModule):

    def __init__(
        self,
        encoder_q,
        epsilon=0.05,
        sinkhorn_iterations=3,
        dim=128,
        T=0.1,
        lr=1e-4,
        stack_frames=2,
        slide=None,
        skill_prior=None,
        skill_prior_encoder=None,
        freeze_prototypes_epoch=1000,
        n_negative_samples=1,
        clutser_T=1,
        reverse_augment=False,
        time_augment=False,
        swav_loss_coef=1,
        steps_per_epoch=None,
        use_lr_scheduler=False,
        use_temperature_scheduler=False,
        cluster_loss_coef=1,
        positive_window=1,
        negative_window=10,
        pretrain_pipeline=None,
    ):

        super(Model, self).__init__()


        self.dim = dim
        self.T = T
        self.clutser_T = clutser_T
        self.positive_window = positive_window
        self.negative_window = negative_window
        self.time_augment = time_augment
        self.stack_frames = stack_frames
        self.slide = slide
        self.sinkhorn_iterations = sinkhorn_iterations
        self.epsilon = epsilon
        self.freeze_prototypes_epoch = freeze_prototypes_epoch
        self.n_negative_samples = n_negative_samples
        self.reverse_augment = reverse_augment

        self.swav_loss_coef = swav_loss_coef

        self.steps_per_epoch = steps_per_epoch
        self.use_lr_scheduler = use_lr_scheduler
        self.cluster_loss_coef = cluster_loss_coef

        self.encoder_q = encoder_q

        self.skill_prior_encoder = skill_prior_encoder
        self.skill_prior = skill_prior

        self.lr = lr
        self.automatic_optimization = False
        self.use_temperature_scheduler = use_temperature_scheduler


        self.pretrain_pipeline = pretrain_pipeline

    # @profile
    def forward(self, im_q, bbox_q, im_k=None, bbox_k=None):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        # vision backbone -> projection [normalized] -> prototype
        zc_q = self.encoder_q(im_q[:, self.stack_frames - 1:], None)  # z: NxC
        zc_k = self.encoder_q(im_k[:, self.stack_frames - 1:], None)  # z: NxC

        return zc_q, zc_k

    # @profile
    def C(self, im_q, bbox_q, zc_q):
        target = torch.softmax(zc_q.detach() / self.T, dim=1)
        # all in one: (bxTxOx4)-> (bxTxf) -> proto logits
        # spatial attention -> temporal attention
        z_logits = self.skill_prior(im_q[:, :self.stack_frames], None)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(z_logits, target)

        return loss

    def configure_optimizers(self):
        e_optimizer = torch.optim.Adam(self.encoder_q.parameters(), lr=self.lr)
        if self.skill_prior_encoder is not None:
            s_optimizer = torch.optim.Adam(
                list(self.skill_prior.parameters()) +
                list(self.skill_prior_encoder.parameters()),
                lr=self.lr)
        else:
            s_optimizer = torch.optim.Adam(self.skill_prior.parameters(),
                                           lr=self.lr)

        if self.use_lr_scheduler:
            e_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                e_optimizer,
                max_lr=1e-3,
                epochs=self.trainer.max_epochs,
                steps_per_epoch=self.steps_per_epoch)
            lambda1 = lambda epoch: 1
            s_scheduler = torch.optim.lr_scheduler.LambdaLR(
                s_optimizer, lr_lambda=[lambda1])
            return (
                {
                    "optimizer": e_optimizer,
                    "lr_scheduler": {
                        "scheduler": e_scheduler,
                    },
                },
                {
                    "optimizer": s_optimizer,
                    "lr_scheduler": s_scheduler
                },
            )
        return e_optimizer, s_optimizer

    def temperature_scheduler(self):
        min_T = 0.1
        max_T = 1

        return min(
            max_T, min_T + (max_T - min_T) /
            (self.trainer.max_epochs / 2) * self.trainer.current_epoch)

    def training_step(self, batch, batch_idx):
        robot_batch, human_batch = batch
        self.training_step_helper(robot_batch, batch_idx)
        self.training_step_helper(human_batch, batch_idx)

    # @profile
    def training_step_helper(self, batch, batch_idx):
        e_opt, s_opt = self.optimizers()
        eps_im, _, _ = batch  #(B,T,3,h,w)
        batch_size = eps_im.shape[0]

        # normalize the prototypes
        with torch.no_grad():
            w = self.encoder_q.prototypes.weight.data.clone()
            w = nn.functional.normalize(w, dim=1, p=2)
            self.encoder_q.prototypes.weight.copy_(w)

            # normalize the skill prior prototypes layer
            if self.skill_prior.normalize:
                v = self.skill_prior.prototypes.weight.data.clone()
                v = nn.functional.normalize(v, dim=1, p=2)
                self.skill_prior.prototypes.weight.copy_(v)

        if self.use_temperature_scheduler:
            self.T = self.temperature_scheduler()

        rep_loss = 0
        cluster_loss = 0
        swav_batch_im_q, swav_batch_im_k = [], []

        for i in range(batch_size):
            im = eps_im[i]
            im_q = torch.stack([
                self.pretrain_pipeline(im[j:j + self.slide + 1])
                for j in range(len(im) - self.slide)
            ])  # (b,slide+1,c,h,w)
            im_k = torch.stack([
                self.pretrain_pipeline(im[j:j + self.slide + 1])
                for j in range(len(im) - self.slide)
            ])  # (b,slide+1,c,h,w)

            if self.reverse_augment:
                reverse_im_q = im_q.flip(dims=[1])
                reverse_im_k = im_k.flip(dims=[1])

                all_im_q = torch.cat([im_q, reverse_im_q], dim=0)
                all_im_k = torch.cat([im_k, reverse_im_k], dim=0)

            else:
                all_im_q = im_q
                all_im_k = im_k

            # collect from different epsisode
            swav_batch_im_q.append(all_im_q)
            swav_batch_im_k.append(all_im_k)

        swav_batch_im_q = torch.cat(swav_batch_im_q, dim=0)
        swav_batch_im_k = torch.cat(swav_batch_im_k, dim=0)

        zc_q, zc_k = self.forward(im_q=swav_batch_im_q,
                                  bbox_q=None,
                                  im_k=swav_batch_im_k,
                                  bbox_k=None)

        if self.reverse_augment:
            chunk_zc_q, chunk_zc_k = torch.chunk(zc_q,
                                                 2 * batch_size), torch.chunk(
                                                     zc_k, 2 * batch_size)
            forward_zc_q, forward_zc_k = chunk_zc_q[::2], chunk_zc_k[::2]
            reverse_zc_q, reverse_zc_k = chunk_zc_q[1::2], chunk_zc_k[1::2]
        else:
            chunk_zc_q, chunk_zc_k = torch.chunk(zc_q,
                                                 batch_size), torch.chunk(
                                                     zc_k, batch_size)
            # forward_zc_q, forward_zc_k, reverse_zc_q, reverse_zc_k = chunk_zc_q, chunk_zc_k, None, None
            forward_zc_q, forward_zc_k, reverse_zc_q, reverse_zc_k = chunk_zc_q, chunk_zc_k, chunk_zc_q, chunk_zc_k

        #swav loss
        with torch.no_grad():
            assignent_q = self.distributed_sinkhorn(zc_k)
            assignent_k = self.distributed_sinkhorn(zc_q)

        rep_loss += 0.5 * (-torch.mean(
            torch.sum(assignent_q * F.log_softmax(zc_q / self.T, dim=1), dim=1)
        ) - torch.mean(
            torch.sum(assignent_k * F.log_softmax(zc_k / self.T, dim=1),
                      dim=1)))

        # cluster loss
        if self.cluster_loss_coef != 0:
            for fz, rz in zip(forward_zc_q, reverse_zc_q):
                totol_segments = len(fz)
                eps_cluster_loss = 0
                for idx in range(totol_segments):
                    positive_proto_idx = np.random.choice(
                        list(np.arange(idx - self.positive_window, idx)) +
                        list(np.arange(idx + 1,
                                       idx + self.positive_window + 1)), 1)
                    positive_proto_idx = np.clip(positive_proto_idx,
                                                 a_min=0,
                                                 a_max=totol_segments - 1)[0]
                    concat_proto_idxs = [[idx, positive_proto_idx]]

                    if self.time_augment:
                        negative_proto_idx = np.random.choice(
                            list(np.arange(idx - self.negative_window)) + list(
                                np.arange(idx + self.negative_window,
                                          totol_segments)),
                            self.n_negative_samples)
                        negative_proto_idx = np.clip(negative_proto_idx,
                                                     a_min=0,
                                                     a_max=totol_segments - 1)
                        concat_proto_idxs.append(negative_proto_idx.tolist())

                    concat_proto_idxs = np.concatenate(concat_proto_idxs)

                    contcat_z = fz[concat_proto_idxs]

                    if self.reverse_augment:
                        revserse_proto_idx = np.random.choice(
                            list(np.arange(idx - self.positive_window, idx)) +
                            list(
                                np.arange(idx + 1,
                                          idx + self.positive_window + 1)),
                            self.n_negative_samples)
                        revserse_proto_idx = np.clip(revserse_proto_idx,
                                                     a_min=0,
                                                     a_max=totol_segments - 1)
                        contcat_z = torch.cat(
                            [contcat_z, rz[revserse_proto_idx]], dim=0)

                    pz, oz = contcat_z[0].unsqueeze(0), contcat_z[1:]

                    logits = torch.mm(pz, oz.t())
                    logits = logits / self.clutser_T
                    label = torch.zeros(1).long().cuda()
                    eps_cluster_loss += nn.CrossEntropyLoss()(logits, label)

                cluster_loss += eps_cluster_loss / totol_segments

        ### optimize the encoder
        # rep_loss = rep_loss / batch_size
        cluster_loss = cluster_loss / batch_size

        encoder_loss = self.swav_loss_coef * rep_loss + self.cluster_loss_coef * cluster_loss
        e_opt.zero_grad()
        self.manual_backward(encoder_loss)
        # cancel gradients for the prototypes
        if self.trainer.current_epoch < self.freeze_prototypes_epoch:
            for name, p in self.encoder_q.named_parameters():
                if "prototypes" in name:
                    p.grad = None
        e_opt.step()

        if self.use_lr_scheduler:
            e_sch, s_sch = self.lr_schedulers()
            e_sch.step()
            s_sch.step()

        # skill prior loss
        if self.reverse_augment:
            prior_im_q = torch.cat(
                torch.chunk(swav_batch_im_q, 2 * batch_size)[::2])
        else:
            prior_im_q = swav_batch_im_q

        prior_z = torch.cat(forward_zc_q)
        skill_prior_loss = self.C(prior_im_q, None, prior_z.detach())
        ### optimize the skill prior
        s_opt.zero_grad()
        self.manual_backward(skill_prior_loss)
        s_opt.step()
        with torch.no_grad():
            wandb.log({
                'encoder_loss':
                encoder_loss,
                'repre_loss':
                rep_loss,
                "cluster_loss":
                cluster_loss,
                'epoch':
                self.trainer.current_epoch,
                'skill_prior_loss':
                skill_prior_loss,
                'encoder_lr':
                e_sch.get_lr()[0] if self.use_lr_scheduler else self.lr,
                'prior_lr':
                s_sch.get_lr()[0] if self.use_lr_scheduler else self.lr,
                'T':
                self.T
            })

    # @profile
    @torch.no_grad()
    def distributed_sinkhorn(self, out):
        Q = torch.exp(out / self.epsilon).t(
        )  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1]  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        Q /= sum_Q

        for it in range(self.sinkhorn_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the colomns must sum to 1 so that Q is an assignment
        return Q.t()