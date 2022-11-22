import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
from math import pi


class CompositeAttack(nn.Module):
    """
    Base class for attacks using the composite attack..
    """

    def __init__(self, model, enabled_attack, mode='eval', local_rank=-1,
                 hue_epsilon=None, sat_epsilon=None, rot_epsilon=None,
                 bright_epsilon=None, contrast_epsilon=None, linf_epsilon=None,
                 dataset='cifar10', early_stop=None,
                 start_num=1, iter_num=5, inner_iter_num=10, multiple_rand_start=True, order_schedule='random'):
        super().__init__()
        self.model = model
        self.local_rank = local_rank
        self.device = 'cuda' if local_rank == -1 else 'cuda:' + str(local_rank)
        self.fixed_order = enabled_attack
        self.enabled_attack = tuple(sorted(enabled_attack))
        self.mode = mode
        self.dataset = dataset
        self.seq_num = len(enabled_attack)  # attack_num
        self.attack_pool = (
        self.caa_hue, self.caa_saturation, self.caa_rotation, self.caa_brightness, self.caa_contrast, self.caa_linf)
        self.attack_pool_normal = (
        kornia.enhance.adjust_hue, kornia.enhance.adjust_saturation, kornia.geometry.transform.rotate,
        kornia.enhance.adjust_brightness, kornia.enhance.adjust_contrast, self.get_linf_perturbation)
        self.attack_dict = tuple([self.attack_pool[i] for i in self.enabled_attack])
        if mode == 'eval':
            self.early_stop = True
        elif mode == 'train' or mode == 'fast_train':
            self.early_stop = False
        elif mode == 'eval_grid':
            self.attack_dict = tuple([self.attack_pool_normal[i] for i in self.enabled_attack])
            self.early_stop = True
        elif mode == 'eval_ensemble':
            self.attack_dict = tuple([self.attack_pool_normal[i] for i in self.enabled_attack])
            self.early_stop = True
        else:
            ValueError()
        self.early_stop = early_stop if early_stop is not None else self.early_stop
        self.linf_idx = self.enabled_attack.index(5) if 5 in self.enabled_attack else None

        self.eps_pool_cifar10 = torch.tensor(
            [(-pi, pi), (0.7, 1.3), (-10, 10), (-0.2, 0.2), (0.7, 1.3), (-8 / 255, 8 / 255)], device=self.device)
        self.eps_pool_imagenet = torch.tensor(
            [(-pi, pi), (0.7, 1.3), (-10, 10), (-0.2, 0.2), (0.7, 1.3), (-4 / 255, 4 / 255)], device=self.device)
        self.eps_pool_custom = [hue_epsilon, sat_epsilon, rot_epsilon, bright_epsilon, contrast_epsilon, linf_epsilon]

        if self.dataset == 'cifar10' or self.dataset == 'svhn':
            self.eps_pool = self.eps_pool_cifar10
        elif self.dataset == 'imagenet':
            self.eps_pool = self.eps_pool_imagenet
        else:
            print("Does not specify dataset. Please use either 'cifar10' or 'imagenet'.")
            raise ValueError

        for i in range(6):
            if self.eps_pool_custom[i] is not None:
                self.eps_pool[i] = torch.tensor(self.eps_pool_custom[i])
        if mode == 'eval_ensemble':
            self.eps_pool_ensemble = [self.eps_pool[i] for i in self.enabled_attack]

        if order_schedule not in ('fixed', 'random', 'scheduled'):
            print("order_schedule: {}, should be either 'fixed', 'random', or 'scheduled'.".format(order_schedule))
            raise ValueError
        else:
            self.order_schedule = order_schedule

        self.start_num = start_num
        self.inner_iter_num = inner_iter_num  # Number of component-wise pgd updates
        self.iter_num = iter_num if self.order_schedule == 'scheduled' else 1  # Number of rescheduled

        if mode == 'eval_ensemble':
            self.step_size_pool_ensemble = [2.5 * ((eps[1] - eps[0]) / 2) / self.inner_iter_num for eps in
                                            self.eps_pool_ensemble]
        self.step_size_pool = [2.5 * ((eps[1] - eps[0]) / 2) / self.inner_iter_num for eps in
                               self.eps_pool]  # 2.5 * ε-test / num_steps
        self.multiple_rand_start = multiple_rand_start  # False: start from little epsilon to the upper bound

        self.batch_size = self.adv_val_pool = self.eps_space = self.adv_val_space = self.curr_dsm = \
            self.curr_seq = self.is_attacked = self.is_not_attacked = self.max_loss = None

    def _setup_attack(self):
        hue_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                self.eps_pool[0][1] - self.eps_pool[0][0]) + self.eps_pool[0][0]
        sat_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                self.eps_pool[1][1] - self.eps_pool[1][0]) + self.eps_pool[1][0]
        rot_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                self.eps_pool[2][1] - self.eps_pool[2][0]) + self.eps_pool[2][0]
        bright_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                self.eps_pool[3][1] - self.eps_pool[3][0]) + self.eps_pool[3][0]
        contrast_space = torch.rand((self.start_num, self.batch_size), device=self.device) * (
                self.eps_pool[4][1] - self.eps_pool[4][0]) + self.eps_pool[4][0]
        linf_space = 0.001 * torch.randn([self.start_num, self.batch_size, 3, 32, 32], device=self.device)
        self.adv_val_pool = [hue_space, sat_space, rot_space, bright_space, contrast_space, linf_space]

        if self.mode == 'eval_ensemble':
            hue_clean = torch.zeros(self.batch_size, device=self.device)
            sat_clean = torch.ones(self.batch_size, device=self.device)
            rot_clean = torch.zeros(self.batch_size, device=self.device)
            bright_clean = torch.zeros(self.batch_size, device=self.device)
            contrast_clean = torch.ones(self.batch_size, device=self.device)
            linf_clean = torch.zeros([self.batch_size, 3, 32, 32],
                                     device=self.device) if self.dataset == 'cifar10' or self.dataset == 'svhn' else torch.zeros(
                [self.batch_size, 3, 256, 256], device=self.device)
            self.adv_val_clean_pool = [hue_clean, sat_clean, rot_clean, bright_clean, contrast_clean, linf_clean]
            self.adv_val_clean_space = [self.adv_val_clean_pool[i] for i in self.enabled_attack]

        self.eps_space = [self.eps_pool[i] for i in self.enabled_attack]
        self.adv_val_space = [self.adv_val_pool[i] for i in self.enabled_attack]

    def forward(self, inputs, labels):
        if self.batch_size != inputs.shape[0]:
            self.batch_size = inputs.shape[0]
        self._setup_attack()
        self.is_attacked = torch.zeros(self.batch_size, device=self.device).bool()
        self.is_not_attacked = torch.ones(self.batch_size, device=self.device).bool()
        if self.mode == 'eval_ensemble':
            return self.ensemble_attack(inputs, labels)
        elif self.mode == 'eval_grid':
            return self.grid_search_attack(inputs, labels)
        else:
            return self.caa_attack(inputs, labels)

    def caa_hue(self, data, hue, labels, dsm_update=False):
        if not self.early_stop or dsm_update:
            hue = hue.detach().clone()
            hue.requires_grad_()
            sur_data = data.detach().requires_grad_()

            if self.mode == 'fast_train':
                return kornia.enhance.adjust_hue(data, hue), hue

            new_data = kornia.enhance.adjust_hue(sur_data, hue)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                hue_grad = torch.autograd.grad(cost, hue)[0]
                hue = torch.clamp(hue + torch.sign(hue_grad) * self.step_size_pool[0], self.eps_pool[0][0],
                                  self.eps_pool[0][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_hue(sur_data, hue)
            new_data = torch.clamp(data + (new_data - sur_data).detach().clone(), min=0., max=1.)

            return new_data, hue
        else:
            hue = hue.detach().clone()
            hue[self.is_attacked] = 0
            hue.requires_grad_()
            sur_data = data.detach().requires_grad_()
            ori_is_attacked = self.is_attacked.clone()

            new_data = kornia.enhance.adjust_hue(sur_data, hue)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                # Get successfully attacked indexes
                self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                hue_grad = torch.autograd.grad(cost, hue)[0]
                hue_grad[self.is_attacked] = 0
                hue = torch.clamp(hue + torch.sign(hue_grad) * self.step_size_pool[0], self.eps_pool[0][0],
                                  self.eps_pool[0][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_hue(sur_data, hue)

            return new_data, hue

    def caa_saturation(self, data, saturation, labels, dsm_update=False):
        if not self.early_stop or dsm_update:
            saturation = saturation.detach().clone()
            saturation.requires_grad_()
            sur_data = data.detach().requires_grad_()

            if self.mode == 'fast_train':
                return kornia.enhance.adjust_saturation(data, saturation), saturation

            new_data = kornia.enhance.adjust_saturation(sur_data, saturation)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, saturation)[0]
                saturation = torch.clamp(saturation + torch.sign(_grad) * self.step_size_pool[1], self.eps_pool[1][0],
                                         self.eps_pool[1][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_saturation(sur_data, saturation)
            new_data = torch.clamp(data + (new_data - sur_data).detach().clone(), min=0., max=1.)

            return new_data, saturation
        else:
            saturation = saturation.detach().clone()
            saturation[self.is_attacked] = 1
            saturation.requires_grad_()
            sur_data = data.detach().requires_grad_()
            ori_is_attacked = self.is_attacked.clone()

            new_data = kornia.enhance.adjust_saturation(sur_data, saturation)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                # Get successfully attacked indexes
                self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, saturation)[0]
                _grad[self.is_attacked] = 0
                saturation = torch.clamp(saturation + torch.sign(_grad) * self.step_size_pool[1], self.eps_pool[1][0],
                                         self.eps_pool[1][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_saturation(sur_data, saturation)

            return new_data, saturation

    def caa_rotation(self, data, theta, labels, dsm_update=False):
        if not self.early_stop or dsm_update:
            theta = theta.detach().clone()
            theta.requires_grad_()
            sur_data = data.detach().clone().requires_grad_()

            if self.mode == 'fast_train':
                return kornia.geometry.transform.rotate(data, theta), theta

            new_data = kornia.geometry.transform.rotate(sur_data, theta)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, theta)[0]
                theta = torch.clamp(theta + torch.sign(_grad) * self.step_size_pool[2], self.eps_pool[2][0],
                                    self.eps_pool[2][1]).detach().requires_grad_()
                new_data = kornia.geometry.transform.rotate(sur_data, theta)
            new_data = torch.clamp(data + (new_data - sur_data).detach().clone(), min=0., max=1.)

            return new_data, theta
        else:
            theta = theta.detach().clone()
            theta[self.is_attacked] = 0
            theta.requires_grad_()
            sur_data = data.detach().requires_grad_()
            ori_is_attacked = self.is_attacked.clone()

            new_data = kornia.geometry.transform.rotate(sur_data, theta)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                # Get successfully attacked indexes
                self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, theta)[0]
                _grad[self.is_attacked] = 0
                theta = torch.clamp(theta + torch.sign(_grad) * self.step_size_pool[2], self.eps_pool[2][0],
                                    self.eps_pool[2][1]).detach().requires_grad_()
                new_data = kornia.geometry.transform.rotate(sur_data, theta)

            return new_data, theta

    def caa_brightness(self, data, brightness, labels, dsm_update=False):
        if not self.early_stop or dsm_update:
            brightness = brightness.detach().clone()
            brightness.requires_grad_()
            sur_data = data.detach().requires_grad_()

            if self.mode == 'fast_train':
                return kornia.enhance.adjust_brightness(data, brightness), brightness

            new_data = kornia.enhance.adjust_brightness(sur_data, brightness)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, brightness)[0]
                brightness = torch.clamp(brightness + torch.sign(_grad) * self.step_size_pool[3], self.eps_pool[3][0],
                                         self.eps_pool[3][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_brightness(sur_data, brightness)
            new_data = torch.clamp(data + (new_data - sur_data).detach().clone(), 0.0, 1.0)

            return new_data, brightness
        else:
            brightness = brightness.detach().clone()
            brightness[self.is_attacked] = 0
            brightness.requires_grad_()
            sur_data = data.detach().requires_grad_()
            ori_is_attacked = self.is_attacked.clone()

            new_data = kornia.enhance.adjust_brightness(sur_data, brightness)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                # Get successfully attacked indexes
                self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, brightness)[0]
                _grad[self.is_attacked] = 0
                brightness = torch.clamp(brightness + torch.sign(_grad) * self.step_size_pool[3], self.eps_pool[3][0],
                                         self.eps_pool[3][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_brightness(sur_data, brightness)

            return new_data, brightness

    def caa_contrast(self, data, contrast, labels, dsm_update=False):
        if not self.early_stop or dsm_update:
            contrast = contrast.detach().clone()
            contrast.requires_grad_()
            sur_data = data.detach().requires_grad_()

            if self.mode == 'fast_train':
                return kornia.enhance.adjust_contrast(data, contrast), contrast

            new_data = kornia.enhance.adjust_contrast(sur_data, contrast)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, contrast)[0]
                contrast = torch.clamp(contrast + torch.sign(_grad) * self.step_size_pool[4], self.eps_pool[4][0],
                                       self.eps_pool[4][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_contrast(sur_data, contrast)
            new_data = torch.clamp(data + (new_data - sur_data).detach().clone(), min=0., max=1.)

            return new_data, contrast
        else:
            contrast = contrast.detach().clone()
            contrast[self.is_attacked] = 1
            contrast.requires_grad_()
            sur_data = data.detach().requires_grad_()
            ori_is_attacked = self.is_attacked.clone()

            new_data = kornia.enhance.adjust_contrast(sur_data, contrast)
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                # Get successfully attacked indexes
                self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                _grad = torch.autograd.grad(cost, contrast)[0]
                _grad[self.is_attacked] = 0
                contrast = torch.clamp(contrast + torch.sign(_grad) * self.step_size_pool[4], self.eps_pool[4][0],
                                       self.eps_pool[4][1]).detach().requires_grad_()
                new_data = kornia.enhance.adjust_contrast(sur_data, contrast)

            return new_data, contrast

    def caa_linf(self, data, labels, dsm_update=False):
        if not self.early_stop or dsm_update:
            sur_data = data.detach()
            new_data = data.detach().requires_grad_()
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                img_grad = torch.autograd.grad(cost, new_data)[0]
                adv_data = new_data + self.step_size_pool[5] * torch.sign(img_grad)
                eta = torch.clamp(adv_data - sur_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
                new_data = torch.clamp(sur_data + eta, min=0., max=1.).detach_().requires_grad_()
            new_data = torch.clamp(data + (new_data - sur_data).detach().clone(), min=0., max=1.)

            return new_data
        else:
            sur_data = data.detach()
            new_data = data.detach().requires_grad_()
            ori_is_attacked = self.is_attacked
            for i in range(self.inner_iter_num):
                outputs = self.model(new_data)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                # Get successfully attacked indexes
                self.is_attacked = torch.logical_or(ori_is_attacked, cur_pred != labels)

                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                img_grad = torch.autograd.grad(cost, new_data)[0]
                img_grad[self.is_attacked] = 0
                adv_data = new_data + self.step_size_pool[5] * torch.sign(img_grad)
                eta = torch.clamp(adv_data - sur_data, min=self.eps_pool[5][0], max=self.eps_pool[5][1])
                new_data = torch.clamp(sur_data + eta, min=0., max=1.).detach_().requires_grad_()

            return new_data

    def get_linf_perturbation(self, data, noise):
        return torch.clamp(data + noise, 0.0, 1.0)

    def update_attack_order(self, images, labels, adv_val=None):
        """
        Updating order
        """

        def hungarian(matrix_batch):
            sol = torch.tensor([-i for i in range(1, matrix_batch.shape[0] + 1)], dtype=torch.int32)
            for i in range(matrix_batch.shape[0]):
                topk = 1
                sol[i] = torch.topk(matrix_batch[i], topk)[1][topk - 1]
                while sol.shape != torch.unique(sol).shape:
                    topk = topk + 1
                    sol[i] = torch.topk(matrix_batch[i], topk)[1][topk - 1]
            return sol

        def sinkhorn_normalization(ori_dsm, n_iters=20):
            for _ in range(n_iters):
                ori_dsm /= ori_dsm.sum(dim=0, keepdim=True)
                ori_dsm /= ori_dsm.sum(dim=1, keepdim=True)
            return ori_dsm

        if self.order_schedule == 'fixed':
            if self.curr_seq is None:
                self.fixed_order = tuple([self.enabled_attack.index(i) for i in self.fixed_order])
                self.curr_seq = torch.tensor(self.fixed_order, device=self.device)
        elif self.order_schedule == 'random':
            self.curr_seq = torch.randperm(self.seq_num)
        elif self.order_schedule == 'scheduled':
            if self.curr_dsm is None:
                self.curr_dsm = sinkhorn_normalization(torch.rand((self.seq_num, self.seq_num)))
                self.curr_seq = hungarian(self.curr_dsm)
            self.curr_dsm = self.curr_dsm.detach().requires_grad_()
            adv_img = images.clone().detach().requires_grad_()
            # Relaxation
            original_iter_num = self.inner_iter_num
            self.inner_iter_num = 3
            for tdx in range(self.seq_num):
                prev_img = adv_img.clone()
                adv_img = torch.zeros_like(adv_img)
                for idx in range(self.seq_num):
                    if idx == self.linf_idx:
                        adv_img = adv_img + self.curr_dsm[tdx][idx] * self.attack_dict[idx](prev_img, labels, True)
                    else:
                        _adv_img, _ = self.attack_dict[idx](prev_img, adv_val[idx], labels, True)
                        adv_img = adv_img + self.curr_dsm[tdx][idx] * _adv_img

            self.inner_iter_num = original_iter_num
            outputs = self.model(adv_img)
            with torch.enable_grad():
                cost = F.cross_entropy(outputs, labels)

            dsm_grad = torch.autograd.grad(cost, self.curr_dsm)[0]

            prev_seq = self.curr_seq.clone()
            dsm_noise = torch.zeros_like(self.curr_dsm)
            while torch.equal(prev_seq, self.curr_seq):
                self.curr_dsm = sinkhorn_normalization(torch.exp(self.curr_dsm + dsm_grad + dsm_noise).detach())
                self.curr_seq = hungarian(self.curr_dsm.detach())
                dsm_noise = (torch.randn_like(self.curr_dsm)+1)*2  # Escaping local optimum
        else:
            raise ValueError()

    def caa_attack(self, images, labels):
        attack = self.attack_dict
        adv_img = images.detach().clone()
        adv_val_saved = torch.zeros((self.seq_num, self.batch_size), device=self.device)

        for i in range(self.start_num):
            adv_val = [self.adv_val_space[idx][i] for idx in range(self.seq_num)]
            if self.is_attacked.sum() > 0:
                for att_id in range(self.seq_num):
                    if att_id == self.linf_idx:
                        continue
                    adv_val[att_id].detach()
                    adv_val[att_id][self.is_attacked] = adv_val_saved[att_id][self.is_attacked]
                    adv_val[att_id].requires_grad_()

            for j in range(self.iter_num):
                self.update_attack_order(images, labels, adv_val)
                if self.early_stop:
                    adv_img = adv_img.detach().clone()
                    self.is_not_attacked = torch.logical_not(self.is_attacked)
                    adv_img[self.is_not_attacked] = images[self.is_not_attacked].clone()
                else:
                    adv_img = images.detach().clone()
                adv_img.requires_grad = True

                for tdx in range(self.seq_num):
                    idx = self.curr_seq[tdx]
                    if idx == self.linf_idx:
                        adv_img = attack[idx](adv_img, labels)
                    else:
                        adv_img, adv_val_updated = attack[idx](adv_img, adv_val[idx], labels)
                        adv_val[idx] = adv_val_updated

                if self.is_attacked.sum() > 0:
                    for att_id in range(self.seq_num):
                        if att_id == self.linf_idx:
                            continue
                        adv_val_saved[att_id][self.is_attacked] = adv_val[att_id][self.is_attacked].detach()

                    if self.early_stop and self.is_attacked.sum() == self.batch_size:
                        break
                if self.early_stop and self.is_attacked.sum() == self.batch_size:
                    break

        return adv_img

    def ensemble_attack(self, images, labels):
        attack = self.attack_dict
        adv_img = images
        adv_val = [self.adv_val_space[idx][0].detach().requires_grad_() for idx in range(self.seq_num)]

        self.update_attack_order(adv_img, labels)

        for j in range(self.iter_num):
            for i in range(self.inner_iter_num):
                self.is_not_attacked = torch.logical_not(self.is_attacked)
                adv_img = adv_img.detach()
                adv_img[self.is_not_attacked] = images.data[self.is_not_attacked]
                adv_img.requires_grad = True

                for tdx in range(self.seq_num):
                    idx = self.curr_seq[tdx]
                    adv_img = attack[idx](adv_img, adv_val[idx])
                    if idx == self.linf_idx:
                        linf_adv_img = adv_img.clone()

                outputs = self.model(adv_img)
                cur_pred = outputs.max(1, keepdim=True)[1].squeeze()
                with torch.enable_grad():
                    cost = F.cross_entropy(outputs, labels)
                adv_val_grad = torch.autograd.grad(cost, adv_val, retain_graph=True)[0]
                if self.linf_idx is not None:
                    inner_outputs = self.model(linf_adv_img)
                    self.model.zero_grad()
                    inner_cost = F.cross_entropy(inner_outputs, labels)
                    adv_img_grad = torch.autograd.grad(inner_cost, linf_adv_img)[0]

                for att_id in range(self.seq_num):
                    if att_id == self.linf_idx:
                        adv_val[att_id] = torch.clamp(
                            adv_val[att_id] + torch.sign(adv_img_grad) *
                            self.step_size_pool_ensemble[att_id], self.eps_pool_ensemble[att_id][0],
                            self.eps_pool_ensemble[att_id][1])
                    else:
                        adv_val[att_id] = torch.clamp(
                            adv_val[att_id] + torch.sign(adv_val_grad[att_id]) *
                            self.step_size_pool_ensemble[att_id], self.eps_pool_ensemble[att_id][0],
                            self.eps_pool_ensemble[att_id][1])
                    adv_val[att_id][self.is_attacked] = self.adv_val_clean_space[att_id][self.is_attacked]
                    adv_val[att_id].detach().requires_grad_()
                self.is_attacked = torch.logical_or(self.is_attacked, cur_pred != labels)
                self.is_not_attacked = torch.logical_not(self.is_attacked)

            if self.is_attacked.sum() == self.batch_size:
                break
            elif 1 < self.iter_num != j:
                self.update_attack_order(adv_img, labels)

        return adv_img

    def grid_search_attack(self, images, labels):
        attack = self.attack_dict
        adv_val = [
            [self.eps_pool[i][0] + j * (self.eps_pool[i][1] - self.eps_pool[i][0]) / (self.inner_iter_num + 1) for j in
             range(1, self.inner_iter_num + 1)] for i in self.curr_seq]

        adv_img = images.detach()
        attacked_adv_img = adv_img.clone()
        ori_adv_img_0 = images.detach()
        for att_val_0 in range(self.inner_iter_num):
            adv_img = attack[self.curr_seq[0]](ori_adv_img_0, adv_val[0][att_val_0])
            ori_adv_img_1 = adv_img.clone()
            for att_val_1 in range(self.inner_iter_num):
                adv_img = attack[self.curr_seq[1]](ori_adv_img_1, adv_val[1][att_val_1])
                ori_adv_img_2 = adv_img.clone()
                for att_val_2 in range(self.inner_iter_num):
                    adv_img = attack[self.curr_seq[2]](ori_adv_img_2, adv_val[2][att_val_2])
                    ori_adv_img_3 = adv_img.clone()
                    for att_val_3 in range(self.inner_iter_num):
                        adv_img = attack[self.curr_seq[3]](ori_adv_img_3, adv_val[3][att_val_3])
                        adv_img[self.is_attacked] = attacked_adv_img[self.is_attacked]
                        outputs = self.model(adv_img)
                        cur_pred = outputs.max(1, keepdim=True)[1].squeeze()

                        self.is_attacked = torch.logical_or(self.is_attacked, cur_pred != labels)
                        self.is_not_attacked = torch.logical_not(self.is_attacked)
                        attacked_adv_img[self.is_attacked] = adv_img[self.is_attacked]

                        if self.is_attacked.sum() == self.batch_size:
                            break
                    if self.is_attacked.sum() == self.batch_size:
                        break
                if self.is_attacked.sum() == self.batch_size:
                    break
            if self.is_attacked.sum() == self.batch_size:
                break
        return adv_img
