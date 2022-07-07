import torch
from torch import nn
import torch.nn.functional as F
import skorch
from skorch.dataset import unpack_data
from skorch.utils import to_device
from skorch.utils import to_numpy
import numpy as np


class MLP(nn.Module):
    def __init__(
            self,
            inputShape=30,
            *args,
            **kwargs
    ):
        # super().__init__(*args, **kwargs)
        super(MLP, self).__init__()
        self.fe1 = nn.Linear(inputShape, 100)
        self.fe2 = nn.Linear(100, 10)

        self.class1 = nn.Linear(10, 2)

    def forward(self, data):
        # print(data.shape)
        x1 = F.relu(self.fe1(data))
        latent = F.relu(self.fe2(x1))
        out = torch.sigmoid(self.class1(latent))
        return {'inp': data, 'pre': out, 'lat': latent}

    # def get_loss(self, y_pred, y_true, *args, **kwargs):
    #     return 0


class TPDNet(skorch.NeuralNet):
    def __init__(self, *args,  **kwargs):
        # make sure to set reduce=False in your criterion, since we need the loss
        # for each sample so that it can be weighted
        super().__init__(*args, **kwargs)

        self.vList = [100, 1]
        self.gammaList = self.CalGammaF(self.vList)

    def get_loss(self, y_pred, y_true, *args, **kwargs):
        # decoded, encoded = y_pred  # <- unpack the tuple that was returned by `forward`
        latent = y_pred['lat']
        data = y_pred['inp']
        y_pred = y_pred['pre']
        loss_unreduced = super().get_loss(y_pred, y_true, *args, **kwargs)

        losselis = self.LossELIS(data, latent)

        # sample_weight = X['sample_weight']
        # loss_reduced = (sample_weight * loss_unreduced).mean()
        lossa = loss_unreduced.mean()
        lossb = losselis / 1000
        # print(lossa, lossb)
        return lossa + lossb

    def LossELIS(self, data, latent):

        loss_ce = self.CE(
            P=self.CalPt(
                dist=self.DistanceSquared(data, data),
                rho=0,
                sigma_array=1,
                gamma=self.gammaList[0],
                v=self.vList[0]),
            Q=self.CalPt(
                dist=self.DistanceSquared(latent, latent),
                rho=0,
                sigma_array=1,
                gamma=self.gammaList[1],
                v=self.vList[1])
        )
        return loss_ce

    def predict_proba(self, X):
        # print(X)
        y_probas = []
        for yp in self.My_forward_iter(X, training=False):
            yp = yp[0] if isinstance(yp, tuple) else yp
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas, 0)
        return y_proba

    def predict_latent(self, X):
        # print(X)
        y_probas = []
        for yp in self.My_forward_iter_latent(X, training=False):
            yp = yp if isinstance(yp, tuple) else yp
            y_probas.append(to_numpy(yp))
        y_proba = np.concatenate(y_probas)
        return y_proba

    def My_forward_iter(self, X, training=False, device='cpu'):

        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for data in iterator:
            # print(data)
            Xi = unpack_data(data)[0]
            yp = self.evaluation_step(Xi, training=training)['pre']
            yield to_device(yp, device=device)

    def My_forward_iter_latent(self, X, training=False, device='cpu'):

        dataset = self.get_dataset(X)
        iterator = self.get_iterator(dataset, training=training)
        for data in iterator:
            # print(data)
            Xi = unpack_data(data)[0]
            yp = self.evaluation_step(Xi, training=training)['lat']
            yield to_device(yp, device=device)

    def DistanceSquared(
        self,
        x,
        y,
        savepath=None,
    ):
        m, n = x.size(0), y.size(0)
        xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
        yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
        dist = xx + yy
        dist.addmm_(x, y.t(), beta=1, alpha=-2)
        # dist.addmm_(1, -2, x, y.t())
        # mat1: Tensor, mat2: Tensor, *, beta: Number=1, alpha: Number=1
        # dist = dist.addmm(mat1=x, mat2=y.t(), beta=1, alpha=-2)
        d = dist.clamp(min=1e-12)
        d[torch.eye(d.shape[0]) == 1] = 1e-12
        # if savepath:
        #     np.save(savepath+'dist.npy', d.detach().cpu().numpy(), )

        return d

    def CE(self, P, Q):
        EPS = 1e-12
        losssum1 = (P * torch.log(Q + EPS)).mean()
        losssum2 = ((1-P) * torch.log(1-Q + EPS)).mean()
        losssum = -1*(losssum1 + losssum2)

        if torch.isnan(losssum):
            input('stop and find nan')
        return losssum

    def CalGammaF(self, vList):
        import scipy
        out = []
        for v in vList:
            a = scipy.special.gamma((v + 1) / 2)
            b = np.sqrt(v * np.pi) * scipy.special.gamma(v / 2)
            out.append(a / b)

        return out

    def CalPt(self, dist, rho, sigma_array, gamma, v=100, split=1):

        if torch.is_tensor(rho):
            dist_rho = (dist - rho.reshape(-1, 1)) / sigma_array.reshape(-1, 1)
        else:
            dist_rho = dist
        dist_rho[dist_rho < 0] = 0
        # print('pass1')
        sample_index_list = torch.linspace(0, dist.shape[0], int(split) + 1)
        # print('pass2')
        for i in range(split):
            # print(i)
            dist_rho_c = dist_rho[int(sample_index_list[i]
                                      ):int(sample_index_list[i + 1])]
            Pij_c = torch.pow(
                gamma * torch.pow((1 + dist_rho_c / v), -1 * (v + 1) / 2) *
                torch.sqrt(torch.tensor(2 * 3.14)), 2)
            if i == 0:
                Pij = Pij_c
            else:
                Pij = torch.cat([Pij, Pij_c], dim=0)
        P = Pij + Pij.t() - torch.mul(Pij, Pij.t())

        return P
