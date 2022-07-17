import torch
from torchid.dynonet.module.lti import SisoLinearDynamicalOperator
from torchid.dynonet.module.static import SisoStaticNonLinearity


class WHNet(torch.nn.Module):
    def __init__(self, nb_1=8, na_1=8, nb_2=8, na_2=8):
        super(WHNet, self).__init__()
        self.nb_1 = nb_1
        self.na_1 = na_1
        self.nb_2 = nb_2
        self.na_2 = na_2
        self.G1 = SisoLinearDynamicalOperator(n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh')
        self.G2 = SisoLinearDynamicalOperator(n_b=self.nb_2, n_a=self.na_2, n_k=0)

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin


class WHNet3(torch.nn.Module):
    def __init__(self, nb_1=3, na_1=3, nb_2=3, na_2=3):
        super(WHNet3, self).__init__()
        self.nb_1 = nb_1
        self.na_1 = na_1
        self.nb_2 = nb_2
        self.na_2 = na_2
        self.G1 = SisoLinearDynamicalOperator(n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = SisoStaticNonLinearity(n_hidden=10, activation='tanh')
        self.G2 = SisoLinearDynamicalOperator(n_b=self.nb_2, n_a=self.na_2, n_k=0)

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin


class DynoWrapper(torch.nn.Module):
    def __init__(self, dyno, n_in, n_out):
        super(DynoWrapper, self).__init__()
        self.dyno = dyno
        self.n_in = n_in
        self.n_out = n_out

    def forward(self, u_in):
        u_in = u_in[None, :, :]  # [bsize, seq_len, n_in]
        y_out = self.dyno(u_in)  # [bsize, seq_len, n_out]
        n_out = y_out.shape[-1]
        y_out_ = y_out.reshape(-1, n_out)
        # output size: [bsize*seq_len, n_out] or [bsize*seq_len, ]
        return y_out_


class WHF(torch.nn.Module):
    def __init__(self):
        super(WHF, self).__init__()

    def forward(self, x):
        y = -torch.nn.functional.elu(-10 / 11 * (x - 0.0), alpha=1.0) + 0.0
        return y


class WHSys(torch.nn.Module):
    def __init__(self, nb_1=3, na_1=3, nb_2=3, na_2=3):
        super(WHSys, self).__init__()
        self.nb_1 = nb_1
        self.na_1 = na_1
        self.nb_2 = nb_2
        self.na_2 = na_2
        self.G1 = SisoLinearDynamicalOperator(n_b=self.nb_1, n_a=self.na_1, n_k=1)
        self.F_nl = WHF()
        self.G2 = SisoLinearDynamicalOperator(n_b=self.nb_2, n_a=self.na_2, n_k=0)

        with torch.no_grad():
            self.G1.a_coeff[:] = torch.tensor([[[-1.2091,  0.2653,  0.1169]]])
            self.G1.b_coeff[:] = torch.tensor([[[0.2323, -0.6494,  0.5902]]])
            self.G2.a_coeff[:] = torch.tensor([[[-1.2303, -0.1718,  0.4465]]])
            self.G2.b_coeff[:] = torch.tensor([[[0.0957, -0.1619,  0.1107]]])

    def forward(self, u):
        y1_lin = self.G1(u)
        y1_nl = self.F_nl(y1_lin)  # B, T, C1
        y2_lin = self.G2(y1_nl)  # B, T, C2

        return y2_lin