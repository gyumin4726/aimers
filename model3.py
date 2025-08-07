import torch
import torch.nn as nn

# ────────────────────────────────────────────────────────────────
# 1) TCNBlock, EncoderTCN, DecoderTCN, Seq2Seq 모델 정의
# ────────────────────────────────────────────────────────────────
class TCNBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1, dropout=0.2):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch, kernel_size,
                                           padding=padding, dilation=dilation)),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
            nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, kernel_size,
                                           padding=padding, dilation=dilation)),
            nn.LeakyReLU(0.1), nn.Dropout(dropout),
        )
        self.chomp = lambda x: x[:, :, :-padding]
        self.downsample = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else None
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.net(x)
        out = self.chomp(out)

        res = x
        if self.downsample is not None:
            res = self.downsample(res)

        # chomp 처리 시 안전하게 자르기
        if res.size(2) > out.size(2):
            res = res[:, :, :out.size(2)]
        elif out.size(2) > res.size(2):
            out = out[:, :, :res.size(2)]

        return self.relu(out + res)


class EncoderTCN(nn.Module):
    def __init__(self, in_ch, channels):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            layers.append(TCNBlock(in_ch, ch, kernel_size=3, dilation=2**i))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, feat, T)
        return self.tcn(x)  # (B, C_lat, T)


class DecoderTCN(nn.Module):
    def __init__(self, m_dim, c_lat, x_dim):
        super().__init__()
        # causal TCN decoder
        self.tcn = EncoderTCN(m_dim, [c_lat, c_lat, c_lat])
        self.out = nn.Conv1d(c_lat, x_dim, kernel_size=1)

    def forward(self, m_seq, h0=None):
        # m_seq: (B, m_dim, T_fut)
        z = self.tcn(m_seq)  # (B, c_lat, T_fut)
        # optional skip-connection from h0 (encoder output)
        if h0 is not None:
            # h0: (B_enc, c_lat, T_ctx) → 평균으로 요약해서 (1, c_lat, 1)
            h_last = h0[:, :, -1:].mean(dim=0, keepdim=True).detach()  # (1, c_lat, 1)
            z = z + h_last  # (B_dec, c_lat, T_fut) + (1, c_lat, 1) → broadcasting 됨
        x_fut = self.out(z)  # (B, x_dim, T_fut)
        return x_fut


class TCNSeq2Seq(nn.Module):
    def __init__(self, x_dim=41, m_dim=11, c_lat=128):
        super().__init__()
        # Encoder takes [x_ctx; m_ctx] -> c_lat channels
        self.encoder = EncoderTCN(x_dim + m_dim, [64, 128, c_lat])
        # Decoder takes m_fut -> x_fut
        self.decoder = DecoderTCN(m_dim, c_lat, x_dim)

    def forward(self, x_ctx_seq, m_ctx_seq, m_fut_seq):
        """
        x_ctx_seq: (B, T_ctx, x_dim)
        m_ctx_seq: (B, T_ctx, m_dim)
        m_fut_seq: (B, T_fut, m_dim)
        """
        B, Tc, _ = x_ctx_seq.shape
        _, Tf, _ = m_fut_seq.shape
        # 1) Encoder: concat along feature dim, then permute to (B, feat, T)
        enc_in = torch.cat([x_ctx_seq, m_ctx_seq], dim=2)      # (B, Tc, x+m)
        enc_in = enc_in.permute(0, 2, 1)                       # (B, x+m, Tc)
        h = self.encoder(enc_in)                              # (B, c_lat, Tc)

        # 2) Decoder: feed m_fut, plus optional h for skip
        m_fut_in = m_fut_seq.permute(0, 2, 1)                  # (B, m_dim, Tf)
        x_fut_out = self.decoder(m_fut_in, h0=h)              # (B, x_dim, Tf)

        # 3) Permute back to (B, Tf, x_dim)
        x_fut_out = x_fut_out.permute(0, 2, 1)                 # (B, Tf, x_dim)
        return x_fut_out


# ────────────────────────────────────────────────────────────────
# 2) 전체 92개 윈도우를 (ctx, W, 52) + (fut, W, 11) → (92, W, 52) 로 예측
# ────────────────────────────────────────────────────────────────
def predict_full_windows(model, x_win, m_win, ctx):
    """
    model: TCNSeq2Seq
    x_win: np.ndarray or tensor of shape (92, W, x_dim)
    m_win: np.ndarray or tensor of shape (92, W, m_dim)
    ctx: int, 과거 윈도우 개수
    returns: np.ndarray (92, W, x_dim+m_dim)
    """
    device = next(model.parameters()).device
    W, x_dim, m_dim = x_win.shape[1], x_win.shape[2], m_win.shape[2]

    # 1) to torch
    xw = torch.from_numpy(x_win).float().to(device)  # (92, W, x_dim)
    mw = torch.from_numpy(m_win).float().to(device)  # (92, W, m_dim)

    # 2) split past/future
    x_ctx = xw[:ctx]      # (ctx, W, x_dim)
    m_ctx = mw[:ctx]      # (ctx, W, m_dim)
    m_fut = mw[ctx:]      # (92-ctx, W, m_dim)

    # 3) reshape to sequences
    x_ctx_seq = x_ctx.reshape(1, ctx * W, x_dim)    # (1, ctx*W, x_dim)
    m_ctx_seq = m_ctx.reshape(1, ctx * W, m_dim)    # (1, ctx*W, m_dim)
    m_fut_seq = m_fut.reshape(1, (92-ctx) * W, m_dim)  # (1, fut*W, m_dim)

    # 4) forward
    model.eval()
    with torch.no_grad():
        x_fut_seq = model(x_ctx_seq, m_ctx_seq, m_fut_seq)  # (1, fut*W, x_dim)

    # 5) reconstruct full feature sequence
    #    - past: x_ctx_seq + m_ctx_seq
    #    - fut:  x_fut_seq + m_fut_seq
    full_x = torch.cat([x_ctx_seq, x_fut_seq], dim=1)  # (1, 92*W, x_dim)
    full_m = torch.cat([m_ctx_seq, m_fut_seq], dim=1)  # (1, 92*W, m_dim)
    full_feat = torch.cat([full_x, full_m], dim=2)     # (1, 92*W, 52)

    # 6) back to windows
    full_win = full_feat.reshape(92, W, x_dim + m_dim)
    return full_win.cpu().numpy()


# ────────────────────────────────────────────────────────────────
# 3) 사용 예시
# ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import numpy as np

    # 예: 92개 윈도우, W=50, x_dim=41, m_dim=11
    x_windows = np.random.randn(92, 50, 41)
    m_windows = np.random.randn(92, 50, 11)
    ctx = 30

    # 모델 생성 & GPU 로드
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = TCNSeq2Seq(x_dim=41, m_dim=11, c_lat=128).to(device)

    # (사전 학습된 체크포인트가 있으면) 불러오기
    # model.load_state_dict(torch.load("tcn_seq2seq.pth"))

    # 예측
    full_windows = predict_full_windows(model, x_windows, m_windows, ctx)
    # full_windows.shape == (92,50,52)

    print("Output windows shape:", full_windows.shape)