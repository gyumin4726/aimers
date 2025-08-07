import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm


class Chomp1d(nn.Module):
    """Conv1d 후 원래 길이로 맞추기 위한 모듈"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """TCN의 기본 블록 - 변수 간 상관관계 학습"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.LeakyReLU(0.1)
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """TCN 네트워크 - 변수 간 상관관계 학습"""
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]
 
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TemporalConvNet2D(nn.Module):
    """2D TCN - 변수 간 상관관계 학습"""
    def __init__(self, in_channels, out_channels, kernel_size=3, dropout=0.2, stride=1, base_dilation=1):
        super(TemporalConvNet2D, self).__init__()

        assert out_channels % 4 == 0
        # 더 작은 커널 크기 사용
        self.conv1 = weight_norm(nn.Conv2d(in_channels, out_channels // 4, kernel_size=3,
                                           stride=stride, padding=1))
        self.relu = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv2d(out_channels // 4, out_channels // 3, kernel_size=3,
                                           stride=stride, padding=1))
        self.dropout2 = nn.Dropout(dropout)

        self.conv3 = weight_norm(nn.Conv2d(out_channels // 3, out_channels // 2, kernel_size=3,
                                           stride=stride, padding=1))
        self.dropout3 = nn.Dropout(dropout)

        self.conv4 = weight_norm(nn.Conv2d(out_channels // 2, out_channels // 1, kernel_size=3,
                                           stride=stride, padding=1))
        self.dropout4 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.relu, self.dropout1,
                                 self.conv2, self.relu, self.dropout2,
                                 self.conv3, self.relu, self.dropout3,
                                 self.conv4, self.relu, self.dropout4)

    def forward(self, x):
        return self.net(x)


class SalesPredictorSeq2Seq(nn.Module):
    """완전한 Conv1D2D + Seq2Seq 구조의 매출 예측 모델"""
    def __init__(self, x_dim=1, m_dim=4, c_lat=128, dropout=0.2):
        super(SalesPredictorSeq2Seq, self).__init__()
        
        # Encoder: 1D TCN + 2D TCN
        self.encoder_1d = TemporalConvNet(x_dim + m_dim, [64, 128, c_lat], kernel_size=3, dropout=dropout)
        self.encoder_2d = TemporalConvNet2D(1, c_lat, kernel_size=3, dropout=dropout)
        
        # Decoder: 1D TCN + 2D TCN
        self.decoder_1d = TemporalConvNet(m_dim, [64, 128, c_lat], kernel_size=3, dropout=dropout)
        self.decoder_2d = TemporalConvNet2D(1, c_lat, kernel_size=3, dropout=dropout)
        
        # 최종 출력 레이어
        self.output = nn.Conv1d(c_lat, x_dim, kernel_size=1)
        
    def forward(self, x_ctx, m_ctx, m_fut):
        """
        x_ctx: (batch_size, 28, 1) - 과거 매출수량
        m_ctx: (batch_size, 28, 4) - 과거 시간특성
        m_fut: (batch_size, 7, 4) - 미래 시간특성
        """
        # 1. Encoder: Conv1D2D
        # 과거 데이터 결합
        ctx_in = torch.cat([x_ctx, m_ctx], dim=2)  # (batch_size, 28, 5)
        ctx_in = ctx_in.transpose(1, 2)  # (batch_size, 5, 28)
        
        # Encoder 1D TCN
        enc_1d_out = self.encoder_1d(ctx_in)  # (batch_size, c_lat, 28)
        
        # Encoder 2D TCN
        enc_2d_input = enc_1d_out.unsqueeze(1)  # (batch_size, 1, c_lat, 28)
        enc_2d_out = self.encoder_2d(enc_2d_input)  # (batch_size, c_lat, H, W)
        
        # Encoder 출력 (마지막 시점 사용)
        h = enc_2d_out[:, :, -1, :]  # (batch_size, c_lat, W)
        
        # 2. Decoder: Conv1D2D
        # 미래 데이터 준비
        m_fut_in = m_fut.transpose(1, 2)  # (batch_size, 4, 7)
        
        # Decoder 1D TCN
        dec_1d_out = self.decoder_1d(m_fut_in)  # (batch_size, c_lat, 7)
        
        # Decoder 2D TCN
        dec_2d_input = dec_1d_out.unsqueeze(1)  # (batch_size, 1, c_lat, 7)
        dec_2d_out = self.decoder_2d(dec_2d_input)  # (batch_size, c_lat, H, W)
        
        # Skip connection: Encoder 출력을 Decoder에 추가
        if h is not None:
            # h: (batch_size, c_lat, W) → 평균으로 요약
            h_summary = h.mean(dim=(0, 2), keepdim=True).detach()  # (1, c_lat, 1)
            # Decoder 출력에 추가 (broadcasting)
            dec_2d_out = dec_2d_out + h_summary.unsqueeze(2)  # (batch_size, c_lat, H, W)
        
        # 3. 최종 출력
        # 2D 출력을 1D로 변환 (마지막 시점 사용)
        final_out = dec_2d_out[:, :, -1, :]  # (batch_size, c_lat, W)
        
        # 최종 예측
        x_fut_out = self.output(final_out)  # (batch_size, 1, W)
        
        # 차원 조정
        x_fut_out = x_fut_out.transpose(1, 2)  # (batch_size, W, 1)
        
        return x_fut_out


class EncoderTCN(nn.Module):
    """TCN Encoder"""
    def __init__(self, in_ch, channels):
        super().__init__()
        layers = []
        for i, ch in enumerate(channels):
            layers.append(TemporalBlock(in_ch, ch, kernel_size=3, stride=1, 
                                      dilation=2**i, padding=(3-1)*2**i, dropout=0.2))
            in_ch = ch
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        return self.tcn(x)


class DecoderTCN(nn.Module):
    """TCN Decoder"""
    def __init__(self, m_dim, c_lat, x_dim):
        super().__init__()
        # causal TCN decoder
        self.tcn = EncoderTCN(m_dim, [c_lat, c_lat, c_lat])
        self.out = nn.Conv1d(c_lat, x_dim, kernel_size=1)

    def forward(self, m_seq, h0=None):
        # m_seq: (batch_size, m_dim, 7)
        z = self.tcn(m_seq)  # (batch_size, c_lat, 7)
        
        # optional skip-connection from h0 (encoder output)
        if h0 is not None:
            # h0: (batch_size, c_lat, 28) → 마지막 시점 사용
            h_last = h0[:, :, -1:].detach()  # (batch_size, c_lat, 1)
            z = z + h_last  # skip connection
        
        x_fut = self.out(z)  # (batch_size, x_dim, 7)
        return x_fut


class SalesTrainer:
    """매출 예측 모델 학습 클래스"""
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, patience=5)
        
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        
        for batch_x, batch_y in tqdm(train_loader, desc="Training"):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)
                
                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))


def load_processed_data():
    """전처리된 데이터 로드"""
    print("데이터 로드 중...")
    
    # 원본 train.csv 로드
    train = pd.read_csv('./data/train/train.csv')
    train['영업일자'] = pd.to_datetime(train['영업일자'])
    
    # pkl 파일에서 메뉴 정보 로드
    with open('./processed_data/menu_info.pkl', 'rb') as f:
        menu_info = pickle.load(f)
    
    print(f"총 메뉴 수: {len(menu_info['processed_menus'])}")
    print(f"총 레코드 수: {len(train)}")
    
    return train, menu_info


def create_time_features(dates):
    """날짜에서 시간 특성 생성"""
    day_of_week = dates.dt.dayofweek
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    month = dates.dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return np.column_stack([day_sin, day_cos, month_sin, month_cos])


def create_sliding_windows(data, time_features, window_size=35, stride=1):
    """슬라이딩 윈도우로 시계열 데이터 생성 - Seq2Seq용"""
    X, y = [], []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window_data = data[i:i + window_size]
        window_time = time_features[i:i + window_size]
        
        # 전체 35일 데이터 준비 (28일 + 7일)
        full_data = np.column_stack([window_data, window_time])  # (35, 5)
        
        X.append(full_data)  # (35, 5) 형태 - 전체 데이터
        y.append(window_data[28:].reshape(-1, 1))  # 뒤 7일을 출력으로 (7, 1) 형태
    
    return np.array(X), np.array(y)


def train_menu_model(menu_data, menu_name, device):
    """특정 메뉴에 대한 TCN 모델 학습"""
    # 해당 메뉴의 데이터만 추출
    menu_df = menu_data[menu_data['영업장명_메뉴명'] == menu_name].copy()
    
    # 날짜순으로 정렬
    menu_df = menu_df.sort_values('영업일자')
    
    # 매출수량만 추출
    sales_data = menu_df['매출수량'].values
    
    # 시간 특성 생성
    time_features = create_time_features(menu_df['영업일자'])
    
    # 슬라이딩 윈도우 생성
    X, y = create_sliding_windows(sales_data, time_features, window_size=35, stride=1)
    
    if len(X) < 10:  # 최소 샘플 수 확인
        return None
    
    # 데이터를 텐서로 변환 - Seq2Seq용
    X_tensor = torch.FloatTensor(X).to(device)  # (N, 35, 5)
    y_tensor = torch.FloatTensor(y).to(device)  # (N, 7, 1)
    
    # 모델 생성
    model = SalesPredictorSeq2Seq(
        x_dim=1,          # 1개 매출수량
        m_dim=4,          # 4개 시간특성
        c_lat=128,        # 128차원 잠재 공간
        dropout=0.2
    ).to(device)
    
    # 학습 설정
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 학습
    model.train()
    for epoch in range(30):  # 각 메뉴별로 30 에포크
        # 배치 학습
        for i in range(0, len(X_tensor), 16):
            batch_x_ctx = X_tensor[i:i+16, :28, 0:1] # (batch_size, 28, 1)
            batch_m_ctx = X_tensor[i:i+16, :28, 1:5] # (batch_size, 28, 4)
            batch_m_fut = X_tensor[i:i+16, 28:35, 1:5] # (batch_size, 7, 4)
            
            optimizer.zero_grad()
            predictions = model(batch_x_ctx, batch_m_ctx, batch_m_fut) # (batch_size, 7, 1)
            loss = criterion(predictions, y_tensor[i:i+16])
            loss.backward()
            optimizer.step()
    
    # 모델을 eval 모드로 설정
    model.eval()
    
    return {
        'model': model,
        'last_sequence': X_tensor[-1:],  # 마지막 28일 데이터
        'n_samples': len(X)
    }


def main():
    """메인 학습 함수 - 메뉴별 개별 학습"""
    print("=== 2D TCN 기반 메뉴별 매출 예측 모델 학습 ===")
    print("변수 간 상관관계 학습 가능!")
    
    # 데이터 로드
    train_data, menu_info = load_processed_data()
    
    # 디바이스 설정
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"디바이스: {device}")
    
    # 메뉴별 모델 학습
    trained_models = {}
    menus = train_data['영업장명_메뉴명'].unique()
    
    print(f"\n총 {len(menus)}개 메뉴에 대해 학습 시작...")
    
    for menu in tqdm(menus, desc="메뉴별 모델 학습"):
        model_info = train_menu_model(train_data, menu, device)
        
        if model_info is not None:
            trained_models[menu] = model_info
            print(f"  {menu}: {model_info['n_samples']} 샘플로 학습 완료")
    
    # 학습된 모델 저장
    torch.save(trained_models, './tcn_menu_models.pth')
    
    print(f"\n학습 완료!")
    print(f"성공적으로 학습된 메뉴 수: {len(trained_models)}")
    print(f"모델이 './tcn_menu_models.pth'에 저장되었습니다.")


def predict_menu_sales(test_df, trained_models, test_prefix):
    """메뉴별 매출 예측"""
    results = []
    device = next(iter(trained_models.values()))['model'].device
    
    for store_menu, store_test in test_df.groupby(['영업장명_메뉴명']):
        if store_menu not in trained_models:
            continue
            
        model_info = trained_models[store_menu]
        model = model_info['model']
        
        # 테스트 데이터 정렬
        store_test_sorted = store_test.sort_values('영업일자')
        
        # 최근 28일 데이터 추출
        recent_sales = store_test_sorted['매출수량'].values[-28:]
        recent_dates = store_test_sorted['영업일자'].values[-28:]
        
        if len(recent_sales) < 28:
            continue
        
        # 시간 특성 생성
        recent_dates = pd.to_datetime(recent_dates)
        time_features = create_time_features(recent_dates)
        
        # 입력 데이터 준비 - Seq2Seq용
        # 28일 과거 데이터
        x_ctx = torch.FloatTensor(recent_sales).unsqueeze(0).unsqueeze(-1).to(device)  # (1, 28, 1)
        m_ctx = torch.FloatTensor(time_features).unsqueeze(0).to(device)  # (1, 28, 4)
        
        # 7일 미래 시간특성 (예측 시점의 시간특성)
        future_dates = pd.date_range(recent_dates[-1] + pd.Timedelta(days=1), periods=7, freq='D')
        future_time_features = create_time_features(future_dates)
        m_fut = torch.FloatTensor(future_time_features).unsqueeze(0).to(device)  # (1, 7, 4)
        
        # 예측
        model.eval()
        with torch.no_grad():
            predictions = model(x_ctx, m_ctx, m_fut)  # (1, 7, 1)
            pred_values = predictions.squeeze().cpu().numpy()
        
        # 예측값을 양수로 제한
        pred_values = np.maximum(pred_values, 0)
        
        # 결과 저장
        pred_dates = [f"{test_prefix}+{i+1}일" for i in range(7)]
        
        for date, pred_val in zip(pred_dates, pred_values):
            results.append({
                '영업일자': date,
                '영업장명_메뉴명': store_menu,
                '매출수량': pred_val
            })
    
    return pd.DataFrame(results)


def convert_to_submission_format(pred_df, sample_submission):
    """제출 형식으로 변환"""
    # (영업일자, 메뉴) → 매출수량 딕셔너리로 변환
    pred_dict = dict(zip(
        zip(pred_df['영업일자'], pred_df['영업장명_메뉴명']),
        pred_df['매출수량']
    ))

    final_df = sample_submission.copy()

    for row_idx in final_df.index:
        date = final_df.loc[row_idx, '영업일자']
        for col in final_df.columns[1:]:  # 메뉴명들
            final_df.loc[row_idx, col] = pred_dict.get((date, col), 0)

    return final_df


def run_prediction():
    """예측 실행 함수"""
    import glob
    import re
    
    print("=== TCN 모델 예측 시작 ===")
    
    # 학습된 모델 로드
    trained_models = torch.load('./tcn_menu_models.pth', map_location='cpu')
    print(f"로드된 모델 수: {len(trained_models)}")
    
    # 모든 test_*.csv 순회
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    all_preds = []
    
    for path in test_files:
        test_df = pd.read_csv(path)
        
        # 파일명에서 접두어 추출 (예: TEST_00)
        filename = os.path.basename(path)
        test_prefix = re.search(r'(TEST_\d+)', filename).group(1)
        
        print(f"예측 중: {filename}")
        pred_df = predict_menu_sales(test_df, trained_models, test_prefix)
        all_preds.append(pred_df)
    
    # 모든 예측 결과 합치기
    full_pred_df = pd.concat(all_preds, ignore_index=True)
    
    # 제출 형식으로 변환
    sample_submission = pd.read_csv('./data/sample_submission.csv')
    submission = convert_to_submission_format(full_pred_df, sample_submission)
    
    # 결과 저장
    submission.to_csv('tcn_submission.csv', index=False, encoding='utf-8-sig')
    print("예측 완료! 결과가 'tcn_submission.csv'에 저장되었습니다.")


if __name__ == "__main__":
    main()
    # 학습 후 예측도 실행하려면 아래 주석 해제
    # run_prediction()
