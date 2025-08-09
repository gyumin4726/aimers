import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import os
import pickle
from tqdm import tqdm

def set_seed(seed=42):
    """랜덤 시드 설정"""
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def create_time_features(dates):
    """날짜에서 시간 특성 생성"""
    day_of_week = dates.dt.dayofweek
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    month = dates.dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return np.column_stack([day_sin, day_cos, month_sin, month_cos])

def load_and_preprocess_data():
    """데이터 로드 및 기본 전처리"""
    print("데이터 로드 중...")
    train = pd.read_csv('./data/train/train.csv')
    
    # 날짜 컬럼을 datetime으로 변환
    train['영업일자'] = pd.to_datetime(train['영업일자'])
    
    # 기본 정보 출력
    print(f"총 레코드 수: {len(train)}")
    print(f"날짜 범위: {train['영업일자'].min()} ~ {train['영업일자'].max()}")
    print(f"메뉴 수: {train['영업장명_메뉴명'].nunique()}")
    
    return train

def create_sliding_windows(data, time_features, window_size=35, stride=1):
    """
    슬라이딩 윈도우로 시계열 데이터 생성
    
    Args:
        data: 시계열 데이터 (numpy array)
        time_features: 시간 특성 (numpy array)
        window_size: 윈도우 크기 (28일 입력 + 7일 출력 = 35)
        stride: 이동 간격
    
    Returns:
        X: 입력 시퀀스 (N, 28, 5) - [매출수량, day_sin, day_cos, month_sin, month_cos]
        y: 출력 시퀀스 (N, 7, 1) - 매출수량 예측값
    """
    X, y = [], []
    
    for i in range(0, len(data) - window_size + 1, stride):
        window_data = data[i:i + window_size]
        window_time = time_features[i:i + window_size]
        
        # 입력: 28일 데이터
        input_data = window_data[:28]
        input_time = window_time[:28]
        
        # 매출수량과 시간 특성 결합
        input_combined = np.column_stack([input_data, input_time])
        
        X.append(input_combined)  # (28, 5) 형태
        y.append(window_data[28:].reshape(-1, 1))  # 뒤 7일을 출력으로 (7, 1) 형태
    
    return np.array(X), np.array(y)

def preprocess_menu_data(train_df, menu_name):
    """
    특정 메뉴에 대한 전처리
    
    Args:
        train_df: 전체 훈련 데이터
        menu_name: 메뉴명
    
    Returns:
        X, y: 슬라이딩 윈도우로 생성된 입력/출력 데이터
    """
    # 해당 메뉴의 데이터만 추출
    menu_data = train_df[train_df['영업장명_메뉴명'] == menu_name].copy()
    
    # 날짜순으로 정렬
    menu_data = menu_data.sort_values('영업일자')
    
    # 매출수량만 추출
    sales_data = menu_data['매출수량'].values
    
    # 시간 특성 생성
    time_features = create_time_features(menu_data['영업일자'])
    
    # 슬라이딩 윈도우 생성
    X, y = create_sliding_windows(sales_data, time_features, window_size=35, stride=1)
    
    return X, y

def main():
    """메인 전처리 함수"""
    set_seed(42)
    
    # 데이터 로드
    train_df = load_and_preprocess_data()
    
    # 결과 저장용 리스트
    all_processed_data = []
    
    # 각 메뉴별로 전처리
    menus = train_df['영업장명_메뉴명'].unique()
    print(f"\n총 {len(menus)}개 메뉴에 대해 전처리 시작...")
    
    total_samples = 0
    
    for menu in tqdm(menus, desc="메뉴별 전처리"):
        # 해당 메뉴의 데이터만 추출
        menu_data = train_df[train_df['영업장명_메뉴명'] == menu].copy()
        
        # 날짜순으로 정렬
        menu_data = menu_data.sort_values('영업일자')
        
        # 시간 특성 생성
        time_features = create_time_features(menu_data['영업일자'])
        
        # 각 날짜별로 데이터 생성
        for i in range(len(menu_data)):
            row = menu_data.iloc[i]
            time_feat = time_features[i]
            
            processed_row = {
                '영업일자': row['영업일자'],
                '영업장명_메뉴명': row['영업장명_메뉴명'],
                '매출수량': row['매출수량'],
                'day_sin': time_feat[0],
                'day_cos': time_feat[1],
                'month_sin': time_feat[2],
                'month_cos': time_feat[3]
            }
            all_processed_data.append(processed_row)
        
        total_samples += len(menu_data)
    
    # DataFrame으로 변환
    processed_df = pd.DataFrame(all_processed_data)
    
    print(f"\n전처리 완료!")
    print(f"총 레코드 수: {len(processed_df)}")
    print(f"처리된 메뉴 수: {len(menus)}")
    
    # 결과 저장
    save_path = './processed_data'
    os.makedirs(save_path, exist_ok=True)
    
    # CSV로 저장
    output_file = os.path.join(save_path, 'train_processed.csv')
    processed_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    
    print(f"데이터가 '{output_file}'에 저장되었습니다.")
    
    return processed_df

if __name__ == "__main__":
    processed_df = main()
