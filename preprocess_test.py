import pandas as pd
import numpy as np
import os
import glob
from tqdm import tqdm

def create_time_features(dates):
    """날짜에서 시간 특성 생성"""
    day_of_week = dates.dt.dayofweek
    day_sin = np.sin(2 * np.pi * day_of_week / 7)
    day_cos = np.cos(2 * np.pi * day_of_week / 7)

    month = dates.dt.month
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)

    return np.column_stack([day_sin, day_cos, month_sin, month_cos])

def preprocess_test_file(file_path):
    """개별 test 파일 전처리"""
    print(f"전처리 중: {os.path.basename(file_path)}")
    
    # 데이터 로드
    test_df = pd.read_csv(file_path)
    test_df['영업일자'] = pd.to_datetime(test_df['영업일자'])
    
    # 각 메뉴별로 시간 특성 추가
    processed_data = []
    
    for menu_name, menu_data in test_df.groupby('영업장명_메뉴명'):
        # 날짜순으로 정렬
        menu_data_sorted = menu_data.sort_values('영업일자')
        
        # 첫 번째 날짜의 요일 파악
        first_date = menu_data_sorted['영업일자'].iloc[0]
        first_day_of_week = first_date.dayofweek  # 0=월요일, 6=일요일
        
        # 순차적으로 요일 sin/cos 생성 (첫 번째 요일부터 시작)
        day_sin_values = []
        day_cos_values = []
        
        for i in range(len(menu_data_sorted)):
            # 첫 번째 요일부터 순차적으로 할당
            current_day_of_week = (first_day_of_week + i) % 7
            day_sin = np.sin(2 * np.pi * current_day_of_week / 7)
            day_cos = np.cos(2 * np.pi * current_day_of_week / 7)
            day_sin_values.append(day_sin)
            day_cos_values.append(day_cos)
        
        # 월별 sin/cos는 실제 월 사용
        month_sin_values = []
        month_cos_values = []
        for date in menu_data_sorted['영업일자']:
            month = date.month
            month_sin = np.sin(2 * np.pi * month / 12)
            month_cos = np.cos(2 * np.pi * month / 12)
            month_sin_values.append(month_sin)
            month_cos_values.append(month_cos)
        
        # 각 날짜별로 데이터 생성
        for i in range(len(menu_data_sorted)):
            row = menu_data_sorted.iloc[i]
            
            processed_row = {
                '영업일자': row['영업일자'],
                '영업장명_메뉴명': row['영업장명_메뉴명'],
                '매출수량': row['매출수량'],
                'day_sin': day_sin_values[i],
                'day_cos': day_cos_values[i],
                'month_sin': month_sin_values[i],
                'month_cos': month_cos_values[i]
            }
            processed_data.append(processed_row)
    
    # DataFrame으로 변환
    processed_df = pd.DataFrame(processed_data)
    
    return processed_df

def main():
    """메인 전처리 함수"""
    print("=== Test 데이터 전처리 시작 ===")
    
    # test 파일들 찾기
    test_files = sorted(glob.glob('./data/test/TEST_*.csv'))
    print(f"처리할 test 파일 수: {len(test_files)}")
    
    # 전처리된 데이터 저장 디렉토리 생성
    processed_dir = './processed_data/test'
    os.makedirs(processed_dir, exist_ok=True)
    
    # 각 test 파일 전처리
    for file_path in tqdm(test_files, desc="Test 파일 전처리"):
        try:
            processed_df = preprocess_test_file(file_path)
            
            # 파일명 추출
            filename = os.path.basename(file_path)
            output_path = os.path.join(processed_dir, filename)
            
            # 전처리된 데이터 저장
            processed_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            
            print(f"   {filename}: {len(processed_df)}개 레코드 처리 완료")
            
        except Exception as e:
            print(f"   {os.path.basename(file_path)} 처리 중 오류: {e}")
    
    print(f"\n전처리 완료!")
    print(f"전처리된 파일들이 '{processed_dir}'에 저장되었습니다.")

if __name__ == "__main__":
    main()
