import pickle
import numpy as np

# pkl 파일 로드
with open('./processed_data/menu_info.pkl', 'rb') as f:
    data = pickle.load(f)

print("=== menu_info 구조 ===")
print("Keys:", list(data.keys()))

print("\n=== menu_indices ===")
print("Type:", type(data['menu_indices']))
print("Length:", len(data['menu_indices']))
print("Sample:", data['menu_indices'][:5])

print("\n=== processed_menus ===")
print("Type:", type(data['processed_menus']))
print("Count:", len(data['processed_menus']))
print("Sample keys:", list(data['processed_menus'].keys())[:3])

print("\n=== 첫 번째 메뉴 정보 ===")
first_menu = list(data['processed_menus'].keys())[0]
print("Menu name:", first_menu)
print("Menu data keys:", list(data['processed_menus'][first_menu].keys()))
print("X shape:", data['processed_menus'][first_menu]['X'].shape)
print("y shape:", data['processed_menus'][first_menu]['y'].shape)

print("\n=== 메뉴별 샘플 수 ===")
sample_counts = []
for menu, menu_data in data['processed_menus'].items():
    sample_counts.append(menu_data['n_samples'])
    if len(sample_counts) <= 5:
        print(f"{menu}: {menu_data['n_samples']} samples")

print(f"\n총 메뉴 수: {len(data['processed_menus'])}")
print(f"평균 샘플 수: {np.mean(sample_counts):.1f}")
print(f"최소 샘플 수: {np.min(sample_counts)}")
print(f"최대 샘플 수: {np.max(sample_counts)}")
