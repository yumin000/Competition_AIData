import pandas as pd
import numpy as np



# 1. 파일 로드 및 초기 설정
# 파일 이름이 TST_test1.csv였다면, TST_test1_wide.csv 등으로 이름을 변경하여 테스트하시는 것을 권장합니다.
file_path = 'filled_liner_TST_test_5.csv'  # 파일 이름을 실제 파일명으로 지정
try:
    # Wide Format 파일 로드 (첫 번째 행은 'ymd', '1', '2', ... 열 이름이 와야 합니다)
    df = pd.read_csv(file_path)
    print(f"✅ Wide Format 파일 로드 성공: {file_path}")
except FileNotFoundError:
    print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해 주세요: {file_path}")
    exit()

# 'ymd'를 datetime 객체로 변환
df["ymd"] = pd.to_datetime(df["ymd"], errors='coerce') # 에러 발생 시 NaT 처리 (예: #####)

# 코드 열 이름 목록 ('1'부터 '12'까지)
code_cols = [str(i) for i in range(1, 13)]

# 2. 데이터 대체 시점 정의
start_date = pd.to_datetime('2021-01-01 00:00:00')
end_date = pd.to_datetime('2021-01-01 23:00:00')
target_date = pd.to_datetime('2021-01-02 00:00:00')

# 3. 2021-01-02 00:00:00 데이터 찾기
# 'ymd' 값이 target_date와 일치하는 행을 찾습니다.
target_row = df[df['ymd'] == target_date].copy()

if target_row.empty:
    print(f"⚠️ 오류: {target_date}의 기준 행을 찾을 수 없습니다. 데이터 추가를 중단합니다.")
    exit()

# 4. 누락된 2021-01-01 행 생성
new_rows_list = []
# 24개 시점 (00:00:00 부터 23:00:00)을 생성
dates_to_add = pd.date_range(start=start_date, end=end_date, freq='H')

# 2021-01-02 00:00:00 시점의 관측소별 값들을 추출합니다.
base_values = target_row[code_cols].iloc[0].to_dict()

# 24개의 새로운 행을 생성합니다.
for date_to_add in dates_to_add:
    new_row = {'ymd': date_to_add}
    # 관측소별 값 복사
    new_row.update(base_values)
    new_rows_list.append(new_row)

print(f"⭐ 2021-01-01 데이터 누락 확인. 24개 행을 새로 생성합니다.")

# 5. 기존 DataFrame에 새로운 행 병합 및 정렬
if new_rows_list:
    df_new_rows = pd.DataFrame(new_rows_list)
    
    # 두 DataFrame을 합치고, 'ymd'를 기준으로 정렬합니다.
    df_combined = pd.concat([df, df_new_rows], ignore_index=True)
    df_combined = df_combined.sort_values(by='ymd').reset_index(drop=True)
    
    # 최종적으로 필요한 열만 유지하고 순서를 맞춥니다.
    df_final = df_combined[['ymd'] + code_cols]
    
    print("\n✅ 누락된 2021-01-01 데이터 행 추가 완료.")
else:
    # 이 경우는 사실상 발생하지 않지만, 코드를 안전하게 유지합니다.
    df_final = df.sort_values(by='ymd').reset_index(drop=True)
    print("\n✅ 추가된 새 행이 없습니다. 원본 데이터프레임 유지.")


# 6. 수정된 데이터를 새 파일로 저장
output_file_path = 'filled_liner_TST_test_5_.csv'
df_final.to_csv(output_file_path, index=False)
print(f"✅ 수정된 파일이 {output_file_path}으로 저장되었습니다.")

# 최종 결과 확인 (누락된 부분이 채워졌는지)
print("\n--- 최종 결과 (첫 30행) ---")
print(df_final.head(30).to_string())
'''

df=pd.read_csv('TST_test5_.csv')
df = df.ffill().bfill()
df.to_csv("filled_TST_test5.csv", index=False)'''






