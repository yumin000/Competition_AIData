import pandas as pd

# 1. 데이터 로드
lst = []
try:
    df = pd.read_csv('total_rename_data/trainData.csv', encoding='utf-8')
    print("파일을 성공적으로 로드했습니다.\n")

    # 2. 'code_new'로 그룹화하여 'elev'의 평균 계산
    # groupby()와 mean()의 결과는 각 code_new를 인덱스로, 평균 elev를 값으로 갖는 Series입니다.
    average_elev_by_code = df.groupby('code_new')['elev'].mean()

    # 3. Series의 값들을 바로 리스트로 변환
    lst = average_elev_by_code.tolist()

    # 4. 결과 출력
    print("--- 관측소(code_new)별 평균 elev 리스트 ---")
    print(lst)
    
    # (참고) 만약 코드와 평균값을 함께 보고 싶다면 아래 주석을 해제하세요.
    # print("\n--- 관측소(code_new)별 평균 elev 상세 ---")
    # print(average_elev_by_code)


except FileNotFoundError:
    print("오류: 'total_rename_data/trainData.csv' 파일을 찾을 수 없습니다. 경로를 확인해주세요.")