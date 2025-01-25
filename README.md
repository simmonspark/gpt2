# gpt2

![image](https://github.com/user-attachments/assets/70ef6296-cf86-48de-a105-fd7e88ea1094)


![스크린샷 2025-01-23 01-32-13](https://github.com/user-attachments/assets/2191d558-a3b9-4c83-a5c2-38476d42c0df)


코드의 generate부분 약 20줄을 제외하고 gpt의 도움 없이 gpt 구현하기 프로젝트

hugging face 의 모델을 분석한 후 직접 구현해서 weight까지 로딩

pad mask도 구현했지만, loss가 nan이 뜨는 문제 발생. logical mask만 구현.

고찰 : for문으로 돌려가며 weight를 custom 매핑 하는 것 보단, 귀찮더라고 keys만 따로 가져와서 1:1로 전부 매핑하는게 디버깅 관점에서 훨씬 효율적이다.
