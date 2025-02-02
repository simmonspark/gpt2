# gpt2

![image](https://github.com/user-attachments/assets/70ef6296-cf86-48de-a105-fd7e88ea1094)


![스크린샷 2025-01-23 01-32-13](https://github.com/user-attachments/assets/2191d558-a3b9-4c83-a5c2-38476d42c0df)


코드의 generate부분 약 20줄을 제외하고 gpt의 도움 없이 gpt 구현하기 프로젝트

hugging face 의 모델을 분석한 후 직접 구현해서 weight까지 로딩(new imple 부분을 hugging face의 파일을 디버깅하며 직접 제작)

causal_mask, pad_mask 또한 구현

- 필자가 알기로는 encoder-decoder attention 없이 self-attention만 진행하는데, 코드를 보는 도중 flash attention이랑 cross attention이 존재함을 발견. 해당 부분은 향후 더 알아볼 예정
