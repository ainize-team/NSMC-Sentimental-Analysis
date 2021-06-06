# NSMC-Sentimental-Analysis
풀잎스쿨 MLOps 스터디를 위한 네이버 영화 리뷰 감정 분석

## NSMC
NSMC(Naver sentiment movie corpus)는 네이버 영화 리뷰 데이터를 이용하여 긍정 부정 분류를 위한 공개 데이터 셋 입니다. 평점이 10, 9인 리뷰에 대해서는 긍정, 평점이 1, 2, 3, 4인 리뷰에 대해서는 부정이라고 정의 합니다. 데이터는 아래의 github 에서 받을 수 있습니다.

* https://github.com/e9t/nsmc

## Sentimental Analysis
입력 리뷰에 대해서 긍정 부정을 판단하는 모델을 학습하기 위해서는 다양한 방법이 있습니다. 현재는 사전 학습된 언어 모델을 기반으로 미세 조정(Fine Tune)을 하는 것이 좋은 방법 중 하나입니다. 본 예제에서는 [이준범](https://github.com/Beomi) 님이 만든 [KcELECTRA](https://github.com/Beomi/KcELECTRA) 를 이용하여 판단하는 모델을 제작 하였습니다. KcELECTRA를 활용하여 학습하는 코드는 [KcELECTRA Fine Tune NSMC.ipynb](./KcELECTRA%20Fine%20Tune%20NSMC.ipynb) 를 참고 하거나 아래의 버튼을 눌러 구글 코랩환경에서 테스트 할 수 있습니다.

* [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1bl16q6GwW4CWpQNtDQRUG0xwubGsb1yV?usp=sharing)

## Firebase
MLOps Pipeline 을 만들기 위해서 본 프로젝트에서는 firebase를 이용합니다. 본 프로젝트에서 firebase를 이용하여 다음과 같은 작업들을 진행 합니다.
1. 학습에 사용할 Base Model 을 FireBase Storage에 저장 하고, 학습을 할 때 다운로드 받아서 이용 한다.
2. 학습에 사용할 데이터를 수집하여 전처리를 진행한 다음 FireBase Realtime Database 에 저장 하고, 이를 불러와서 학습에 사용 한다.
3. 학습 모델을 FireBase Storage에 저장하고, 이를 향후 배포에 활용 한다.
4. 학습에 사용할 tasks와 파라메터를 저장하고 이를 활용하여 학습을 한다.

## How to Use
먼저 해당 프로젝트를 사용하기 위해서는 FireBase 설정 파일을  `src/keys/mlops-crawler-firebase.json` 와 `src/keys/firebase-config.json` 에 넣어주신 다음 Docker 파일을 빌드해주셔야 합니다.

그 다음, 크롤러를 이용하여 데이터 베이스에 학습에 사용할 데이터를 수집 합니다. 크롤러는 아래에 있는 크롤러를 사용하여 수집 하였습니다.
그 이후 아래와 같이 storage 에 학습에 사용할 base model을 업로드 합니다.

![baseModel](./imgs/01.png)
![baseModel2](./imgs/02.png)

그 이후 데이터 베이스 `trainTasks` 를 만들고 다음과 같이 학습에 사용할 파라메터를 지정해줍니다.

![trainTaks](./imgs/03.png)

그 다음 도커파일을 빌드하고 실행하게 되면 trainTasks에 있는 내용을 학습하기 시작 합니다.

* 크롤러 : https://github.com/ainize-team/NMRC
## Reference
* https://huggingface.co/beomi/KcELECTRA-bases
