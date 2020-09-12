# AWS에서 eKoNLPy   다운받기

## 1. AWS 연결 

F1 -> Remote - SSH: Connection to Host ->  + Configure SSH Hosts,,, 

> Host AWS               # ID_1
>
> ​	HostName 13.209.166.28
>
> ​	User lab21
>
> ​	IdentityFile C:\Users\johc5\Downloads\AWS\multi-nlp-a.pem
>
> 
>
> Host AWS_bok       # ID_2
>
> ​	HostName 13.124.96.100
>
> ​	User lab21
>
> ​	IdentityFile C:\Users\johc5\Downloads\AWS\multi-nlp-a.pem

## 2. AWS 연결 -> terminal 창 열기 -> 가상환경 만들기

   ```prompt 
   # (base) -> (가상환경 name) 만들어 이동
   conda create --name multi python=3.6 # 가상환경만들기
   conda activate multi # 가상환경으로 서버 이동
   pip list  #pip 된 파일들 목록
   python  # 파이썬 실행
   jupyter notebook  # 쥬피터 노트북 실행 (cntrl + 클릭)
   
   exit()
   ```

## 3. eKONLpy 다운받기

   https://github.com/entelecheia/eKoNLPy   (install 부분 보기)

   ```prompt
   git clone https://github.com/entelecheia/eKoNLPy.git
   cd eKoNLPy
   pip install .
   pip install . --upgrade
   
   ```

 ## 4. error  사항

    참고( https://github.com/seunghyunmoon2/ProgrammingTips/blob/master/mecab_tagger()_error.md)

> mecab인스톨 완료 후에 eKoNLPy를 사용할 때 tagger()에러가 뜰 때가 었습니다.
>
> ```
> curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh
> ```
> 잘 동작하는지 확인하기 위해 다음 코드를 실행합니다.
>
> ```prompt
> from ekonlpy.tag import Mecab
> mecab = Mecab()
> mecab.pos('금통위는 따라서 물가안정과 병행, 경기상황에 유의하는 금리정책을 펼쳐나가기로 했다고 밝혔다.')
> 
> 결과값이 나오면 성공! 
> ```



### <참고사항>

```prompt
cd .. # 그 전의 폴더로 돌아간다.
ls # 거기에 해당하는 list 목록들 보여주기

<폴더 생성>
mkdir test 하면 파란색으로 폴더가 생성된 것을 알 수 있다.

<파일 생성> - test.py / test.txt를 만들어보자 
touch.test.py 
touch.test.txt

<저장하는 키>
:wq 

<code로 파일 열기>
code test.txt

<메모장으로 열기>
vi test.py -> insert 누르기 -> :wq(저장)
```

