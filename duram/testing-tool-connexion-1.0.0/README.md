# testing-tool-connexion

testing-tool-connexion은 Connexion 문제를 테스팅 하기 위해서 만들어진 도구입니다.

## 실행 방법

`testing-tool-connexion.py`는 다음과 같은 커맨드라인 인자를 받을 수 있습니다.

- `-h`, `--help`: 도움말을 출력합니다.
- `-c CONFIG`, `--config CONFIG`: **설정 파일**로 `CONFIG`를 사용합니다.
- `-i INPUT`, `--input INPUT`: **입력 파일**로 `INPUT`을 사용합니다.
- `-l LOG`, `--log LOG`: **로그 파일**로 `LOG`를 사용합니다.
- `-s`, `--stdio`: **입력 파일**이나 **로그 파일**이 주어지지 않은 경우, 표준 입출력을 대신 사용합니다.
- `-a EXEC1, --exec1 EXEC1`: 선공 플레이어의 실행 커맨드로 `EXEC1`을 사용합니다.
- `-b EXEC2, --exec2 EXEC2`: 후공 플레이어의 실행 커맨드로 `EXEC2`를 사용합니다.
- `--cwd1 CWD1`: 선공 플레이어의 작업 디렉토리로 `CWD1`을 사용합니다. 프로그램 내부에서 `data.bin`을 사용하고 싶을 때 경로를 설정할 수 있습니다. (선택)
- `--cwd2 CWD2`: 후공 플레이어의 작업 디렉토리로 `CWD2`을 사용합니다. 프로그램 내부에서 `data.bin`을 사용하고 싶을 때 경로를 설정할 수 있습니다. (선택)

예를 들어 입력 파일을 `input.txt`, 로그 파일을 `log.txt`, 선공 플레이어의 실행 커맨드를 `python3 sample-code.py P1`, 후공 플레이어의 실행 커맨드를 `python3 sample-code.py P2`, 선공과 후공 플레이어의 작업 디렉토리를 `./sample`로 사용하고 싶으면 다음과 같이 실행합니다.

```bash
python3 testing-tool-connexion.py -i input.txt -l log.txt -a "python3 sample-code.py P1" -b "python3 sample-code.py P2" --cwd1 "./sample" --cwd2 "./sample"
```

### 설정 파일

설정 파일은 command-line argument를 간단하게 사용하기 위한 방법으로, 다음과 같은 내용을 작성할 수 있습니다.

```
INPUT=<입력 파일 경로>
LOG=<로그 파일 경로>
EXEC1=<선공 플레이어의 프로그램 실행 커맨드>
EXEC2=<후공 플레이어의 프로그램 실행 커맨드>
CWD1=<선공 플레이어의 작업 디렉토리>
CWD2=<후공 플레이어의 작업 디렉토리>
```

단, 커맨드라인 인자와 내용이 충돌하는 경우 커맨드라인 인자가 우선 실행됩니다.

예를 들어 입력 파일을 `input.txt`, 로그 파일을 `log.txt`, 선공 플레이어의 실행 커맨드를 `python3 sample-code.py P1`, 후공 플레이어의 실행 커맨드를 `python3 sample-code.py P2`와 같이 사용하고 싶으면 `config.ini`를 다음과 같이 작성합니다.

```
INPUT=input.txt
LOG=log.txt
EXEC1=python3 sample-code.py P1
EXEC2=python3 sample-code.py P2
CWD1=./sample
CWD2=./sample
```

그 이후 다음 명령어를 사용합니다.

```bash
python3 testing-tool-connexion.py -c config.ini
```

### 입력 파일

테스팅 툴의 입력 파일은 각 주머니에서 타일을 뽑아오는 순서를 나타냅니다.
첫 줄은 선공이 뽑아오는 타일 32개, 둘째 줄은 후공이 뽑아오는 타일 32개를 공백으로 구분하여 작성하여야 합니다.
자세한 내용은 예시로 주어진 `input.txt` 파일을 참고하세요.

### 로그 파일

로그 파일에는 게임에 대한 다음 정보를 출력합니다.

- `[<FIRST/SECOND> "<player>"]`
  - 1P혹은 2P가 어떤 커맨드를 실행했는지를 나타냅니다.
- `[RESULT "<result>"]`
  - 게임의 결과를 나타냅니다. `1-0`은 선공 승, `1/2-1/2`는 무승부, `0-1`은 후공 승을 나타냅니다.
- `INIT <A₁> <A₂> <A₃> <A₄> <A₅> <B₁> <B₂> <B₃> <B₄> <B₅>`
  - 준비 단계에서 선공이 뽑은 타일이  `<A₁>`, `<A₂>`, `<A₃>`, `<A₄>`, `<A₅>`이고 후공이 뽑은 타일이 `<B₁>`, `<B₂>`, `<B₃>`, `<B₄>`, `<B₅>`임을 의미합니다.
- `<FIRST/SECOND> <p> <T₁> <T₂> <t>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 `<p>` 위치에 `<T₁>` 타일을 놓은 후 주머니에서 `<T₂>` 타일을 뽑았고, `<t>` 밀리초를 사용했음을 의미합니다. 타일을 뽑아오지 않은 경우 `<T₂>`는 `X0`으로 주어집니다.
- `FINISH`
  - 게임의 정상적인 종료를 나타냅니다.
- `SCORE<FIRST/SECOND> <score>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 받은 점수가 `<score>`임을 나타냅니다.
- `ABORT <FIRST/SECOND> <reason>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 `<reason>`을 이유로 프로그램이 비정상종료했음을 의미합니다. 다음과 같은 이유가 있습니다.
  - `TLE`: 제한시간 안에 출력을 하지 못했음을 의미합니다.
  - `INVALID READY MESSAGE`: `READY`에 대해 올바르게 `OK`를 출력하지 않았음을 의미합니다.
  - `PUT PARSE FAILED`: `PUT` 명령에 대해 올바르지 않은 출력을 했음을 의미합니다.
  - `CELL PARSE FAILED`: 위치에 대해 올바르지 않은 출력을 했음을 의미합니다.
  - `TILE PARSE FAILED`: 타일에 대해 올바르지 않은 출력을 했음을 의미합니다.
  - `NO TILE`: 가지고 있지 않은 타일을 배치하려고 시도한 경우입니다.
  - `NOT EMPTY`: 이미 타일이 위치해있는 칸에 타일을 추가로 배치하려고 시도한 경우입니다.
- `# Debug <FIRST/SECOND>: <msg>`
  - 선공 (`FIRST`) 혹은 후공 (`SECOND`)이 표준 에러 출력(`stderr`)에 출력한 줄입니다.
