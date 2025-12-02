# vs_agent.py
import os
import subprocess
import random
import argparse
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_TOOL = os.path.join(BASE_DIR, "testing-tool-connexion.py")

# 16가지 타일 타입 (문자열 기준)
COLORS = ["R", "G", "B", "Y"]
SYMS = ["1", "2", "3", "4"]
ALL_TILE_TYPES = [c + s for c in COLORS for s in SYMS]


def make_random_tile_sequence():
    """
    한 플레이어용 32장 타일 시퀀스를 랜덤 생성.
    - 16가지 타입 * 2장씩
    - 순서는 랜덤
    반환: ["R1", "Y3", ..., 32개]
    """
    tiles = []
    for t in ALL_TILE_TYPES:
        tiles.append(t)
        tiles.append(t)
    random.shuffle(tiles)
    return tiles


def write_input_file(path):
    """
    두 플레이어용 랜덤 input.txt 하나 생성.
    """
    first_tiles = make_random_tile_sequence()
    second_tiles = make_random_tile_sequence()

    with open(path, "w", encoding="utf-8") as f:
        f.write(" ".join(first_tiles) + "\n")
        f.write(" ".join(second_tiles) + "\n")


def write_config_file(path, input_path, log_path,
                      ppo_first: bool,
                      ppo_cmd: str, ppo_cwd: str,
                      opp_cmd: str, opp_cwd: str):
    """
    config.ini 파일을 동적으로 생성.
    ppo_first=True면 PPO가 선공(FIRST),
    False면 PPO가 후공(SECOND)이 되도록 EXEC1/EXEC2를 스왑.
    """
    if ppo_first:
        exec1 = ppo_cmd
        cwd1 = ppo_cwd
        exec2 = opp_cmd
        cwd2 = opp_cwd
    else:
        exec1 = opp_cmd
        cwd1 = opp_cwd
        exec2 = ppo_cmd
        cwd2 = ppo_cwd

    # INPUT/LOG는 testing-tool 기준에서의 경로 (BASE_DIR 기준 상대경로)
    input_rel = os.path.relpath(input_path, BASE_DIR)
    log_rel = os.path.relpath(log_path, BASE_DIR)

    config_text = f"""INPUT={input_rel}
LOG={log_rel}
EXEC1={exec1}
EXEC2={exec2}
CWD1={cwd1}
CWD2={cwd2}
"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(config_text)


def parse_result_from_log(log_path, ppo_first: bool):
    """
    로그 파일에서 [RESULT "..."], SCOREFIRST, SCORESECOND를 파싱.
    PPO 관점에서 승/무/패 및 점수차(my_score - opp_score)를 반환.
    """
    result_str = None
    score_first = None
    score_second = None

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("[RESULT"):
                # 예: [RESULT "1-0"]
                try:
                    result_str = line.split('"')[1]
                except Exception:
                    pass
            elif line.startswith("SCOREFIRST"):
                parts = line.split()
                if len(parts) == 2:
                    score_first = int(parts[1])
            elif line.startswith("SCORESECOND"):
                parts = line.split()
                if len(parts) == 2:
                    score_second = int(parts[1])

    if result_str is None:
        return "abort", 0

    if ppo_first:
        # FIRST = PPO
        if result_str == "1-0":
            outcome = "win"
        elif result_str == "0-1":
            outcome = "lose"
        elif result_str == "1/2-1/2":
            outcome = "draw"
        else:
            outcome = "unknown"
        my_score = score_first
        opp_score = score_second
    else:
        # SECOND = PPO
        if result_str == "1-0":
            outcome = "lose"
        elif result_str == "0-1":
            outcome = "win"
        elif result_str == "1/2-1/2":
            outcome = "draw"
        else:
            outcome = "unknown"
        my_score = score_second
        opp_score = score_first

    score_diff = 0
    if my_score is not None and opp_score is not None:
        score_diff = my_score - opp_score

    return outcome, score_diff


def run_many_games(num_games: int,
                   ppo_cmd: str, ppo_cwd: str,
                   opp_cmd: str, opp_cwd: str,
                   log_prefix: str,
                   output_dir: str):
    """
    여러 번 게임을 돌리고, 로그는 전부 output_dir 안에 저장.
    """
    wins = draws = losses = aborts = 0
    score_diffs = []

    for i in range(num_games):
        print(f"=== Game {i+1} / {num_games} ===")

        # 선/후공 번갈아가며
        ppo_first = (i % 2 == 0)

        input_path = os.path.join(BASE_DIR, "input_auto.txt")
        log_path = os.path.join(output_dir, f"{log_prefix}_{i+1}.txt")
        config_path = os.path.join(BASE_DIR, "config_auto.ini")

        # 1) input 생성
        write_input_file(input_path)
        # 2) config 생성
        write_config_file(config_path, input_path, log_path,
                          ppo_first, ppo_cmd, ppo_cwd, opp_cmd, opp_cwd)

        # 3) testing-tool 호출
        cmd = ["python", TEST_TOOL, "-c", config_path]
        subprocess.run(cmd, cwd=BASE_DIR)

        # 4) 결과 파싱
        if not os.path.exists(log_path):
            print("  >> log file not found, treat as abort")
            aborts += 1
            continue

        outcome, sd = parse_result_from_log(log_path, ppo_first)
        score_diffs.append(sd)

        print(f"  PPO as {'FIRST' if ppo_first else 'SECOND'} "
              f"-> {outcome}, score_diff={sd}")

        if outcome == "win":
            wins += 1
        elif outcome == "lose":
            losses += 1
        elif outcome == "draw":
            draws += 1
        else:
            aborts += 1

    print("\n=== Summary ===")
    print(f"games     : {num_games}")
    print(f"wins      : {wins}")
    print(f"draws     : {draws}")
    print(f"losses    : {losses}")
    print(f"aborts    : {aborts}")
    avg_sd = None
    if score_diffs:
        avg_sd = sum(score_diffs) / len(score_diffs)
        print(f"avg score diff (PPO - opp): {avg_sd:.2f}")

    # summary.txt로도 저장
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("=== Summary ===\n")
        f.write(f"games     : {num_games}\n")
        f.write(f"wins      : {wins}\n")
        f.write(f"draws     : {draws}\n")
        f.write(f"losses    : {losses}\n")
        f.write(f"aborts    : {aborts}\n")
        if avg_sd is not None:
            f.write(f"avg score diff (PPO - opp): {avg_sd:.2f}\n")


def infer_agent_name(cmd: str) -> str:
    """
    'python ppo_agent_cli.py' -> 'ppo_agent_cli'
    'python ./sample/sample-code.py P2' -> 'sample-code'
    같은 느낌으로, 커맨드 문자열에서 스크립트 이름만 뽑아 라벨로 사용.
    """
    parts = cmd.strip().split()
    if not parts:
        return "unknown"

    # 마지막 토큰부터 거꾸로 보고, .py로 끝나거나 경로처럼 보이는 애를 찾자.
    for tok in reversed(parts):
        base = os.path.basename(tok)
        name, ext = os.path.splitext(base)
        if ext == ".py":
            return name

    # 못 찾으면 그냥 마지막 토큰 베이스이름
    base = os.path.basename(parts[-1])
    name, _ = os.path.splitext(base)
    return name or "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Run multiple Connexion games: PPO vs Opponent"
    )
    parser.add_argument(
        "--games", "-n",
        type=int,
        default=20,
        help="number of games to run (default: 20)",
    )

    # PPO 에이전트 실행 커맨드 / 작업 디렉토리
    parser.add_argument(
        "--ppo-cmd",
        type=str,
        default="python ppo_agent_cli.py",
        help='PPO agent command (default: "python ppo_agent_cli.py")',
    )
    parser.add_argument(
        "--ppo-cwd",
        type=str,
        default=".",
        help='PPO agent working directory (default: current dir ".")',
    )

    # 상대 에이전트 실행 커맨드 / 작업 디렉토리
    parser.add_argument(
        "--opp-cmd",
        type=str,
        required=True,
        help='Opponent agent command, e.g. "python random_agent.py"',
    )
    parser.add_argument(
        "--opp-cwd",
        type=str,
        default=".",
        help='Opponent working directory (default: current dir ".")',
    )

    parser.add_argument(
        "--log-prefix",
        type=str,
        default="log_game",
        help='Prefix for per-game log files (default: "log_game")',
    )

    args = parser.parse_args()

    # cwd는 BASE_DIR 기준으로 해석해서 절대경로로 바꿔두는 게 안전
    ppo_cwd_abs = os.path.abspath(os.path.join(BASE_DIR, args.ppo_cwd))
    opp_cwd_abs = os.path.abspath(os.path.join(BASE_DIR, args.opp_cwd))

    # 에이전트 이름 추출해서 폴더 이름 만들기
    ppo_name = infer_agent_name(args.ppo_cmd)
    opp_name = infer_agent_name(args.opp_cmd)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    dir_name = f"Log_{ppo_name}_vs_{opp_name}_{now_str}"
    output_dir = os.path.join(BASE_DIR, dir_name)
    os.makedirs(output_dir, exist_ok=True)

    print("PPO CMD :", args.ppo_cmd)
    print("PPO CWD :", ppo_cwd_abs)
    print("OPP CMD :", args.opp_cmd)
    print("OPP CWD :", opp_cwd_abs)
    print("LOG DIR :", output_dir)

    run_many_games(
        num_games=args.games,
        ppo_cmd=args.ppo_cmd,
        ppo_cwd=ppo_cwd_abs,
        opp_cmd=args.opp_cmd,
        opp_cwd=opp_cwd_abs,
        log_prefix=args.log_prefix,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
