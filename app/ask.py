"""第1回: 素のLLMに1回だけ質問するスクリプト。

使い方:
    uv run python -m app.ask "LoRAとは何ですか？"
    uv run python -m app.ask            # 質問を省略すると、講座に固有の質問をする
    uv run python -m app.ask --chat     # 会話履歴を保つチャット

ツールを持たない「素の」呼び出し。
講座に固有のことを聞くと、「分からない」と答えるか、一般論でもっともらしく答える
（講座の実際とは違う）ことを確かめよう。
（第3回で、用語集を検索できるエージェントにして改善する）
"""

import argparse
import time

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key

# 講座に固有の質問。正しい答えは data/glossary.json の homework-levels にある
DEFAULT_QUESTION = "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
STOP_COMMANDS = {"exit", "quit", "end", "終了"}


def ask(question: str) -> str:
    client = genai.Client(api_key=get_api_key())
    started_at = time.perf_counter()
    response = client.models.generate_content(model=MODEL, contents=question)
    elapsed = time.perf_counter() - started_at
    usage = response.usage_metadata

    print(
        "[使用量] "
        f"入力: {usage.prompt_token_count:,} tokens / "
        f"出力: {usage.candidates_token_count:,} tokens / "
        f"応答時間: {elapsed:.2f}秒"
    )

    return response.text


def chat() -> None:
    """会話履歴を保ちながら、停止指示までモデルと対話する。"""
    client = genai.Client(api_key=get_api_key())
    session = client.chats.create(model=MODEL)

    print("チャットを開始しました。終了するには exit または 終了 と入力してください。")
    while True:
        try:
            question = input("あなた: ").strip()
        except EOFError:
            print()
            return

        if question.casefold() in STOP_COMMANDS:
            print("チャットを終了します。")
            return
        if not question:
            continue

        started_at = time.perf_counter()
        response = session.send_message(question)
        elapsed = time.perf_counter() - started_at
        usage = response.usage_metadata

        print(f"モデル: {response.text}")
        print(
            "[使用量] "
            f"入力: {usage.prompt_token_count:,} tokens / "
            f"出力: {usage.candidates_token_count:,} tokens / "
            f"応答時間: {elapsed:.2f}秒\n"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="素のLLMに質問する")
    parser.add_argument("--chat", action="store_true", help="会話履歴を保つチャットを開始する")
    parser.add_argument("question", nargs="?", default=DEFAULT_QUESTION)
    args = parser.parse_args()

    print(f"[model] {MODEL}")
    try:
        if args.chat:
            chat()
        else:
            print(f"[質問] {args.question}\n")
            print(ask(args.question))
    except errors.APIError as e:
        print(api_error_message(e.code, e.message))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
