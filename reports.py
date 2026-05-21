def print_score_bar(score: int) -> None:
    filled = int(score / 10)
    empty = 10 - filled
    bar = "█" * filled + "░" * empty
    print(f"[{bar}] {score}/100")
