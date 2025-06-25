from .agent import build_engine, run_cli

if __name__ == "__main__":
    eng = build_engine()
    run_cli(eng, None)
