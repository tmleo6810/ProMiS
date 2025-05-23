import argparse
from pathlib import Path
import pandas as pd
import logging
import tiktoken
from tqdm import tqdm
from typing import Any, Tuple
import json
import openai
from collections import deque
import time

class RateLimiter:
    def __init__(self):
        self.max_rpm = 500
        self.max_tpm = 200000
        self._req_ts: deque[float] = deque()           # timestamps of requests (sec)
        self._tok_ts: deque[tuple[float, int]] = deque()  # (timestamp, tokens)

    def _purge(self, now: float) -> None:
        window_start = now - 60.0
        while self._req_ts and self._req_ts[0] < window_start:
            self._req_ts.popleft()
        while self._tok_ts and self._tok_ts[0][0] < window_start:
            self._tok_ts.popleft()

    def _current_tokens(self) -> int:
        return sum(t for _, t in self._tok_ts)

    def acquire(self, estimated_tokens: int) -> None:
        '''Block until sending `tokens` would respect both caps.'''
        while True:
            now = time.time()
            self._purge(now)

            need_sleep = 0.0
            if len(self._req_ts) >= self.max_rpm:
                need_sleep = max(need_sleep, self._req_ts[0] + 60.0 - now)

            if self._current_tokens() + estimated_tokens > self.max_tpm and self._tok_ts:
                need_sleep = max(need_sleep, self._tok_ts[0][0] + 60.0 - now)

            if need_sleep > 0:
                time.sleep(need_sleep)
                continue
            return
    
    def record(self, processed_tokens: int) -> None:
        now = time.time()
        self._purge(now)
        self._req_ts.append(now)
        self._tok_ts.append((now, processed_tokens))


class GPT35Turbo:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125"):
        self.model_name = model_name
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.context_window = 16385
        self.client = openai.OpenAI()

    def count_tokens(self, text: str) -> int:
        n = len(self.encoder.encode(text))
        return min(n, 16209) # 16209 is the max article token length

    def _truncate_article(self, article: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(article)
        if len(tokens) <= max_tokens:
            return article
        return self.encoder.decode(tokens[:max_tokens])

    def prompt(self, article: str, question: str, system_context: str, max_new_tokens: int) -> Tuple[str, int]:
        reserved = max_new_tokens + 160
        truncate_to = self.context_window - reserved
        article = self._truncate_article(article, truncate_to)

        response = self.client.responses.create(
            model=self.model_name,
            input=[
                {"role": "developer", "content": system_context},
                {
                    "role": "user",
                    "content": f"{article.strip()}\n\n{question.strip()}"
                }
            ],
            max_output_tokens=max_new_tokens,
            temperature=0
        )
        return response.output_text, response.usage.total_tokens
    
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="gpt35_turbo")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--rationales", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()

def load_cache(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        return []
    with cache_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]
    
def category_mapping(answer: str) -> int:
    answer_lc = answer.lower().lstrip()
    if answer_lc.startswith("yes"):
        return 1
    if answer_lc.startswith("no"):
        return 0
    if answer_lc.startswith("unsure"):
        return -1
    raise ValueError(f"Unrecognized answer '{answer}'")

def dump_cache(record: dict[str, Any], path: Path) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")

def process(model, df: pd.DataFrame, signals_df: pd.DataFrame, *, verbose: bool = False, rationales: bool = False, cache_path: Path, logger: logging.Logger, limiter: RateLimiter) -> None:
    system_context = (
        "You are a helpful and unbiased news verification assistant."
        " You will be provided with the title and the full body of text of a news article."
        " Then, you will answer further questions related to the given article."
        " Ensure that your answers are grounded in reality, truthful and reliable."
        " You are expeted to answer with 'Yes' or 'No', but you are also allowed to answer with 'Unsure' if you do not have enough information or context to provide a reliable answer."
    )
    if rationales:
        system_context += " Afterwards, explain your answer by providing a rationale."
    input_format = """Title: {title}\nText: {text}"""
    existing_article_ids = {row["article_id"] for row in load_cache(cache_path)}

    total_latency = 0.0
    total_tokens = 0
    request_latencies: list[float] = []

    with tqdm(total=len(df) * len(signals_df), unit="q") as pbar:
        for article_row in df.itertuples():
            if article_row.article_id in existing_article_ids:
                pbar.update(len(signals_df))
                continue
            article = input_format.format(title=article_row.title, text=article_row.text)
            gold_label = article_row.objective
            signals_count = len(signals_df)

            processed: dict[str, Any] = {}
            for i, question_row in enumerate(signals_df.itertuples()):
                question = "Question: "+ question_row.Question + " (Yes/Unsure/No)"

                # Token counting & rate-limit enforcement
                estimated_tokens = (
                    123
                    + model.count_tokens(article+"\n\n")
                    + model.count_tokens(question)
                )
                limiter.acquire(estimated_tokens)

                start = time.perf_counter()
                try:
                    answer, tokens = model.prompt(
                        article=article,
                        question=question,
                        system_context=system_context,
                        max_new_tokens=256 if rationales else 16
                    )
                except openai.OpenAIError as e:
                    msg = (
                        f"[{type(e).__name__}] | "
                        f"id = {article_row.article_id} | "
                        f"{question_row.Question} | {e}"
                    )
                    logger.error(msg)
                    pbar.update(signals_count - i)
                    break
                latency = time.perf_counter() - start

                limiter.record(tokens)

                total_latency += latency
                total_tokens += tokens
                request_latencies.append(latency)

                try:
                    cat_ws = category_mapping(answer)
                except ValueError as e:
                    msg = (
                        f"[{type(e).__name__}] | "
                        f"id = {article_row.article_id} | "
                        f"{question_row.Question} | {e}"
                    )
                    logger.error(msg)
                    pbar.update(signals_count - i)
                    continue

                processed[question_row._2] = cat_ws
                if rationales:
                    processed[question_row._2 + "_rationale"] = answer
                if verbose:
                    tqdm.write(
                        f"id = {article_row.article_id} | "
                        f"{question_row.Question} → {cat_ws} / {answer}"
                    )
                pbar.update(1)
            processed.update(
                {
                    "objective_true": gold_label,
                    "article_id": article_row.article_id
                }
            )
            dump_cache(processed, cache_path)
    
    print('=' * 60)
    print(f'Total requests            : {len(request_latencies)}')
    print(f"Total tokens: {total_tokens}")
    print(f'Total latency (sum)       : {total_latency:.2f} seconds')
    if request_latencies:
        print(f'Average latency/request   : {total_latency / len(request_latencies):.2f} seconds')
    print('=' * 60)

if __name__ == "__main__":
    args = parse_arguments()

    MODEL_NAME = args.model_name
    DATASET = args.dataset

    BASE_DIR = Path(__file__).resolve().parent.parent
    CACHE_DIR = BASE_DIR / "data" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH = CACHE_DIR / f"{DATASET}.jsonl"

    LOG_DIR = BASE_DIR / "data" / "logs"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    LOG_PATH = LOG_DIR / f"errors_{DATASET}.log"

    logging.basicConfig(
        filename=LOG_PATH,
        encoding="utf-8",
        level=logging.ERROR,
        format="%(asctime)s - %(levelname)s : %(message)s"
    )
    logger = logging.getLogger(__name__)

    DATASET_PATH = BASE_DIR / "data" / "dataset" / f"{DATASET}.csv"
    # Test dataset
    # DATASET_PATH = BASE_DIR / "data" / "dataset" / f"{DATASET}_sample.csv"
    SIGNALS_PATH = BASE_DIR / "data" / "signals.csv"
    
    assert DATASET_PATH.exists(), f"Dataset CSV not found: {DATASET_PATH}"
    assert SIGNALS_PATH.exists(), f"Signals CSV not found: {SIGNALS_PATH}"

    df = pd.read_csv(DATASET_PATH)
    signals_df = pd.read_csv(SIGNALS_PATH)

    if MODEL_NAME == "gpt35_turbo":
        model = GPT35Turbo()
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")
    
    limiter = RateLimiter()
    
    print(f"Dataset: {DATASET} | Model: {MODEL_NAME}")
    if args.verbose:
        print(f"Rationales: {args.rationales}")
    
    process(
        model,
        df,
        signals_df,
        verbose=args.verbose,
        rationales=args.rationales,
        cache_path=CACHE_PATH,
        logger=logger,
        limiter=limiter
    )
