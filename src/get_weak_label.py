import argparse
from pathlib import Path
import pandas as pd
from openai import OpenAI
import tiktoken
from tqdm import tqdm
from typing import Any
import json

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name", type=str, default="gpt35_turbo")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--rationales", action="store_true")
    parser.add_argument("--verbose", action="store_true")

    return parser.parse_args()

class GPT35Turbo:
    def __init__(self, model_name: str = "gpt-3.5-turbo-0125"):
        self.model_name = model_name
        self.encoder = tiktoken.encoding_for_model(model_name)
        self.context_window = 16385

    def _truncate_article(self, article: str, max_tokens: int) -> str:
        tokens = self.encoder.encode(article)
        if len(tokens) <= max_tokens:
            return article
        return self.encoder.decode(tokens[:max_tokens])

    def prompt(self, article: str, question: str, system_context: str, max_new_tokens: int) -> str:
        reserved = max_new_tokens + 160
        truncate_to = self.context_window - reserved
        article = self._truncate_article(article, truncate_to)

        client = OpenAI()
        response = client.responses.create(
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
        return response.output_text

def load_cache(cache_path: Path) -> list[dict[str, Any]]:
    if not cache_path.exists():
        return []
    with cache_path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]

def process(model, df: pd.DataFrame, signals_df: pd.DataFrame, *, verbose: bool = False, rationales: bool = False, cache_path: Path) -> None:
    system_context = (
        "You are a helpful and unbiased news verification assistant."
        " You will be provided with the title and the full body of text of a news article."
        " Then, you will answer further questions related to the given article."
        " Ensure that your answers are grounded in reality, truthful and reliable."
        " You are expeted to answer with 'Yes' or 'No', but you are also allowed to answer with 'Unsure' if you do not have enough information or context to provide a reliable answer."
    )
    input_format = """Title: {title}\nText: {text}"""

    existing_article_ids = {row["article_id"] for row in load_cache(cache_path)}

    with tqdm(total=len(df) * len(signals_df), unit="q") as pbar:
        for article_row in df.itertuples():
            if article_row.article_id in existing_article_ids:
                pbar.update(len(signals_df))
                continue
            input = input_format.format(title=article_row.title, text=article_row.text)
            for question_row in signals_df.itertuples():
                if rationales:
                    system_context += " Afterwards, explain your answer by providing a rationale."
                question = question_row.Question + " (Yes/Unsure/No)"

                try:

if __name__ == "__main__":
    args = parse_arguments()

    MODEL_NAME = args.model_name
    DATASET = args.dataset

    BASE_DIR = Path(__file__).resolve().parent.parent
    CACHE_DIR = BASE_DIR / "data" / "cache"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_PATH = CACHE_DIR / f"{DATASET}.jsonl"

    DATASET_PATH = BASE_DIR / "data" / "dataset" / f"{DATASET}.csv"
    SIGNALS_PATH = BASE_DIR / "data" / "signals.csv"
    
    assert DATASET_PATH.exists(), f"Dataset CSV not found: {DATASET_PATH}"
    assert SIGNALS_PATH.exists(), f"Signals CSV not found: {SIGNALS_PATH}"

    df = pd.read_csv(DATASET_PATH)
    signals_df = pd.read_csv(SIGNALS_PATH)

    if MODEL_NAME == "gpt35_turbo":
        model = GPT35Turbo()
    else:
        raise ValueError(f"Unsupported model: {MODEL_NAME}")
    
    print(f"Dataset: {DATASET} | Model: {MODEL_NAME}")
    if args.verbose:
        print(f"Rationales: {args.rationales}")
    
    process(
        model,
        df,
        signals_df,
        verbose=args.verbose,
        rationales=args.rationales,
        cache_path=CACHE_PATH
    )
