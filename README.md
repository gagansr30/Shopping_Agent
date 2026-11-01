# AI Shopping Assistant

A Streamlit shopping assistant that searches a local product catalog, retrieves
customer ratings, analyzes uploaded product images, and can place demo orders.
The assistant uses LangChain with Groq-hosted language models. Product,
review, and order data are stored in SQLite.

## Features

- Search products by keyword, maximum price, and organic status.
- Show product ratings and review counts.
- Upload a JPG, JPEG, PNG, or WebP image to identify a product and find
  catalog matches.
- Place a demo order after explicitly confirming a product.
- Retrieve order history and save organic/maximum-price preferences per shopper.
- Reject requests unrelated to shopping before they reach the shopping agent.
- Keep the product catalog, reviews, orders, and preferences in the local
  `store.db` database.

## Requirements

- Python 3.14 or newer
- [uv](https://docs.astral.sh/uv/)
- A Groq API key

## Setup

1. Clone the repository and enter the project directory:

   ```powershell
   git clone https://github.com/gagansr30/Shopping_Agent.git
   cd Shopping_Agent
   ```

2. Install the project dependencies:

   ```powershell
   uv sync
   ```

3. Create a `.env` file in the project root and add your Groq API key:

   ```dotenv
   GROQ_API_KEY=your_groq_api_key
   ```

   Keep `.env` private; it is excluded from Git.

## Run the Streamlit app

From the project directory, run:

```powershell
uv run streamlit run setup_db.py
```

Streamlit prints a local URL (usually `http://localhost:8501`) when the app is
ready.

## Deploy to Streamlit Community Cloud

1. Connect the GitHub repository in Streamlit Community Cloud.
2. Set the app's main file path to `setup_db.py`.
3. Add `GROQ_API_KEY` under the app's **Settings → Secrets** using TOML syntax:

   ```toml
   GROQ_API_KEY = "your_groq_api_key"
   ```

The root-level `requirements.txt` lists the app's runtime dependencies and is
used by Streamlit Community Cloud during deployment.

## Run the command-line example

```powershell
uv run python shopping_agent.py
```

The CLI example searches for organic honey under $20 with a rating of at least
4.5 stars.

## Run the evaluations

With `GROQ_API_KEY` configured, run:

```powershell
uv run python evaluations.py
```

The evaluation checks expected tool calls and parameters for a small shopping
query set, then uses an LLM judge to score relevance, product correctness, and
numbered-list format compliance. It does not place orders.

## Project files

- `setup_db.py` — Streamlit chat and image-upload interface.
- `shopping_agent.py` — LangChain agent and product, rating, checkout, and
  image-analysis tools.
- `evaluations.py` — live tool-call assertions and LLM-as-judge response checks.
- `reviews_api.py` — SQLite queries for product ratings.
- `store.db` — local SQLite database containing the product catalog, reviews,
  and orders.
- `pyproject.toml` / `uv.lock` — Python project metadata and dependency lock.

## Notes

- Image results depend on the configured model's image-understanding
  capabilities. If an image cannot be identified, the assistant should not
  search the catalog based on a guessed product.
- Checkout writes a demo order to the local database; it does not process a
  payment or contact a shipping provider.