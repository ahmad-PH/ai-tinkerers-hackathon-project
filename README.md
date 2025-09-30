# AI Tinkerers Hackathon Project

This project is developed for the AI Tinkerers x Google Cloud Hackathon.

## Project Structure

- `src/`: Contains the main source code for the application.
- `tests/`: Contains test cases for the application using PyTest.
- `notebooks/`: Jupyter notebooks for experiments and analysis.

## Setup Instructions

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd ai-tinkerers-hackathon
   ```

2. **Install Dependencies**

   ```bash
   uv sync --dev
   ```

3. **Set environment variables**
   Set your environment variables in `.env`. Use `.env.example` as a template:
   ```bash
   cp -v .env.example .env
   ```

4. **Pre-commit checks**
   To ensure code quality and consistency, please run pre-commit checks before pushing your code. This will automatically format code, check for linting errors, and catch common issues early.

   Run all pre-commit checks manually:
   ```bash
   pre-commit run --all-files
   ```

5. **Run tests**
   ```bash
   uv run pytest
   ```
