# Makefile
.PHONY: install run clean

# Load environment variables from .env
ifneq (,$(wildcard .env))
  include .env
  export
else
  $(warning ⚠️  No .env file found — some commands may fail)
endif

install:
	@echo "📦 Installing dependencies..."
	uv sync --dev
	uv tool install "git+https://github.com/54yyyu/zotero-mcp.git"
	@echo "⚙️  Setting up Zotero MCP..."
	zotero-mcp setup \
	  --api-key $(ZOTERO_API_KEY) \
	  --library-id $(ZOTERO_USER_ID) \
	  --no-claude \
	  --no-local
	@echo "✅ Install complete"

run:
	@echo "🚀 Running Streamlit app..."
	streamlit run src/app.py
