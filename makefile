# Makefile
.PHONY: install run

export $(shell sed 's/=.*//' .env)

install:
	uv sync --dev && \
	uv tool install "git+https://github.com/54yyyu/zotero-mcp.git" && \
	zotero-mcp setup \
	  --api-key $$ZOTERO_API_KEY \
	  --library-id $$ZOTERO_USER_ID \
	  --no-claude \
	  --no-local

run:
	streamlit run app.py
