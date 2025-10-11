# Repository Guidelines

## Struktura projektu i moduły
Repozytorium skupia logikę aplikacji w `ui/app.py`, który definiuje pipeline ingestowania i interfejs Gradio. Dane robocze trafiają do `data/<nazwa_kolekcji>` oraz trwałej bazy `chroma_db/`; przy współpracy dbaj, by katalogi robocze nie lądowały w git. Artefakty dokumentacyjne znajdują się w `docs/` i `img/`. Skrypty pomocnicze trzymaj w `spec/`, scenariusze testowe w `test/`. Zależności i konfigurację dopisuj w `pyproject.toml`.

## Budowanie, testy i środowisko deweloperskie
Narzędziem zarządzającym jest `uv`. Po klonowaniu wykonaj synchronizację środowiska:
```bash
uv sync
```
Do uruchomienia UI użyj:
```bash
OLLAMA_HOST=... ollama serve &
uv run ui/app.py
```
Podczas pracy aplikacja wypisuje w konsoli fragmenty kontekstu przekazywane do LLM wraz z informacją, czy rerank jest aktywny. Ingestion można odtworzyć na przykładzie:
```bash
uv run test/embeddings.py
```
Skrypty w katalogu `test/` traktuj jako referencję do tworzenia własnych scenariuszy próbnych. Dla szybkiej inspekcji bazy wektorów skorzystaj z `chroma_db/` wraz z klientem `chromadb`.

## Styl kodu i konwencje nazewnicze
Kod w repozytorium wykorzystuje Python ≥3.11, 4-spacowe wcięcia i `snake_case` dla funkcji oraz zmiennych. Klasy i konfiguracje modułów nazywamy w `PascalCase`, a flagi środowiskowe wielkimi literami (`RERANK_MODEL_NAME`). Konfiguracje trzymaj w jednym miejscu (sekcje stałych na górze pliku) i dodawaj komentarze tylko przy nieoczywistych fragmentach. Rekomendowany formatter to `uv run python -m black ui test` przed zgłoszeniem zmian.

## Wytyczne testowe
Preferujemy testy scenariuszowe uruchamiane poleceniami `uv run test/<obszar>.py`. Nazwy skryptów opisują funkcję (`query.py`, `qwen_query.py`); nowe pliki zachowują schemat `test/<feature>_*.py`. Testy powinny korzystać z małych, jawnie opisanych zbiorów w `data/`, a wyniki tymczasowe zapisuj do `.gitignore`owanych plików. Przy zmianach pipeline’u zweryfikuj liczbę wygenerowanych nodów oraz poprawność zapisu do ChromaDB.

## Commity i pull requesty
Historia wykorzystuje przedrostki `add:`, `fix:`, `chore:` itp. – stosuj je w kolejnych commitach i pisz zwięzłe opisy w języku polskim. Każdy PR musi zawierać streszczenie celu, instrukcję testów (`uv run ...`) oraz link do powiązanego zadania. Załącz zrzuty ekranu UI w sekcji wyników, gdy zmieniasz warstwę prezentacji. Przed wysłaniem PR upewnij się, że katalogi robocze (`chroma_db`, `data`) nie zawierają prywatnych danych użytkowników.

## Konfiguracja modeli i bezpieczeństwo
Aplikacja polega na lokalnym serwerze Ollama oraz modelach `gemma3:4b-it-qat`, `qwen3:4b` i embeddingach `embeddinggemma:latest`. Zmieniaj domyślne nazwy lub parametry rerankera przez zmienne środowiskowe (`RERANK_DEVICE`, `RERANK_TOP_N`) zamiast modyfikacji kodu. Interfejs UI udostępnia przełącznik „Rerank”, który domyślnie włącza SentenceTransformer i przepuszcza 20–32 fragmenty; po wyłączeniu model otrzymuje maksymalnie 4 konteksty KNN. Wrażliwe dane wejściowe przechowuj poza repozytorium i czyść katalog `data/` przed commitami, aby uniknąć przypadkowego publikowania dokumentów klientów.
