# Local RAG

Lokalna biblioteka dokumentów z odpowiedziami opartymi na cytatach. Frontend to
zwykły HTML, CSS i JavaScript; API działa w FastAPI. Bez Gradio, Node.js,
budowania frontendu oraz zewnętrznych skryptów i fontów.

![Interfejs Local RAG](img/local-rag-desktop.jpg)

## Uruchomienie

Wymagania: Python 3.11+, [uv](https://docs.astral.sh/uv/) i lokalna
[Ollama](https://ollama.com/).

```bash
uv sync
ollama pull gemma4:e2b-it-qat
ollama pull embeddinggemma:latest
```

Uruchom aplikację Ollama albo `ollama serve`, a następnie z katalogu repozytorium:

```bash
uv run ui/app.py
```

Otwórz [127.0.0.1:7860](http://127.0.0.1:7860). Serwer nasłuchuje wyłącznie
lokalnie. Port możesz zmienić przez `PORT=7861 uv run ui/app.py`.

## Korzystanie

- Wybierz kolekcję z bocznej biblioteki i wpisz pytanie. Enter wysyła, Shift+Enter
  dodaje nową linię. Każde pytanie wyszukuje źródła samodzielnie; poprzednie
  wiadomości nie są kontekstem modelu.
- „Dokładne wyszukiwanie” włącza reranker. Lekki MiniLM ocenia 24 kandydatów i wybiera
  4 fragmenty. Po wyłączeniu wyszukiwane są 4 fragmenty KNN.
- Odpowiedź pojawia się po kontroli źródeł; cytaty rozwiniesz pod twierdzeniami.
  Podczas pracy interfejs pokazuje aktualny etap analizy i licznik czasu.
- „Dodaj dokumenty” tworzy nową kolekcję. Formaty: PDF, EPUB, DOCX, TXT, MD,
  HTML, CSV; do 30 plików i 100 MB łącznie. PDF musi zawierać warstwę tekstową;
  aplikacja nie wykonuje OCR skanów.
- Opcjonalne wzbogacanie generuje tytuły i pytania. Wydłuża import i domyślnie
  używa tego samego modelu co odpowiedzi.
- Import nie nadpisuje istniejącej kolekcji. Usunięcie kolekcji wymaga
  potwierdzenia w oknie i usuwa jej indeks oraz kopie dokumentów w `data/`.

Dotychczasowe kolekcje w `chroma_db/` działają bez ponownego indeksowania.
Rozmowy pozostają w pamięci strony i znikają po odświeżeniu lub zmianie kolekcji.
W jednym procesie trwa najwyżej jedna kosztowna operacja; kolejne żądanie z innej
karty otrzyma komunikat o zajętości. Nie uruchamiaj kilku procesów API na tej
samej bazie. UI nie wykonuje zewnętrznych żądań, ale biblioteki mogą pobierać
modele i pliki pomocnicze przy pierwszym uruchomieniu.

## Profil dla Maca M1 z 16 GB RAM

Domyślny model to `gemma4:e2b-it-qat`. Odpowiedzi i kontrola źródeł mają
`think=false`, kontekst 4096 oraz limity 384/128 tokenów. Rozumowanie
przed odpowiedzią nie jest potrzebne do każdego krótkiego pytania i w poprzednim
profilu Ornith 9B potrafiło zużyć cały limit 4096 tokenów bez wyniku.
Przełącznik **Think** w UI włącza lub wyłącza rozumowanie dla odpowiedzi, próby pomocniczej i weryfikacji. Wybór jest zapamiętywany, domyślnie wyłączony. Wymaga modelu obsługującego thinking i może wydłużyć odpowiedź. Tryb włączony ma limit 1536 tokenów generacji oraz 512 dla weryfikacji (zmienne `THINK_NUM_PREDICT` i `THINK_VERIFY_NUM_PREDICT`). Dla wywołań bez jawnego wyboru `RAG_THINKING=default` przywraca ustawienie
modelu. Przy takim eksperymencie trzeba też odpowiednio dobrać budżet generacji.
[Dokumentacja thinking](https://docs.ollama.com/capabilities/thinking).

Reranker to wielojęzyczny MiniLM, przetwarzający małe partie na CPU, z 4 wątkami.
Nie zajmuje GPU podczas generowania odpowiedzi. Ładuje się dopiero przy pierwszym
użyciu i pozostaje w pamięci procesu. Model embeddingów jest zwalniany z Ollamy
po embeddingu pytania; import nadal korzysta z niego seryjnie.
[Karta MiniLM](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).

Model wskazuje rzeczywiste fragmenty źródłowe zamiast pojedynczych wyrwanych
z dialogu zdań. Aplikacja sprawdza identyfikatory, kopiuje cytaty i osobno ocenia
każde twierdzenie: czy jest poparte źródłem oraz czy odpowiada na pytanie.
Krótka ocena przed werdyktem pomaga odrzucać odpowiedzi, które tylko powtarzają
pytanie. Nie jest wyświetlana użytkownikowi. To kontrola modelowa, nie gwarancja
poprawności: szczególnie trudne pozostają metafory, aluzje oraz pytania o wiele scen.

Gdy pierwsza odpowiedź jest pusta albo nie przejdzie kontroli, aplikacja wykonuje
jedną próbę prostym tekstem na tych samych fragmentach. Pomaga to małej Gemmie,
która czasem pomija poszukiwany fakt przy generowaniu JSON. Próba wymaga poprawnego
identyfikatora źródła i zawsze przechodzi weryfikację; nie wyłącza zabezpieczeń ani
nie korzysta z wiedzy spoza dokumentów. Może wydłużyć pytania bez odpowiedzi.
Pytanie „Gdzie Aomame zabiła lidera?” po tej zmianie dało „w apartamencie w hotelu
Okura” z cytatem z tomu 3, z rerankerem i bez niego (21,3 s i 14,0 s w kolejnych
próbach, z modelem już załadowanym; nie są to pomiary zimnego startu).

Budżet kontekstu uwzględnia prompt, pytanie i odpowiedź. Przycinanie zachowuje
pełne zdania; cytat nigdy nie obejmuje usuniętego końca źródła. Limit generacji
lub czasu daje wyraźny błąd, nie fałszywe stwierdzenie, że książka nie zawiera danych.
Logi pokazują oddzielnie embedding, wyszukiwanie, reranking, przetwarzanie promptu,
generowanie oraz weryfikację. Przejrzystość etapów nie wymaga wyświetlania rozumowania.

## Pomiary na M1 16 GB (2026-10-03)

Końcowy test przez UI po zatrzymaniu modelu w Ollamie i ponownym uruchomieniu
`uv run ui/app.py`: pytanie „Jak nazywał się mały detektyw który ścigał Aomame
po zabiciu lidera?” dało **Ushikawa**, z cytatem z tomu 3, w **26,4 s**.

| Etap zimnego startu | Czas |
|---|---:|
| Embedding i wyszukiwanie | 1,1 s |
| Ładowanie rerankera i selekcja | 7,0 s |
| Ładowanie LLM, prompt i odpowiedź | 13,9 s |
| Weryfikacja twierdzenia | 4,3 s |

Wcześniejszy log Ornith 9B pokazywał 30,7 s przetwarzania promptu oraz
411,6 s generowania 4096 tokenów, zakończonego limitem. Był używany w 100% GPU.
Po optymalizacji Ollama raportowała 3,6 GB dla Gemmy E2B (wcześniej 5,8 GB dla
Ornith). To pomiary konkretnych wywołań, nie gwarancja czasu każdego pytania.
Ponowne identyczne pytania mogą korzystać z cache promptu i nie są miarodajnym
pomiarem zimnego startu.

Porównano też Qwen 3.5 4B/9B, Gemmę E4B, większy reranker na MPS,
szersze konteksty i warianty wyszukiwania. Nie uzasadniły zastąpienia szybkiego
profilu w tych próbach; modele pobrane wyłącznie do porównania usunięto.

**Ograniczenia jakości wydania v1.12:** w ścisłej regresji książkowej 2 z 4 przypadków
były kompletne (nazwisko i prawidłowy brak numeru konta). Opis księżyców pomijał
żółty kolor dużego księżyca, a odpowiedź o przejściu między światami opisywała
przyczynę fabularną zamiast sceny ze schodami. Skrypt `grounding_live.py`
sygnalizuje te dwie niepełne odpowiedzi jako FAIL. Testy techniczne API, Chroma,
ograniczeń źródeł i ustawień modeli przechodzą. Przyspieszenie nie oznacza
pełnej poprawności interpretacji książki.

Po dodaniu próby pomocniczej i dwóch pytań o miejsce zabójstwa regresja daje
4/6 PASS: oba warianty miejsca, nazwisko i brak numeru konta. Dwa wcześniejsze
problemy (księżyce i przejście między światami) nadal pozostają nierozwiązane.

Kolekcje pozostały niezmienione: `1Q84_full` — 1476 fragmentów,
`sztuka_wojny` — 315. Nie wykonywano ponownego indeksowania danych użytkownika.

![Test odpowiedzi po optymalizacji](img/performance-desktop.jpg)

## Konfiguracja

| Zmienna | Domyślnie | Znaczenie |
|---|---|---|
| `STANDARD_MODEL` | `gemma4:e2b-it-qat` | Zalecany model odpowiedzi i weryfikacji |
| `QUESTION_MODEL` | wartość `STANDARD_MODEL` | Model opcjonalnego wzbogacania importu |
| `OLLAMA_NUM_CTX` | `4096` | Okno kontekstu odpowiedzi |
| `OLLAMA_NUM_PREDICT` | `384` | Limit generacji odpowiedzi |
| `VERIFY_NUM_PREDICT` | `128` | Limit oceny jednego twierdzenia |
| `RAG_THINKING` | `false` | `false`, `true` lub `default` |
| `RAG_TIMEOUT` | `90` | Limit sekund pojedynczej generacji, nie całego pytania |
| `VERIFY_ANSWERS` | `true` | Osobna kontrola twierdzeń; zalecana |
| `ENRICH_NUM_CTX` | `8192` | Kontekst opcjonalnego wzbogacania |
| `ENRICH_NUM_PREDICT` | `4096` | Budżet opcjonalnego wzbogacania |
| `RERANK_MODEL_NAME` | `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` | Reranker |
| `RERANK_CANDIDATES` | `24` | Kandydaci do rerankingu |
| `RERANK_TOP_N` | `4` | Wybrane źródła |
| `KNN_TOP_K` | `4` | Źródła bez rerankingu |
| `RERANK_DEVICE` | `cpu` | Urządzenie rerankera |
| `RERANK_THREADS` | `4` | Wątki PyTorch |
| `RERANK_MAX_LENGTH` | `512` | Długość pary pytanie–fragment |
| `DEBUG_CONTEXT` | `false` | Wypisywanie kontekstu w terminalu |

Zmiana modelu odpowiedzi nie wymaga ponownego indeksowania. Każdy model
ma inną szybkość i trafność, więc wybranie większego modelu nadal może wydłużyć czas.

## Struktura i testy

- `ui/app.py` — pipeline RAG i punkt uruchomienia.
- `ui/server.py` — lokalne API, upload i blokada współbieżnych operacji.
- `ui/static/` — interfejs bez frameworka.
- `data/`, `chroma_db/` — lokalne dokumenty i baza, wykluczone z Git.

Testy bez uruchamiania modeli:

```bash
uv run test/performance_scenario.py
uv run test/web_scenario.py
uv run test/model_defaults.py
```

Pierwszy sprawdza zapis Chroma, liczbę nodów, oba tryby wyszukiwania i walidację
źródeł. Drugi sprawdza HTTP, upload, usuwanie, walidację, blokadę i frontend.
Trzeci sprawdza szybki profil, opcję domyślnego thinking, ukrycie jego śladu i limit generacji.

Opcjonalna regresja na istniejącej kolekcji „1Q84”, z rzeczywistymi modelami:

```bash
uv run test/grounding_live.py --collection 1Q84_full
```

Scenariusz nie zmienia kolekcji; sprawdza detektywa, księżyce, przejście Aomame
i pytanie bez dowodów. Ocenia tekst odpowiedzi bez cytatów, żeby cytat z właściwym
słowem nie maskował błędnej odpowiedzi. Nie zastępuje to ręcznej oceny jakości.
Opcja `--model NAZWA` pozwala porównać inny model.

### Wybór modelu

Pod polem pytania wybierz model dostępny w lokalnej Ollamie. Lista pokazuje modele
obsługujące generowanie tekstu (bez modeli służących wyłącznie do embeddingów).
Przycisk ↻ odświeża listę po pobraniu lub usunięciu modelu. Przeglądarka zapamiętuje
wybór; dotyczy on odpowiedzi i weryfikacji źródeł. Profil aplikacji kontroluje
reasoning przez `RAG_THINKING`. Pierwsze otwarcie nowego profilu wybiera model
zalecany; późniejsze wybory są zapamiętywane. Model embeddingów i opcjonalne wzbogacanie importu
pozostają konfigurowane osobno. Połączenie z Ollamą ustawia `OLLAMA_HOST`
(domyślnie `http://localhost:11434`).

## Porównanie promptów i kontrola jakości (2026-10-03)

Przeprowadzono **189 porównawczych wywołań modeli oraz 12 pełnych zapytań RAG**
(dwa przebiegi po 6 pytań). Sprawdzono 7 wariantów promptu generatora, 3 warianty
weryfikatora, ograniczenie kontekstu do wybranych akapitów oraz Think on/off.
Dodatkowo wykonano trzy orientacyjne próby na lokalnym Ornith 9B.

**Decyzja: zachowano dotychczasowy prompt i próbę pomocniczą.** Kandydaci poprawiali
pojedyncze odpowiedzi, ale tracili inne; nie potwierdzono stabilnej przewagi.
Nie zmieniono domyślnego modelu ani nie włączono automatycznie Think.
Komunikat odmowy mówi teraz o niemożności potwierdzenia odpowiedzi — nie przesądza,
że w dokumentach nie ma poszukiwanej informacji.

Weryfikator testowano osobno na poprawnych i błędnych twierdzeniach:

| Wariant | Seria rozwojowa | Nowa seria kontrolna | Mediana czasu kontroli |
|---|---:|---:|---:|
| Obecny, Think off | 13/14 | 7/8 | 2,83 s |
| Ostrzejszy, Think off | 14/14 | 6/8 | 3,80 s |
| Obecny, Think on | — | 7/8 | 14,06 s |
| Ostrzejszy, Think on | — | 7/8 | 13,69 s |

Ostrzejszy wariant bez Think odrzucał poprawne sprostowanie błędnego założenia
pytania. Z Think jeden przypadek skończył się niekompletnym JSON po limicie.
Obecny weryfikator błędnie akceptował dosłowną interpretację metafory. Takie wyniki
nie uzasadniają deklaracji, że weryfikacja gwarantuje prawdziwość odpowiedzi.

Oba końcowe przebiegi `1Q84_full` dały **4/6**: hotel (dwa sformułowania), Ushikawa
i brak numeru konta. Opis księżyców nadal był niepełny/niejednoznaczny, a odpowiedź
o przejściu podawała przyczynę fabularną zamiast sceny ze schodami. W wybranych
źródłach do tego ostatniego pytania nie było sceny zejścia z autostrady. To wymaga
poprawy wyszukiwania; prompt nie powinien uzupełniać brakującego dowodu z pamięci.

### Jak odtworzyć porównanie

```bash
# Syntetyczne źródła, niezależnie od jakości wyszukiwania:
uv run test/prompt_eval.py --split dev --variants baseline concise extract --think off --output .codex/prompt-eval/new-dev.jsonl
uv run test/prompt_eval.py --split holdout --variants baseline --think off --output .codex/prompt-eval/new-holdout.jsonl
uv run test/verifier_eval.py --split holdout --variants baseline strict --think off --output .codex/prompt-eval/new-verifier.jsonl

# Zapis lokalnych fragmentów książki, następnie identyczne źródła dla wariantów:
uv run test/prompt_eval.py --freeze-book --book-file .codex/prompt-eval/new-book.json
uv run test/prompt_eval.py --split book --book-file .codex/prompt-eval/new-book.json --variants baseline minimal --think on --output .codex/prompt-eval/new-book-on.jsonl

# Pełny pipeline, z wyszukiwaniem, weryfikacją i próbą pomocniczą:
uv run test/grounding_live.py --think off --output .codex/prompt-eval/new-e2e.jsonl
```

Używaj nowego pliku wyników dla każdej serii; JSONL jest dopisywany. Uruchamiaj
porównania seryjnie, bez równoległego zadawania pytań w UI. Pliki ze źródłami książki
pozostają w ignorowanym `.codex/`. `grounding_live.py` kończy się błędem, jeśli któryś
przypadek nie przejdzie; skrypty porównawcze zapisują także nieudane próby.

[Raport pomiarów, decyzje i uwagi z ręcznego przeglądu](test/prompt_eval_results.json).
Zapisano wersję Ollamy, identyfikator modelu, hashe kontekstu i promptu, liczbę
tokenów, czas ładowania/prefill/generowania i błędy. Wyniki `rubric_passed` to
**heurystyka tekstowa**, nie automatyczny dowód poprawności: samo wystąpienie nazwy
hotelu może dotyczyć wywiezienia zwłok zamiast odpowiedzi o miejscu zabójstwa.
Raport wskazuje również zbyt surowe reguły oraz poprawne odmowy.

Pomiary generatora nie obejmują wyszukiwania i weryfikacji. Cache promptu był aktywny;
nie są to czasy zimnego startu ani gwarancja czasu w UI. Pierwszy końcowy test RAG
nakładał się z testami technicznymi i nie służy porównaniu czasu; drugi wykonano
osobno. Mały zestaw, jedna książka i parafrazy nie uzasadniają deklaracji
uniwersalnej trafności lub statystycznie istotnej przewagi.

Podstawy konfiguracji: [Ollama — structured outputs](https://docs.ollama.com/capabilities/structured-outputs)
i [format promptów Gemma 4](https://ai.google.dev/gemma/docs/core/prompt-formatting-gemma4).
Schemat zapewnia strukturę danych; poprawność treści sprawdzamy osobno.
