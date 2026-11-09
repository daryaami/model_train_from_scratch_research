# Сравнение OCR и прогон химического корпуса

Здесь проверялось, какой движок даёт русский текст и формулы, пригодные для обучения. Выборка — шесть страниц из `data/chem_dataset/chem_extracted`. Разбор цифр и цитат — в [`ОТЧЕТ.md`](ОТЧЕТ.md). Ниже — как устроены скрипты и как повторить прогон.

Итог выборки: для корпуса страниц брался Marker. Дальше книги пошли через MinerU. Облачный API и локальный `mineru-kit` пишут в один каталог и в один журнал, готовую книгу второй проход не трогает.

Движки не ставятся в одно окружение. У Marker, Docling, MinerU 3.4, MinerU 4 и пары DeepSeek/dots разные сборки torch и transformers. `requirements.txt` в этом каталоге — только линейное сравнение (`run_bench.py`). Остальные команды ниже создают отдельные venv.

## Что лежит в каталоге

| Путь | Зачем |
|---|---|
| `sample/manifest.json` | Шесть страниц: id, книга, номер страницы, есть ли эталон |
| `sample/pages/` | PNG, PDF и эталон. Каталог в `.gitignore`, его собирает `prepare_sample.py` |
| `sample/docs/` | Ещё четыре PDF: cathode, kormachev, labtech, tables |
| `results/` | Ответы движков. `results/scores.json` — recall и CER линейных движков |
| `results/text/` | Tesseract, RapidOCR, Docling. Каталог в `.gitignore` |
| `.venv*` | Окружения движков, в git не входят |
| `.cache/`, `.tools/` | Веса и CUDA-сборка llama.cpp для Marker |

Шесть страниц выборки: Мингулина (сплошной текст), Крестов (таблица), Смит (ИК), Фролов (практикум), фармакопея без текстового слоя, журнальная страница с аллилированием.

## Общие бинарники

Их нет в pip.

- DjVuLibre: `../.tools/djvulibre/usr/bin` (`ddjvu`, `djvused`, `djvutxt`). Без них не собирается выборка и не режется корпус.
- Tesseract 5 с `rus` и `eng`: `../.tools/tesseract`. `run_bench.py` и `run_docling.py` сами выставляют `LD_LIBRARY_PATH` и `TESSDATA_PREFIX`.
- Poppler в `PATH`: `pdftotext` и `pdfinfo`. Облачный прогон смотрит ими текст PDF, локальный MinerU считает страницы через `pdfinfo`.
- CUDA llama.cpp сборки b11838: `.tools/llama-cuda/llama-b11838`. Её поднимает `run_marker.sh`.

Книги читаются из `data/chem_dataset/chem_extracted`.

## Выборка

Из корня репозитория, в окружении с Pillow (`requirements.txt` корня):

```bash
python ocr_research/prepare_sample.py
```

Скрипт рисует PNG и PDF и, где слой DjVu непустой, кладёт его в `sample/pages/<id>.gt.txt`.

## Линейные движки

Отдельное окружение, зависимости из `requirements.txt` этого каталога: RapidOCR 3.10.0, RapidFuzz 3.14.6, onnxruntime 1.31.0.

```bash
cd ocr_research
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python run_bench.py line
```

`line` гоняет Tesseract `rus+eng` и RapidOCR PP-OCRv5: распознаватели `eslav` и `cyrillic`, детекторы mobile и server, плюс чтение по двум колонкам. Готовый файл не пересчитывается. `run_bench.py score` только пересчитывает `results/scores.json` по текстам в `results/text/`.

## Marker

Окружение, в котором снималась выборка: `marker-pdf==2.0.0` (Surya 0.22.1). Ставить в `.venv-marker`, не в `.venv`.

```bash
cd ocr_research
python3.12 -m venv .venv-marker
.venv-marker/bin/pip install marker-pdf==2.0.0
bash run_marker.sh
```

`run_marker.sh` берёт `sample/pages/{general,thermo,ir,colloid}.pdf`, включает OCR и пишет Markdown в `results/marker/<id>/`. Нужны свободные примерно 10 ГБ на GPU под llama.cpp и каталог `.tools/llama-cuda/llama-b11838`.

Marker пишет индексы как `$SiH_4$`. Превью Markdown режет такое подчёркивание раньше KaTeX. Замена только обрамления, без правки самой формулы:

```bash
python fix_marker_math.py results/marker/general/general/general.md
python fix_marker_math.py --check results/marker/general/general/general.md
```

Четыре PDF из `sample/docs/` гонялись тем же `marker_single` в `results/marker_docs/`.

## Docling

Текст здесь снимает тот же Tesseract. Docling добавляет раскладку, таблицы и свою модель формул. На выборке формулы вышли слабее Marker. Окружение: `docling==2.142.1`.

```bash
cd ocr_research
python3.12 -m venv .venv-docling
.venv-docling/bin/pip install docling==2.142.1
.venv-docling/bin/python run_docling.py
```

Скрипт сам прячет GPU (`CUDA_VISIBLE_DEVICES` пустой) и пишет `results/text/docling/<id>.md`.

## MinerU на выборке

Две разные поставки.

Pipeline с русским языком, MinerU 3.4.5:

```bash
cd ocr_research
python3.12 -m venv .venv-mineru34
.venv-mineru34/bin/pip install mineru==3.4.5
.venv-mineru34/bin/mineru -p sample/pages/general.pdf -o results/mineru34/general \
  -b pipeline -l east_slavic -m ocr
```

VLM без выбора языка, MinerU 4.0.12, уровень `standard`. В этой сборке `mineru-llama-cpp` без CUDA, поэтому VLM на выборке шёл на CPU.

```bash
cd ocr_research
python3.12 -m venv .venv-mineru
.venv-mineru/bin/pip install mineru==4.0.12
MINERU_MODEL_VLM_ENGINE=llama-cpp MINERU_MODEL_SMALL_BACKEND=onnx \
  .venv-mineru/bin/mineru-kit parse sample/pages/general.pdf \
  -o results/mineru/general_vlm.md --tier standard --ocr-mode ocr
```

MinerU 4 на уровне `basic` в этом прогоне взял китайский PP-OCRv6 и кириллицу потерял. Для русского pipeline это не тот запуск. Ответы выборки лежат в `results/mineru/` и `results/mineru34/`. Четыре PDF из `sample/docs/` — в `results/mineru_docs/`.

## DeepSeek-OCR-2 и dots.ocr

Оба скрипта рассчитаны на `.venv-vlm`: torch `2.11.0+cu128` и transformers `4.51.3`. В общий `.venv` их не ставить. Код DeepSeek-OCR-2 рассчитан на более старый transformers и всё ещё импортирует `LlamaFlashAttention2`. `run_deepseek.py` подставляет на её место обычный `LlamaAttention`. Внимание при этом eager, flash-attn не нужен. dots.ocr считает зрение через SDPA: на полном разрешении матрица внимания просила около 22 ГБ и в свободные 10 ГБ не влезла.

```bash
cd ocr_research
python3.12 -m venv .venv-vlm
.venv-vlm/bin/pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 \
  --extra-index-url https://download.pytorch.org/whl/cu128
.venv-vlm/bin/pip install transformers==4.51.3 qwen-vl-utils==0.0.14 einops==0.8.2 addict==2.4.0
.venv-vlm/bin/python run_deepseek.py
.venv-vlm/bin/python run_dots.py
```

Веса DeepSeek качаются как `deepseek-ai/DeepSeek-OCR-2`. dots.ocr читает локальную копию `rednote-hilab/dots.ocr` из `.cache/dots_ocr_local`. Кэш Hugging Face обоих скриптов — `.cache/hf`. Аргументом можно передать id страниц, иначе идут все шесть. Результаты: `results/deepseek/` и `results/dots/`.

## Книги целиком

Оба прогона берут DjVu и PDF из `data/chem_dataset/chem_extracted`, оставляют книги с кириллицей в имени или в тексте и пишут Markdown в `data/chem_dataset/chem_text_mineru`. Журнал один: `data/chem_dataset/chem_text_mineru/_manifest.jsonl`. Книга с успешной записью пропускается. Облако идёт по списку с начала, локальный прогон — с конца, чтобы не разбирать одну книгу дважды.

Облако ходит в `https://mineru.net`. Ключи читаются из окружения и из `.env` в корне репозитория: `MINERU_API_KEY`, `MINERU_API_KEY_2` и дальше `MINERU_API_KEY_<имя>`. Лимит запроса — 200 страниц и 200 МБ. Более толстую книгу скрипт режет на части и склеивает Markdown обратно. На первой ошибке API прогон останавливается. Повтор той же команды продолжает с журнала.

Запускать из корня репозитория окружением, где есть `pypdf` (корневой `requirements.txt`):

```bash
python ocr_research/run_mineru_cloud.py --workers 6
```

`--workers` — сколько задач держать в полёте на один ключ. Журнал хода пишут в `logs/mineru_cloud.log`. Рядом можно держать сторож: на строке `ERROR` или `STOP` он останавливает только этот процесс.

```bash
python ocr_research/stop_cloud_on_error.py
```

Один файл без корпуса:

```bash
python ocr_research/mineru_api.py path/to/file.pdf -o out.md --language east_slavic
```

Локальный корпус вызывает `.venv-mineru/bin/mineru-kit` (MinerU 4.0.12, `--tier standard`). DjVu сначала рисуется во временный PDF. Картинки `data:image` из Markdown выкидываются.

```bash
python ocr_research/run_mineru_corpus.py
```

Дальше пилот очистки Markdown — `scripts/preprocess_mineru_pilot.py` и `configs/preprocess_v001.yaml`, это уже в корневом README.
