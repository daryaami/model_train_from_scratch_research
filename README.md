# research_chemistry

Небольшая decoder-only модель на русском и подготовка химического корпуса: книги в DjVu и PDF превращаются в текст, на котором эту модель можно учить.

Обучение живёт в `lm/`. Сравнение OCR и прогон MinerU по книгам — в [`ocr_research/README.md`](ocr_research/README.md).

## Каталоги

| Путь | Зачем |
|---|---|
| `lm/` | Сеть, цикл обучения, генерация, RuBLiMP и LanguageTool |
| `configs/` | Размер сети, корпус и расписание. `baseline.yaml` — старт |
| `scripts/` | Сборка корпусов, токенный поток, черновой DjVu, пилот предобработки |
| `notebooks/` | Разбор FineWeb, обучение BPE, пилот на 10M |
| `tokenizers/fineweb2_rus_v001/` | BPE 8k, 16k и 32k. В обучении стоит 16k из `chosen/` |
| `data/` | Корпуса и химические книги. В git каталог пустой |
| `ocr_research/` | Выборка страниц, движки OCR, облачный и локальный MinerU |
| `checkpoints/`, `logs/` | Прогоны обучения и журнал облачного MinerU |

## Окружение

Python 3.12. Torch ставится с колёсами CUDA 12.8, индекс уже прописан в `requirements.txt`.

```bash
conda env create -f environment.yml
conda activate research_chemistry
```

`environment.yml` ставит зависимости из `requirements.txt`. Если окружение уже есть, обновление зависимостей:

```bash
conda activate research_chemistry
pip install -r requirements.txt
```

`lm.russian_eval` ходит в LanguageTool и хочет Java. JRE лежит в `.tools/jdk-17.0.21+9-jre`, если системной Java нет.

## Корпуса

FineWeb-2, русский, срез `data/fineweb2/rus_Cyrl/v001`:

```bash
python scripts/build_fineweb2_rus.py
```

Cultura-Ru-Edu того же объёма, `data/cultura_ru_edu/v001`:

```bash
python scripts/build_cultura_ru_edu.py
```

Оба скрипта качают parquet с Hugging Face и пропускают уже записанные части. `--refresh` стирает срез и начинает заново.

Токенный поток 98/1/1, BPE 16k. Готовый манифест не переписывается без `--refresh`.

```bash
python scripts/prepare_dataset.py
python scripts/prepare_dataset.py --corpus data/cultura_ru_edu/v001
```

На выходе `dataset/tokens/{train,validation,test}.bin` и `manifest.json`. Хеши частей FineWeb сверяются с `tokenizers/fineweb2_rus_v001/chosen/choice.json`.

## Обучение

Короткий шаг без длинного прогона:

```bash
python -m lm.train --check
```

Полный проход. Каждый запуск создаёт свою папку в `checkpoints/` и чужие прогоны не трогает.

```bash
python -m lm.train --config configs/baseline.yaml
python -m lm.train --config configs/50m.yaml
python -m lm.train --config configs/10m_cultura.yaml
python -m lm.train --resume checkpoints/<прогон>/last.pt
```

`configs/baseline.yaml` — около 10.9M, окно 512, FineWeb. `configs/50m.yaml` — 30 слоёв и ширина 320 на том же корпусе. `configs/10m_cultura.yaml` — та же сеть, что baseline, корпус Cultura. `configs/50m_long.yaml` — та же форма, что 50M, окно 2048.

Оценка и генерация:

```bash
python -m lm.evaluate --config configs/baseline.yaml --checkpoint checkpoints/<прогон>/best.pt --full
python -m lm.generate --checkpoint checkpoints/<прогон>/best.pt --prompt "Однажды в"
python -m lm.russian_eval --checkpoint checkpoints/<прогон>/best.pt
```

Сводка Cultura против FineWeb собирается так. Два числа loss — это `lm.evaluate --full` каждой модели на validation чужого корпуса.

```bash
python scripts/compare_runs.py \
  --cultura-run checkpoints/<cultura> \
  --fineweb-run checkpoints/<fineweb> \
  --cultura-dataset data/cultura_ru_edu/v001/dataset/manifest.json \
  --fineweb-dataset data/fineweb2/rus_Cyrl/v001/dataset/manifest.json \
  --cultura-on-fineweb-val <loss> \
  --fineweb-on-cultura-val <loss>
```

## Химический корпус

Книги лежат в `data/chem_dataset/chem_extracted` (DjVu и PDF). Черновой текст, сначала текстовый слой, потом Tesseract `rus+eng`:

```bash
python scripts/djvu_to_text.py
```

Скрипт ждёт бинарники DjVuLibre в `.tools/djvulibre/usr/bin` и Tesseract в `.tools/tesseract`. Готовые `.txt` пропускает.

Для обучения этот черновик слабый на формулах. Сравнение движков и выбранный прогон MinerU описаны в [`ocr_research/README.md`](ocr_research/README.md). Markdown книг пишется в `data/chem_dataset/chem_text_mineru`. Пилот очистки этого Markdown:

```bash
python scripts/preprocess_mineru_pilot.py
python -m unittest scripts/test_preprocess_mineru_pilot.py
```

Правила пилота — `configs/preprocess_v001.yaml`. Исходный Markdown MinerU скрипт не переписывает.
