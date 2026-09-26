# Efficient Models — ITMO

Материалы курса Efficient NN.

## Локальное окружение

Для первичного создания окружения:

```bash
conda env create -f environment.yml
conda activate efficient-models-itmo
poetry install
```

После повторного открытия терминала достаточно активировать окружение:

```bash
conda activate efficient-models-itmo
```

Conda управляет версией Python и самим Poetry. Poetry устанавливает и фиксирует
зависимости проекта в `poetry.lock`. Дополнительное вложенное окружение Poetry
не создаётся.
Ширшов Дмитрий - М4155
