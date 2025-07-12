## Mygpt

A learning project to understand how GPT work.

## Installation

git clone

## Usage

#### Download a model:

```
from utils import load_encoder_hparams_and_params
encoder, hparams, params = load_encoder_hparams_and_params("124M", "models")
```

#### CLI:

```
python main.py "How many seas are in the world?"
```

## Sources:
1. Часть 1. URL: https://habr.com/ru/articles/716902/
2. Часть 2. URL: https://habr.com/ru/articles/717644/

## License

[MIT](https://choosealicense.com/licenses/mit/)