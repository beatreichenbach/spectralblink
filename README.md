# Spectral Blink

## Installation

Requires [Python 3.10](https://www.python.org/downloads)

Create a virtual environment:
```shell
py -3.10 -m venv venv
```

Install the package:
```shell
pip install spectralblink@https://github.com/beatreichenbach/spectralblink/archive/refs/heads/main.zip
```

## Usage

```shell
python -m spectralblink --input "models/aces2065_1.coeff" --output "output/"
```

## Contributing

Create a virtual environment:
```shell
python -m venv venv
```

Install the development package:
```shell
python -m pip install -e .[dev]
```