import argparse
import logging
import sys

from spectralblink.rgb2spec import write_model


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog='Spectral Blink',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--input',
        type=str,
        help='path to model in .coeffs format',
    )
    parser.add_argument(
        '--output',
        type=str,
        help='path to output dir',
    )
    parser.add_argument(
        '--log',
        type=str,
        default='INFO',
        help='logging level',
    )
    return parser


def main() -> None:
    parser = argument_parser()
    args = parser.parse_args(sys.argv[1:])
    logging.basicConfig(level=args.log, force=True)

    write_model(args.input, args.output)


if __name__ == '__main__':
    main()
