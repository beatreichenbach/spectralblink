import logging
import os.path

from spectralblink.rgb2spec import write_model, reconstruct_pattern


def test_write_model():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_dir, 'models', 'aces2065_1.coeff')
    output_path = os.path.join(project_dir, 'output')
    write_model(model_path, output_path)


def test_reconstruct_pattern():
    project_dir = os.path.dirname(os.path.dirname(__file__))
    model_path = os.path.join(project_dir, 'models', 'srgb.coeff')
    output_path = os.path.join(project_dir, 'output', 'pattern.exr')
    reconstruct_pattern(model_path, output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, force=True)
    # test_write_model()
    test_reconstruct_pattern()
