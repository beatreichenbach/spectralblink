import dataclasses
import logging
import math
import os
import struct

import numpy as np

from spectralblink import color

os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
import cv2


RGB2SPEC_N_COEFFS = 3


@dataclasses.dataclass
class RGB2Spec:
    res: int
    scale: np.ndarray
    data: np.ndarray


def rgb2spec_find_interval(values, size, x):
    left = 0
    last_interval = size - 2
    size = last_interval

    while size > 0:
        half = size >> 1
        middle = left + half + 1

        if values[middle] < x:
            left = middle
            size -= half + 1
        else:
            size = half

    return min(left, last_interval)


def rgb2spec_fetch(model, rgb):
    assert all(0.0 <= c <= 1.0 for c in rgb)

    i = np.argmax(rgb)
    res = model.res
    logging.debug(f'{res=}')

    z = rgb[i]
    scale = (res - 1) / z if z > 0 else 0
    logging.debug(f'{scale=}')

    x = rgb[(i + 1) % 3] * scale
    y = rgb[(i + 2) % 3] * scale
    logging.debug(f'{x=}')
    logging.debug(f'{y=}')

    xi = min(int(x), res - 2)
    yi = min(int(y), res - 2)
    zi = rgb2spec_find_interval(model.scale, res, z)
    offset = (((i * res + zi) * res + yi) * res + xi) * RGB2SPEC_N_COEFFS
    logging.debug(f'{offset=}')
    dx = RGB2SPEC_N_COEFFS
    dy = RGB2SPEC_N_COEFFS * res
    dz = RGB2SPEC_N_COEFFS * res * res

    x1 = x - xi
    x0 = 1.0 - x1
    y1 = y - yi
    y0 = 1.0 - y1
    z1 = (z - model.scale[zi]) / (model.scale[zi + 1] - model.scale[zi])
    z0 = 1.0 - z1

    out = np.zeros(RGB2SPEC_N_COEFFS)
    for j in range(RGB2SPEC_N_COEFFS):
        out[j] = (
            (model.data[offset] * x0 + model.data[offset + dx] * x1) * y0
            + (model.data[offset + dy] * x0 + model.data[offset + dy + dx] * x1) * y1
        ) * z0 + (
            (model.data[offset + dz] * x0 + model.data[offset + dz + dx] * x1) * y0
            + (
                model.data[offset + dz + dy] * x0
                + model.data[offset + dz + dy + dx] * x1
            )
            * y1
        ) * z1
        offset += 1

    return out


def rgb2spec_fma(a, b, c):
    return a * b + c


def rgb2spec_eval_precise(coeff, lambda_):
    x = rgb2spec_fma(rgb2spec_fma(coeff[0], lambda_, coeff[1]), lambda_, coeff[2])
    y = 1.0 / math.sqrt(rgb2spec_fma(x, x, 1.0))
    return rgb2spec_fma(0.5 * x, y, 0.5)


def rgb2spec_eval_fast(coeff, lambda_):
    x = rgb2spec_fma(rgb2spec_fma(coeff[0], lambda_, coeff[1]), lambda_, coeff[2])
    y = 1.0 / math.sqrt(rgb2spec_fma(x, x, 1.0))  # Using math.sqrt for simplicity
    return rgb2spec_fma(0.5 * x, y, 0.5)


def reconstruct(array: np.ndarray, model: RGB2Spec) -> np.ndarray:
    lambda_min = 360
    lambda_max = 830
    lambda_count = 10
    standard_illuminant = 'D65'
    cmfs_variation = 'CIE 2015 2 Degree Standard Observer'

    whitepoint = color.get_whitepoint(standard_illuminant=standard_illuminant)
    logging.info(f'{whitepoint=}')

    lambdas = np.linspace(lambda_min, lambda_max, lambda_count)
    logging.info(f'{lambdas=}')

    cmfs = color.get_cmfs(cmfs_variation, lambdas)
    logging.info(f'{cmfs=}')

    illuminant = color.get_illuminant(standard_illuminant, lambdas)
    logging.info(f'{illuminant=}')

    xyz_table = cmfs * illuminant[:, np.newaxis]
    xyz_table /= np.sum(xyz_table, axis=0)

    reconstructed = np.zeros(array.shape)
    k = np.sum(cmfs * illuminant[:, np.newaxis], axis=0)
    for y in range(array.shape[0]):
        for x in range(array.shape[1]):
            rgb = array[y, x]

            coefficients = rgb2spec_fetch(model, rgb)
            sd = np.array([rgb2spec_eval_precise(coefficients, w) for w in lambdas])

            rgb = np.dot(sd * illuminant, cmfs) / k
            reconstructed[y, x] = rgb
    return reconstructed


def read_model(model_path: str) -> RGB2Spec | None:
    with open(model_path, 'rb') as file:
        header = file.read(4)
        if header != b"SPEC":
            return None

        logging.info(f'Loading model: {model_path}')
        res = struct.unpack('I', file.read(4))[0]

        size_scale = res
        size_data = res * res * res * 3 * RGB2SPEC_N_COEFFS

        scale = np.fromfile(file, dtype=np.float32, count=size_scale)
        data = np.fromfile(file, dtype=np.float32, count=size_data)

    if scale.size != size_scale or data.size != size_data:
        return None

    return RGB2Spec(res, scale, data)


def write_model(model_path: str, output_path: str) -> None:
    model = read_model(model_path)

    if model is None:
        logging.error(f'Could not read model: {model_path}')
        return

    filename = os.path.basename(model_path)
    base, ext = os.path.splitext(filename)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model_output_path = os.path.join(os.path.abspath(output_path), f'{base}.model.exr')
    scale_output_path = os.path.join(os.path.abspath(output_path), f'{base}.scale.exr')

    resolution = int(np.sqrt(model.res**3 * 3 * RGB2SPEC_N_COEFFS))
    array = model.data.reshape((resolution, resolution, 1))
    cv2.imwrite(model_output_path, array)
    logging.info(f'Written image: {model_output_path}')

    array = model.scale.reshape((model.res, 1))
    cv2.imwrite(scale_output_path, array)
    logging.info(f'Written image: {scale_output_path}')


def test_pattern(resolution: int) -> np.ndarray:
    space = np.linspace(0, 1, resolution)
    pattern_3d = np.stack(np.meshgrid(space, space, space), axis=3)

    shape = (resolution, resolution**2, 3)
    pattern_2d = pattern_3d.flatten()
    pattern_2d.resize(shape, refcheck=False)
    return pattern_2d


def reconstruct_pattern(model_path: str, output: str, resolution: int = 8) -> None:
    pattern = test_pattern(resolution)
    model = read_model(model_path)

    reconstructed = reconstruct(pattern, model)
    diff = np.abs(pattern - reconstructed)
    array = np.vstack((pattern, reconstructed, diff))

    array = array.astype(np.float32)
    image_bgr = cv2.cvtColor(array, cv2.COLOR_RGBA2BGR)
    cv2.imwrite(output, image_bgr)
    logging.info(f'Written image: {output}')
