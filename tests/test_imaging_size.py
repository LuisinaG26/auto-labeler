from PIL import Image

from autolabel.imaging import image_size


def save(path, size, orientation):
    exif = Image.Exif()
    exif[274] = orientation
    Image.new("RGB", size).save(path, exif=exif)


def test_image_size_applies_orientation_without_decoding(tmp_path):
    save(tmp_path / "rot.jpg", (400, 300), 6)
    save(tmp_path / "plain.jpg", (400, 300), 1)
    assert image_size(tmp_path / "rot.jpg") == (300, 400)
    assert image_size(tmp_path / "plain.jpg") == (400, 300)
