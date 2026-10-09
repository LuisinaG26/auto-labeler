from PIL import Image

from autolabel.imaging import load_image


def save_with_orientation(path, size, orientation):
    img = Image.new("RGB", size, (200, 30, 30))
    exif = Image.Exif()
    exif[274] = orientation
    img.save(path, exif=exif)


def test_exif_rotation_is_applied(tmp_path):
    p = tmp_path / "a.jpg"
    save_with_orientation(p, (400, 300), 6)      # guardada en horizontal, se muestra en vertical
    assert load_image(p).size == (300, 400)


def test_without_orientation_the_size_is_unchanged(tmp_path):
    p = tmp_path / "b.jpg"
    save_with_orientation(p, (400, 300), 1)
    assert load_image(p).size == (400, 300)


def test_max_side_keeps_aspect_and_orientation(tmp_path):
    p = tmp_path / "c.jpg"
    save_with_orientation(p, (800, 600), 6)
    w, h = load_image(p, max_side=200).size
    assert (w, h) == (150, 200)


def test_small_images_are_not_upscaled(tmp_path):
    p = tmp_path / "d.png"
    Image.new("RGB", (100, 50)).save(p)
    assert load_image(p, max_side=500).size == (100, 50)
