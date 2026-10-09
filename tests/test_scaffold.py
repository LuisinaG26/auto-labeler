import argparse

import pytest

from autolabel.cli import run_init
from autolabel.config import load_project
from autolabel.scaffold import init_project, project_template


def test_generated_project_loads_with_the_strict_schema(tmp_path):
    target = init_project(
        tmp_path / "mi proyecto", ["casco", "chaleco reflectivo"], scene="obra de construcción"
    )
    cfg = load_project(target.parent)
    assert cfg.class_names == ("casco", "chaleco reflectivo")
    assert cfg.name == "mi-proyecto"
    assert cfg.scene == "obra de construcción"
    assert cfg.quality.blur_min is None                       # los cortes se calibran, no vienen fijos
    strategies = [s.model_id.rsplit("-", 1)[-1] for s in cfg.strategies()]
    assert strategies == ["tiny"]                              # una sola ronda por defecto
    assert (target.parent / "images").is_dir()


def test_it_refuses_to_overwrite_unless_forced(tmp_path):
    init_project(tmp_path / "p", ["a"])
    with pytest.raises(FileExistsError, match="--force"):
        init_project(tmp_path / "p", ["b"])
    init_project(tmp_path / "p", ["b"], force=True)
    assert load_project(tmp_path / "p").class_names == ("b",)


def test_class_names_with_quotes_and_accents_stay_valid_yaml(tmp_path):
    names = ['rueda "delantera"', "señal: stop", "a#b"]
    target = init_project(tmp_path / "p", names)
    assert load_project(target.parent).class_names == tuple(names)


@pytest.mark.parametrize("classes", [[], ["a", "a"]])
def test_invalid_class_lists_are_rejected(classes):
    with pytest.raises(ValueError):
        project_template("x", classes, "./images", "")


def test_a_custom_images_folder_is_not_created_inside_the_project(tmp_path):
    target = init_project(tmp_path / "p", ["a"], images="/data/fotos")
    assert not (target.parent / "images").exists()
    assert load_project(target.parent).images == "/data/fotos"


def test_the_command_reports_errors_without_a_traceback(tmp_path):
    args = argparse.Namespace(
        folder=str(tmp_path / "p"), classes="a, ,b", images="./images", scene="", force=False
    )
    run_init(args)
    assert load_project(tmp_path / "p").class_names == ("a", "b")
    with pytest.raises(SystemExit, match="ya existe"):
        run_init(args)
