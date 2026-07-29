import pathlib

from src.research import colab_paths


def test_drive_not_mounted_on_this_machine():
    # bu depo Colab degil - Drive'in ebeveyni yerel makinede yok
    assert colab_paths.drive_mounted() is False


def test_in_colab_false_on_this_machine():
    assert colab_paths.in_colab() is False


def test_research_root_falls_back_to_local_when_drive_not_mounted():
    root = colab_paths.research_root()
    assert root == pathlib.Path("artifacts/research")
    assert root.exists()


def test_local_scratch_root_never_under_drive():
    root = colab_paths.local_scratch_root()
    assert "drive" not in str(root).lower() or "MyDrive" not in str(root)
    assert str(colab_paths.DRIVE_ROOT) not in str(root)


def test_dataset_root_is_subdir_of_research_root():
    root = colab_paths.dataset_root("auair")
    assert root == colab_paths.research_root() / "datasets" / "auair"


def test_drive_root_is_single_source_of_truth_path():
    assert colab_paths.DRIVE_ROOT.as_posix() == "/content/drive/MyDrive/VidEmbedd/phase6_mrl_vector_backend"


def test_embeddings_and_checkpoints_and_results_roots_are_subdirs():
    assert colab_paths.embeddings_root() == colab_paths.research_root() / "embeddings"
    assert colab_paths.checkpoints_root() == colab_paths.research_root() / "checkpoints"
    assert colab_paths.results_root() == colab_paths.research_root() / "results"
