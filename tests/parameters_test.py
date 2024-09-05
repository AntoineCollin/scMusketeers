import pytest

given = pytest.mark.parametrize


@given("class_key,key", [("celltype", "celltype")])
def test_param(class_key, key):
    print("Run test")
    # atlas_seurat.check_seurat_install()
    assert class_key == key
