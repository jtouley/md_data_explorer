"""An upload's data dictionary reaches the version's contract, and the layer points at it."""

from __future__ import annotations

from clinical_analytics.contracts.context import load_contract_context
from clinical_analytics.datasets.uploaded.definition import UploadedDataset

DICTIONARY = b"""subject: The participant's study number.
smoker: Ever smoked. \xe2\x80\x9cYes\xe2\x80\x9d (0.35%) if the person ever smoked.
age_band: Age in categories.
0: under 40
1: 40 or more
"""


def _csv() -> bytes:
    rows = ["subject,smoker,age_band"] + [f"S{i:03d},{'yes' if i % 3 else 'no'},{i % 2}" for i in range(120)]
    return ("\n".join(rows) + "\n").encode()


def test_saveUpload_dictionaryWithFile_contractDocumentsColumnsAndLayerReadsIt(upload_storage):
    # Arrange / Act
    ok, message, upload_id = upload_storage.save_upload(
        file_bytes=_csv(),
        original_filename="cohort.csv",
        metadata={
            "dataset_name": "cohort",
            "external_pdf_bytes": DICTIONARY,
            "external_pdf_filename": "data_dictionary.txt",
        },
    )
    assert ok is True, message
    layer = UploadedDataset(upload_id, storage=upload_storage).get_semantic_layer()
    context = load_contract_context(layer.contract_path)

    # Assert
    assert context is not None
    by_name = {c.name: c for c in context.columns}
    assert by_name["smoker"].description == "Ever smoked. “Yes” if the person ever smoked."
    assert by_name["age_band"].codebook == {"0": "under 40", "1": "40 or more"}
    assert by_name[context.key].description == "The participant's study number."
