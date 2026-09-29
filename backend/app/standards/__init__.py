"""The verified Indian Standards dataset: schema, loading, text and indexing.

The package keeps one responsibility per module:

* :mod:`app.standards.dataset_schema`  - file schema + strict validation
* :mod:`app.standards.dataset_loader`  - resolve path, read JSON, map to entities
* :mod:`app.standards.documents`       - record -> searchable text / metadata
* :mod:`app.standards.indexing_service`- dataset -> chunks -> embeddings -> Chroma

No module here invents standards data, and none of them talks HTTP.
"""

from app.standards.dataset_loader import (
    StandardsDatasetLoader,
    get_standards_dataset_loader,
    to_standard_entities,
    to_standard_entity,
)
from app.standards.dataset_schema import (
    StandardDatasetRecord,
    StandardsDataset,
    StandardsDatasetMetadata,
    designation_key,
    designation_year,
    validate_standards_dataset,
)
from app.standards.documents import (
    StandardDocument,
    build_standard_document,
    build_standard_documents,
    standard_chunk_metadata,
    standard_source_id,
    standard_to_searchable_text,
)
from app.standards.indexing_service import (
    STATUS_DATASET_INVALID,
    STATUS_DATASET_NOT_AVAILABLE,
    STATUS_EMBEDDING_NOT_CONFIGURED,
    STATUS_INDEXED,
    STATUS_VECTOR_STORE_UNAVAILABLE,
    IndexingReport,
    StandardsIndexingService,
    get_standards_indexing_service,
)

__all__ = [
    "STATUS_DATASET_INVALID",
    "STATUS_DATASET_NOT_AVAILABLE",
    "STATUS_EMBEDDING_NOT_CONFIGURED",
    "STATUS_INDEXED",
    "STATUS_VECTOR_STORE_UNAVAILABLE",
    "IndexingReport",
    "StandardDatasetRecord",
    "StandardDocument",
    "StandardsDataset",
    "StandardsDatasetLoader",
    "StandardsDatasetMetadata",
    "StandardsIndexingService",
    "build_standard_document",
    "build_standard_documents",
    "designation_key",
    "designation_year",
    "get_standards_dataset_loader",
    "get_standards_indexing_service",
    "standard_chunk_metadata",
    "standard_source_id",
    "standard_to_searchable_text",
    "to_standard_entities",
    "to_standard_entity",
    "validate_standards_dataset",
]
