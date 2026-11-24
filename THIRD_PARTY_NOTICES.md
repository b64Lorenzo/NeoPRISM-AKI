# Third-party materials

The source project used a parameter-mapping workbook (`organIQ_relevant_parameters.xlsx`), but that workbook is not included here. Parameter catalogs remain INI-configured; `config/pipeline.xml` orchestrates stages and does not embed a workbook-derived catalog. The repository maintainer confirms that the derived catalog material included here is cleared for redistribution in this repository. This does not claim or change the workbook's own license or terms, and the project's MIT license does not apply to third-party source material or copied text.

Do not add patient-level cohort CSVs or generated patient-level outputs to version control unless the data owner has explicitly approved that use. Keep them in approved storage, and verify that custom output directories are ignored or outside the repository before committing.
