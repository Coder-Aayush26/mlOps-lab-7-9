# Validation checklists (A4, B5, C7) with evidence

Status key: **VERIFIED** = executed while this project was generated (outputs in `evidence/`);
**RUN ON YOUR MACHINE** = needs Docker, which was not available in the generation environment (one command produces the evidence).

## A4. Pipeline validation
| Check | Expected result | Evidence | Status |
|---|---|---|---|
| Run from a terminal | `python -m src.train` completes | `evidence/local/train_log.txt` | VERIFIED |
| Import has no side effects | importing modules starts no training and writes no files | `tests/test_data.py::test_importing_modules_does_not_start_training` | VERIFIED |
| No test-data leakage | scaler statistics equal TRAIN-split statistics; train/test rows disjoint | `tests/test_pipeline.py::test_test_data_not_used_to_fit_preprocessing` | VERIFIED |
| Load model in a separate process | `python -m src.predict` loads `models/model.joblib` in a new interpreter | `tests/test_pipeline.py::test_model_loads_and_predicts_in_a_separate_python_process`, `evidence/local/predict_5_records.txt` | VERIFIED |
| >= 5 predictions | 5 valid predictions printed | `evidence/local/predict_5_records.txt` | VERIFIED |
| >= 2 tests | all tests pass | `evidence/local/pytest_verbose.txt` (28 passed) | VERIFIED |
| Data loading | expected file and columns | `test_load_data_missing_file_raises`, `test_validate_schema_reports_missing_columns` | VERIFIED |
| Preprocessing | duplicates, invalid, bad formats handled | `test_preprocess_cleans_duplicates_invalid_and_bad_formats` | VERIFIED |
| Training | model saved | `models/model.joblib`, `models/model_metadata.json` | VERIFIED |
| Evaluation | metrics on held-out data | `reports/metrics.json`, `reports/evaluation_report.md` (accuracy 0.9561, ROC-AUC 0.9917) | VERIFIED |
| Prediction | valid output for samples | `test_predict_output_shape_and_values` | VERIFIED |

## B5. API validation
| Test | Expected | Observed (live uvicorn run, `evidence/api/curl_results.txt`) | Status |
|---|---|---|---|
| Valid input | prediction returned | HTTP 200 `{"prediction":1,"label":"benign","probability":0.778...}` | VERIFIED |
| Missing field | validation error | HTTP 422, one `missing` error per absent field | VERIFIED |
| Wrong type | validation error | HTTP 422 `float_parsing` for `mean_radius="large"` | VERIFIED |
| Invalid business value | controlled error | HTTP 422 `greater_than` (radius -5) and `less_than_equal` (smoothness 1.7) | VERIFIED |
| Health check | healthy | HTTP 200 `{"status":"ok","model_loaded":true}` | VERIFIED |
| Internal error | no internals leaked | HTTP 500 `{"detail":"Internal server error."}` (`test_internal_errors_do_not_leak_details`) | VERIFIED |
| Model missing | controlled error | HTTP 503, `/health` still 200 (`test_model_missing_gives_503_not_crash`) | VERIFIED |
| Swagger UI screenshots / Postman | screenshots | open http://127.0.0.1:8000/docs and take screenshots into `evidence/api/screenshots/` | RUN ON YOUR MACHINE |

## C7. Container validation
| Check | Command | Status |
|---|---|---|
| 14 Build image | `docker build -t ml-inference-api:1.0 .` | RUN ON YOUR MACHINE |
| 15 Run + `/health` | `docker run -d --name ml-api -p 8000:8000 ml-inference-api:1.0` | RUN ON YOUR MACHINE |
| 16 Prediction from host | `scripts/api_demo.sh http://localhost:8000` | RUN ON YOUR MACHINE |
| 17 Logs | `docker logs ml-api` | RUN ON YOUR MACHINE |
| 18 Remove/recreate | `docker rm -f ml-api && docker run ...` | RUN ON YOUR MACHINE |
| 19 Compose connectivity + persistence | `scripts/compose_validate.sh` | RUN ON YOUR MACHINE |
| Local vs container | `python scripts/compare_predictions.py --url http://localhost:8000 --label container` | RUN ON YOUR MACHINE (local vs *local API* already VERIFIED: 10/10 identical, `evidence/api/local_vs_api_local.md`) |

`scripts/docker_validate.sh` runs all of C7 + the Part D comparison and writes `evidence/docker/docker_validation.txt`.
