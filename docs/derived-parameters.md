# Extensible derived parameters

The extractor supports **deterministic patient-local derivations**. This is distinct from statistical imputation (median/KNN/MICE), which would require fitting on training patients only. The design follows the observed-first BMI behavior documented in the [reference project](https://github.com/b64Lorenzo/NeoPRISM-AKI/blob/xml_version/docs/preoperative-parameters.md), but uses a registry to add new calculations without editing the extractor.

## Architecture

`config/parameters.ini` declares a parameter and `derive = method_name`. `src/preprocessing/parameters/config.py` validates its definition. `src/preprocessing/parameters/derivation.py` contains a registry of functions, a patient-local context, and a generic dispatcher. `extractor.py` passes the patient observations and event boundary timestamps to that dispatcher. Each function returns `DerivedValue` or `None`; `None` means insufficient evidence. The dispatcher preserves recorded values, records provenance in `value_origin` and `source_field`, and can resolve chained derivations over multiple passes.

## BMI

The formula is `BMI = weight_kg / height_m²`. If recorded BMI exists it is preserved. Otherwise the function uses a valid latest timestamped preoperative weight and height, or unambiguous untimed measurements. It never uses postoperative values. The `weight_unit` and `height_unit` options explicitly declare units (kg/g/lb and m/cm/in). Verify these against the data dictionary before clinical use.

```ini
[parameter.bmi]
phase = preoperative
source_fields = epic.DEMOGRAPHICS.BMI
type = numeric
unit = kg/m^2
timed = false
derive = bmi
weight_parameter_id = weight
height_parameter_id = height
weight_unit = kg
height_unit = cm
```

The `weight` and `height` dependencies must also be defined as preoperative numeric parameters. Derived BMI is an untimed patient feature and is marked `derived_bmi_from_weight_height`.

## Operation duration

The formula is `(operation_end - operation_start) / 60` minutes, using the **configured boundary events** rather than the first and last observations. If either boundary is unavailable, or the end does not follow the start, the parameter remains missing. Supported output units are `minutes`, `min`, `hours`, `h`, `days`, `d`. The resulting value is an untimed *intraoperative* summary and must not be used as a preoperative predictor.

```ini
[parameter.operation_duration]
phase = intraoperative
source_fields =
type = numeric
unit = minutes
timed = false
derive = operation_duration
```

The value's provenance is `derived_operation_duration`. The operation boundaries may represent anesthesia/OR markers rather than incision-to-closure duration; label and interpret accordingly.

## Add a new derivation

1. Add a function to `src/preprocessing/parameters/derivation.py` using `@register("my_method")`.
2. Give it the signature `(context: DerivationContext, parameter: Parameter) -> DerivedValue | None`.
3. Read only the patient-local `context.rows` and `context.boundaries`. Return `None` when required inputs are missing; reject invalid units and nonfinite results.
4. Add `[parameter.my_feature]` with `derive = my_method`, its phase, type, units, and any method-specific options.
5. Add tests for observed-value precedence, missing inputs, unit conversion, phase correctness and provenance.

Example:

```python
@register("minutes_since_start")
def minutes_since_start(context, parameter):
    start = context.boundaries.get("operation_start")
    end = context.boundaries.get("operation_end")
    if start is None or end is None or end <= start:
        return None
    return DerivedValue((end - start) / 60, "derived:boundaries", "derived_minutes_since_start")
```

Generic INI options are available as `dict(parameter.derive_options)`. Registering the function is sufficient to recognize its `derive` name; add method-specific validation in `config.py` when necessary. Do not import untrusted derivation names dynamically or execute expressions from INI files.

## Data provenance and safeguards

Extracted CSVs retain `time_minutes,parameter_id,value,unit,type,source_field,value_origin`. `observed` denotes recorded values; derivations receive specific origin labels. Untimed derived features have blank `time_minutes`, which does **not** mean zero minutes. Patient-level derivation is not a substitute for clinical validation. Statistical imputation should occur later, after train/validation splitting, with fitted statistics learned only on training data.
