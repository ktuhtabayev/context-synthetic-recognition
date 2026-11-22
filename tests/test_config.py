"""Typed configuration: defaults, presets, deviations from the article, formats, validation."""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.config import (
    DEVIATIONS,
    AdditionalFeaturesConfig,
    ClassMeans,
    ConfigFormat,
    ExperimentConfig,
    OperatorConfig,
    PluginSpec,
    PresetName,
    RIPSConfig,
    active_deviations,
    config_hash,
    deviation,
    dumps,
    load_config,
    loads,
    matching_preset,
    plugin,
    preset,
    save_config,
)
from context_synthetic_recognition.config.io import format_for, from_data
from context_synthetic_recognition.errors import ConfigError

from .conftest import CONFIGS, article_switches, excel_experiment

# ---------------------------------------------------------------- defaults = the article


def test_the_defaults_follow_the_article() -> None:
    config = ExperimentConfig()
    assert config.additional_features.class_means is ClassMeans.FINAL
    assert config.additional_features.majorizing_passes == 1
    assert config.admissible_k == plugin("min-class")
    assert config.rips == RIPSConfig()
    assert (config.rips.training, config.rips.keep_blocked_cover) == ("remaining", False)
    assert config.synthetic_features.zero_value is False
    assert config.synthetic_features.original_features == ()
    assert active_deviations(config) == []
    assert matching_preset(config) is PresetName.DEFAULT


def test_parameters_the_article_does_not_fix() -> None:
    additional = ExperimentConfig().additional_features
    assert (additional.alpha, additional.delta, additional.kappa, additional.cr1) == (
        0.3,
        0.1,
        5,
        10.0,
    )
    assert additional.majorizing_function == plugin("sigmoid")


def test_default_operators_are_zhuravlyov_on_all_i_and_j() -> None:
    operators = ExperimentConfig().local_context.operators
    assert [(o.label, o.metric.name, o.features, o.normalizer) for o in operators] == [
        ("ρ", "zhuravlyov", "all", None),
        ("ρ_I", "zhuravlyov", "quantitative", None),
        ("ρ_J", "zhuravlyov", "nominal", None),
    ]


def test_numeric_conventions() -> None:
    config = ExperimentConfig()
    assert config.local_context.distance_decimals == 10
    assert config.local_context.tie_break == "smaller-index"
    assert config.rips.distance_decimals == 10
    assert config.evaluation.score_decimals == 10
    assert config.evaluation.positive_class == 1


# ---------------------------------------------------------------- presets and deviations


def test_the_presets() -> None:
    default, workbook = ExperimentConfig(), excel_experiment()
    assert preset("default") == default
    assert preset("alpha-0.1").additional_features.alpha == 0.1
    assert preset("alpha-0.2").additional_features.alpha == 0.2
    for name in ("alpha-0.1", "alpha-0.2"):
        changed = preset(name).model_copy(
            update={"additional_features": default.additional_features, "name": default.name}
        )
        assert changed == default
    assert workbook.admissible_k == plugin("twice-min-class")
    assert workbook.additional_features.class_means is ClassMeans.RUNNING
    assert workbook.additional_features.majorizing_passes == 2
    assert workbook.additional_features.alpha == 0.3


def test_the_article_switches_differ_from_the_workbook_in_exactly_two_fields() -> None:
    workbook, article = excel_experiment(), article_switches()
    changed = {
        key
        for key, value in article.additional_features.model_dump().items()
        if value != workbook.additional_features.model_dump()[key]
    }
    assert changed == {"class_means", "majorizing_passes"}


def test_deviations_are_active_only_at_their_deviating_values() -> None:
    assert [d.key for d in active_deviations(excel_experiment())] == [
        "class_means",
        "majorizing_passes",
    ]
    assert active_deviations(article_switches()) == []
    only_means = ExperimentConfig(
        additional_features=AdditionalFeaturesConfig(class_means=ClassMeans.RUNNING)
    )
    assert [d.key for d in active_deviations(only_means)] == ["class_means"]
    assert matching_preset(only_means) is None
    rips = ExperimentConfig(rips=RIPSConfig(keep_blocked_cover=True, training="moved"))
    assert [d.key for d in active_deviations(rips)] == ["keep_blocked_cover", "training"]


def test_deviation_statements_are_the_agreed_wording() -> None:
    assert [d.statement for d in DEVIATIONS] == [
        (
            "in R-IPS a blocked object keeps covering other objects with |r(S)| although "
            "blocking makes r(S) negative"
        ),
        (
            "the objects R-IPS moves to D form the training sample instead of the objects that "
            "remain in E₁"
        ),
        (
            "θ and γ are measured from running partial class means "
            "instead of the final class means M₁ and M₂"
        ),
        "in Step 4 the majorizing function is applied twice instead of once",
    ]
    assert [d.adr for d in DEVIATIONS] == ["ADR-061", "ADR-061", "ADR-002", "ADR-003"]
    assert deviation("class_means").field == "additional_features.class_means"
    with pytest.raises(KeyError):
        deviation("nothing")


def test_non_method_fields_do_not_change_the_preset() -> None:
    config = preset("alpha-0.1").model_copy(update={"name": "mine", "seed": 7})
    assert matching_preset(config) is PresetName.ALPHA_01


@pytest.mark.parametrize("name", list(PresetName))
def test_shipped_config_files_equal_the_presets(name: PresetName) -> None:
    assert load_config(CONFIGS / f"{name.value}.yaml") == preset(name)


# ---------------------------------------------------------------- formats, round trip, hash


@pytest.mark.parametrize("fmt", list(ConfigFormat))
@pytest.mark.parametrize("name", list(PresetName))
def test_round_trip_in_every_format(fmt: ConfigFormat, name: PresetName) -> None:
    config = preset(name)
    assert loads(dumps(config, fmt), fmt) == config


def test_yaml_and_toml_headers_highlight_the_deviations() -> None:
    for fmt in (ConfigFormat.YAML, ConfigFormat.TOML):
        text = dumps(excel_experiment(), fmt)
        assert "# preset: " in text
        assert text.count("⚠ deviates from the article") == 2
        assert "⚠" not in dumps(article_switches(), fmt)


def test_hash_is_stable_and_format_independent() -> None:
    config = excel_experiment()
    expected = config_hash(config)
    assert len(expected) == 64
    for fmt in ConfigFormat:
        assert config_hash(loads(dumps(config, fmt), fmt)) == expected
    assert config_hash(article_switches()) != expected


@settings(max_examples=40, deadline=None)
@given(
    alpha=st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
    delta=st.floats(min_value=0.001, max_value=0.499, allow_nan=False),
    kappa=st.integers(min_value=2, max_value=50),
    passes=st.sampled_from([1, 2]),
    class_means=st.sampled_from(list(ClassMeans)),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_any_valid_setting_round_trips(
    alpha: float, delta: float, kappa: int, passes: int, class_means: ClassMeans, seed: int
) -> None:
    config = ExperimentConfig(
        additional_features=AdditionalFeaturesConfig(
            alpha=alpha, delta=delta, kappa=kappa, majorizing_passes=passes, class_means=class_means
        ),
        seed=seed,
    )
    for fmt in ConfigFormat:
        restored = loads(dumps(config, fmt), fmt)
        assert restored == config
        assert config_hash(restored) == config_hash(config)


# ---------------------------------------------------------------- validation messages


@pytest.mark.parametrize(
    ("data", "location"),
    [
        ({"additional_features": {"alpha": 1.0}}, "additional_features.alpha"),
        ({"additional_features": {"alpha": 0}}, "additional_features.alpha"),
        ({"additional_features": {"delta": 0.5}}, "additional_features.delta"),
        ({"additional_features": {"kappa": 1}}, "additional_features.kappa"),
        (
            {"additional_features": {"majorizing_passes": 3}},
            "additional_features.majorizing_passes",
        ),
        ({"additional_features": {"class_means": "partial"}}, "additional_features.class_means"),
        ({"additional_features": {"centre": "final"}}, "additional_features.centre"),
        ({"local_context": {"distance_decimals": 16}}, "local_context.distance_decimals"),
        ({"local_context": {"operators": []}}, "local_context.operators"),
        ({"seed": -1}, "seed"),
        ({"schema_version": 1}, "schema_version"),
    ],
)
def test_invalid_values_name_their_location(data: dict[str, object], location: str) -> None:
    with pytest.raises(ConfigError, match=rf"(?m)^  {location}: "):
        from_data(data, source="test")


def test_operator_labels_must_be_unique() -> None:
    with pytest.raises(ConfigError, match="repeated: ρ"):
        from_data({"local_context": {"operators": [{"label": "ρ"}, {"label": "ρ"}]}})


@pytest.mark.parametrize("features", [[], ["x1", "x1"]])
def test_explicit_feature_subsets_are_non_empty_sets(features: list[str]) -> None:
    with pytest.raises(ConfigError, match=r"local_context\.operators\.0\.features"):
        from_data({"local_context": {"operators": [{"label": "ρ", "features": features}]}})


def test_explicit_feature_subset_is_accepted() -> None:
    config = from_data({"local_context": {"operators": [{"label": "ρ", "features": ["x1", "x4"]}]}})
    assert config.local_context.operators == (OperatorConfig(label="ρ", features=("x1", "x4")),)


def test_plugin_shorthand_accepts_a_bare_name() -> None:
    config = from_data(
        {
            "local_context": {"normalizer": "z-score"},
            "additional_features": {"majorizing_function": "tanh"},
        }
    )
    assert config.local_context.normalizer == PluginSpec(name="z-score")
    assert config.additional_features.majorizing_function == plugin("tanh")


def test_empty_document_gives_the_defaults() -> None:
    assert loads("", ConfigFormat.YAML) == ExperimentConfig()


def test_top_level_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="expected a mapping"):
        loads("- 1\n- 2\n", ConfigFormat.YAML)


@pytest.mark.parametrize(
    ("fmt", "text"),
    [("yaml", "rips: [unclosed"), ("toml", "rips = "), ("json", "{")],
)
def test_syntax_errors_are_config_errors(fmt: str, text: str) -> None:
    with pytest.raises(ConfigError, match=f"not valid {fmt.upper()}"):
        loads(text, fmt)


def test_configs_are_immutable() -> None:
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.additional_features.alpha = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------- files


def test_save_and_load_by_extension(tmp_path: Path) -> None:
    for suffix in (".yaml", ".yml", ".toml", ".json"):
        path = save_config(article_switches(), tmp_path / f"exp{suffix}")
        assert load_config(path) == article_switches()


def test_save_refuses_to_overwrite_unless_asked(tmp_path: Path) -> None:
    path = save_config(excel_experiment(), tmp_path / "exp.yaml")
    with pytest.raises(ConfigError, match="already exists"):
        save_config(article_switches(), path)
    save_config(article_switches(), path, overwrite=True)
    assert load_config(path) == article_switches()


def test_unknown_extension_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown configuration extension"):
        format_for(tmp_path / "exp.ini")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
