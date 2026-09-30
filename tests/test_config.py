"""Typed configuration: defaults, presets, template deviations, formats, hash, validation."""

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from context_synthetic_recognition.config import (
    DEVIATIONS,
    CentreMode,
    ConfigFormat,
    ExperimentConfig,
    HAGConfig,
    OperatorConfig,
    PluginSpec,
    PresetName,
    active_deviations,
    config_hash,
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

from .conftest import CONFIGS

# ---------------------------------------------------------------- defaults = template (ADR-004)


def test_defaults_are_the_template_setting() -> None:
    config = ExperimentConfig()
    assert config == preset("template")
    assert config.hag.centres is CentreMode.RUNNING
    assert config.hag.step4_passes == 2
    assert matching_preset(config) is PresetName.TEMPLATE


def test_template_parameters_of_the_experiment() -> None:
    hag = ExperimentConfig().hag
    assert (hag.alpha, hag.delta, hag.kappa, hag.cr1) == (0.3, 0.1, 5, 10.0)
    assert hag.majorizer == plugin("sigmoid")


def test_default_operators_are_zhuravlyov_on_all_i_and_j() -> None:
    operators = ExperimentConfig().context.operators
    assert [(o.label, o.metric.name, o.features) for o in operators] == [
        ("ρ", "zhuravlyov", "all"),
        ("ρ_I", "zhuravlyov", "quantitative"),
        ("ρ_J", "zhuravlyov", "nominal"),
    ]


def test_numeric_conventions() -> None:
    config = ExperimentConfig()
    assert config.context.distance_decimals == 10
    assert config.context.tie_break == "smaller-index"
    assert config.evaluation.score_decimals == 10
    assert config.evaluation.positive_class == 1
    assert config.k.name == "formula"


# ---------------------------------------------------------------- presets and deviations


def test_article_preset_flips_exactly_the_two_switches() -> None:
    template, article = preset("template"), preset("article")
    assert article.hag.centres is CentreMode.FINAL
    assert article.hag.step4_passes == 1
    changed = {
        key
        for key, value in article.hag.model_dump().items()
        if value != template.hag.model_dump()[key]
    }
    assert changed == {"centres", "step4_passes"}
    assert article.model_copy(update={"hag": template.hag, "name": template.name}) == template


def test_deviations_are_active_only_at_template_values() -> None:
    assert [d.key for d in active_deviations(preset("template"))] == ["centres", "step4_passes"]
    assert active_deviations(preset("article")) == []
    only_centres = ExperimentConfig(hag=HAGConfig(step4_passes=1))
    assert [d.key for d in active_deviations(only_centres)] == ["centres"]
    assert matching_preset(only_centres) is None


def test_deviation_statements_are_the_agreed_wording() -> None:
    statements = [d.statement for d in DEVIATIONS]
    assert statements == [
        (
            "θ and γ are measured from running partial class means "
            "instead of the final class means M₁ and M₂"
        ),
        "in STEP 4 the majorizer is applied twice instead of once",
    ]
    assert [d.adr for d in DEVIATIONS] == ["ADR-002", "ADR-003"]


def test_non_method_fields_do_not_change_the_preset() -> None:
    config = preset("article").model_copy(update={"name": "mine", "seed": 7})
    assert matching_preset(config) is PresetName.ARTICLE


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
        text = dumps(preset("template"), fmt)
        assert "# preset: template" in text
        assert text.count("⚠ template calculation") == 2
        assert "⚠" not in dumps(preset("article"), fmt)


def test_hash_is_stable_and_format_independent() -> None:
    config = preset("template")
    expected = config_hash(config)
    assert len(expected) == 64
    for fmt in ConfigFormat:
        assert config_hash(loads(dumps(config, fmt), fmt)) == expected
    assert config_hash(preset("article")) != expected


@settings(max_examples=40, deadline=None)
@given(
    alpha=st.floats(min_value=0.001, max_value=0.999, allow_nan=False),
    delta=st.floats(min_value=0.001, max_value=0.499, allow_nan=False),
    kappa=st.integers(min_value=2, max_value=50),
    passes=st.sampled_from([1, 2]),
    centres=st.sampled_from(list(CentreMode)),
    seed=st.integers(min_value=0, max_value=2**31),
)
def test_any_valid_setting_round_trips(
    alpha: float, delta: float, kappa: int, passes: int, centres: CentreMode, seed: int
) -> None:
    config = ExperimentConfig(
        hag=HAGConfig(alpha=alpha, delta=delta, kappa=kappa, step4_passes=passes, centres=centres),
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
        ({"hag": {"alpha": 1.0}}, "hag.alpha"),
        ({"hag": {"alpha": 0}}, "hag.alpha"),
        ({"hag": {"delta": 0.5}}, "hag.delta"),
        ({"hag": {"kappa": 1}}, "hag.kappa"),
        ({"hag": {"step4_passes": 3}}, "hag.step4_passes"),
        ({"hag": {"centres": "partial"}}, "hag.centres"),
        ({"hag": {"centre": "final"}}, "hag.centre"),
        ({"context": {"distance_decimals": 16}}, "context.distance_decimals"),
        ({"context": {"operators": []}}, "context.operators"),
        ({"seed": -1}, "seed"),
        ({"schema_version": 2}, "schema_version"),
    ],
)
def test_invalid_values_name_their_location(data: dict[str, object], location: str) -> None:
    with pytest.raises(ConfigError, match=rf"(?m)^  {location}: "):
        from_data(data, source="test")


def test_operator_labels_must_be_unique() -> None:
    with pytest.raises(ConfigError, match="repeated: ρ"):
        from_data({"context": {"operators": [{"label": "ρ"}, {"label": "ρ"}]}})


@pytest.mark.parametrize("features", [[], ["x1", "x1"]])
def test_explicit_feature_subsets_are_non_empty_sets(features: list[str]) -> None:
    with pytest.raises(ConfigError, match=r"context\.operators\.0\.features"):
        from_data({"context": {"operators": [{"label": "ρ", "features": features}]}})


def test_explicit_feature_subset_is_accepted() -> None:
    config = from_data({"context": {"operators": [{"label": "ρ", "features": ["x1", "x4"]}]}})
    assert config.context.operators == (OperatorConfig(label="ρ", features=("x1", "x4")),)


def test_plugin_shorthand_accepts_a_bare_name() -> None:
    config = from_data({"preprocessing": {"normalizer": "z-score"}, "hag": {"majorizer": "tanh"}})
    assert config.preprocessing.normalizer == PluginSpec(name="z-score")
    assert config.hag.majorizer == plugin("tanh")


def test_empty_document_gives_the_defaults() -> None:
    assert loads("", ConfigFormat.YAML) == ExperimentConfig()


def test_top_level_must_be_a_mapping() -> None:
    with pytest.raises(ConfigError, match="expected a mapping"):
        loads("- 1\n- 2\n", ConfigFormat.YAML)


@pytest.mark.parametrize(
    ("fmt", "text"),
    [("yaml", "hag: [unclosed"), ("toml", "hag = "), ("json", "{")],
)
def test_syntax_errors_are_config_errors(fmt: str, text: str) -> None:
    with pytest.raises(ConfigError, match=f"not valid {fmt.upper()}"):
        loads(text, fmt)


def test_configs_are_immutable() -> None:
    config = ExperimentConfig()
    with pytest.raises(ValueError, match="frozen"):
        config.hag.alpha = 0.5  # type: ignore[misc]


# ---------------------------------------------------------------- files


def test_save_and_load_by_extension(tmp_path: Path) -> None:
    for suffix in (".yaml", ".yml", ".toml", ".json"):
        path = save_config(preset("article"), tmp_path / f"exp{suffix}")
        assert load_config(path) == preset("article")


def test_save_refuses_to_overwrite_unless_asked(tmp_path: Path) -> None:
    path = save_config(preset("template"), tmp_path / "exp.yaml")
    with pytest.raises(ConfigError, match="already exists"):
        save_config(preset("article"), path)
    save_config(preset("article"), path, overwrite=True)
    assert load_config(path) == preset("article")


def test_unknown_extension_and_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="unknown configuration extension"):
        format_for(tmp_path / "exp.ini")
    with pytest.raises(ConfigError, match="cannot read"):
        load_config(tmp_path / "missing.yaml")
