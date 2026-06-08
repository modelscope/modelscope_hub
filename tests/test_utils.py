"""Unit tests for modelscope_hub.utils modules."""

from __future__ import annotations

import time
import zoneinfo
from datetime import datetime, timezone

import pytest

from modelscope_hub.utils.format import format_size, format_timesince, tabulate
from modelscope_hub.utils.media import encode_media_to_base64
from modelscope_hub.utils.patterns import extract_common_prefix, normalize_patterns
from modelscope_hub.utils.time_utils import parse_timestamp


class TestFormatSize:
    def test_zero(self):
        assert format_size(0) == "0 B"

    def test_iec_integer(self):
        assert format_size(1024) == "1 KiB"

    def test_iec_fractional(self):
        assert format_size(1536) == "1.5 KiB"

    def test_iec_gib(self):
        assert format_size(1073741824) == "1 GiB"

    def test_si_integer(self):
        assert format_size(1000, unit_system="si") == "1 KB"

    def test_si_fractional(self):
        assert format_size(1500, unit_system="si") == "1.5 KB"

    def test_very_large_value(self):
        # PiB-level value should not crash
        result = format_size(2 * 1024**5)
        assert "PiB" in result

    def test_invalid_unit_system(self):
        with pytest.raises(ValueError):
            format_size(100, unit_system="unknown")


class TestFormatTimesince:
    def test_few_seconds_ago(self):
        ts = time.time() - 5
        assert format_timesince(ts) == "a few seconds ago"

    def test_one_minute_ago(self):
        ts = time.time() - 60
        assert format_timesince(ts) == "1 minute ago"

    def test_one_hour_ago(self):
        ts = time.time() - 3600
        assert format_timesince(ts) == "1 hour ago"

    def test_plural_hours(self):
        ts = time.time() - 7200
        assert format_timesince(ts) == "2 hours ago"

    def test_boundary_few_seconds(self):
        # Exactly at boundary (19 seconds)
        ts = time.time() - 19
        assert format_timesince(ts) == "a few seconds ago"


class TestTabulate:
    def test_basic_output(self):
        rows = [["Alice", 30], ["Bob", 25]]
        result = tabulate(rows, headers=["Name", "Age"])
        lines = result.split("\n")
        assert len(lines) == 4  # header + separator + 2 data rows
        assert "Name" in lines[0]
        assert "---" in lines[1] or "---" in lines[1].replace(" ", "")
        assert "Alice" in lines[2]

    def test_none_renders_as_dash(self):
        rows = [["x", None]]
        result = tabulate(rows, headers=["A", "B"])
        assert "-" in result.split("\n")[2]

    def test_truncation(self):
        rows = [["a" * 200]]
        result = tabulate(rows, headers=["Col"], max_width=10)
        lines = result.split("\n")
        # Data row should be truncated
        assert "\u2026" in lines[2]
        assert len(lines[2].strip()) <= 10

    def test_empty_rows(self):
        result = tabulate([], headers=["X", "Y"])
        lines = result.split("\n")
        # header + separator only
        assert len(lines) == 2
        assert "X" in lines[0]


class TestNormalizePatterns:
    def test_none(self):
        assert normalize_patterns(None) is None

    def test_empty_string(self):
        assert normalize_patterns("") is None

    def test_single_pattern(self):
        assert normalize_patterns("*.bin") == ["*.bin"]

    def test_comma_separated(self):
        assert normalize_patterns("*.bin, *.safetensors") == ["*.bin", "*.safetensors"]

    def test_list_passthrough(self):
        assert normalize_patterns(["a", "b"]) == ["a", "b"]

    def test_list_with_inline_comma(self):
        assert normalize_patterns(["a, b", "c"]) == ["a", "b", "c"]

    def test_empty_list(self):
        assert normalize_patterns([]) is None


class TestExtractCommonPrefix:
    def test_single_pattern_with_dir(self):
        assert extract_common_prefix(["TacExo/*"]) == "TacExo"

    def test_nested_dir(self):
        assert extract_common_prefix(["data/train/*.parquet"]) == "data/train"

    def test_multiple_patterns_common(self):
        assert extract_common_prefix(["data/train/*", "data/valid/*"]) == "data"

    def test_no_dir_prefix(self):
        assert extract_common_prefix(["*.safetensors"]) is None

    def test_different_top_dirs(self):
        assert extract_common_prefix(["TacExo/*", "OtherDir/*"]) is None

    def test_none_input(self):
        assert extract_common_prefix(None) is None


class TestEncodeMediaToBase64:
    def test_success(self, tmp_path):
        img = tmp_path / "test.png"
        # Minimal valid PNG (1x1 transparent pixel)
        img.write_bytes(
            b"\x89PNG\r\n\x1a\n"
            b"\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
            b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89"
            b"\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01"
            b"\r\n\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
        )
        result = encode_media_to_base64(img)
        assert result.startswith("data:image/png;base64,")

    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            encode_media_to_base64(tmp_path / "nonexist.png")

    def test_directory_raises(self, tmp_path):
        with pytest.raises(ValueError):
            encode_media_to_base64(tmp_path)


class TestParseTimestamp:
    def test_none(self):
        assert parse_timestamp(None) is None

    def test_unix_int(self):
        dt = parse_timestamp(0)
        assert isinstance(dt, datetime)
        assert dt.tzinfo is not None
        # UNIX 0 is 1970-01-01 00:00:00 UTC
        assert dt.astimezone(timezone.utc).year == 1970

    def test_utc_string(self):
        dt = parse_timestamp("2024-01-01T00:00:00Z")
        assert dt is not None
        assert dt.tzinfo is not None

    def test_nanosecond_truncation(self):
        dt = parse_timestamp("2024-01-01T00:00:00.123456789Z")
        assert dt is not None
        assert dt.microsecond == 123456

    def test_naive_string(self):
        dt = parse_timestamp("2024-01-01 10:30:00")
        assert dt is not None
        assert dt.tzinfo is not None
        sh_tz = zoneinfo.ZoneInfo("Asia/Shanghai")
        assert dt.tzinfo == sh_tz

    def test_datetime_passthrough(self):
        original = datetime(2024, 6, 1, 12, 0, tzinfo=timezone.utc)
        result = parse_timestamp(original)
        assert result is original

    def test_invalid_string(self):
        with pytest.raises(ValueError):
            parse_timestamp("not-a-date")

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            parse_timestamp([1, 2, 3])
