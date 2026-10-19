"""Source header boundaries and physical-line coordinates; invented values only."""
import ast

import pytest


class TestSecretHeaderSourceReferences:
    @pytest.mark.parametrize("header", [
        "x-api-key", "x-goog-api-key", "api-key", "apikey",
        "x-api-token", "x-auth-token", "x-access-token",
    ])
    def test_source_reference_boundaries_keep_literal_headers_protected(self, header):
        from agent import redact as R

        for text in (
            "out = '" + header + ": '+reference\n",
            "send('" + header + ": ' + session_credential + '\\n')\n",
            "clé = '" + header + ": '+clé\n",
            "out='curl -H \"" + header + ": '+reference+'\"\\n'\n",
        ):
            assert R.redact_sensitive_text(text, force=True, file_read=True) == text
        secret = "opaque-synthetic-0123456789AzBy"
        for literal in (
            header + ": " + secret, header + ': "' + secret + '"',
            header + ": '" + secret + "'",
            'curl -H "' + header + ': ' + secret + '" https://example.invalid',
            "curl -H '" + header + ": " + secret + "' https://example.invalid",
        ):
            assert secret not in R.redact_sensitive_text(literal, force=True, file_read=True)
        for gap in ("", " "):
            first = "send('" + header + ": '+reference)"
            text = first + ";send(" + repr(header + ":" + gap + secret) + ")\n"
            result = R.redact_sensitive_text(text, force=True, file_read=True)
            assert result.startswith(first)
            assert secret not in result
        for text in (
            "out = '" + header + ": '+reference+'literal'\n",
            "out = '" + header + ": prefix'+reference\n",
            "out = '" + header + ": '+obj.reference\n",
            "out = '" + header + ": '+str(reference)\n",
            "out = '" + header + ": '+reference+\n",
        ):
            assert R.redact_sensitive_text(text, force=True, file_read=True) != text
        pure = "out='" + header + ": '+reference\n"
        assert R.redact_sensitive_text(pure, force=True, file_read=True, secret_file=True) != pure
        assert R.redact_sensitive_text(pure, force=True) != pure
        try:
            R.register_vault_redaction_value("reference")
            assert "reference" not in R.redact_sensitive_text(pure, force=True, file_read=True)
        finally:
            R.clear_vault_redaction_values()

    @pytest.mark.parametrize("separator", ["\u2028", "\u2029", "\v", "\f", "\x85", "\x1c", "\x1d", "\x1e"])
    @pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"])
    @pytest.mark.parametrize("prefix", ["", "é" * 9, "αβ鍵"])
    def test_source_reference_coordinates_follow_python_physical_lines(self, separator, newline, prefix):
        from agent import redact as R

        secret = "secret-literal-0123456789"
        pure = "'X-Api-Key: '+token"
        text = "#" + separator + prefix + " X-Api-Key: '" + secret + "'" + newline + pure + newline
        ast.parse(text)
        result = R.redact_sensitive_text(text, force=True, file_read=True)
        assert secret not in result
        assert pure in result
        assert result.count(newline) == text.count(newline)

    @pytest.mark.parametrize("volume", [16, 64, 256, 1024, 4096])
    def test_dense_reference_work_does_not_rescan_line_prefixes_or_values(self, monkeypatch, volume):
        import sys
        from agent import redact as R

        text = "v='X-Api-Key:'+reference;" * volume
        pattern = R._SECRET_HEADER_RE
        work = {"whole_line_encodes": 0, "matched_value_characters": 0}

        class ObservedPattern:
            def record(self, match):
                if match is not None:
                    work["matched_value_characters"] += match.end(2) - match.start(2)
                return match

            def search(self, *args, **kwargs):
                return self.record(pattern.search(*args, **kwargs))

            def match(self, *args, **kwargs):
                return self.record(pattern.match(*args, **kwargs))

            def finditer(self, *args, **kwargs):
                for match in pattern.finditer(*args, **kwargs):
                    yield self.record(match)

            def __getattr__(self, name):
                return getattr(pattern, name)

        def observe(frame, event, arg):
            if (event == "c_call" and getattr(arg, "__name__", "") == "encode"
                    and getattr(arg, "__self__", None) == text):
                work["whole_line_encodes"] += 1

        monkeypatch.setattr(R, "_SECRET_HEADER_RE", ObservedPattern())
        previous = sys.getprofile()
        try:
            sys.setprofile(observe)
            result = R.redact_sensitive_text(text, force=True, file_read=True)
        finally:
            sys.setprofile(previous)
        assert result == text
        # Work bounds, not wall-clock thresholds: each physical line once and
        # no repeated greedy traversal of the same long value suffix.
        assert work["whole_line_encodes"] <= 1
        assert work["matched_value_characters"] <= len(text)

    @pytest.mark.parametrize("volume", [16, 64, 256, 1024, 4096])
    @pytest.mark.parametrize("atom", ["v='X-Api-Key:'+reference;", "clé='X-Api-Key:'+clé;"])
    @pytest.mark.parametrize("newline", ["", "\r\n"])
    def test_dense_references_keep_a_final_literal_protected(self, volume, atom, newline):
        from agent import redact as R

        pure = newline.join([atom] * volume)
        secret = "opaque-synthetic-final-0123456789AzBy"
        text = pure + newline + "z='X-Api-Key:" + secret + "';"
        result = R.redact_sensitive_text(text, force=True, file_read=True)
        assert result.startswith(pure)
        assert secret not in result
