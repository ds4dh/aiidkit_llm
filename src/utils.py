import json
import os
import re
import sys
from pathlib import Path
import yaml

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class _UTF8LogTee:
    """Line-buffered UTF-8 log tee that strips binary/control bytes so editors always open it as text.

    Fault-tolerant: if the secondary log file hits a transient filesystem/quota error (e.g. ENOSPC on /home),
    logging degrades gracefully to the primary stream without crashing training or masking stderr tracebacks.
    """

    def __init__(self, stream, log_file):
        self._stream = stream
        self._log_file = log_file
        self._disabled = False

    def write(self, data):
        res = self._stream.write(data)
        if data and not self._disabled and self._log_file is not None:
            try:
                cleaned = _ANSI_RE.sub("", str(data)).replace("\x00", "").replace("\r\n", "\n")
                wrote = False
                for chunk in cleaned.split("\r"):
                    if not chunk or not chunk.strip():
                        continue
                    if "%|" in chunk and "100%|" not in chunk:
                        continue
                    line = chunk.lstrip(" ") if chunk.startswith("          ") else chunk
                    if not line.endswith("\n"):
                        line += "\n"
                    self._log_file.write(line)
                    wrote = True
                if wrote:
                    self._log_file.flush()
            except OSError as exc:
                self._disabled = True
                try:
                    self._stream.write(f"\n[WARN] Disabling secondary UTF-8 log tee due to OSError: {exc}\n")
                    self._stream.flush()
                except Exception:
                    pass
        return res

    def flush(self):
        try:
            self._stream.flush()
        except OSError:
            pass
        if not self._disabled and self._log_file is not None:
            try:
                self._log_file.flush()
            except OSError:
                self._disabled = True

    def isatty(self):
        return False

    def __getattr__(self, name):
        return getattr(self._stream, name)


def setup_utf8_log_tee(log_path: str | Path | None = None):
    """If log_path or AIIDKIT_LOG_FILE is set, tee stdout and stderr to a clean UTF-8 text file."""
    target = log_path or os.environ.get("AIIDKIT_LOG_FILE")
    if not target or isinstance(sys.stdout, _UTF8LogTee):
        return
    p = Path(target)
    p.parent.mkdir(parents=True, exist_ok=True)
    log_f = open(p, "a", encoding="utf-8", newline="\n", buffering=1)
    sys.stdout = _UTF8LogTee(sys.stdout, log_f)
    sys.stderr = _UTF8LogTee(sys.stderr, log_f)


setup_utf8_log_tee()


def apply_config_overrides(config, overrides_input):
    """Recursively updates config dictionary with overrides."""
    if not overrides_input:
        return config

    if isinstance(overrides_input, list):
        cleaned_list = [x for x in overrides_input if x not in ("--overrides", "-o")]
        if not cleaned_list:
            return config

        # Check if single joined string parses as dict (e.g. space-split JSON/YAML)
        joined_str = " ".join(str(x) for x in cleaned_list).strip("'\"")
        parsed_joined = None
        try:
            parsed_joined = json.loads(joined_str)
        except Exception:
            try:
                p = yaml.safe_load(joined_str)
                if isinstance(p, dict):
                    parsed_joined = p
            except Exception:
                pass

        if isinstance(parsed_joined, dict):
            overrides_input = [parsed_joined]
        else:
            for item in cleaned_list:
                config = apply_config_overrides(config, item)
            return config

    if isinstance(overrides_input, list) and len(overrides_input) == 1 and isinstance(overrides_input[0], dict):
        overrides = overrides_input[0]
        overrides_str = ""
    else:
        overrides_str = str(overrides_input).strip("'\"")
        if not overrides_str or overrides_str == "{}":
            return config
        overrides = {}

    if not overrides:
        try:
            overrides = json.loads(overrides_str)
        except Exception:
            try:
                parsed = yaml.safe_load(overrides_str)
                if isinstance(parsed, dict):
                    overrides = parsed
            except Exception:
                pass

    if not overrides and "=" in overrides_str:
        overrides = {}
        for item in overrides_str.split():
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    v = json.loads(v)
                except Exception:
                    try:
                        v = yaml.safe_load(re.sub(r":(?=\S)", ": ", v))
                    except Exception:
                        pass
                overrides[k.strip()] = v

    def update(d, u):
        for k, v in u.items():
            if "." in str(k):
                head, tail = str(k).split(".", 1)
                d[head] = update(d.get(head, {}) if isinstance(d.get(head), dict) else {}, {tail: v})
            elif isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(config, overrides)


def load_optuna_json_overrides(config: dict, optuna_json_paths: list[str] | None) -> dict:
    """Apply `portable_overrides` from one or more exported Optuna HPO JSON files onto `config`."""
    for opt_json_path in (optuna_json_paths or []):
        if not opt_json_path:
            continue
        p = Path(opt_json_path)
        if not p.exists():
            raise FileNotFoundError(f"--optuna-json file not found: {p}")
        with open(p, "r", encoding="utf-8") as f:
            payload = json.load(f)
        portable = payload.get("portable_overrides", payload)
        print(f">>> Applying Optuna HPO overrides from: {p}")
        config = apply_config_overrides(config, [portable])
    return config
