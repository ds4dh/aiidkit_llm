import json
import yaml

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
                try: v = json.loads(v)
                except Exception: pass
                overrides[k.strip()] = v

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(config, overrides)