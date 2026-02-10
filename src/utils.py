import json

def apply_config_overrides(config, overrides_str):
    """Recursively updates config dictionary with overrides."""
    try:
        overrides = json.loads(overrides_str)
    except json.JSONDecodeError:
        print("Error decoding overrides JSON. Using default config.")
        return config

    def update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = update(d.get(k, {}), v)
            else:
                d[k] = v
        return d

    return update(config, overrides)