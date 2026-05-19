"""
Poor Man's Configurator. Probably a terrible idea. Example usage:
$ python train.py config/override_file.py --batch_size=32
this will first run config/override_file.py, then override batch_size to 32

The code in this file will be run as follows from e.g. train.py:
>>> exec(open('configurator.py').read())

So it's not a Python module, it's just shuttling this code away from train.py
The code in this script then overrides the globals()

I know people are not going to love this, I just really dislike configuration
complexity and having to prepend config. to every single variable. If someone
comes up with a better simple Python solution I am all ears.
"""

import sys
from ast import literal_eval

# short flag aliases: -x maps to the equivalent --long_key
SHORT_FLAGS = {
    '-d': 'dataset',
}

def _set_key(key, val):
    if key not in globals():
        raise ValueError(f"Unknown config key: {key}")
    try:
        attempt = literal_eval(val)
    except (SyntaxError, ValueError):
        attempt = val
    assert type(attempt) == type(globals()[key]), \
        f"Type mismatch for {key}: expected {type(globals()[key])}, got {type(attempt)}"
    print(f"Overriding: {key} = {attempt}")
    globals()[key] = attempt

args = sys.argv[1:]
idx = 0
while idx < len(args):
    arg = args[idx]
    if arg in SHORT_FLAGS:
        # -d value  (two-token form)
        idx += 1
        if idx >= len(args):
            raise ValueError(f"Short flag {arg} requires a value")
        _set_key(SHORT_FLAGS[arg], args[idx])
    elif '=' not in arg:
        # assume it's the name of a config file
        assert not arg.startswith('--'), f"Unknown flag: {arg}"
        config_file = arg
        print(f"Overriding config with {config_file}:")
        with open(config_file) as f:
            print(f.read())
        exec(open(config_file).read())
    else:
        # --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=', 1)
        key = key[2:]
        _set_key(key, val)
    idx += 1
