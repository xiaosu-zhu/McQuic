"""
`get_install_command`, `get_install_paths`
are export from
https://github.com/jythontools/wheel

"wheel" copyright (c) 2012-2014 Daniel Holth <dholth@fastmail.fm> and
contributors.
"""
import sys
import os
import distutils.dist as dist
import distutils.command.install as install

def get_install_command(name):
    # late binding due to potential monkeypatching
    d = dist.Distribution({'name':name})
    i = install.install(d)
    i.finalize_options()
    return i

def get_install_paths(name):
    """
    Return the (distutils) install paths for the named dist.

    A dict with ('purelib', 'platlib', 'headers', 'scripts', 'data') keys.
    """
    paths = {}

    i = get_install_command(name)

    for key in install.SCHEME_KEYS:
        paths[key] = getattr(i, 'install_' + key)

    # pip uses a similar path as an alternative to the system's (read-only)
    # include directory:
    if hasattr(sys, 'real_prefix'):  # virtualenv
        paths['headers'] = os.path.join(sys.prefix,
                                        'include',
                                        'site',
                                        'python' + sys.version[:3],
                                        name)

    return paths

from distutils.core import setup

distribution = setup()


# mcquic
PKG_NAME = distribution.get_option_dict("metadata")["name"][1]


def parseEntryPoints(entryPoints):
    result = dict()
    entries = entryPoints.strip().split("\n")
    for ent in entries:
        if len(ent) < 1:
            continue
        key, value = ent.strip().split("=")
        result[key.strip()] = value.strip()
    return result


# {'mcquic': 'mcquic.cli:entryPoint', 'mcquic-train': 'mcquic.train.cli:entryPoint'}
ENTRY_POINTS = parseEntryPoints(distribution.get_option_dict("options.entry_points")["console_scripts"][1])


BIN_PATH = get_install_paths(PKG_NAME)["scripts"]

# Add `-O` (Optimizer mode) arg to shebang
# if WINDOWS:
#     targetPY = f"{entry}-script.py"
# else:
#     targetPY = f"{entry}"
