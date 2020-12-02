from os.path import expanduser, expandvars, realpath
from pathlib import Path
from typing import overload

from clustertools.shared.typing import PathLike


@overload
def cleanpath(path: str) -> str:
    ...
@overload
def cleanpath(path: Path) -> Path:
    ...
def cleanpath(path: PathLike) -> PathLike:
    return type(path)(str(realpath(expanduser(expandvars(path)))))