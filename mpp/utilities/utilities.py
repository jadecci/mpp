from typing import Union
from pathlib import Path


class SimgCmd:
    def __init__(self, simg: Union[Path, None], work_dir: Path) -> None:
        if simg is None:
            self.cmd = None
        else:
            self.cmd = f'singularity run -B {work_dir}:{work_dir} {simg}'

    def run_cmd(self, cmd: str) -> str:
        if self.cmd is None:
            return cmd
        else:
            return f'{self.cmd} {cmd}'
