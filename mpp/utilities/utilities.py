from typing import Union
from pathlib import Path

base_dir = Path(__file__).resolve().parent.parent.parent


class SimgCmd:
    def __init__(self, simg: Union[str, None], work_dir: Path,
                 out_dir: Path) -> None:
        if simg is None:
            self.cmd = None
        else:
            self.cmd = (f'singularity run -B {work_dir}:{work_dir},'
                        f'{out_dir}:{out_dir},{base_dir}:{base_dir}')
            self._simg = simg

    def run_cmd(self, cmd: str, options: Union[str, None] = None) -> str:
        if self.cmd is None:
            run_cmd = cmd
        else:
            if options is None:
                run_cmd = f'{self.cmd} {self._simg} {cmd}'
            else:
                run_cmd = f'{self.cmd} {options} {self._simg} {cmd}'

        return run_cmd
