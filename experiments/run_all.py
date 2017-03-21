from pathlib import Path
from subprocess import run


for param_file in Path('.').glob('*.json'):
    print(f'Run {param_file.stem}')
    run(['python', 'run.py', str(param_file)])
