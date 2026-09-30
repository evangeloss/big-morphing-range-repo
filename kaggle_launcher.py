"""Paste into one Kaggle cell, or import the accompanying notebook."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import urlparse

REPO_URL='https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git'
BRANCH=''
PROJECT_SUBDIR=''             # Auto-detect the folder containing train.py and bridge.py.
QUICK=False                   # True = tiny end-to-end wiring check, not scientific evidence.
LARGE_ALPHA=0.5
SMALL_ALPHA=0.02
SEEDS=[11]                    # Four models. Use [11,22,33] for a 12-model replication.
MAX_EPOCHS=100
STEPS_PER_EPOCH=64
BATCH_SIZE=32
VALIDATION_SCENES=256
TEST_SCENES=300
QUADRATURE_ATOMS=1024
BRIDGE_RANK=384
RESUME_OUTPUT=''              # Existing /kaggle/working/.../results folder; identical settings required.


def main():
    u=urlparse(REPO_URL)
    if (u.scheme!='https' or u.hostname!='github.com' or u.username or u.password or u.query or u.fragment
        or 'YOUR_' in REPO_URL or len(u.path.strip('/').split('/'))!=2):
        raise ValueError('Set REPO_URL to your public GitHub repository URL, not a file URL.')
    working=Path('/kaggle/working')
    if not working.exists():raise RuntimeError('This launcher runs in Kaggle.')
    folder=Path(tempfile.mkdtemp(prefix='large_morphing_',dir=working));repo=folder/'repo'
    output=Path(RESUME_OUTPUT).resolve() if RESUME_OUTPUT else folder/'results'
    if RESUME_OUTPUT and (not output.is_relative_to(working.resolve()) or not output.is_dir()):
        raise ValueError('RESUME_OUTPUT must be an existing results folder under /kaggle/working')
    env=os.environ.copy();env.update(GIT_TERMINAL_PROMPT='0',PYTHONUNBUFFERED='1',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    command=['git','clone','--depth','1']+(['--branch',BRANCH] if BRANCH else [])+['--',REPO_URL,str(repo)]
    subprocess.run(command,env=env,check=True)
    if PROJECT_SUBDIR:
        project=(repo/PROJECT_SUBDIR).resolve()
        if not project.is_relative_to(repo.resolve()):raise ValueError('Invalid project folder')
        candidates=[project] if (project/'bridge.py').is_file() and (project/'train.py').is_file() else []
    else:candidates=[p.parent for p in repo.rglob('train.py') if (p.parent/'bridge.py').is_file() and (p.parent/'simulator.py').is_file()]
    if len(candidates)!=1:raise RuntimeError('Upload the extracted project files, not only the ZIP. Set PROJECT_SUBDIR if multiple copies exist.')
    project=candidates[0]
    subprocess.run([sys.executable,'-m','pip','install','--quiet','-r',str(project/'requirements.txt')],check=True)
    import torch
    print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU (enable GPU for full training)')
    subprocess.run([sys.executable,str(project/'verify.py')],cwd=project,env=env,check=True)
    command=[sys.executable,'-u',str(project/'train.py'),'--output',str(output),
             '--large-alpha',str(LARGE_ALPHA),'--small-alpha',str(SMALL_ALPHA),'--epochs',str(MAX_EPOCHS),
             '--steps',str(STEPS_PER_EPOCH),'--batch-size',str(BATCH_SIZE),'--validation-scenes',str(VALIDATION_SCENES),
             '--test-scenes',str(TEST_SCENES),'--atoms',str(QUADRATURE_ATOMS),'--rank',str(BRIDGE_RANK),'--seeds',*map(str,SEEDS)]
    if QUICK:command+=['--quick']
    if RESUME_OUTPUT:command+=['--resume']
    success=False
    try:
        with (folder/'run.log').open('w',encoding='utf-8') as log:
            process=subprocess.Popen(command,cwd=project,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            try:
                for line in process.stdout:print(line,end='',flush=True);log.write(line);log.flush()
                code=process.wait()
            finally:
                if process.poll() is None:process.terminate();process.wait()
            if code:raise subprocess.CalledProcessError(code,command)
        success=True
    finally:
        output.mkdir(exist_ok=True)
        shutil.copy2(folder/'run.log',output/f'launcher_{folder.name}.log')
        provenance={'repo':REPO_URL,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
                    'completed':success,'resume':bool(RESUME_OUTPUT),'command':command}
        (output/f'provenance_{folder.name}.json').write_text(json.dumps(provenance,indent=2))
        # Compact report ZIP omits optimizer/model weights; all checkpoints remain in results.
        import zipfile
        with zipfile.ZipFile(folder/'results_for_review.zip','w',zipfile.ZIP_DEFLATED) as z:
            for f in output.rglob('*'):
                if f.is_file() and f.suffix in ('.csv','.json','.md','.log','.png'):z.write(f,f.relative_to(output))
        print('\nReview ZIP:',folder/'results_for_review.zip')
        print('Checkpoints and resume folder:',output)
        from IPython.display import FileLink,display
        display(FileLink(str((folder/'results_for_review.zip').relative_to(working))))
    print('\nPaste back:\n'+(output/'PASTE_BACK.md').read_text())


if __name__=='__main__':main()
