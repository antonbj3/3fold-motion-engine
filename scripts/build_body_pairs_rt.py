"""Build the optional RT backend using external CUDA and OptiX headers."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
root=Path(__file__).resolve().parents[1]
native=root/'src/motion_engine/body_pairs_rt_native'
output=Path(sys.argv[1]);output.mkdir(parents=True,exist_ok=True)
include=Path(os.environ['OPTIX_INCLUDE']).resolve()
cuda=Path(shutil.which('nvcc')).resolve().parents[1]
subprocess.run(['g++','-shared','-fPIC','-pthread','-O2','-std=c++17',
    '-I'+str(include),'-I'+str(cuda/'include'),str(native/'api.cpp'),
    '-L'+str(cuda/'lib64'),'-Wl,-rpath,'+str(cuda/'lib64'),'-lcudart','-ldl','-o',str(output/'pairs.so')],check=True)
subprocess.run(['nvcc','-ptx','--fmad=false','--ftz=false','-std=c++17','-arch=compute_89',
    '-I'+str(include),str(native/'program.cu'),'-o',str(output/'pairs.ptx')],check=True)
