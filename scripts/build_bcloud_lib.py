import sys
import os
import shutil

dirname = sys.argv[1]
bc_dirname = sys.argv[2]

if os.path.exists(bc_dirname):
    raise Exception("Path {} is already exists.".format(bc_dirname))

os.makedirs(bc_dirname)

# copy include files
shutil.copytree(os.path.join(dirname, "include"), os.path.join(bc_dirname, "include"))

# copy libraries
shutil.copytree(os.path.join(dirname, "lib"), os.path.join(bc_dirname, "lib"))

third_libs = os.path.join(dirname, "third_libs")

for root, dirs, files in os.walk(third_libs):
    for f in files:
        if f.strip().count(".so") > 0 or f.strip() == "plugins.xml":
            full_path = os.path.join(root, f)
            shutil.copy(full_path, os.path.join(bc_dirname, "lib"), follow_symlinks=False)  
