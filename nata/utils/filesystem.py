from pathlib import Path

def get_files(path=None, pattern=None):
    if pattern is not None:
        files = path.glob(pattern)
    else:
        files = path.glob('*')
    
    # todo: check that returned results are files
    # for file in files:
    #     if not file.is_file():
    #         del file
    
    return files

def get_dirs(path=None, pattern=None):
    if pattern is not None:
        dirs = path.glob(pattern)
    else:
        dirs = path.glob('*')
    
    # todo: check that returned results are dirs
    # for dir in dirs:
    #     if not file.is_file():
    #         del file
    
    return dirs