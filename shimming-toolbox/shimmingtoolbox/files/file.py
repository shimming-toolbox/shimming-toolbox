import nibabel as nib
from __future__ import annotations
from functools import wraps


def safe_getter(default_value=None):
    """Decorator that catches errors in getter functions and returns a default value."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                return func(self, *args, **kwargs)
            except Exception as e:
                print(f"Error in {func.__name__}: {e}")
                # terminate the program if the error is critical
                if isinstance(e, (NameError, ValueError)):
                    raise e
                return default_value
        return wrapper
    return decorator


class NiftiFile:
    def __init__(self, filename_nii, filename_json=None):
        self.filename = None
        self.data = None
        self.header = None

    def load(self):
        # Placeholder for loading NIfTI file data and header
        pass

    def save(self):
        # Placeholder for saving NIfTI file data and header
        pass
    
    def find_json(self):
        # Placeholder for finding associated JSON file
        pass
    
    @safe_getter(default_value="unknown")
    def get_extension(self):
        # Placeholder for getting file extension
        if self.filename.endswith('.nii.gz'):
            return 'nii.gz'
        else:
            raise NameError("Unsupported file extension for file " + self.filename)
    
    @safe_getter(default_value="")
    def get_path(self):
        # takes the whole path except the filename
        path = self.filename.rsplit('/', 1)[0] if '/' in self.filename else ''
        
        if not path:
            raise ValueError("Path is empty for file " + self.filename)
        
        return path