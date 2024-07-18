from abc import ABC


class BaseAsset(ABC):
    def __init__(self, asset_name, asset_file, loading_options):
        self.name = asset_name
        self.file = asset_file
        # save loading options as a class instance
        self.options = type("LoadingOptions", (object,), loading_options)

    def load_from_file(self, asset_file):
        raise NotImplementedError("load_from_file not implemented")
