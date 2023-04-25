class DatasetError(Exception):
    """Raised when input dataset value is wrong"""
    def __init__(self, message="dataset must be 'HCP-A' or 'HCP-D'"):
        self.message = message
        super().__init__(self.message)
