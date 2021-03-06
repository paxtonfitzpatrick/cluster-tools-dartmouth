class ClusterToolsError(Exception):
    pass


class SSHConnectionError(ClusterToolsError):
    def __init__(self, message: str = ''):
        super().__init__(message)
        self.message = message


class SSHProcessError(ClusterToolsError, ProcessLookupError):
    def __init__(self, message: str = ''):
        super().__init__(message)
        self.message = message


class ClusterToolsProjectError(ClusterToolsError):
    def __init__(self, message: str = ''):
        super().__init__(message)
        self.message = message


class ProjectConfigurationError(ClusterToolsProjectError):
    def __init__(self, message: str = ''):
        super().__init__(message)
        self.message = message