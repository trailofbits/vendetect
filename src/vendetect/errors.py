class VendetectError(Exception):
    pass


class VendetectRuntimeError(RuntimeError, VendetectError):
    pass
