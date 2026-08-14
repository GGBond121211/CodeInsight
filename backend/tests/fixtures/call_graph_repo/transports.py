class BaseTransport:
    def handle_request(self, method: str) -> str:
        raise NotImplementedError


class HTTPTransport(BaseTransport):
    def handle_request(self, method: str) -> str:
        return method


class OtherTransport(BaseTransport):
    def handle_request(self, method: str) -> str:
        return method.lower()
