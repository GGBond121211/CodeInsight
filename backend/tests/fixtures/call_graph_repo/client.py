from transports import BaseTransport


class Client:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def _transport_for_url(self) -> BaseTransport:
        raise NotImplementedError

    def request(self, method: str) -> str:
        return self._send_single_request(method)

    def _send_single_request(self, method: str) -> str:
        transport = self._transport_for_url()
        return transport.handle_request(method)
